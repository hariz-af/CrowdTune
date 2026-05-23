import csv
import math
import re
import sys
import traceback
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from crowd_physics import CrowdPhysicsAnalyzer, compute_regime_tendency
from frame_source import VideoFileSource
from instability_forecast import create_forecaster
from lbp_density import LBPDensity
from risk_index import RiskIndex
from scene_profile import SceneProfile
from stability_index import StabilityIndex
from tune_config import DENSITY, FORECAST, FUSION, REGIME, STABILITY

try:
    from scipy.stats import pearsonr, spearmanr
except Exception:
    pearsonr = None
    spearmanr = None


def _safe_corrcoef(x, y):
    if len(x) < 2 or len(y) < 2:
        return float('nan')
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if np.std(x_arr) == 0.0 or np.std(y_arr) == 0.0:
        return float('nan')
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _rankdata(values):
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr)
    ranks = np.empty(len(arr), dtype=float)
    i = 0
    while i < len(arr):
        j = i
        while j + 1 < len(arr) and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks


def compute_pearson(x, y):
    if pearsonr is not None:
        try:
            r, p = pearsonr(x, y)
            return float(r), float(p)
        except Exception:
            pass
    return _safe_corrcoef(x, y), float('nan')


def compute_spearman(x, y):
    if spearmanr is not None:
        try:
            rho, p = spearmanr(x, y)
            return float(rho), float(p)
        except Exception:
            pass
    return _safe_corrcoef(_rankdata(x), _rankdata(y)), float('nan')


def infer_direction(delta, threshold):
    if delta > threshold:
        return 'Escalating'
    if delta < -threshold:
        return 'Dissipating'
    return 'Stable'


def trailing_rolling_mean(values, window):
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return arr
    window = max(1, int(window))
    out = np.empty(len(arr), dtype=float)
    for idx in range(len(arr)):
        start = max(0, idx - window + 1)
        out[idx] = float(np.mean(arr[start:idx + 1]))
    return out


def compute_forecast_lead_times(valid_rows, stable_threshold, horizon_frames, trend_window):
    if len(valid_rows) < (trend_window * 2 + 1):
        return [], {}

    rolling_current = np.asarray([float(row['Rolling_CCI_5']) for row in valid_rows], dtype=float)
    spike_threshold = float(np.percentile(rolling_current, 90))
    lead_times = []
    lead_time_by_frame = {}
    max_lookback = max(horizon_frames * 3, trend_window * 2)

    for idx in range(trend_window, len(valid_rows) - trend_window):
        value = rolling_current[idx]
        if value < spike_threshold:
            continue
        if value < rolling_current[idx - 1] or value < rolling_current[idx + 1]:
            continue

        earliest_trigger = None
        search_start = max(0, idx - max_lookback)
        for j in range(search_start, idx):
            forecast_delta = float(valid_rows[j]['Projected_CCI']) - float(valid_rows[j]['Rolling_CCI_5'])
            if forecast_delta > stable_threshold:
                earliest_trigger = j
                break

        if earliest_trigger is not None:
            lead_time = idx - earliest_trigger
            lead_times.append(lead_time)
            spike_frame = int(valid_rows[idx]['frame'])
            lead_time_by_frame[spike_frame] = float(lead_time)

    return lead_times, lead_time_by_frame


def compute_forecast_validation(rows, horizon_frames, stable_threshold, trend_window=5):
    valid_rows = []
    if horizon_frames <= 0:
        return valid_rows, []

    cci_series = [float(row['CCI']) for row in rows]
    rolling_cci = trailing_rolling_mean(cci_series, trend_window)

    for idx in range(0, len(rows) - horizon_frames):
        row = dict(rows[idx])
        projected_cci = float(row['Projected_CCI'])
        actual_future_cci = float(rows[idx + horizon_frames]['CCI'])
        current_trend_value = float(rolling_cci[idx])
        actual_future_trend_value = float(rolling_cci[idx + horizon_frames])
        actual_delta = actual_future_trend_value - current_trend_value
        forecast_delta = projected_cci - current_trend_value

        row['Actual_Future_CCI'] = actual_future_cci
        row['Forecast_Error'] = projected_cci - actual_future_cci
        row['Rolling_CCI_5'] = current_trend_value
        row['Rolling_Future_CCI_5'] = actual_future_trend_value
        row['Forecast_Direction'] = infer_direction(forecast_delta, stable_threshold)
        row['Actual_Direction'] = infer_direction(actual_delta, stable_threshold)
        row['Lead_Time_Frames'] = ''
        valid_rows.append(row)

    lead_times, lead_time_by_frame = compute_forecast_lead_times(
        valid_rows,
        stable_threshold,
        horizon_frames,
        trend_window,
    )
    for row in valid_rows:
        frame_id = int(row['frame'])
        if frame_id in lead_time_by_frame:
            row['Lead_Time_Frames'] = lead_time_by_frame[frame_id]

    return valid_rows, lead_times


class ValidationWorker(QThread):
    progress_signal = pyqtSignal(int)
    status_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, video_path: str, case_name: str, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.case_name = case_name
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    def run(self):
        try:
            self.status_signal.emit('Initializing analysis pipeline...')
            # Use a temporary directory for outputs; UI will export on user action.
            import tempfile
            out_dir_path = tempfile.mkdtemp(prefix=f"{self.case_name}_")
            out_dir = Path(out_dir_path)

            source = VideoFileSource(self.video_path)
            total_frames = int(source.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if hasattr(source, 'cap') else 0

            density_engine = LBPDensity(
                radius=DENSITY.radius,
                n_points=DENSITY.n_points,
                grid_size=DENSITY.grid_size,
            )
            crowd_engine = CrowdPhysicsAnalyzer()
            risk_engine = RiskIndex()
            stability_engine = StabilityIndex(window=STABILITY.window)
            forecast_engine = create_forecaster(FORECAST)

            profile = SceneProfile.load_for_video(self.video_path, Path(__file__).resolve().parent / 'scene_profiles')
            spatial_mask = None

            regime_g = REGIME.initial_gas
            regime_f = REGIME.initial_fluid
            regime_gr = REGIME.initial_granular
            warmup_frames = max(STABILITY.window, int(FORECAST.signal_fps * 2))
            horizon_frames = max(1, int(round(FORECAST.horizon_seconds * FORECAST.signal_fps)))

            input_fps = 0.0
            if hasattr(source, 'cap'):
                input_fps = float(source.cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if input_fps <= 0.0:
                input_fps = float(FORECAST.signal_fps)

            rows = []
            frame_idx = 0

            while not self._stop_requested:
                ret, frame = source.read()
                if not ret:
                    break

                if profile is not None and (
                    spatial_mask is None or spatial_mask.shape[:2] != frame.shape[:2]
                ):
                    spatial_mask = profile.get_spatial_analysis_mask(frame.shape)

                _, s_raw = density_engine.compute(frame, analysis_mask=spatial_mask)
                s_norm = (float(s_raw) - DENSITY.s_min) / (DENSITY.s_max - DENSITY.s_min)
                s_norm = float(np.clip(s_norm, 0.0, 1.0) ** DENSITY.s_gamma)

                metrics = crowd_engine.update(frame, analysis_mask=spatial_mask)
                k_val = float(metrics.get('K', 0.0))
                c_val = float(metrics.get('C', 0.0))

                r_gas, r_fluid, r_granular = compute_regime_tendency(s_norm, c_val)
                alpha = REGIME.smooth_alpha
                regime_g = alpha * r_gas + (1 - alpha) * regime_g
                regime_f = alpha * r_fluid + (1 - alpha) * regime_f
                regime_gr = alpha * r_granular + (1 - alpha) * regime_gr

                cci = s_norm * (FUSION.k_weight * k_val + FUSION.incoherence_weight * (1.0 - c_val))
                cci = float(np.clip(cci, 0.0, 1.0))
                si = float(stability_engine.update(cci))
                projected_cci, slope = forecast_engine.update(cci)
                projected_cci = float(np.clip(projected_cci, 0.0, 1.0))
                slope = float(slope)

                risk = risk_engine.update(cci_raw=cci)
                risk_value = float(risk.get('risk', 0.0))

                if frame_idx >= warmup_frames:
                    rows.append({
                        'frame': frame_idx,
                        'time_s': frame_idx / max(1.0, input_fps),
                        'S': s_norm,
                        'K': k_val,
                        'C': c_val,
                        'CCI': cci,
                        'Risk': risk_value,
                        'SI': si,
                        'Projected_CCI': projected_cci,
                        'Forecast_Slope': slope,
                        'R_gas': regime_g,
                        'R_fluid': regime_f,
                        'R_granular': regime_gr,
                    })

                frame_idx += 1
                if total_frames > 0 and frame_idx % 10 == 0:
                    progress = int((frame_idx / total_frames) * 100)
                    self.progress_signal.emit(max(0, min(progress, 100)))

            source.release()
            if not rows:
                raise RuntimeError(f'No frames remained after warm-up exclusion ({warmup_frames} frames).')

            forecast_rows, lead_times = compute_forecast_validation(rows, horizon_frames, FORECAST.stable_slope_threshold, trend_window=5)
            if not forecast_rows:
                raise RuntimeError(f'No frames remained after applying the {horizon_frames}-frame forecast horizon.')

            self.progress_signal.emit(100)
            self.status_signal.emit(
                f'Loading evaluation metrics and graph results after excluding the first {warmup_frames} warm-up frames...'
            )

            self._save_csv(forecast_rows, out_dir)
            metrics = self._compute_validation_metrics(forecast_rows, lead_times)
            self._save_metrics(metrics, out_dir)
            self._save_summary(forecast_rows, metrics, out_dir, warmup_frames, horizon_frames)
            self._save_plots(forecast_rows, out_dir)

            self.finished_signal.emit(str(out_dir))
        except Exception:
            self.error_signal.emit(traceback.format_exc())

    def _save_csv(self, rows, out_dir: Path):
        fieldnames = list(rows[0].keys())
        for filename in ('metrics.csv', 'data.csv'):
            csv_path = out_dir / filename
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

    def _compute_validation_metrics(self, rows, lead_times):
        projected = np.asarray([float(r['Projected_CCI']) for r in rows], dtype=float)
        actual_future = np.asarray([float(r['Actual_Future_CCI']) for r in rows], dtype=float)
        errors = projected - actual_future
        pearson_r, _ = compute_pearson(projected, actual_future)
        spearman_rho, _ = compute_spearman(projected, actual_future)
        direction_accuracy = float(np.mean([
            r['Forecast_Direction'] == r['Actual_Direction'] for r in rows
        ]))

        metric_arrays = {
            'S': np.asarray([float(r['S']) for r in rows], dtype=float),
            'K': np.asarray([float(r['K']) for r in rows], dtype=float),
            'C': np.asarray([float(r['C']) for r in rows], dtype=float),
            'CCI': np.asarray([float(r['CCI']) for r in rows], dtype=float),
            'SI': np.asarray([float(r['SI']) for r in rows], dtype=float),
        }

        si_values = metric_arrays['SI']
        cci_values = metric_arrays['CCI']
        si_pearson_r, si_pearson_p = compute_pearson(si_values, cci_values)
        si_spearman_rho, si_spearman_p = compute_spearman(si_values, cci_values)

        lead_times = self._extract_lead_times(rows, lead_times)
        lead_time_mean = float(np.mean(lead_times)) if lead_times else float('nan')
        lead_time_median = float(np.median(lead_times)) if lead_times else float('nan')

        metrics = {
            'Forecast Pearson r': pearson_r,
            'Forecast Spearman rho': spearman_rho,
            'Forecast MAE': float(np.mean(np.abs(errors))),
            'Forecast RMSE': float(np.sqrt(np.mean(errors ** 2))),
            'Forecast Direction Accuracy': direction_accuracy,
            'Forecast Lead Time Mean Frames': lead_time_mean,
            'Forecast Lead Time Median Frames': lead_time_median,
            'Forecast Lead Event Count': len(lead_times),
            'SI Std': float(np.std(si_values)),
            'SI vs CCI Pearson r': si_pearson_r,
            'SI vs CCI Pearson p': si_pearson_p,
            'SI vs CCI Spearman rho': si_spearman_rho,
            'SI vs CCI Spearman p': si_spearman_p,
            'Retained Frames': len(rows),
        }

        for metric_name, values in metric_arrays.items():
            metrics[f'{metric_name} Mean'] = float(np.mean(values))
            metrics[f'{metric_name} Min'] = float(np.min(values))
            metrics[f'{metric_name} Max'] = float(np.max(values))

        return metrics

    @staticmethod
    def _extract_lead_times(rows, lead_times=None):
        extracted = []
        source = lead_times if lead_times else None
        if source:
            for value in source:
                try:
                    extracted.append(float(value))
                except (TypeError, ValueError):
                    continue
            return extracted

        for row in rows:
            value = row.get('Lead_Time_Frames', '')
            if value in ('', None):
                continue
            try:
                extracted.append(float(value))
            except (TypeError, ValueError):
                continue
        return extracted

    def _save_metrics(self, metrics, out_dir: Path):
        rows = []
        for key, value in metrics.items():
            rows.append({'metric': key, 'value': value})

        for filename in ('summary_metrics.csv', 'validation_metrics.csv'):
            csv_path = out_dir / filename
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['metric', 'value'])
                writer.writeheader()
                writer.writerows(rows)

    def _save_summary(self, rows, metrics, out_dir: Path, warmup_frames: int, horizon_frames: int):
        lines = [
            'CrowdTune Per-Video Metrics Summary',
            f'Video: {self.video_path}',
            f'Frames retained after warm-up exclusion: {len(rows)}',
            f'Warm-up frames excluded: {warmup_frames}',
            f'Forecast horizon frames: {horizon_frames}',
            '',
            'Core Signal Summary:',
            f"- Mean S = {metrics['S Mean']:.4f}",
            f"- Min S = {metrics['S Min']:.4f}",
            f"- Max S = {metrics['S Max']:.4f}",
            f"- Mean K = {metrics['K Mean']:.4f}",
            f"- Min K = {metrics['K Min']:.4f}",
            f"- Max K = {metrics['K Max']:.4f}",
            f"- Mean C = {metrics['C Mean']:.4f}",
            f"- Min C = {metrics['C Min']:.4f}",
            f"- Max C = {metrics['C Max']:.4f}",
            f"- Mean CCI = {metrics['CCI Mean']:.4f}",
            f"- Min CCI = {metrics['CCI Min']:.4f}",
            f"- Max CCI = {metrics['CCI Max']:.4f}",
            f"- Mean SI = {metrics['SI Mean']:.4f}",
            f"- Min SI = {metrics['SI Min']:.4f}",
            f"- Max SI = {metrics['SI Max']:.4f}",
            '',
            'Forecast Validation:',
            f"- Pearson r = {metrics['Forecast Pearson r']:.4f}",
            f"- Spearman rho = {metrics['Forecast Spearman rho']:.4f}",
            f"- MAE = {metrics['Forecast MAE']:.4f}",
            f"- RMSE = {metrics['Forecast RMSE']:.4f}",
            f"- Direction accuracy (5-frame rolling trend) = {metrics['Forecast Direction Accuracy']:.4f}",
            f"- Mean lead time = {metrics['Forecast Lead Time Mean Frames']:.2f} frames" if not math.isnan(metrics['Forecast Lead Time Mean Frames']) else '- Mean lead time = n/a',
            f"- Median lead time = {metrics['Forecast Lead Time Median Frames']:.2f} frames" if not math.isnan(metrics['Forecast Lead Time Median Frames']) else '- Median lead time = n/a',
            f"- Lead-time events = {int(metrics['Forecast Lead Event Count'])}",
            '',
            'Stability Validation:',
            f"- Mean SI = {metrics['SI Mean']:.4f}",
            f"- Min SI = {metrics['SI Min']:.4f}",
            f"- Max SI = {metrics['SI Max']:.4f}",
            f"- Std SI = {metrics['SI Std']:.4f}",
            f"- SI vs CCI Pearson r = {metrics['SI vs CCI Pearson r']:.4f}, p = {metrics['SI vs CCI Pearson p']:.4g}" if not math.isnan(metrics['SI vs CCI Pearson p']) else f"- SI vs CCI Pearson r = {metrics['SI vs CCI Pearson r']:.4f}, p = n/a",
            f"- SI vs CCI Spearman rho = {metrics['SI vs CCI Spearman rho']:.4f}, p = {metrics['SI vs CCI Spearman p']:.4g}" if not math.isnan(metrics['SI vs CCI Spearman p']) else f"- SI vs CCI Spearman rho = {metrics['SI vs CCI Spearman rho']:.4f}, p = n/a",
        ]
        (out_dir / 'summary.txt').write_text('\n'.join(lines), encoding='utf-8')

    def _save_plots(self, rows, out_dir: Path):
        self._save_timeseries(rows, out_dir)
        self._save_scatter(
            rows,
            'Projected_CCI',
            'Actual_Future_CCI',
            'Projected CCI vs Realized Future CCI',
            out_dir / 'forecast_vs_actual_future_cci.png',
        )
        self._save_direction_plot(rows, out_dir / 'forecast_direction_comparison.png')

    def _save_timeseries(self, rows, out_dir: Path):
        frames = np.asarray([float(r['frame']) for r in rows], dtype=float)
        plot_series = [
            ('S', 'Spatial Constraint S(t)', '#1f77b4'),
            ('K', 'Kinematic Activity K(t)', '#ff7f0e'),
            ('C', 'Collective Coherence C(t)', '#2ca02c'),
            ('CCI', 'Crowd Constraint Index CCI(t)', '#d62728'),
            ('SI', 'Stability Index SI(t)', '#9467bd'),
            ('Projected_CCI', 'Forecast / Projected CCI(t)', '#8c564b'),
        ]

        fig, axes = plt.subplots(len(plot_series), 1, figsize=(12, 14), sharex=True)
        for ax, (key, title, color) in zip(axes, plot_series):
            values = np.asarray([float(r[key]) for r in rows], dtype=float)
            ax.plot(frames, values, color=color, linewidth=1.2)
            ax.set_ylabel(key)
            ax.set_title(title)
            ax.grid(alpha=0.25)

        axes[-1].set_xlabel('Frame')
        fig.suptitle('CrowdTune Per-Video Metric Time Series', fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fig.savefig(out_dir / 'plots.png', dpi=150)
        fig.savefig(out_dir / 'timeseries.png', dpi=150)
        plt.close(fig)

    def _save_scatter(self, rows, x_key, y_key, title, out_path: Path):
        x = np.asarray([r[x_key] for r in rows], dtype=float)
        y = np.asarray([r[y_key] for r in rows], dtype=float)
        plt.figure(figsize=(6, 6))
        plt.scatter(x, y, s=10, alpha=0.4)
        if len(x) >= 2 and np.std(x) > 0.0:
            coeffs = np.polyfit(x, y, 1)
            line_x = np.linspace(float(np.min(x)), float(np.max(x)), 100)
            line_y = coeffs[0] * line_x + coeffs[1]
            plt.plot(line_x, line_y, color='red', linewidth=1.5)
        plt.xlabel(x_key.replace('_', ' '))
        plt.ylabel(y_key.replace('_', ' '))
        plt.title(title)
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()

    def _save_direction_plot(self, rows, out_path: Path):
        mapping = {'Dissipating': -1, 'Stable': 0, 'Escalating': 1}
        frames = [r['frame'] for r in rows]
        predicted = [mapping[r['Forecast_Direction']] for r in rows]
        actual = [mapping[r['Actual_Direction']] for r in rows]
        plt.figure(figsize=(12, 4))
        plt.plot(frames, predicted, label='Forecast Direction', linewidth=1.0)
        plt.plot(frames, actual, label='Actual Direction', linewidth=1.0)
        plt.yticks([-1, 0, 1], ['Dissipating', 'Stable', 'Escalating'])
        plt.xlabel('Frame')
        plt.ylabel('Direction')
        plt.title('Forecast Direction vs Realized Direction (5-frame trend)')
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()


class ValidationUI(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        self._build_ui()

    def _build_ui(self):
        self.setWindowTitle('CrowdTune SI and Forecast Validation')
        self.setMinimumSize(820, 460)

        self.video_path_input = QLineEdit()
        self.video_path_input.setPlaceholderText('Select validation video file...')
        self.video_path_input.setReadOnly(True)

        self.case_name_input = QLineEdit()
        self.case_name_input.setPlaceholderText('Enter output case name')

        self.profile_status_label = QLabel('Scene Profile: auto-detect by video name')
        self.status_label = QLabel('Idle')
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        self.preview_label = QLabel('Video Preview')
        self.preview_label.setMinimumSize(640, 360)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet('background-color: black; color: white;')

        self.btn_browse = QPushButton('Browse Video')
        self.btn_start = QPushButton('Run Analysis')
        self.btn_combine = QPushButton('Combine Results')
        self.btn_barcharts = QPushButton('Bar Charts')
        self.btn_stop = QPushButton('Stop')
        self.btn_stop.setEnabled(False)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)

        form = QGridLayout()
        form.addWidget(QLabel('Video:'), 0, 0)
        form.addWidget(self.video_path_input, 0, 1)
        form.addWidget(self.btn_browse, 0, 2)
        form.addWidget(QLabel('Case Name:'), 1, 0)
        form.addWidget(self.case_name_input, 1, 1, 1, 2)
        form.addWidget(self.profile_status_label, 2, 0, 1, 3)

        buttons = QHBoxLayout()
        buttons.addWidget(self.btn_start)
        buttons.addWidget(self.btn_combine)
        buttons.addWidget(self.btn_barcharts)
        buttons.addWidget(self.btn_stop)
        buttons.addStretch()

        root = QVBoxLayout()
        root.addLayout(form)
        root.addWidget(self.preview_label)
        root.addLayout(buttons)
        root.addWidget(self.progress)
        root.addWidget(self.status_label)
        root.addWidget(self.log_box)
        self.setLayout(root)

        self.btn_browse.clicked.connect(self._browse_video)
        self.btn_start.clicked.connect(self._start_validation)
        self.btn_combine.clicked.connect(self._combine_results)
        self.btn_barcharts.clicked.connect(self._create_bar_charts)
        self.btn_stop.clicked.connect(self._stop_validation)

    def _browse_video(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Select Validation Video', '', 'Video Files (*.mp4 *.avi)')
        if path:
            self.video_path_input.setText(path)
            profile_path = Path(__file__).resolve().parent / 'scene_profiles' / f'{Path(path).stem}.json'
            if profile_path.exists():
                self.profile_status_label.setText(f'Scene Profile: {profile_path.name}')
            else:
                self.profile_status_label.setText('Scene Profile: none found, full-frame analysis will be used')
            self._load_preview_frame(path)

    def _load_preview_frame(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.preview_label.setText('Preview unavailable')
            return
        ret, frame = cap.read()
        cap.release()
        if not ret:
            self.preview_label.setText('Preview unavailable')
            return

        profile = SceneProfile.load_for_video(video_path, Path(__file__).resolve().parent / 'scene_profiles')
        if profile is not None:
            frame = profile.render_overlay(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg.copy())
        pix = pix.scaled(
            self.preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_label.setPixmap(pix)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        video_path = self.video_path_input.text().strip()
        if video_path and self.preview_label.pixmap() is not None:
            self._load_preview_frame(video_path)

    @staticmethod
    def _sanitize_case_name(text):
        cleaned = re.sub(r'[^A-Za-z0-9._-]+', '_', text.strip())
        return cleaned.strip('._-')

    def _start_validation(self):
        video_path = self.video_path_input.text().strip()
        case_name = self._sanitize_case_name(self.case_name_input.text().strip())

        if not video_path:
            QMessageBox.warning(self, 'Missing Video', 'Please select a validation video file.')
            return
        if not case_name:
            QMessageBox.warning(self, 'Missing Case Name', 'Please enter a valid case name.')
            return

        self.worker = ValidationWorker(video_path=video_path, case_name=case_name)
        self.worker.progress_signal.connect(self.progress.setValue)
        self.worker.status_signal.connect(self._on_status)
        self.worker.finished_signal.connect(self._on_finished)
        self.worker.error_signal.connect(self._on_error)

        self.progress.setValue(0)
        self.log_box.clear()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._append_log('SI and forecast validation started.')
        self.worker.start()

    def _stop_validation(self):
        if self.worker and self.worker.isRunning():
            self.worker.request_stop()
            self._append_log('Stop requested. Finishing current processing loop...')
            self.status_label.setText('Stopping...')
            self.btn_stop.setEnabled(False)

    def _on_status(self, text):
        self.status_label.setText(text)
        self._append_log(text)

    def _on_finished(self, out_dir):
        self.status_label.setText('Completed')
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._append_log(f'Validation completed. Results saved to: {out_dir}')
        QMessageBox.information(self, 'Validation Complete', f'Results saved to:\n{out_dir}')

    def _on_error(self, err):
        self.status_label.setText('Failed')
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._append_log('Validation failed.')
        self._append_log(err)
        QMessageBox.critical(self, 'Validation Failed', err)

    def _append_log(self, text):
        self.log_box.append(text)

    def _combine_results(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            'Select Validation CSV Files',
            str(Path.cwd() / 'results'),
            'CSV Files (data.csv *.csv)'
        )
        if not paths:
            return

        rows = []
        for path in paths:
            with open(path, 'r', newline='', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                source_name = Path(path).parent.name
                for row in reader:
                    row['source_case'] = source_name
                    rows.append(row)

        if not rows:
            QMessageBox.warning(self, 'No Data', 'The selected CSV files did not contain any rows.')
            return

        combined_name = self._sanitize_case_name(self.case_name_input.text().strip()) or 'combined_validation'
        out_dir = Path.cwd() / 'results' / f'{combined_name}_combined'
        out_dir.mkdir(parents=True, exist_ok=True)

        self._save_combined_csv(rows, out_dir)
        metrics = self._compute_combined_metrics(rows)
        per_case_metrics = self._compute_per_case_metrics(rows)
        self._save_combined_metrics(metrics, out_dir)
        self._save_per_case_metrics(per_case_metrics, out_dir)
        self._save_combined_summary(rows, metrics, out_dir, paths)
        self._save_combined_plots(rows, out_dir, per_case_metrics)

        self._append_log(f'Combined analysis saved to: {out_dir}')
        QMessageBox.information(self, 'Combine Complete', f'Combined results saved to:\n{out_dir}')

    def _create_bar_charts(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            'Select Experiment CSV Files',
            str(Path.cwd() / 'results'),
            'CSV Files (data.csv *.csv)'
        )
        if not paths:
            return

        rows = []
        for path in paths:
            with open(path, 'r', newline='', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                source_name = Path(path).parent.name
                fieldnames = reader.fieldnames or []
                if 'Projected_CCI' not in fieldnames or 'Actual_Future_CCI' not in fieldnames or 'SI' not in fieldnames:
                    continue
                for row in reader:
                    row['source_case'] = source_name
                    rows.append(row)

        if not rows:
            QMessageBox.warning(self, 'No Data', 'The selected CSV files did not contain compatible validation rows.')
            return

        per_case_metrics = self._compute_per_case_metrics(rows)
        if not per_case_metrics:
            QMessageBox.warning(self, 'No Metrics', 'Unable to compute per-experiment metrics from the selected CSV files.')
            return

        out_name = self._sanitize_case_name(self.case_name_input.text().strip()) or 'experiment_bar_charts'
        out_dir = Path.cwd() / 'results' / f'{out_name}_bar_charts'
        out_dir.mkdir(parents=True, exist_ok=True)

        self._save_per_case_metrics(per_case_metrics, out_dir)
        self._save_forecast_pearson_bar_chart(per_case_metrics, out_dir / 'forecast_pearson_r_by_experiment.png')
        self._save_si_mean_bar_chart(per_case_metrics, out_dir / 'si_mean_by_experiment.png')

        self._append_log(f'Bar charts saved to: {out_dir}')
        QMessageBox.information(self, 'Bar Charts Complete', f'Bar charts saved to:\n{out_dir}')
    def _compute_combined_metrics(self, rows):
        projected = np.asarray([float(r['Projected_CCI']) for r in rows], dtype=float)
        actual_future = np.asarray([float(r['Actual_Future_CCI']) for r in rows], dtype=float)
        errors = projected - actual_future
        pearson_r, _ = compute_pearson(projected, actual_future)
        spearman_rho, _ = compute_spearman(projected, actual_future)
        direction_accuracy = float(np.mean([
            r['Forecast_Direction'] == r['Actual_Direction'] for r in rows
        ]))
        si_values = np.asarray([float(r['SI']) for r in rows], dtype=float)
        cci_values = np.asarray([float(r['CCI']) for r in rows], dtype=float)
        si_pearson_r, si_pearson_p = compute_pearson(si_values, cci_values)
        si_spearman_rho, si_spearman_p = compute_spearman(si_values, cci_values)

        lead_times = self._extract_lead_times(rows)
        lead_time_mean = float(np.mean(lead_times)) if lead_times else float('nan')
        lead_time_median = float(np.median(lead_times)) if lead_times else float('nan')

        return {
            'Forecast Pearson r': pearson_r,
            'Forecast Spearman rho': spearman_rho,
            'Forecast MAE': float(np.mean(np.abs(errors))),
            'Forecast RMSE': float(np.sqrt(np.mean(errors ** 2))),
            'Forecast Direction Accuracy': direction_accuracy,
            'Forecast Lead Time Mean Frames': lead_time_mean,
            'Forecast Lead Time Median Frames': lead_time_median,
            'Forecast Lead Event Count': len(lead_times),
            'SI Mean': float(np.mean(si_values)),
            'SI Min': float(np.min(si_values)),
            'SI Max': float(np.max(si_values)),
            'SI Std': float(np.std(si_values)),
            'SI vs CCI Pearson r': si_pearson_r,
            'SI vs CCI Pearson p': si_pearson_p,
            'SI vs CCI Spearman rho': si_spearman_rho,
            'SI vs CCI Spearman p': si_spearman_p,
            'Retained Frames': len(rows),
        }

    def _save_combined_csv(self, rows, out_dir: Path):
        csv_path = out_dir / 'combined_data.csv'
        fieldnames = list(rows[0].keys())
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _save_combined_metrics(self, metrics, out_dir: Path):
        csv_path = out_dir / 'combined_validation_metrics.csv'
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            for key, value in metrics.items():
                writer.writerow([key, value])

    def _save_combined_summary(self, rows, metrics, out_dir: Path, paths):
        lines = [
            'CrowdTune Combined SI and Forecast Validation',
            f'Combined source files: {len(paths)}',
            f'Total rows: {len(rows)}',
            '',
            'Input files:',
        ]
        lines.extend([f'- {path}' for path in paths])
        lines.extend([
            '',
            'Forecast Validation:',
            f"- Pearson r = {metrics['Forecast Pearson r']:.4f}",
            f"- Spearman rho = {metrics['Forecast Spearman rho']:.4f}",
            f"- MAE = {metrics['Forecast MAE']:.4f}",
            f"- RMSE = {metrics['Forecast RMSE']:.4f}",
            f"- Direction accuracy (5-frame rolling trend) = {metrics['Forecast Direction Accuracy']:.4f}",
            f"- Mean lead time = {metrics['Forecast Lead Time Mean Frames']:.2f} frames" if not math.isnan(metrics['Forecast Lead Time Mean Frames']) else '- Mean lead time = n/a',
            f"- Median lead time = {metrics['Forecast Lead Time Median Frames']:.2f} frames" if not math.isnan(metrics['Forecast Lead Time Median Frames']) else '- Median lead time = n/a',
            f"- Lead-time events = {int(metrics['Forecast Lead Event Count'])}",
            '',
            'Stability Index Summary:',
            f"- Mean SI = {metrics['SI Mean']:.4f}",
            f"- Min SI = {metrics['SI Min']:.4f}",
            f"- Max SI = {metrics['SI Max']:.4f}",
            f"- Std SI = {metrics['SI Std']:.4f}",
            f"- SI vs CCI Pearson r = {metrics['SI vs CCI Pearson r']:.4f}, p = {metrics['SI vs CCI Pearson p']:.4g}" if not math.isnan(metrics['SI vs CCI Pearson p']) else f"- SI vs CCI Pearson r = {metrics['SI vs CCI Pearson r']:.4f}, p = n/a",
            f"- SI vs CCI Spearman rho = {metrics['SI vs CCI Spearman rho']:.4f}, p = {metrics['SI vs CCI Spearman p']:.4g}" if not math.isnan(metrics['SI vs CCI Spearman p']) else f"- SI vs CCI Spearman rho = {metrics['SI vs CCI Spearman rho']:.4f}, p = n/a",
        ])
        (out_dir / 'combined_summary.txt').write_text('\n'.join(lines), encoding='utf-8')

    def _compute_per_case_metrics(self, rows):
        grouped = {}
        for row in rows:
            case_name = row.get('source_case', 'unknown_case')
            grouped.setdefault(case_name, []).append(row)

        per_case_metrics = []
        for case_name, case_rows in grouped.items():
            case_metrics = self._compute_combined_metrics(case_rows)
            per_case_metrics.append({
                'source_case': case_name,
                'Forecast Pearson r': case_metrics['Forecast Pearson r'],
                'Forecast Spearman rho': case_metrics['Forecast Spearman rho'],
                'Forecast MAE': case_metrics['Forecast MAE'],
                'Forecast RMSE': case_metrics['Forecast RMSE'],
                'Forecast Direction Accuracy': case_metrics['Forecast Direction Accuracy'],
                'Forecast Lead Time Mean Frames': case_metrics['Forecast Lead Time Mean Frames'],
                'Forecast Lead Time Median Frames': case_metrics['Forecast Lead Time Median Frames'],
                'Forecast Lead Event Count': case_metrics['Forecast Lead Event Count'],
                'SI Mean': case_metrics['SI Mean'],
                'Retained Frames': case_metrics['Retained Frames'],
            })

        per_case_metrics.sort(key=lambda item: item['source_case'].lower())
        return per_case_metrics


    def _save_per_case_metrics(self, per_case_metrics, out_dir: Path):
        if not per_case_metrics:
            return
        csv_path = out_dir / 'per_case_validation_metrics.csv'
        fieldnames = list(per_case_metrics[0].keys())
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_case_metrics)

    def _save_combined_plots(self, rows, out_dir: Path, per_case_metrics):
        self._save_combined_scatter(rows, 'Projected_CCI', 'Actual_Future_CCI', 'Projected CCI vs Realized Future CCI', out_dir / 'combined_forecast_vs_actual_future_cci.png')
        self._save_forecast_pearson_bar_chart(per_case_metrics, out_dir / 'forecast_pearson_r_by_experiment.png')
        self._save_si_mean_bar_chart(per_case_metrics, out_dir / 'si_mean_by_experiment.png')

    def _save_forecast_pearson_bar_chart(self, per_case_metrics, out_path: Path):
        if not per_case_metrics:
            return

        labels = [item['source_case'] for item in per_case_metrics]
        forecast_r = [float(item['Forecast Pearson r']) for item in per_case_metrics]
        x = np.arange(len(labels), dtype=float)

        plt.figure(figsize=(max(8, len(labels) * 1.3), 5))
        plt.bar(x, forecast_r, width=0.55, color='#2a9d8f')
        plt.axhline(0.0, color='black', linewidth=0.8, alpha=0.5)
        plt.xticks(x, labels, rotation=20, ha='right')
        plt.ylabel('Pearson r')
        plt.title('Forecast Pearson r Across Experiments')
        plt.ylim(-1.0, 1.0)
        plt.grid(axis='y', alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()

    def _save_si_mean_bar_chart(self, per_case_metrics, out_path: Path):
        if not per_case_metrics:
            return

        labels = [item['source_case'] for item in per_case_metrics]
        si_mean = [float(item['SI Mean']) for item in per_case_metrics]
        x = np.arange(len(labels), dtype=float)

        plt.figure(figsize=(max(8, len(labels) * 1.3), 5))
        plt.bar(x, si_mean, width=0.55, color='#e9c46a')
        plt.xticks(x, labels, rotation=20, ha='right')
        plt.ylabel('Mean SI')
        plt.title('Mean Stability Index Across Experiments')
        plt.ylim(0.0, 1.0)
        plt.grid(axis='y', alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()

    def _save_combined_scatter(self, rows, x_key, y_key, title, out_path: Path):
        paired = []
        for row in rows:
            try:
                paired.append((float(row[x_key]), float(row[y_key])))
            except Exception:
                continue
        if not paired:
            return
        x = np.asarray([item[0] for item in paired], dtype=float)
        y = np.asarray([item[1] for item in paired], dtype=float)
        plt.figure(figsize=(6, 6))
        plt.scatter(x, y, s=10, alpha=0.35)
        if len(x) >= 2 and np.std(x) > 0.0:
            coeffs = np.polyfit(x, y, 1)
            line_x = np.linspace(float(np.min(x)), float(np.max(x)), 100)
            line_y = coeffs[0] * line_x + coeffs[1]
            plt.plot(line_x, line_y, color='red', linewidth=1.5)
        plt.xlabel(x_key.replace('_', ' '))
        plt.ylabel(y_key.replace('_', ' '))
        plt.title(title + ' (Combined)')
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = ValidationUI()
    ui.show()
    sys.exit(app.exec())










