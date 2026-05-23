# cam_thread.py

from pathlib import Path

import cv2
import time
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage

from crowd_physics import CrowdPhysicsAnalyzer, compute_regime_tendency
from risk_index import RiskIndex
from lbp_density import LBPDensity
from stability_index import StabilityIndex
from instability_forecast import create_forecaster
from scene_profile import SceneProfile
from tune_config import DENSITY, FORECAST, FUSION, PLAYBACK, REGIME, STABILITY


class CameraThread(QThread):

    frame_signal = pyqtSignal(QImage)
    cci_signal = pyqtSignal(float)
    heatmap_signal = pyqtSignal(dict)
    risk_signal = pyqtSignal(dict)
    stability_signal = pyqtSignal(float)
    forecast_signal = pyqtSignal(dict)
    playback_position_signal = pyqtSignal(int, int)
    skc_signal = pyqtSignal(float, float, float)
    runtime_metrics_signal = pyqtSignal(float, int, float)

    def __init__(self, source):
        super().__init__()
        self.source = source
        self.running = True
        self.paused = False
        self.file_mode = bool(getattr(self.source, "is_seekable", lambda: False)())

        self.crowd = CrowdPhysicsAnalyzer()
        self.density_engine = LBPDensity(
            radius=DENSITY.radius,
            n_points=DENSITY.n_points,
            grid_size=DENSITY.grid_size,
        )
        self.risk_engine = RiskIndex()
        self.stability = StabilityIndex(window=STABILITY.window)
        self.forecaster = create_forecaster(FORECAST)

        self.overlay_enabled = False
        self.tracking_overlay_enabled = False
        self.last_density_map = None
        self.last_density_score = 0.0

        self.frame_idx = 0
        self.R_gas_s = REGIME.initial_gas
        self.R_fluid_s = REGIME.initial_fluid
        self.R_granular_s = REGIME.initial_granular

        self.regime_alpha = REGIME.smooth_alpha
        self.last_frame_ts = None
        self.fps_ema = 0.0
        self.warmup_seconds = PLAYBACK.warmup_seconds
        self._stream_start_ts = None
        self._last_crowd_metrics = {"K": 0.0, "C": 0.0, "track_prev": None, "track_next": None}
        self._pending_seek_seconds = None
        self._eof_reached = False
        self.scene_profile = None
        self.spatial_analysis_mask = None
        self._mask_log_printed = False
        if self.file_mode and hasattr(self.source, "path"):
            profiles_dir = Path(__file__).resolve().parent / "scene_profiles"
            self.scene_profile = SceneProfile.load_for_video(self.source.path, profiles_dir)

        self.target_frame_dt = 1.0 / PLAYBACK.target_fps_fallback
        self.analytics_stride = 1
        if self.file_mode and hasattr(self.source, "cap"):
            fps = float(self.source.cap.get(cv2.CAP_PROP_FPS))
            if fps > 1.0:
                self.target_frame_dt = 1.0 / fps
            self.analytics_stride = PLAYBACK.file_analytics_stride

    def set_overlay_enabled(self, enabled: bool):
        self.overlay_enabled = enabled

    def set_tracking_overlay_enabled(self, enabled: bool):
        self.tracking_overlay_enabled = enabled

    def pause_playback(self):
        if getattr(self.source, "is_seekable", lambda: False)():
            self.paused = True
            self._eof_reached = False

    def resume_playback(self):
        if getattr(self.source, "is_seekable", lambda: False)():
            if self._eof_reached:
                self._pending_seek_seconds = 0
                self._eof_reached = False
            self.paused = False

    def seek_seconds(self, seconds: int):
        if not getattr(self.source, "is_seekable", lambda: False)():
            return
        self._pending_seek_seconds = max(0, int(seconds))

    def _reset_temporal_state(self):
        self.crowd.prev_gray = None
        self.crowd.prev_pts = None
        self.crowd.k_median_history.clear()
        self.crowd.c_smoothed = None
        self.stability.history.clear()
        self.forecaster.reset()
        self.last_density_map = None
        self.last_density_score = 0.0
        self._last_crowd_metrics = {"K": 0.0, "C": 0.0, "track_prev": None, "track_next": None}
        self.frame_idx = 0
        self.R_gas_s = REGIME.initial_gas
        self.R_fluid_s = REGIME.initial_fluid
        self.R_granular_s = REGIME.initial_granular
        self.last_frame_ts = None
        self.fps_ema = 0.0
        self._stream_start_ts = None
        self.spatial_analysis_mask = None

    def _apply_pending_seek(self):
        if self._pending_seek_seconds is None:
            return
        target_seconds = self._pending_seek_seconds
        self._pending_seek_seconds = None
        if self.source.seek_seconds(target_seconds):
            self._reset_temporal_state()
            self._eof_reached = False
            self._emit_playback_position()

    def _emit_playback_position(self):
        if not getattr(self.source, "is_seekable", lambda: False)():
            return
        duration = self.source.get_duration_seconds()
        position = self.source.get_position_seconds()
        if self._eof_reached:
            position = duration
        self.playback_position_signal.emit(position, duration)

    def _draw_tracking_overlay(self, frame, crowd_metrics):
        prev_pts = crowd_metrics.get("track_prev")
        next_pts = crowd_metrics.get("track_next")
        if prev_pts is None or next_pts is None:
            return

        max_tracks = min(len(prev_pts), 120)
        for i in range(max_tracks):
            x0, y0 = prev_pts[i]
            x1, y1 = next_pts[i]
            p0 = (int(x0), int(y0))
            p1 = (int(x1), int(y1))
            cv2.line(frame, p0, p1, (255, 210, 0), 1, cv2.LINE_AA)
            cv2.circle(frame, p1, 2, (0, 255, 255), -1, cv2.LINE_AA)

    def run(self):
        while self.running:
            self._apply_pending_seek()
            if self.paused:
                self._emit_playback_position()
                time.sleep(PLAYBACK.paused_sleep_seconds)
                continue

            frame_start_ts = time.perf_counter()
            ret, frame = self.source.read()
            if not ret:
                if self.file_mode:
                    self._eof_reached = True
                    self.paused = True
                    self._emit_playback_position()
                    time.sleep(PLAYBACK.paused_sleep_seconds)
                continue

            if self._stream_start_ts is None:
                self._stream_start_ts = time.perf_counter()

            if self.scene_profile is not None and (
                self.spatial_analysis_mask is None
                or self.spatial_analysis_mask.shape[:2] != frame.shape[:2]
            ):
                self.spatial_analysis_mask = self.scene_profile.get_spatial_analysis_mask(frame.shape)
                if not self._mask_log_printed:
                    try:
                        print(f"[CAM_THREAD] applied scene_profile mask shape={self.spatial_analysis_mask.shape} for frame_shape={frame.shape}")
                    except Exception:
                        print("[CAM_THREAD] applied scene_profile mask (shape unknown)")
                    self._mask_log_printed = True

            self.frame_idx += 1
            run_analytics = (not self.file_mode) or ((self.frame_idx % self.analytics_stride) == 0)

            if getattr(self.source, "mirror", False):
                frame = cv2.flip(frame, 1)

            crowd_metrics = self._last_crowd_metrics
            if run_analytics:
                try:
                    mask_info = None
                    if self.spatial_analysis_mask is None:
                        mask_info = "None"
                    else:
                        try:
                            mask_info = f"shape={self.spatial_analysis_mask.shape}, sum={int(self.spatial_analysis_mask.sum())}"
                        except Exception:
                            mask_info = "present"
                    print(f"[CAM_THREAD] run loop: run_analytics=True, spatial_analysis_mask={mask_info}")
                except Exception:
                    pass
                crowd_metrics = self.crowd.update(frame, analysis_mask=self.spatial_analysis_mask)
                self._last_crowd_metrics = crowd_metrics

                if self.frame_idx % DENSITY.compute_every_n_frames == 0:
                    density_map, s_sobel = self.density_engine.compute(
                        frame,
                        analysis_mask=self.spatial_analysis_mask,
                    )
                    self.last_density_map = density_map
                    self.last_density_score = float(s_sobel)
                    self.heatmap_signal.emit({
                        "density": density_map,
                        "frame_shape": frame.shape,
                    })

                spatial_score = self.last_density_score
                s_raw = spatial_score
                s = (s_raw - DENSITY.s_min) / (DENSITY.s_max - DENSITY.s_min)
                s = np.clip(s, 0.0, 1.0)
                s = s ** DENSITY.s_gamma

                k = crowd_metrics["K"]
                c = crowd_metrics["C"]
                self.skc_signal.emit(float(s), float(k), float(c))

                r_gas, r_fluid, r_granular = compute_regime_tendency(s, c)
                alpha = self.regime_alpha
                self.R_gas_s = alpha * r_gas + (1 - alpha) * self.R_gas_s
                self.R_fluid_s = alpha * r_fluid + (1 - alpha) * self.R_fluid_s
                self.R_granular_s = alpha * r_granular + (1 - alpha) * self.R_granular_s

                cci = s * (FUSION.k_weight * k + FUSION.incoherence_weight * (1 - c))
                cci = np.clip(cci, 0.0, 1.0)
                self.cci_signal.emit(cci)

                elapsed = time.perf_counter() - self._stream_start_ts
                in_warmup = elapsed < self.warmup_seconds
                si = self.stability.update(cci)
                projected_cci, slope = self.forecaster.update(cci)

                stable_flag = False
                slope_threshold = FORECAST.stable_slope_threshold
                if si > FORECAST.stable_si_threshold and abs(slope) < slope_threshold:
                    projected_cci = cci
                    stable_flag = True
                projected_cci = np.clip(projected_cci, 0.0, 1.0)

                if not in_warmup:
                    self.stability_signal.emit(si)
                    self.forecast_signal.emit({
                        "projected_cci": projected_cci,
                        "current_cci": cci,
                        "slope": slope,
                        "stable": stable_flag,
                    })

                risk = self.risk_engine.update(cci_raw=cci)
                risk["R_gas"] = self.R_gas_s
                risk["R_fluid"] = self.R_fluid_s
                risk["R_granular"] = self.R_granular_s
                self.risk_signal.emit(risk)

            if self.overlay_enabled and self.last_density_map is not None:
                h, w = frame.shape[:2]
                heat = cv2.resize(self.last_density_map, (w, h), interpolation=cv2.INTER_LINEAR)
                heat = np.clip(heat, 0.0, 1.0)
                heat = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
                frame = cv2.addWeighted(frame, 0.65, heat_color, 0.35, 0)

            if self.tracking_overlay_enabled:
                self._draw_tracking_overlay(frame, crowd_metrics)

            if self.scene_profile is not None:
                frame = self.scene_profile.render_overlay(frame)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            self.frame_signal.emit(qimg.copy())
            self._emit_playback_position()

            frame_end_ts = time.perf_counter()
            processing_ms = (frame_end_ts - frame_start_ts) * 1000.0
            if self.last_frame_ts is not None:
                dt = frame_end_ts - self.last_frame_ts
                if dt > 0:
                    fps_inst = 1.0 / dt
                    alpha = 0.2
                    if self.fps_ema <= 0:
                        self.fps_ema = fps_inst
                    else:
                        self.fps_ema = alpha * fps_inst + (1.0 - alpha) * self.fps_ema
            self.last_frame_ts = frame_end_ts

            track_next = crowd_metrics.get("track_next")
            feature_count = int(len(track_next)) if track_next is not None else 0
            self.runtime_metrics_signal.emit(self.fps_ema, feature_count, processing_ms)

            if self.file_mode:
                sleep_s = self.target_frame_dt - (time.perf_counter() - frame_start_ts)
                if sleep_s > 0:
                    time.sleep(sleep_s)
            else:
                time.sleep(PLAYBACK.live_loop_sleep_seconds)

    def stop(self):
        self.running = False
        self.source.release()


