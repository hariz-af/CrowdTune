# interface.py
import json
import re
import sys
import time
from pathlib import Path

import cv2
import os
import csv
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QGroupBox, QCheckBox,
    QStackedWidget, QSlider, QButtonGroup, QSizePolicy, QFrame, QToolButton,
    QComboBox, QGraphicsDropShadowEffect, QDialog, QSplitter, QProgressBar, QMessageBox
)

from PyQt6.QtWidgets import QInputDialog
import shutil

from PyQt6.QtGui import QImage, QPixmap, QIcon, QColor, QPainter, QPen, QPolygon
from PyQt6.QtCore import Qt, pyqtSignal, QPoint, QThread, QTimer

from cam_thread import CameraThread
from frame_source import CameraSource, VideoFileSource

from cci_widget import CCIWidget
from risk_widget import RiskIndexWidget
from stability_widget import StabilityWidget
from forecast_widget import ForecastWidget
from ternary_widget import TernaryPlotWidget
from regime_state_map_widget import RegimeStateMapWidget
from advisory_widget import AdvisoryWidget
from scene_profile import SceneProfile, ZONE_COLORS
from validation_ui import ValidationWorker
from tune_config import RISK, UI_SMOOTHING


ANNOTATION_ZONE_GROUPS = {
    "crowd_space": "zones",
    "walls_barriers": "boundaries",
    "ignore_background": "ignore_regions",
}

ANNOTATION_ZONE_OPTIONS = [
    "crowd_space",
    "walls_barriers",
    "ignore_background",
]

ANNOTATION_ZONE_LABELS = {
    "crowd_space": "Crowd Space",
    "walls_barriers": "Walls & Barriers",
    "ignore_background": "Ignore Background",
}


class AnnotatableVideoLabel(QLabel):
    clicked = pyqtSignal(int, int)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            self.clicked.emit(int(pos.x()), int(pos.y()))
        super().mousePressEvent(event)


class SceneMaskPreviewLabel(QLabel):
    point_clicked = pyqtSignal(int, int)
    points_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(720, 405)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #050505; color: #cfd8dc; border: 1px solid rgba(255,255,255,0.12);")
        self._frame_qimage = None
        self._frame_size = None
        self._points = []
        self._display_rect = None

    def set_frame(self, qimage: QImage):
        self._frame_qimage = qimage.copy()
        self._frame_size = (qimage.width(), qimage.height())
        self._refresh()

    def set_points(self, points):
        self._points = [[int(x), int(y)] for x, y in points]
        self._refresh()
        self.points_changed.emit()

    def points(self):
        return [list(p) for p in self._points]

    def undo_point(self):
        if self._points:
            self._points.pop()
            self._refresh()
            self.points_changed.emit()

    def clear_points(self):
        self._points = []
        self._refresh()
        self.points_changed.emit()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            mapped = self._map_widget_to_frame(event.position().x(), event.position().y())
            if mapped is not None:
                x, y = mapped
                self._points.append([x, y])
                self.point_clicked.emit(x, y)
                self._refresh()
                self.points_changed.emit()
        super().mousePressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh()

    def _map_widget_to_frame(self, widget_x, widget_y):
        if self._frame_size is None or self._display_rect is None:
            return None
        x0, y0, display_w, display_h = self._display_rect
        if widget_x < x0 or widget_x > x0 + display_w or widget_y < y0 or widget_y > y0 + display_h:
            return None
        frame_w, frame_h = self._frame_size
        rel_x = (widget_x - x0) / max(1, display_w)
        rel_y = (widget_y - y0) / max(1, display_h)
        frame_x = int(max(0, min(frame_w - 1, rel_x * frame_w)))
        frame_y = int(max(0, min(frame_h - 1, rel_y * frame_h)))
        return frame_x, frame_y

    def _frame_to_display_point(self, x, y):
        if self._frame_size is None or self._display_rect is None:
            return QPoint(int(x), int(y))
        x0, y0, display_w, display_h = self._display_rect
        frame_w, frame_h = self._frame_size
        px = x0 + (float(x) / max(1, frame_w)) * display_w
        py = y0 + (float(y) / max(1, frame_h)) * display_h
        return QPoint(int(round(px)), int(round(py)))

    def _refresh(self):
        if self._frame_qimage is None or self._frame_qimage.isNull():
            self.clear()
            self.setText("Scene preview unavailable")
            self._display_rect = None
            return

        scaled = QPixmap.fromImage(self._frame_qimage).scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        x0 = (self.width() - scaled.width()) / 2.0
        y0 = (self.height() - scaled.height()) / 2.0
        self._display_rect = (x0, y0, scaled.width(), scaled.height())

        canvas = QPixmap(self.size())
        canvas.fill(QColor(0, 0, 0))
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.drawPixmap(int(round(x0)), int(round(y0)), scaled)

        if self._points:
            display_points = [self._frame_to_display_point(x, y) for x, y in self._points]
            outline = QColor(72, 190, 85, 230)
            fill = QColor(72, 190, 85, 55)
            painter.setPen(QPen(outline, 3))
            if len(display_points) >= 3:
                painter.setBrush(fill)
                painter.drawPolygon(QPolygon(display_points))
            else:
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawPolyline(QPolygon(display_points))
            painter.setBrush(outline)
            for point in display_points:
                painter.drawEllipse(point, 5, 5)
        painter.end()
        self.setPixmap(canvas)


class SceneMaskingDialog(QDialog):
    def __init__(self, video_path: str, profiles_dir: Path, parent=None):
        super().__init__(parent)
        self.video_path = Path(video_path)
        self.profiles_dir = Path(profiles_dir)
        self.profile_path = self.profiles_dir / f"{self.video_path.stem}.json"
        self.frame_size = None
        self.mask_saved = False

        self.setWindowTitle("Scene Masking")
        self.setMinimumSize(860, 650)
        self.setStyleSheet("""
            QDialog { background-color: #161616; color: #e8eaed; }
            QLabel { color: #e8eaed; }
            QPushButton { background-color: #2b2b2b; color: white; border: 1px solid rgba(255,255,255,0.14); border-radius: 4px; padding: 6px 12px; }
            QPushButton:hover { border-color: rgba(255,255,255,0.30); }
            QPushButton:disabled { color: rgba(255,255,255,0.28); background-color: #242424; }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(8)

        instruction = QLabel("Please define the Crowd Region of Interest (ROI) before analysis begins.")
        instruction.setStyleSheet("font-size: 13px; font-weight: 600;")
        layout.addWidget(instruction)

        helper = QLabel("Click around the crowd area to create a polygon. The polygon closes automatically after three or more points.")
        helper.setStyleSheet("color: rgba(220,220,220,0.62); font-size: 11px;")
        layout.addWidget(helper)

        notes = QLabel(
            "Notes:\n"
            "- Use footage from a fixed camera (no panning, zooming, or shaking)\n"
            "- Ensure the crowd region remains consistent throughout the video\n"
            "- Avoid videos with large perspective or viewpoint changes\n"
            "- Mask only the active crowd area (exclude background and empty regions)"
        )
        notes.setStyleSheet("color: rgba(220,220,220,0.52); font-size: 10px;")
        layout.addWidget(notes)

        self.preview = SceneMaskPreviewLabel(self)
        layout.addWidget(self.preview, 1)
        self.preview.points_changed.connect(self._mark_mask_dirty)

        button_row = QHBoxLayout()
        self.load_profile_btn = QPushButton("Load Existing Profile")
        self.undo_btn = QPushButton("Undo Point")
        self.clear_btn = QPushButton("Clear Points")
        self.save_btn = QPushButton("Save Mask")
        self.continue_btn = QPushButton("Continue Analysis")
        self.cancel_btn = QPushButton("Cancel")
        self.continue_btn.setEnabled(False)

        button_row.addWidget(self.load_profile_btn)
        button_row.addWidget(self.undo_btn)
        button_row.addWidget(self.clear_btn)
        button_row.addStretch()
        button_row.addWidget(self.save_btn)
        button_row.addWidget(self.continue_btn)
        button_row.addWidget(self.cancel_btn)
        layout.addLayout(button_row)

        self.load_profile_btn.clicked.connect(self.load_existing_profile)
        self.undo_btn.clicked.connect(self.preview.undo_point)
        self.clear_btn.clicked.connect(self.preview.clear_points)
        self.save_btn.clicked.connect(self.save_mask_with_confirmation)
        self.continue_btn.clicked.connect(self.continue_analysis)
        self.cancel_btn.clicked.connect(self.reject)

        if not self._load_representative_frame():
            self.save_btn.setEnabled(False)
            QMessageBox.critical(
                self,
                "Scene Masking",
                "CrowdTune could not load a representative frame for scene masking. Please select a valid video file."
            )

    def _load_representative_frame(self):
        cap = None
        try:
            cap = cv2.VideoCapture(str(self.video_path))
            if not cap.isOpened():
                return False
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if frame_count > 10:
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count // 3))
            ret, frame = cap.read()
            if not ret or frame is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            if not ret or frame is None:
                return False
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            self.frame_size = (w, h)
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
            self.preview.set_frame(qimg)
            return True
        except Exception:
            return False
        finally:
            if cap is not None:
                cap.release()

    def load_existing_profile(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Scene Profile",
            str(self.profiles_dir),
            "Scene Profiles (*.json)"
        )

        if not path:
            return
        try:
            profile = SceneProfile.from_json(path)
            zones = profile.export_annotation_group("zones")
            points = zones.get("crowd_space", [])
            if not self._valid_polygon(points):
                QMessageBox.information(self, "Scene Masking", "Selected profile does not contain a valid Crowd Space polygon.")
                return
            if self.frame_size is not None:
                frame_w, frame_h = self.frame_size
                sx = frame_w / max(1, profile.frame_width)
                sy = frame_h / max(1, profile.frame_height)
                points = [[int(round(x * sx)), int(round(y * sy))] for x, y in points]
            self.preview.set_points(points)
        except Exception as exc:
            QMessageBox.critical(self, "Scene Masking", f"Could not load scene profile:\n{exc}")

    def save_mask_with_confirmation(self):
        if self.save_mask():
            self.mask_saved = True
            self._update_continue_state()
            QMessageBox.information(self, "Scene Masking", "Scene masking profile saved.")

    def continue_analysis(self):
        if not self.mask_saved:
            QMessageBox.information(self, "Scene Masking", "Please save the ROI mask before continuing analysis.")
            return
        if not self.profile_path.exists():
            QMessageBox.information(self, "Scene Masking", "Please save the ROI mask before continuing analysis.")
            return
        if not self._profile_has_valid_mask(self.profile_path):
            QMessageBox.information(self, "Scene Masking", "The saved profile does not contain a valid Crowd Space polygon.")
            return
        self.accept()

    def _profile_has_valid_mask(self, profile_path: Path):
        try:
            profile = SceneProfile.from_json(profile_path)
            zones = profile.export_annotation_group("zones")
            return self._valid_polygon(zones.get("crowd_space", []))
        except Exception:
            return False

    def _mark_mask_dirty(self):
        self.mask_saved = False
        self._update_continue_state()

    def _update_continue_state(self):
        has_saved_profile = self.profile_path.exists() and self._profile_has_valid_mask(self.profile_path)
        self.continue_btn.setEnabled(bool(self.mask_saved and has_saved_profile))

    def save_mask(self):
        points = self.preview.points()
        if not self._valid_polygon(points):
            QMessageBox.information(self, "Scene Masking", "Please define at least three ROI points before continuing.")
            return False
        if self.frame_size is None:
            QMessageBox.information(self, "Scene Masking", "Scene preview is unavailable, so the mask cannot be saved.")
            return False
        frame_w, frame_h = self.frame_size
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "profile_name": self.video_path.stem,
            "frame_size": {"width": frame_w, "height": frame_h},
            "zones": {"crowd_space": points},
            "boundaries": {},
            "ignore_regions": {},
            "semantic_weights": {"crowd_space": 1.0},
            "risk_modifiers": {
                "wall_proximity_weight": 1.3,
                "boundary_exclusion_weight": 1.0,
            },
            "notes": {
                "flow_direction": "unspecified",
                "scene_type": "validation_scene_mask_dialog",
                "comment": "Profile created before CrowdTune validation analysis.",
            },
        }
        self.profile_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return True

    @staticmethod
    def _valid_polygon(points):
        return isinstance(points, list) and len(points) >= 3

class LiveSceneMaskingDialog(QDialog):
    def __init__(self, frame_qimage: QImage, parent=None):
        super().__init__(parent)
        self.frame_size = (frame_qimage.width(), frame_qimage.height()) if frame_qimage is not None else None
        self.mask_saved = False
        self.setWindowTitle("Live Scene Masking")
        self.setMinimumSize(860, 650)
        self.setStyleSheet("""
            QDialog { background-color: #161616; color: #e8eaed; }
            QLabel { color: #e8eaed; }
            QPushButton { background-color: #2b2b2b; color: white; border: 1px solid rgba(255,255,255,0.14); border-radius: 4px; padding: 6px 12px; }
            QPushButton:hover { border-color: rgba(255,255,255,0.30); }
            QPushButton:disabled { color: rgba(255,255,255,0.28); background-color: #242424; }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(8)

        instruction = QLabel("Please define the live Crowd Region of Interest (ROI) before starting the session.")
        instruction.setStyleSheet("font-size: 13px; font-weight: 600;")
        layout.addWidget(instruction)

        helper = QLabel("This uses a frozen camera frame. The mask is temporary and will be cleared when the session ends.")
        helper.setStyleSheet("color: rgba(220,220,220,0.62); font-size: 11px;")
        layout.addWidget(helper)

        notes = QLabel(
            "Notes:\n"
            "- Use a fixed camera position for the session\n"
            "- Mask only the active crowd area\n"
            "- Recreate the mask whenever the camera viewpoint changes"
        )
        notes.setStyleSheet("color: rgba(220,220,220,0.52); font-size: 10px;")
        layout.addWidget(notes)

        self.preview = SceneMaskPreviewLabel(self)
        layout.addWidget(self.preview, 1)
        if frame_qimage is not None and not frame_qimage.isNull():
            self.preview.set_frame(frame_qimage)
        self.preview.points_changed.connect(self._mark_mask_dirty)

        button_row = QHBoxLayout()
        self.undo_btn = QPushButton("Undo Point")
        self.clear_btn = QPushButton("Clear Points")
        self.save_btn = QPushButton("Save Mask")
        self.continue_btn = QPushButton("Continue Session")
        self.cancel_btn = QPushButton("Cancel")
        self.continue_btn.setEnabled(False)

        button_row.addWidget(self.undo_btn)
        button_row.addWidget(self.clear_btn)
        button_row.addStretch()
        button_row.addWidget(self.save_btn)
        button_row.addWidget(self.continue_btn)
        button_row.addWidget(self.cancel_btn)
        layout.addLayout(button_row)

        self.undo_btn.clicked.connect(self.preview.undo_point)
        self.clear_btn.clicked.connect(self.preview.clear_points)
        self.save_btn.clicked.connect(self.save_mask_with_confirmation)
        self.continue_btn.clicked.connect(self.continue_session)
        self.cancel_btn.clicked.connect(self.reject)

    def points(self):
        return self.preview.points()

    def _mark_mask_dirty(self):
        self.mask_saved = False
        self.continue_btn.setEnabled(False)

    def save_mask_with_confirmation(self):
        if not self._valid_polygon(self.preview.points()):
            QMessageBox.information(self, "Live Scene Masking", "Please define at least three ROI points before continuing.")
            return
        self.mask_saved = True
        self.continue_btn.setEnabled(True)
        QMessageBox.information(self, "Live Scene Masking", "Temporary live scene mask saved for this session.")

    def continue_session(self):
        if not self.mask_saved or not self._valid_polygon(self.preview.points()):
            QMessageBox.information(self, "Live Scene Masking", "Please save the ROI mask before continuing.")
            return
        self.accept()

    @staticmethod
    def _valid_polygon(points):
        return isinstance(points, list) and len(points) >= 3

class ControlsPopup(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.setObjectName("controls_popup")
        self.setStyleSheet(
            "QDialog#controls_popup { background-color: #1b1b1b; border: 1px solid rgba(255,255,255,0.18); border-radius: 6px; }"
            "QLabel { color: #cfd8dc; }"
        )
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(8)


class CameraSetupPopup(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint)
        self.setObjectName("camera_setup_popup")
        self.setStyleSheet(
            "QDialog#camera_setup_popup { background-color: #1b1b1b; border: 1px solid rgba(255,255,255,0.12); border-radius: 6px; }"
            "QLabel { color: #cfd8dc; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(4)

        title = QLabel("Camera Setup Guideline")
        title.setStyleSheet("color: #e8eaed; font-weight: 700; font-size: 13px;")
        layout.addWidget(title)

        # optional illustrative image beneath the title
        try:
            img_path = Path(__file__).resolve().parent / "cam_angle.png"
            if img_path.exists():
                pix = QPixmap(str(img_path))
                if not pix.isNull():
                    img_lbl = QLabel()
                    img_lbl.setPixmap(pix.scaledToWidth(318, Qt.TransformationMode.SmoothTransformation))
                    img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    img_lbl.setContentsMargins(0, 6, 0, 6)
                    layout.addWidget(img_lbl)
        except Exception:
            pass

        tips = [
            "Place camera above shoulder level",
            "Angle preferably at around 45 degrees",
            "Keep the crowd region fully visible",
            "Avoid camera shaking or movement",
            "Ensure lighting is sufficient",
            "Do not move camera after ROI masking",
        ]

        for t in tips:
            lbl = QLabel(f"- {t}")
            lbl.setWordWrap(True)
            lbl.setContentsMargins(0, 0, 0, 0)
            lbl.setStyleSheet("color: rgba(220,220,220,0.86); font-size: 11px; margin:0px; padding:0px;")
            layout.addWidget(lbl)

        self.setFixedWidth(347)
class VideoSourcePanel(QWidget):
    start_camera_requested = pyqtSignal()
    open_camera_requested = pyqtSignal()
    stop_camera_requested = pyqtSignal()
    overlay_toggled = pyqtSignal(bool)
    tracking_overlay_toggled = pyqtSignal(bool)
    export_session_requested = pyqtSignal()

    load_video_requested = pyqtSignal()
    play_requested = pyqtSignal()
    pause_requested = pyqtSignal()
    stop_video_requested = pyqtSignal()
    seek_requested = pyqtSignal(int)

    annotation_mode_toggled = pyqtSignal(bool)
    annotation_zone_changed = pyqtSignal(str)
    undo_annotation_requested = pyqtSignal()
    clear_annotation_requested = pyqtSignal()
    save_annotation_requested = pyqtSignal()
    annotation_point_clicked = pyqtSignal(int, int)
    mode_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._duration_seconds = 0
        self._is_updating_slider = False
        self._syncing_overlay = False
        self._syncing_tracking_overlay = False
        self._live_indicator_active = False
        self._live_indicator_level = 0.0
        self._live_indicator_dir = 1
        self._live_indicator_timer = QTimer(self)
        self._live_indicator_timer.setInterval(60)
        self._live_indicator_timer.timeout.connect(self._on_live_indicator_pulse)
        self._build_ui()
        self._connect_signals()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        mode_row = QHBoxLayout()
        self.live_mode_btn = QPushButton("Live Camera")
        self.video_mode_btn = QPushButton("Video File")
        self.validation_mode_btn = QPushButton("Analysis")
        self.live_mode_btn.setCheckable(True)
        self.video_mode_btn.setCheckable(True)
        self.validation_mode_btn.setCheckable(True)
        self.live_mode_btn.setChecked(True)

        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_group.addButton(self.live_mode_btn, 0)
        self.mode_group.addButton(self.video_mode_btn, 1)
        self.mode_group.addButton(self.validation_mode_btn, 2)

        mode_row.addWidget(self.live_mode_btn)
        mode_row.addWidget(self.video_mode_btn)
        mode_row.addWidget(self.validation_mode_btn)
        mode_row.addStretch()
        main_layout.addLayout(mode_row)

        self.video_display = AnnotatableVideoLabel("Video Preview")
        self.video_display.setMinimumSize(640, 280)
        self.video_display.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.video_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_display.setStyleSheet("background-color: black; color: white;")
        self.live_indicator_badge = QLabel("LIVE", self.video_display)
        self._apply_live_indicator_style(active=False, level=0.0)
        self.live_indicator_badge.adjustSize()
        self._update_live_indicator_position()
        main_layout.addWidget(self.video_display)

        self.control_stack = QStackedWidget()
        self.control_stack.addWidget(self._create_live_camera_controls())
        self.control_stack.addWidget(self._create_video_file_controls())
        self.control_stack.addWidget(self._create_validation_controls())
        self.control_stack.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Fixed
        )
        self.control_stack.setCurrentIndex(0)
        self._sync_control_stack_height()
        if self.control_stack.currentIndex() != 1 and self.video_tools_popup.isVisible():
            self.video_tools_popup.hide()
        if self.control_stack.currentIndex() != 0 and self.live_tools_popup.isVisible():
            self.live_tools_popup.hide()
        main_layout.addWidget(self.control_stack)

    def _create_live_camera_controls(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(8)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.live_status_label = QLabel("FPS: 00.0   |   Features: 000   |   Processing: 00 ms")
        self.live_status_label.setStyleSheet("color: rgba(220,220,220,0.50); font-size: 12px;")

        button_row = QHBoxLayout()
        self.open_camera_btn = QPushButton("Open Camera")
        self.start_camera_btn = QPushButton("Start Session")
        self.stop_camera_btn = QPushButton("Stop Session")
        self.export_session_btn = QPushButton("Export Session")
        self.export_session_btn.setEnabled(False)
        self.live_time_label = QLabel("00:00")
        self.live_time_label.setStyleSheet("color: #d0d0d0; font-size: 12px;")
        self.live_tools_btn = QToolButton()
        self.live_tools_btn.setText("...")
        self.live_tools_btn.setAutoRaise(True)
        self.live_tools_btn.setFixedSize(24, 24)
        self.live_tools_btn.setStyleSheet(
            "QToolButton { color: #cfd8dc; border: 1px solid rgba(255,255,255,0.14); border-radius: 4px; font-size: 16px; }"
            "QToolButton:hover { border-color: rgba(255,255,255,0.28); color: #ffffff; }"
        )
        button_row.addWidget(self.open_camera_btn)
        button_row.addWidget(self.start_camera_btn)
        button_row.addWidget(self.stop_camera_btn)
        button_row.addWidget(self.export_session_btn)
        button_row.addStretch()
        button_row.addWidget(self.live_time_label, 0, Qt.AlignmentFlag.AlignRight)
        button_row.addWidget(self.live_tools_btn)

        self.live_overlay_heatmap_chk = QCheckBox("Spatial Heatmap")
        self.live_tracking_overlay_chk = QCheckBox("Shi-Tomasi + RLOF Tracking")
        self.live_tools_popup = ControlsPopup(self)
        self.live_tools_popup.layout.addWidget(QLabel("Overlays"))
        self.live_tools_popup.layout.addWidget(self.live_overlay_heatmap_chk)
        self.live_tools_popup.layout.addWidget(self.live_tracking_overlay_chk)
        layout.addWidget(self.live_status_label)
        layout.addLayout(button_row)
        layout.addStretch(1)
        # Start Session should be unavailable until a preview is opened.
        self.start_camera_btn.setEnabled(False)
        # Stop Session should also be unavailable until a preview is opened (after Open Camera).
        self.stop_camera_btn.setEnabled(False)
        return page

    def _create_video_file_controls(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(8)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setObjectName("video_seek_progress")
        self.seek_slider.setRange(0, 0)
        self.seek_slider.setValue(0)
        # Display-only progress indicator: updated by playback, not user input.
        self.seek_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.seek_slider.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.seek_slider.setStyleSheet("""
            QSlider#video_seek_progress::groove:horizontal {
                height: 4px;
                background: #9a9a9a;
                border-radius: 2px;
            }
            QSlider#video_seek_progress::sub-page:horizontal {
                background: #f08a8a;
                border-radius: 2px;
            }
            QSlider#video_seek_progress::add-page:horizontal {
                background: #9a9a9a;
                border-radius: 2px;
            }
            QSlider#video_seek_progress::handle:horizontal {
                width: 1px;
                height: 1px;
                margin: -6px 0px;
                background: rgba(0, 0, 0, 0);
                border: 0px solid transparent;
                image: none;
            }
        """)

        button_row = QHBoxLayout()
        self.load_video_btn = QPushButton("Upload Video")
        self.play_video_btn = QPushButton("Play")
        self.pause_video_btn = QPushButton("Pause")
        self.stop_video_btn = QPushButton("Stop")
        self.export_results_btn = QPushButton("Export Results")
        self.export_results_btn.setEnabled(False)
        # Playback controls should be unavailable until a video is uploaded.
        self.play_video_btn.setEnabled(False)
        self.pause_video_btn.setEnabled(False)
        self.stop_video_btn.setEnabled(False)
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setStyleSheet("color: #d0d0d0; font-size: 12px;")
        self.video_tools_btn = QToolButton()
        self.video_tools_btn.setText("...")
        self.video_tools_btn.setAutoRaise(True)
        self.video_tools_btn.setFixedSize(24, 24)
        self.video_tools_btn.setStyleSheet(
            "QToolButton { color: #cfd8dc; border: 1px solid rgba(255,255,255,0.14); border-radius: 4px; font-size: 16px; }"
            "QToolButton:hover { border-color: rgba(255,255,255,0.28); color: #ffffff; }"
        )
        button_row.addWidget(self.load_video_btn)
        button_row.addWidget(self.play_video_btn)
        button_row.addWidget(self.pause_video_btn)
        button_row.addWidget(self.stop_video_btn)
        button_row.addWidget(self.export_results_btn)
        button_row.addStretch()
        button_row.addWidget(self.time_label, 0, Qt.AlignmentFlag.AlignRight)
        button_row.addWidget(self.video_tools_btn)

        self.video_overlay_heatmap_chk = QCheckBox("Spatial Heatmap")
        self.video_tracking_overlay_chk = QCheckBox("Shi-Tomasi + RLOF Tracking")
        self.annotation_mode_btn = QPushButton("Annotate Scene")
        self.annotation_mode_btn.setCheckable(True)
        self.annotation_zone_combo = QComboBox()
        for zone_name in ANNOTATION_ZONE_OPTIONS:
            self.annotation_zone_combo.addItem(ANNOTATION_ZONE_LABELS[zone_name], zone_name)
        self.undo_annotation_btn = QPushButton("Undo Point")
        self.clear_annotation_btn = QPushButton("Clear Zone")
        self.save_annotation_btn = QPushButton("Save Profile")

        self.video_tools_popup = ControlsPopup(self)
        self.video_tools_popup.layout.addWidget(QLabel("Overlays"))
        self.video_tools_popup.layout.addWidget(self.video_overlay_heatmap_chk)
        self.video_tools_popup.layout.addWidget(self.video_tracking_overlay_chk)
        self.video_tools_popup.layout.addSpacing(4)
        self.video_tools_popup.layout.addWidget(QLabel("Scene Annotation"))
        self.video_tools_popup.layout.addWidget(self.annotation_mode_btn)
        self.video_tools_popup.layout.addWidget(self.annotation_zone_combo)
        annotation_actions = QHBoxLayout()
        annotation_actions.addWidget(self.undo_annotation_btn)
        annotation_actions.addWidget(self.clear_annotation_btn)
        self.video_tools_popup.layout.addLayout(annotation_actions)
        self.video_tools_popup.layout.addWidget(self.save_annotation_btn)

        layout.addWidget(self.seek_slider)
        layout.addLayout(button_row)
        layout.addStretch(1)
        return page

    def _create_validation_controls(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(8)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.validation_progress_bar = QProgressBar()
        self.validation_progress_bar.setRange(0, 100)
        self.validation_progress_bar.setValue(0)
        self.validation_progress_bar.setTextVisible(False)
        self.validation_progress_bar.setFixedHeight(8)
        self.validation_progress_bar.setStyleSheet(
            "QProgressBar { border: none; background-color: #9a9a9a; border-radius: 1px; margin: 4px 12px 0 12px; }"
            "QProgressBar::chunk { background-color: #f08a8a; border-radius: 1px; }"
        )

        button_row = QHBoxLayout()
        self.validation_load_video_btn = QPushButton("Upload Video")
        self.validation_play_btn = QPushButton("Run Analysis")
        self.validation_export_btn = QPushButton("Export Results")
        self.validation_export_btn.setEnabled(False)
        self.validation_time_label = QLabel("0%")
        self.validation_time_label.setStyleSheet("color: #d0d0d0; font-size: 12px;")
        button_row.addWidget(self.validation_load_video_btn)
        button_row.addWidget(self.validation_play_btn)
        button_row.addWidget(self.validation_export_btn)
        button_row.addStretch()
        button_row.addWidget(self.validation_time_label, 0, Qt.AlignmentFlag.AlignRight)

        layout.addWidget(self.validation_progress_bar)
        layout.addSpacing(6)
        layout.addLayout(button_row)
        layout.addStretch(1)
        return page


    def _connect_signals(self):
        self.mode_group.idClicked.connect(self.set_mode)

        self.open_camera_btn.clicked.connect(self.open_camera_requested.emit)
        # Enable Stop Session button when Open Camera is clicked
        self.open_camera_btn.clicked.connect(lambda: self.stop_camera_btn.setEnabled(True))
        self.start_camera_btn.clicked.connect(self.start_camera_requested.emit)
        self.stop_camera_btn.clicked.connect(self.stop_camera_requested.emit)
        # Disable Stop Session button after it is clicked
        self.stop_camera_btn.clicked.connect(lambda: self.stop_camera_btn.setEnabled(False))
        # Enable export button after stop is clicked (UI-level)
        self.stop_camera_btn.clicked.connect(lambda: self.export_session_btn.setEnabled(True))
        self.live_overlay_heatmap_chk.toggled.connect(self._on_live_overlay_toggled)
        self.video_overlay_heatmap_chk.toggled.connect(self._on_video_overlay_toggled)
        self.live_tracking_overlay_chk.toggled.connect(self._on_live_tracking_overlay_toggled)
        self.video_tracking_overlay_chk.toggled.connect(self._on_video_tracking_overlay_toggled)

        self.load_video_btn.clicked.connect(self.load_video_requested.emit)
        # Enable playback controls once a video is uploaded
        self.load_video_btn.clicked.connect(lambda: (self.play_video_btn.setEnabled(True), self.pause_video_btn.setEnabled(True), self.stop_video_btn.setEnabled(True), self.export_results_btn.setEnabled(False)))
        self.play_video_btn.clicked.connect(self.play_requested.emit)
        self.pause_video_btn.clicked.connect(self.pause_requested.emit)
        self.stop_video_btn.clicked.connect(self.stop_video_requested.emit)
        # Disable playback controls after Stop is clicked
        self.stop_video_btn.clicked.connect(lambda: (self.play_video_btn.setEnabled(False), self.pause_video_btn.setEnabled(False), self.stop_video_btn.setEnabled(False)))

        self.seek_slider.sliderMoved.connect(self._on_slider_preview)
        self.seek_slider.sliderReleased.connect(self._on_slider_released)

        self.annotation_mode_btn.toggled.connect(self.annotation_mode_toggled.emit)
        self.annotation_zone_combo.currentIndexChanged.connect(lambda _: self.annotation_zone_changed.emit(self.annotation_zone_combo.currentData()))
        self.undo_annotation_btn.clicked.connect(self.undo_annotation_requested.emit)
        self.clear_annotation_btn.clicked.connect(self.clear_annotation_requested.emit)
        self.save_annotation_btn.clicked.connect(self.save_annotation_requested.emit)
        self.live_tools_btn.clicked.connect(self._toggle_live_tools_popup)
        self.video_tools_btn.clicked.connect(self._toggle_video_tools_popup)
        self.video_display.clicked.connect(self.annotation_point_clicked.emit)
        # Export session data
        if hasattr(self, "export_session_btn"):
            self.export_session_btn.clicked.connect(self.export_session_requested.emit)
        # Export results for video files
        if hasattr(self, "export_results_btn"):
            self.export_results_btn.clicked.connect(self.export_session_requested.emit)

    def _toggle_live_tools_popup(self):
        if self.live_tools_popup.isVisible():
            self.live_tools_popup.hide()
            return
        if self.video_tools_popup.isVisible():
            self.video_tools_popup.hide()
        anchor = self.live_tools_btn.mapToGlobal(self.live_tools_btn.rect().bottomRight())
        self.live_tools_popup.adjustSize()
        popup_pos = anchor - QPoint(self.live_tools_popup.width(), -6)
        self.live_tools_popup.move(popup_pos)
        self.live_tools_popup.show()

    def _toggle_video_tools_popup(self):
        if self.video_tools_popup.isVisible():
            self.video_tools_popup.hide()
            return
        if self.live_tools_popup.isVisible():
            self.live_tools_popup.hide()
        anchor = self.video_tools_btn.mapToGlobal(self.video_tools_btn.rect().bottomRight())
        self.video_tools_popup.adjustSize()
        popup_pos = anchor - QPoint(self.video_tools_popup.width(), -6)
        self.video_tools_popup.move(popup_pos)
        self.video_tools_popup.show()

    def set_mode(self, index: int):
        self.control_stack.setCurrentIndex(index)
        self.live_mode_btn.setChecked(index == 0)
        self.video_mode_btn.setChecked(index == 1)
        self.validation_mode_btn.setChecked(index == 2)
        self.live_indicator_badge.setVisible(index == 0 and self.control_stack.currentIndex() == 0)
        self._update_live_indicator_position()
        self._sync_control_stack_height()
        if self.control_stack.currentIndex() != 1 and self.video_tools_popup.isVisible():
            self.video_tools_popup.hide()
        if self.control_stack.currentIndex() != 0 and self.live_tools_popup.isVisible():
            self.live_tools_popup.hide()
        self.mode_changed.emit(index)

    def set_annotation_mode(self, enabled: bool):
        # Update both video and live annotation toggles; live annotations are transient
        try:
            self.annotation_mode_btn.setChecked(enabled)
        except Exception:
            pass
        if hasattr(self, "live_annotation_mode_btn"):
            try:
                self.live_annotation_mode_btn.setChecked(enabled)
            except Exception:
                pass
        if not enabled:
            if self.video_tools_popup.isVisible():
                self.video_tools_popup.hide()
            if hasattr(self, "live_tools_popup") and self.live_tools_popup.isVisible():
                self.live_tools_popup.hide()
    def set_duration_seconds(self, total_seconds: int):
        self._duration_seconds = max(0, int(total_seconds))
        self.seek_slider.setRange(0, self._duration_seconds)
        self._update_time_labels(self.seek_slider.value())

    def set_current_position_seconds(self, seconds: int):
        clamped = max(0, min(int(seconds), self._duration_seconds))
        self._is_updating_slider = True
        self.seek_slider.setValue(clamped)
        self._is_updating_slider = False
        self._update_time_labels(clamped)

    def _on_slider_preview(self, value: int):
        self._update_time_labels(value)

    def _on_slider_released(self):
        if self._is_updating_slider:
            return
        value = self.seek_slider.value()
        self._update_time_labels(value)
        self.seek_requested.emit(value)

    def map_display_to_frame_point(self, widget_x: int, widget_y: int, frame_size: tuple[int, int]):
        frame_w, frame_h = frame_size
        if frame_w <= 0 or frame_h <= 0:
            return None

        label_w = self.video_display.width()
        label_h = self.video_display.height()
        scale = min(label_w / frame_w, label_h / frame_h)
        display_w = int(frame_w * scale)
        display_h = int(frame_h * scale)
        x0 = (label_w - display_w) / 2.0
        y0 = (label_h - display_h) / 2.0

        if widget_x < x0 or widget_x > x0 + display_w or widget_y < y0 or widget_y > y0 + display_h:
            return None

        rel_x = (widget_x - x0) / max(1, display_w)
        rel_y = (widget_y - y0) / max(1, display_h)
        frame_x = int(rel_x * frame_w)
        frame_y = int(rel_y * frame_h)
        frame_x = max(0, min(frame_w - 1, frame_x))
        frame_y = max(0, min(frame_h - 1, frame_y))
        return frame_x, frame_y
    def _update_time_labels(self, current_seconds: int):
        current = self._format_time(current_seconds)
        total = self._format_time(self._duration_seconds)
        self.time_label.setText(f"{current} / {total}")
        
    def _sync_control_stack_height(self):
        page = self.control_stack.currentWidget()
        if page is None:
            return
        self.control_stack.setFixedHeight(page.sizeHint().height())

    def _update_live_indicator_position(self):
        self.live_indicator_badge.adjustSize()
        x = self.video_display.width() - self.live_indicator_badge.width() - 8
        self.live_indicator_badge.move(max(0, x), 8)

    def _on_live_overlay_toggled(self, checked: bool):
        if self._syncing_overlay:
            return
        self._syncing_overlay = True
        self.video_overlay_heatmap_chk.setChecked(checked)
        self._syncing_overlay = False
        self.overlay_toggled.emit(checked)

    def _on_video_overlay_toggled(self, checked: bool):
        if self._syncing_overlay:
            return
        self._syncing_overlay = True
        self.live_overlay_heatmap_chk.setChecked(checked)
        self._syncing_overlay = False
        self.overlay_toggled.emit(checked)

    def is_overlay_enabled(self):
        return self.live_overlay_heatmap_chk.isChecked()

    def _on_live_tracking_overlay_toggled(self, checked: bool):
        if self._syncing_tracking_overlay:
            return
        self._syncing_tracking_overlay = True
        self.video_tracking_overlay_chk.setChecked(checked)
        self._syncing_tracking_overlay = False
        self.tracking_overlay_toggled.emit(checked)

    def _on_video_tracking_overlay_toggled(self, checked: bool):
        if self._syncing_tracking_overlay:
            return
        self._syncing_tracking_overlay = True
        self.live_tracking_overlay_chk.setChecked(checked)
        self._syncing_tracking_overlay = False
        self.tracking_overlay_toggled.emit(checked)

    def is_tracking_overlay_enabled(self):
        return self.live_tracking_overlay_chk.isChecked()

    def set_live_elapsed_seconds(self, seconds: int):
        self.live_time_label.setText(f"{self._format_time(seconds)}")

    def set_live_indicator_active(self, active: bool):
        self._live_indicator_active = bool(active)
        self._live_indicator_level = 0.0
        self._live_indicator_dir = 1
        if self._live_indicator_active:
            self._live_indicator_timer.start()
        else:
            self._live_indicator_timer.stop()
        self._apply_live_indicator_style(self._live_indicator_active, self._live_indicator_level)

    def _on_live_indicator_pulse(self):
        if not self._live_indicator_active:
            return
        step = 0.06
        self._live_indicator_level += step * self._live_indicator_dir
        if self._live_indicator_level >= 1.0:
            self._live_indicator_level = 1.0
            self._live_indicator_dir = -1
        elif self._live_indicator_level <= 0.0:
            self._live_indicator_level = 0.0
            self._live_indicator_dir = 1
        self._apply_live_indicator_style(True, self._live_indicator_level)

    def _apply_live_indicator_style(self, active: bool, level: float):
        if not active:
            bg = "rgba(90, 90, 90, 180)"
        else:
            level = max(0.0, min(1.0, float(level)))
            red = int(130 + (100 * level))
            alpha = int(180 + (60 * level))
            bg = f"rgba({red}, 0, 0, {alpha})"
        self.live_indicator_badge.setStyleSheet(
            f"background-color: {bg}; color: white; "
            "font-size: 11px; font-weight: 700; padding: 2px 6px; border-radius: 4px;"
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_live_indicator_position()

    @staticmethod
    def _format_time(total_seconds: int):
        total_seconds = max(0, int(total_seconds))
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes:02d}:{seconds:02d}"

def boxed(title, widget, info_text=None, tier="secondary"):
    tier_styles = {
        "primary": {"border": "rgba(255,255,255,.70)", "thickness": 1, "title_color": "#e8eaed"},
        "secondary": {"border": "rgba(255,255,255,0.40)", "thickness": 1, "title_color": "#cfd8dc"},
        "tertiary": {"border": "rgba(255,255,255,0.30)", "thickness": 1, "title_color": "#b0bec5"},
    }
    style = tier_styles.get(tier, tier_styles["secondary"])

    box = QGroupBox(title)
    box.setObjectName("panel_box")
    box.setStyleSheet(f"""
        QGroupBox#panel_box {{
            background-color: #1E1E1E;
            border: {style["thickness"]}px solid {style["border"]};
            border-radius: 4px;
            margin-top: 8px;
        }}
        QGroupBox#panel_box::title {{
            subcontrol-origin: margin;
            left: 8px;
            padding: 0 2px;
            color: {style["title_color"]};
        }}
    """)

    if tier == "primary":
        glow = QGraphicsDropShadowEffect(box)
        glow.setBlurRadius(20)
        glow.setOffset(0, 0)
        glow.setColor(QColor(255, 255, 255, 20))
        box.setGraphicsEffect(glow)
    layout = QVBoxLayout()
    if info_text:
        info_row = QHBoxLayout()
        info_row.setContentsMargins(0, 0, 0, 0)
        info_row.addStretch()

        info_btn = QToolButton()
        info_btn.setText("i")
        info_btn.setToolTip(info_text)
        info_btn.setAutoRaise(True)
        info_btn.setCursor(Qt.CursorShape.ArrowCursor)
        info_btn.setFixedSize(16, 16)
        info_btn.setStyleSheet(
            "QToolButton { color: #9aa0a6; border: none; font-weight: bold; }"
            "QToolButton:hover { color: #e8eaed; }"
        )
        info_row.addWidget(info_btn, 0, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(info_row)

    layout.addWidget(widget)
    box.setLayout(layout)
    return box

def placeholder_widget(title, subtitle="Awaiting validation analysis"):
    panel = QWidget()
    panel.setObjectName("validation_placeholder")
    panel.setStyleSheet("""
        QWidget#validation_placeholder {
            background-color: rgba(0, 0, 0, 0.10);
            border: 1px dashed rgba(255, 255, 255, 0.14);
            border-radius: 4px;
        }
    """)
    layout = QVBoxLayout(panel)
    layout.setContentsMargins(12, 12, 12, 12)
    layout.setSpacing(4)
    layout.addStretch()

    title_label = QLabel(title)
    title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    title_label.setStyleSheet("color: #cfd8dc; font-size: 14px; font-weight: 700;")
    layout.addWidget(title_label)

    subtitle_label = QLabel(subtitle)
    subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    subtitle_label.setWordWrap(True)
    subtitle_label.setStyleSheet("color: rgba(220, 220, 220, 0.48); font-size: 12px;")
    layout.addWidget(subtitle_label)
    layout.addStretch()
    return panel

class ResultImageLabel(QLabel):
    def __init__(
        self,
        title: str,
        subtitle: str = "Awaiting validation analysis",
        parent=None,
        fill_to_bounds: bool = False,
    ):
        super().__init__(parent)
        self._source_pixmap = None
        self._placeholder_title = title
        self._placeholder_subtitle = subtitle
        self._fill_to_bounds = bool(fill_to_bounds)

        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)
        self.setMinimumSize(160, 120)

        if self._fill_to_bounds:
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self.setScaledContents(True)
            self.setStyleSheet("""
                QLabel {
                    background-color: transparent;
                    border: none;
                    padding: 0px;
                }
            """)
        else:
            self.setScaledContents(False)
            self.setStyleSheet("""
                QLabel {
                    background-color: rgba(0, 0, 0, 0.10);
                    border: 1px dashed rgba(255, 255, 255, 0.14);
                    border-radius: 4px;
                    color: rgba(220, 220, 220, 0.58);
                    font-size: 12px;
                    padding: 12px;
                }
            """)

        self.set_placeholder(title, subtitle)
    def set_placeholder(self, title: str = None, subtitle: str = None):
        if title is not None:
            self._placeholder_title = title
        if subtitle is not None:
            self._placeholder_subtitle = subtitle
        self._source_pixmap = None
        self.clear()

        title_text = (self._placeholder_title or "").strip()
        subtitle_text = (self._placeholder_subtitle or "").strip()

        if not title_text and not subtitle_text:
            self.setText("")
        elif title_text and subtitle_text:
            self.setText(f"<b>{title_text}</b><br>{subtitle_text}")
        elif title_text:
            self.setText(f"<b>{title_text}</b>")
        else:
            self.setText(subtitle_text)
    def set_result_pixmap(self, pixmap: QPixmap):
        self._source_pixmap = pixmap
        self._refresh_pixmap()

    def _refresh_pixmap(self):
        if self._source_pixmap is None or self._source_pixmap.isNull():
            return

        if self._fill_to_bounds:
            self.setPixmap(self._source_pixmap)
            return

        scaled = self._source_pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_pixmap()


class ValidationSummaryCard(QWidget):
    """Compact validation summary card shown in the validation tab.

    The widget is deliberately compact and styled to match the dark theme.
    It can populate itself from an output directory produced by the validation
    worker by heuristically parsing any metrics CSV found there.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("validation_summary_card")
        self.setMinimumSize(320, 160)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("""
            QWidget#validation_summary_card {
                background-color: rgba(0,0,0,0.06);
                border: 1px dashed rgba(255,255,255,0.12);
                border-radius: 4px;
                color: rgba(220,220,220,0.9);
                padding: 10px;
            }
            QLabel#vs_title {
                color: #e8eaed;
                font-weight: 700;
                font-size: 13px;
            }
            QLabel#vs_section {
                color: rgba(220,220,220,0.82);
                font-size: 11px;
            }
            QLabel.vs_mono { font-family: monospace; font-size: 11px; }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 6)
        layout.setSpacing(6)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.dom_regime = QLabel("<b>Dominant Regime:</b> <span class=vs_mono>Unknown</span>")
        self.dom_regime.setObjectName("vs_section")
        self.dom_regime.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self.dom_regime)

        self.transitions = QLabel("<b>Transition Observed:</b> <span class=vs_mono>-</span>")
        self.transitions.setObjectName("vs_section")
        self.transitions.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self.transitions)

        self.evidence = QLabel()
        self.evidence.setObjectName("vs_section")
        self.evidence.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self.evidence)

        self.interpretation = QLabel()
        self.interpretation.setObjectName("vs_section")
        self.interpretation.setWordWrap(True)
        self.interpretation.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self.interpretation)

        bottom_row = QHBoxLayout()
        self.confidence = QLabel("<b>Confidence:</b> <span class=vs_mono>-</span>")
        self.confidence.setObjectName("vs_section")
        self.confidence.setTextFormat(Qt.TextFormat.RichText)
        bottom_row.addWidget(self.confidence)
        layout.addLayout(bottom_row)
        layout.addStretch(1)

        self.set_placeholder()

    def set_placeholder(self, dom_regime: str = "Unknown", transitions: str = "-"):
        self.dom_regime.setText(f"<b>Dominant Regime:</b> <span class=vs_mono>{dom_regime}</span>")
        self.transitions.setText(f"<b>Transition Observed:</b> <span class=vs_mono>{transitions}</span>")
        self.evidence.setText("<b>Key Evidence:</b><br>- No validation metrics available")
        self.interpretation.setText("<b>Interpretation:</b> <i>Awaiting analysis results for a concise interpretation.</i>")
        self.confidence.setText("<b>Confidence:</b> <span class=vs_mono>-</span>")

    def update_from_dir(self, out_dir: Path):
        if out_dir is None or not out_dir.exists():
            self.set_placeholder()
            return

        data_file = None
        for name in ("data.csv", "metrics.csv"):
            candidate = out_dir / name
            if candidate.exists():
                data_file = candidate
                break

        if data_file is None:
            self.set_placeholder()
            return

        rows = []
        try:
            with open(data_file, newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                rows = list(reader)
        except Exception:
            self.set_placeholder()
            return

        if not rows:
            self.set_placeholder()
            return

        def first_series(keys):
            for key in keys:
                vals = []
                for r in rows:
                    v = r.get(key)
                    if v is None or str(v).strip() == "":
                        continue
                    try:
                        vals.append(float(v))
                    except Exception:
                        pass
                if vals:
                    return vals
            return []

        s_vals = first_series(["S", "s", "s_mean", "spatial"])
        c_vals = first_series(["C", "c", "c_mean", "coherence"])
        cci_vals = first_series(["CCI", "cci", "cci_mean"])
        si_vals = first_series(["SI", "si", "stability"])
        proj_vals = first_series(["Projected_CCI", "projected_cci", "Forecast", "forecast"])

        if not (s_vals and c_vals and cci_vals and si_vals and proj_vals):
            self.set_placeholder()
            return

        n = min(len(s_vals), len(c_vals), len(cci_vals), len(si_vals), len(proj_vals))
        s_vals = s_vals[:n]
        c_vals = c_vals[:n]
        cci_vals = cci_vals[:n]
        si_vals = si_vals[:n]
        proj_vals = proj_vals[:n]

        if n < 8:
            self.set_placeholder()
            return

        def mean(values):
            return sum(values) / max(1, len(values))

        def window_trend(values, threshold):
            w = max(3, len(values) // 5)
            start_mean = mean(values[:w])
            end_mean = mean(values[-w:])
            delta = end_mean - start_mean
            if delta > threshold:
                label = "rising"
            elif delta < -threshold:
                label = "declining"
            else:
                label = "stable"
            return label, delta

        avg_s = mean(s_vals)
        avg_c = mean(c_vals)
        avg_cci = mean(cci_vals)
        avg_si = mean(si_vals)

        c_label, c_delta = window_trend(c_vals, threshold=0.02)
        cci_label, cci_delta = window_trend(cci_vals, threshold=0.01)
        si_label, si_delta = window_trend(si_vals, threshold=0.01)
        forecast_label, forecast_delta = window_trend(proj_vals, threshold=0.01)

        # Use regime tendencies when present; otherwise infer from signal behavior.
        gas_t = first_series(["R_gas", "r_gas"])
        fluid_t = first_series(["R_fluid", "r_fluid"])
        granular_t = first_series(["R_granular", "r_granular"])

        def regime_name(name):
            if name == "gas":
                return "Gas-like"
            if name == "fluid":
                return "Fluid-like"
            return "Granular-like"

        def infer_regime_by_signals(s_mean, c_mean, cci_mean, si_mean, c_tr, cci_tr, si_tr):
            if s_mean >= 0.55 and c_tr <= -0.03 and cci_tr >= 0.015 and si_tr <= -0.01:
                return "granular"
            if s_mean <= 0.30 and cci_mean <= 0.18 and si_mean >= 0.85:
                return "gas"
            if c_mean >= 0.70 and s_mean >= 0.35 and abs(cci_tr) <= 0.02 and si_mean >= 0.80:
                return "fluid"

            gas_score = (1.0 - s_mean) + (1.0 - cci_mean) + si_mean
            fluid_score = c_mean + (1.0 - abs(cci_tr) * 8.0) + si_mean
            granular_score = s_mean + max(0.0, -c_tr) * 4.0 + max(0.0, cci_tr) * 8.0 + max(0.0, -si_tr) * 8.0

            if gas_score >= fluid_score and gas_score >= granular_score:
                return "gas"
            if fluid_score >= granular_score:
                return "fluid"
            return "granular"

        if gas_t and fluid_t and granular_t:
            m_g = mean(gas_t)
            m_f = mean(fluid_t)
            m_gr = mean(granular_t)
            tendency_map = {"gas": m_g, "fluid": m_f, "granular": m_gr}
            ranked = sorted(tendency_map.items(), key=lambda kv: kv[1], reverse=True)
            dom_key, dom_val = ranked[0]
            second_val = ranked[1][1]
            dominant = regime_name(dom_key)

            # Confidence from tendency separation.
            margin = max(0.0, dom_val - second_val)
            confidence_pct = int(max(55, min(99, 55 + margin * 100)))

            seg = max(5, n // 3)
            early_map = {
                "gas": mean(gas_t[:seg]),
                "fluid": mean(fluid_t[:seg]),
                "granular": mean(granular_t[:seg]),
            }
            late_map = {
                "gas": mean(gas_t[-seg:]),
                "fluid": mean(fluid_t[-seg:]),
                "granular": mean(granular_t[-seg:]),
            }
            early_key = max(early_map, key=early_map.get)
            late_key = max(late_map, key=late_map.get)
            transition_text = f"{regime_name(early_key)} -> {regime_name(late_key)}"
        else:
            dom_key = infer_regime_by_signals(avg_s, avg_c, avg_cci, avg_si, c_delta, cci_delta, si_delta)
            dominant = regime_name(dom_key)

            seg = max(5, n // 3)
            early_key = infer_regime_by_signals(
                mean(s_vals[:seg]), mean(c_vals[:seg]), mean(cci_vals[:seg]), mean(si_vals[:seg]),
                window_trend(c_vals[:seg], 0.02)[1], window_trend(cci_vals[:seg], 0.01)[1], window_trend(si_vals[:seg], 0.01)[1]
            )
            late_key = infer_regime_by_signals(
                mean(s_vals[-seg:]), mean(c_vals[-seg:]), mean(cci_vals[-seg:]), mean(si_vals[-seg:]),
                window_trend(c_vals[-seg:], 0.02)[1], window_trend(cci_vals[-seg:], 0.01)[1], window_trend(si_vals[-seg:], 0.01)[1]
            )
            transition_text = f"{regime_name(early_key)} -> {regime_name(late_key)}"

            # Confidence from rule consistency strength.
            consistency = 0
            consistency += 1 if cci_label in ("Rising", "Declining") else 0
            consistency += 1 if si_label in ("Rising", "Declining") else 0
            consistency += 1 if c_label in ("Rising", "Declining") else 0
            consistency += 1 if abs(forecast_delta) >= 0.01 else 0
            confidence_pct = int(max(58, min(92, 58 + consistency * 8)))

        evidence_lines = [
            f"- Mean S: {avg_s:.3f}",
            f"- Mean C: {avg_c:.3f} ({c_label}, {c_delta:+.3f})",
            f"- Mean CCI: {avg_cci:.3f} ({cci_label}, {cci_delta:+.3f})",
            f"- Mean SI: {avg_si:.3f} ({si_label}, {si_delta:+.3f})",
            f"- Forecast Trend: {forecast_label} ({forecast_delta:+.3f})",
        ]

        if dominant == "Fluid-like":
            interpretation = (
                "High coherence with moderate-to-high spatial constraint, non-escalating pressure trajectory, "
                "and sustained stability indicates coordinated collective motion consistent with fluid-like behavior."
            )
        elif dominant == "Gas-like":
            interpretation = (
                "Low spatial constraint and low crowd pressure with high stability indicate weak coupling between agents, "
                "consistent with gas-like crowd dynamics."
            )
        else:
            interpretation = (
                "Elevated spatial constraint with coherence decline, pressure growth, and stability reduction indicates "
                "increasing contact-dominated interactions consistent with granular-like behavior."
            )

        if transition_text.split(" -> ")[0] != transition_text.split(" -> ")[1]:
            interpretation += f" Temporal evolution indicates a transition from {transition_text.split(' -> ')[0]} to {transition_text.split(' -> ')[1]}."
        self.dom_regime.setText(f"<b>Dominant Regime:</b> <span class=vs_mono>{dominant}</span>")
        self.transitions.setText(f"<b>Transition Observed:</b> <span class=vs_mono>{transition_text}</span>")
        self.evidence.setText("<b>Key Evidence:</b><br>" + "<br>".join(evidence_lines))
        self.interpretation.setText(f"<b>Interpretation:</b> {interpretation}")
        self.confidence.setText(f"<b>Confidence:</b> <span class=vs_mono>{confidence_pct}%</span>")

class PreviewThread(QThread):
    """Simple preview thread that only captures frames and emits QImage frames.

    Used by the 'Open Camera' action to display live video without running analytics.
    """
    frame_signal = pyqtSignal(QImage)

    def __init__(self, source):
        super().__init__()
        self.source = source
        self._running = True

    def run(self):
        import time
        while self._running:
            ret, frame = self.source.read()
            if not ret:
                time.sleep(0.03)
                continue
            if getattr(self.source, "mirror", False):
                frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            try:
                self.frame_signal.emit(qimg.copy())
            except Exception:
                pass
            time.sleep(0.03)

    def stop(self):
        self._running = False
        # Do not release the source here; ownership may be transferred to the
        # analytics thread (CameraThread) to avoid reopening the device.
        return

class CrowdTuneUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CrowdTune")
        self.setObjectName("crowdtune_root")
        self.setStyleSheet("""
            QWidget#crowdtune_root {
                background-color: #141414;
            }
            QPushButton:checked {
                background-color: #f08a8a;
                color: white;
            }
        """)
        icon_path = Path(__file__).resolve().parent / "CrowdTune Icon - White.ico"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        self.setMinimumSize(1200, 700)

        # --------- Video Source Panel ---------
        self.video_panel = VideoSourcePanel()
        self.video_panel.start_camera_requested.connect(self.start_camera)
        self.video_panel.open_camera_requested.connect(self.open_camera)
        self.video_panel.stop_camera_requested.connect(self.stop_source)
        self.video_panel.overlay_toggled.connect(self.on_overlay_toggled)
        self.video_panel.tracking_overlay_toggled.connect(self.on_tracking_overlay_toggled)
        self.video_panel.load_video_requested.connect(self.load_video)
        self.video_panel.play_requested.connect(self.play_video)
        self.video_panel.pause_requested.connect(self.pause_video)
        self.video_panel.stop_video_requested.connect(self.stop_source)
        self.video_panel.seek_requested.connect(self.seek_video)
        self.video_panel.annotation_mode_toggled.connect(self.on_annotation_mode_toggled)
        self.video_panel.annotation_zone_changed.connect(self.on_annotation_zone_changed)
        self.video_panel.undo_annotation_requested.connect(self.undo_annotation_point)
        self.video_panel.clear_annotation_requested.connect(self.clear_annotation_zone)
        self.video_panel.save_annotation_requested.connect(self.save_annotation_profile)
        self.video_panel.export_session_requested.connect(self.export_session)
        self.video_panel.annotation_point_clicked.connect(self.on_annotation_point_clicked)
        self.video_panel.mode_changed.connect(self.on_source_mode_changed)
        self.video_panel.validation_load_video_btn.clicked.connect(self.load_validation_video)
        self.video_panel.validation_play_btn.clicked.connect(self.run_validation_analysis)
        # Validation 'Run Analysis' should be disabled until a validation video is uploaded and previewed.
        self.video_panel.validation_play_btn.setEnabled(False)
        # Wire validation export signal to handler
        self.video_panel.validation_export_btn.clicked.connect(lambda: self.export_validation_results())

        # --------- Widgets ---------
        self.risk_widget = RiskIndexWidget()
        self.advisory_widget = AdvisoryWidget()
        self.cci_widget = CCIWidget()
        self.stability_widget = StabilityWidget()
        self.forecast_widget = ForecastWidget()
        self.ternary_widget = TernaryPlotWidget()
        self.regime_map_widget = RegimeStateMapWidget()
        self.s_value_label = QLabel("S: 0.000")
        self.k_value_label = QLabel("K: 0.000")
        self.c_value_label = QLabel("C: 0.000")

        self.s_value_label.setStyleSheet("color: rgba(220,220,220,0.50); font-size: 11px;")
        self.k_value_label.setStyleSheet("color: rgba(220,220,220,0.50); font-size: 11px;")
        self.c_value_label.setStyleSheet("color: rgba(220,220,220,0.50); font-size: 11px;")
        self.ternary_title_label = QLabel("Signal Composition (S-K-C)")
        self.regime_map_title_label = QLabel("Regime State Map")
        self.ternary_title_label.setStyleSheet("color: #9aa0a6; font-size: 13px; font-weight: 600;")
        self.regime_map_title_label.setStyleSheet("color: #9aa0a6; font-size: 13px; font-weight: 600;")
        self.ternary_title_label.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom)
        self.regime_map_title_label.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom)

        skc_values_row = QHBoxLayout()
        skc_values_row.setContentsMargins(0, 0, 0, 0)
        skc_values_row.setSpacing(22)
        skc_values_row.addStretch()
        skc_values_row.addWidget(self.s_value_label)
        skc_values_row.addWidget(self.k_value_label)
        skc_values_row.addWidget(self.c_value_label)
        skc_values_row.addStretch()

        ternary_block = QWidget()
        ternary_block_layout = QVBoxLayout(ternary_block)
        ternary_block_layout.setContentsMargins(0, 0, 0, 0)
        ternary_block_layout.setSpacing(4)
        ternary_block_layout.addLayout(skc_values_row)
        ternary_block_layout.addWidget(self.ternary_title_label)
        ternary_block_layout.addWidget(self.ternary_widget)

        regime_block = QWidget()
        regime_block_layout = QVBoxLayout(regime_block)
        regime_block_layout.setContentsMargins(0, 0, 0, 0)
        regime_block_layout.setSpacing(4)
        regime_block_layout.addWidget(self.regime_map_title_label)
        regime_block_layout.addWidget(self.regime_map_widget)

        skc_diagrams_row = QHBoxLayout()
        skc_diagrams_row.setContentsMargins(0, 0, 0, 0)
        skc_diagrams_row.setSpacing(12)
        skc_diagrams_row.addWidget(ternary_block, 9)
        skc_diagrams_row.addStretch(1)
        skc_diagrams_row.addWidget(regime_block, 11)

        skc_panel = QWidget()
        skc_layout = QVBoxLayout(skc_panel)
        skc_layout.setContentsMargins(0, 0, 0, 0)
        skc_layout.setSpacing(8)
        skc_layout.addLayout(skc_diagrams_row)

        self.validation_summary_card = ValidationSummaryCard(self)

        video_feed_tip = (
            "Input source for crowd analysis. "
            "Optional overlays display density maps and feature tracking."
        )
        ternary_tip = (
            "Ternary plot showing the interaction of Spatial constraint (S), Kinetic activity (K) "
            "and Collective Coherence (C) defining crowd regime."
        )
        risk_tip = (
            "Composite indicator measuring the proximity of the crowd to structural instability."
        )
        advisory_tip = (
            "Operational guidance generated from crowd signals, CCI levels and short-term forecasts."
        )
        stability_tip = (
            "Time series showing temporal stability of crowd structure. "
            "Lower values indicate unstable or turbulent dynamics."
        )
        cci_tip = (
            "Normalized measure of crowd pressure. "
            "Higher values indicate increasing congestion and compression risk."
        )
        forecast_tip = (
            "Projection of crowd structural instability over the next 3 seconds based on current dynamics."
        )


        validation_results_tip = (
            "Aggregated validation visualization for SKC, CCI, SI, Risk and Forecast signals produced from the uploaded video analysis run."
        )

        validation_summary_tip = (
            "Summary of the video analysis including dominant regime, mean values for SKC and Forecast trends."
        )
        risk_box = boxed("Risk Index", self.risk_widget, risk_tip, tier="primary")
        advisory_box = boxed("Advisory", self.advisory_widget, advisory_tip, tier="tertiary")
        stability_box = boxed("Stability Index", self.stability_widget, stability_tip, tier="primary")
        cci_box = boxed("Crowd Constraint Index", self.cci_widget, cci_tip, tier="secondary")
        forecast_box = boxed("Short-Horizon Instability Forecast", self.forecast_widget, forecast_tip, tier="tertiary")
        video_feed_box = boxed("Video Feed", self.video_panel, video_feed_tip, tier="primary")
        composition_box = boxed("Crowd State Space", skc_panel, ternary_tip, tier="secondary")
        validation_summary_box = boxed("Analysis Summary", self.validation_summary_card, validation_summary_tip, tier="secondary")
        self.left_lower_stack = QStackedWidget()
        self.left_lower_stack.addWidget(composition_box)
        self.left_lower_stack.addWidget(validation_summary_box)

        self.risk_splitter = QSplitter(Qt.Orientation.Vertical)
        self.risk_splitter.setChildrenCollapsible(False)
        self.risk_splitter.addWidget(risk_box)
        self.risk_splitter.addWidget(advisory_box)
        self.risk_splitter.setStretchFactor(0, 11)
        self.risk_splitter.setStretchFactor(1, 9)

        self.top_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.top_splitter.setChildrenCollapsible(False)
        self.top_splitter.addWidget(self.risk_splitter)
        self.top_splitter.addWidget(stability_box)
        self.top_splitter.setStretchFactor(0, 1)
        self.top_splitter.setStretchFactor(1, 2)

        self.bottom_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.bottom_splitter.setChildrenCollapsible(False)
        self.bottom_splitter.addWidget(cci_box)
        self.bottom_splitter.addWidget(forecast_box)
        self.bottom_splitter.setStretchFactor(0, 1)
        self.bottom_splitter.setStretchFactor(1, 1)

        right_panel = QWidget()
        right = QVBoxLayout(right_panel)
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(10)
        right.addWidget(self.top_splitter, 2)
        right.addWidget(self.bottom_splitter, 1)

        self.validation_results_label = ResultImageLabel("", "", fill_to_bounds=True)
        validation_results_box = boxed(
            "Analysis Results",
            self.validation_results_label,
            validation_results_tip,
            tier="primary"
        )
        validation_right_panel = QWidget()
        validation_right = QVBoxLayout(validation_right_panel)
        validation_right.setContentsMargins(0, 0, 0, 0)
        validation_right.setSpacing(0)
        validation_right.addWidget(validation_results_box)

        self.right_stack = QStackedWidget()
        self.right_stack.addWidget(right_panel)
        self.right_stack.addWidget(validation_right_panel)

        self.system_logo_label = QLabel()
        self.system_logo_label.setFixedSize(200, 50)
        self.system_logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_path = Path(__file__).resolve().parent / "CrowdTune Alt Icon - T.png"
        logo_pix = QPixmap(str(logo_path))
        if not logo_pix.isNull():
            self.system_logo_label.setPixmap(
                logo_pix.scaled(
                    self.system_logo_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
            )
        else:
            self.system_logo_label.hide()
        
        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(10)
        header_row.addWidget(self.system_logo_label, 0, Qt.AlignmentFlag.AlignVCenter)
        header_row.addStretch()

        section_divider = QFrame()
        section_divider.setFrameShape(QFrame.Shape.HLine)
        section_divider.setFrameShadow(QFrame.Shadow.Sunken)
        section_divider.setFixedHeight(1)
        section_divider.setStyleSheet("background-color: #4a4a4a; border: none;")

        self.left_splitter = QSplitter(Qt.Orientation.Vertical)
        self.left_splitter.setChildrenCollapsible(False)
        self.left_splitter.addWidget(video_feed_box)
        self.left_splitter.addWidget(self.left_lower_stack)
        self.left_splitter.setStretchFactor(0, 1)
        self.left_splitter.setStretchFactor(1, 1)

        left_panel = QWidget()
        left = QVBoxLayout(left_panel)
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(0)
        left.addLayout(header_row)
        left.addSpacing(15)
        left.addWidget(section_divider)
        left.addSpacing(15)
        left.addWidget(self.left_splitter, 1)
        left.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.addWidget(left_panel)
        self.main_splitter.addWidget(self.right_stack)
        self.main_splitter.setStretchFactor(0, 2)
        self.main_splitter.setStretchFactor(1, 3)

        main = QVBoxLayout()
        main.setContentsMargins(0, 0, 0, 0)
        main.addWidget(self.main_splitter)

        self.setLayout(main)

        self.thread = None
        self.preview_thread = None
        self.preview_active = False
        self.preview_source = None
        self.validation_worker = None
        self.validation_video_path = None
        self.validation_output_dir = None
        self.source_active = False
        self.live_start_ts = None
        self.loaded_video_path = None
        self.latest_frame_qimage = None
        self.latest_frame_size = None
        self.annotation_mode = False
        self.annotation_zone = ANNOTATION_ZONE_OPTIONS[0]
        self.annotation_polygons = {"zones": {}, "boundaries": {}, "ignore_regions": {}}
        self.latest_cci = 0.0
        self.latest_risk = 0.0
        self.latest_projected_cci = 0.0
        self.latest_slope = 0.0
        self.latest_s = 0.0
        self.latest_k = 0.0
        self.latest_c = 0.0
        self.latest_stability = 1.0
        self.latest_stability_trend = 0.0
        self.has_stability_signal = False
        self.has_forecast_signal = False
        self.latest_regime_phase = "Unknown"
        self.latest_forecast_direction = "Stable"
        self.latest_risk_data = {}

        # UI stabilization layer: EMA + hysteresis + throttled repaint.
        self.ui_refresh_ms = UI_SMOOTHING.refresh_ms
        self.ema_alpha = UI_SMOOTHING.ema_alpha
        self.ema_fast_alpha = UI_SMOOTHING.ema_fast_alpha
        self.ema_fast_delta = UI_SMOOTHING.ema_fast_delta
        self.hysteresis_margin = RISK.hysteresis_margin
        self.risk_band_state = "Low"

        self.smooth_cci = None
        self.smooth_risk = None
        self.smooth_projected_cci = None
        self.smooth_slope = None
        self.smooth_stability = None
        self.smooth_stability_trend = 0.0

        self._ui_values_dirty = False
        self.ui_update_timer = QTimer(self)
        self.ui_update_timer.setInterval(self.ui_refresh_ms)
        self.ui_update_timer.timeout.connect(self._flush_smoothed_ui)

        self._default_splitters_applied = False
        QTimer.singleShot(0, self._apply_default_splitter_sizes)
        # session recording (populated when a live session starts)
        self._session_records = []
        self._session_frame_count = 0

    def _apply_default_splitter_sizes(self):
        if self._default_splitters_applied:
            return

        self._default_splitters_applied = True

        self.main_splitter.setSizes([920, 1080])
        self.left_splitter.setSizes([520, 340])
        self.risk_splitter.setSizes([320, 310])
        self.top_splitter.setSizes([430, 760])
        self.bottom_splitter.setSizes([380, 530])

        self._update_video_preview_height()

    def _update_video_preview_height(self):
        window_h = max(700, self.height())
        target = int(window_h * 0.33)
        target = max(280, min(460, target))
        self.video_panel.video_display.setMinimumHeight(target)
        self.video_panel.video_display.setMaximumHeight(target)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_video_preview_height()

    def on_source_mode_changed(self, index: int):
        validation_enabled = index == 2
        if validation_enabled and self.source_active:
            self.stop_source()
        self._set_validation_mode(validation_enabled)

    def _set_validation_mode(self, enabled: bool):
        self.left_lower_stack.setCurrentIndex(1 if enabled else 0)
        self.right_stack.setCurrentIndex(1 if enabled else 0)
        if enabled:
            self.annotation_mode = False
            self.video_panel.set_annotation_mode(False)
            self.video_panel.set_live_indicator_active(False)
            self.video_panel.validation_progress_bar.setRange(0, 100)
            self._set_validation_progress(0)
            self._reset_validation_outputs()
            if self.validation_video_path:
                self._load_validation_preview_frame(self.validation_video_path)
            else:
                self.latest_frame_qimage = None
                self.latest_frame_size = None
                self.video_panel.video_display.clear()
                self.video_panel.video_display.setText("Video Preview")
        else:
            if self.latest_frame_qimage is None and not self.source_active:
                self.blackout()

    def load_validation_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Analysis Video", "", "Video Files (*.mp4 *.avi)"
        )
        if not path:
            return

        ok, error_text = self._validate_video_file(path)
        if not ok:
            self.validation_video_path = None
            self.validation_output_dir = None
            self.video_panel.validation_play_btn.setEnabled(False)
            self.video_panel.validation_export_btn.setEnabled(False)
            self.video_panel.set_mode(2)
            self._set_validation_progress(0)
            self._reset_validation_outputs()
            self.video_panel.video_display.clear()
            self.video_panel.video_display.setText("Validation Preview")
            QMessageBox.critical(self, "Invalid Video File", error_text)
            return

        self.validation_video_path = path
        self.validation_output_dir = None
        self.video_panel.set_mode(2)
        self.video_panel.validation_play_btn.setEnabled(False)
        self.video_panel.validation_export_btn.setEnabled(False)
        self._set_validation_progress(0)
        self._reset_validation_outputs()
        # Attempt to load a preview frame; only enable Run Analysis if preview succeeds.
        if self._load_validation_preview_frame(path):
            self.video_panel.validation_play_btn.setEnabled(True)
        else:
            self.validation_video_path = None
            QMessageBox.critical(
                self,
                "Invalid Video File",
                "CrowdTune could not decode a preview frame from this file. Please select a valid video file."
            )

    def run_validation_analysis(self):
        if not self.validation_video_path:
            QMessageBox.warning(self, "Missing Video", "Please upload an analysis video first.")
            return
        if self.validation_worker and self.validation_worker.isRunning():
            return

        ok, error_text = self._validate_video_file(self.validation_video_path)
        if not ok:
            self.video_panel.validation_play_btn.setEnabled(False)
            QMessageBox.critical(self, "Invalid Video File", error_text)
            return

        if not self._ensure_validation_scene_mask():
            return
        case_name = self._validation_case_name(self.validation_video_path)
        self.validation_worker = ValidationWorker(
            video_path=self.validation_video_path,
            case_name=case_name,
            parent=self,
        )
        self.validation_worker.progress_signal.connect(self._set_validation_progress)
        self.validation_worker.status_signal.connect(self._on_validation_status)
        self.validation_worker.finished_signal.connect(self._on_validation_finished)
        self.validation_worker.error_signal.connect(self._on_validation_error)

        self.video_panel.validation_play_btn.setEnabled(False)
        self.video_panel.validation_load_video_btn.setEnabled(False)
        self._set_validation_progress(0)
        self.validation_results_label.set_placeholder(
            "CCI / SI / Risk / Forecast",
            "Running analysis..."
        )
        self.validation_worker.start()

    def _validate_video_file(self, video_path: str):
        path = Path(video_path)
        if not path.exists() or not path.is_file():
            return False, "The selected file does not exist. Please select a valid video file."
        if path.stat().st_size <= 0:
            return False, "The selected file is empty. Please select a valid video file."

        cap = None
        try:
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                return False, (
                    "CrowdTune could not open this file as a video.\n\n"
                    "The file may be corrupted, unsupported, or not a real video file even if it has an .mp4/.avi extension."
                )

            ret, frame = cap.read()
            if not ret or frame is None or getattr(frame, "size", 0) == 0:
                return False, (
                    "CrowdTune could not decode any video frames from this file.\n\n"
                    "Please select a valid video file encoded with a supported codec."
                )

            height, width = frame.shape[:2]
            if width <= 0 or height <= 0:
                return False, "The selected video has invalid frame dimensions."

            return True, ""
        except Exception as exc:
            return False, f"CrowdTune could not validate this video file.\n\nDetails: {exc}"
        finally:
            if cap is not None:
                cap.release()

    def _scene_profiles_dir(self):
        return Path(__file__).resolve().parent / "scene_profiles"

    def _scene_profile_path_for_video(self, video_path: str):
        return self._scene_profiles_dir() / f"{Path(video_path).stem}.json"

    def _has_valid_scene_profile(self, video_path: str):
        profile_path = self._scene_profile_path_for_video(video_path)
        if not profile_path.exists():
            return False
        try:
            profile = SceneProfile.from_json(profile_path)
            zones = profile.export_annotation_group("zones")
            points = zones.get("crowd_space", [])
            return isinstance(points, list) and len(points) >= 3
        except Exception:
            return False

    def _ensure_validation_scene_mask(self):
        if not self.validation_video_path:
            return False
        if self._has_valid_scene_profile(self.validation_video_path):
            return True

        QMessageBox.information(
            self,
            "Scene Profile Required",
            "No saved scene masking profile was found for this video.\n\n"
            "Please define the Crowd Region of Interest (ROI) before analysis begins."
        )

        try:
            dialog = SceneMaskingDialog(
                self.validation_video_path,
                self._scene_profiles_dir(),
                parent=self,
            )
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return False
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Scene Masking Error",
                f"CrowdTune could not open the scene masking workflow.\n\nDetails: {exc}"
            )
            return False

        if not self._has_valid_scene_profile(self.validation_video_path):
            QMessageBox.information(
                self,
                "Scene Profile Required",
                "Analysis cannot continue until a valid Crowd Space ROI mask is saved."
            )
            return False
        self._load_validation_preview_frame(self.validation_video_path)
        return True
    def _load_validation_preview_frame(self, video_path: str):
        self.latest_frame_qimage = None
        self.latest_frame_size = None
        cap = None
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.video_panel.video_display.clear()
                self.video_panel.video_display.setText("Preview unavailable")
                return False
            ret, frame = cap.read()
            if not ret or frame is None:
                self.video_panel.video_display.clear()
                self.video_panel.video_display.setText("Preview unavailable")
                return False

            profile = SceneProfile.load_for_video(video_path, Path(__file__).resolve().parent / "scene_profiles")
            if profile is not None:
                frame = profile.render_overlay(frame)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
            self.latest_frame_qimage = qimg
            self.latest_frame_size = (w, h)
            self._render_current_frame()
            return True
        except Exception:
            self.video_panel.video_display.clear()
            self.video_panel.video_display.setText("Preview unavailable")
            self.latest_frame_qimage = None
            self.latest_frame_size = None
            return False
        finally:
            if cap is not None:
                cap.release()
    def _set_validation_progress(self, value: int):
        value = max(0, min(100, int(value)))
        self.video_panel.validation_progress_bar.setValue(value)
        self.video_panel.validation_time_label.setText(f"{value}%")

    def _on_validation_status(self, text: str):
        self.validation_results_label.set_placeholder(
            "CCI / SI / Risk / Forecast",
            text
        )

    def _on_validation_finished(self, out_dir: str):
        self.validation_output_dir = Path(out_dir)
        self.video_panel.validation_play_btn.setEnabled(True)
        self.video_panel.validation_load_video_btn.setEnabled(True)
        # enable export button so user can choose where to save outputs
        try:
            if hasattr(self.video_panel, "validation_export_btn"):
                self.video_panel.validation_export_btn.setEnabled(True)
        except Exception:
            pass
        self._set_validation_progress(100)
        self._load_validation_outputs(self.validation_output_dir)

    def _on_validation_error(self, error_text: str):
        self.video_panel.validation_play_btn.setEnabled(True)
        self.video_panel.validation_load_video_btn.setEnabled(True)
        self.validation_results_label.set_placeholder(
            "Analysis Failed",
            "See terminal output for the traceback."
        )
        print(error_text)
        QMessageBox.critical(self, "Analysis Failed", error_text)

    def _load_validation_outputs(self, out_dir: Path):
        timeseries_path = out_dir / "timeseries.png"
        if timeseries_path.exists():
            self.validation_results_label.set_result_pixmap(QPixmap(str(timeseries_path)))
        else:
            self.validation_results_label.set_placeholder(
                "Analysis Complete",
                f"Results saved to {out_dir}"
            )

        # Update the compact validation summary card from the output directory.
        try:
            if hasattr(self, "validation_summary_card"):
                self.validation_summary_card.update_from_dir(out_dir)
        except Exception:
            # Best-effort: do not raise UI errors
            pass

    def _reset_validation_outputs(self):
        self.validation_results_label.set_placeholder("", "")
        # Reset compact validation summary card as well.
        if hasattr(self, "validation_summary_card"):
            try:
                self.validation_summary_card.set_placeholder()
            except Exception:
                pass
    @staticmethod
    def _validation_case_name(video_path: str):
        stem = Path(video_path).stem
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
        return cleaned or "analysis_case"

    # --------- Source Control ---------
    def start_camera(self):
        self.video_panel.set_mode(0)
        if not self._ensure_live_scene_mask():
            return
        # If a lightweight preview is running, stop it before starting full analytics
        preview_src = None
        if self.preview_thread is not None and getattr(self.preview_thread, "isRunning", lambda: False)():
            try:
                # stop reading but keep capture open for handoff
                self.preview_thread.stop()
                self.preview_thread.wait(500)
            except Exception:
                pass
            self.preview_thread = None
            self.preview_active = False
            # prefer stored preview_source if available
            preview_src = getattr(self, "preview_source", None)
            self.preview_source = None

        # Update badge to LIVE and pulsing style
        try:
            self.video_panel.live_indicator_badge.setText("LIVE")
            self.video_panel.set_live_indicator_active(True)
        except Exception:
            pass

        # If we have an existing preview CameraSource, reuse it to avoid reopening
        # the camera. Otherwise create a new CameraSource.
        if preview_src is not None:
            self.start_source(preview_src)
        else:
            self.start_source(CameraSource(0))
        # Close the camera setup popup when full session starts
        try:
            if hasattr(self, "camera_setup_popup") and self.camera_setup_popup is not None:
                self.camera_setup_popup.close()
                self.camera_setup_popup = None
        except Exception:
            pass

    def open_camera(self):
        """Open camera for preview only (no analytics)."""
        # Live camera masks are session-local. Opening a new preview starts clean.
        self.annotation_polygons = {"zones": {}, "boundaries": {}, "ignore_regions": {}}
        self.annotation_mode = False
        # If full analytics is already running, do nothing.
        if self.thread is not None and getattr(self.thread, "isRunning", lambda: False)():
            return
        # If a preview is already running, do nothing.
        if self.preview_thread is not None and getattr(self.preview_thread, "isRunning", lambda: False)():
            return
        # Start a lightweight preview thread that only emits frames.
        src = CameraSource(0)
        # store preview source so it can be handed off to analytics without
        # reopening the camera device
        self.preview_source = src
        self.preview_thread = PreviewThread(src)
        self.preview_thread.frame_signal.connect(self.update_frame)
        self.preview_thread.start()
        self.preview_active = True
        self.video_panel.set_mode(0)
        # Indicate preview mode: show a PREVIEW badge and a static style.
        try:
            self.video_panel.live_indicator_badge.setText("PREVIEW")
            self.video_panel.live_indicator_badge.setStyleSheet(
                "background-color: rgba(96,96,96,0.95); color: white; font-size: 11px; font-weight: 700; padding: 2px 6px; border-radius: 4px;"
            )
            self.video_panel.live_indicator_badge.setVisible(True)
        except Exception:
            pass
        # Enable starting a full session now that preview is active.
        try:
            self.video_panel.start_camera_btn.setEnabled(True)
        except Exception:
            pass
        # Show camera setup popup to the right of the video display
        try:
            # Close existing popup if present
            if hasattr(self, "camera_setup_popup") and self.camera_setup_popup is not None:
                try:
                    self.camera_setup_popup.close()
                except Exception:
                    pass
            self.camera_setup_popup = CameraSetupPopup(self)
            anchor = self.video_panel.video_display.mapToGlobal(self.video_panel.video_display.rect().topRight())
            self.camera_setup_popup.adjustSize()
            popup_pos = anchor + QPoint(8, 0)
            self.camera_setup_popup.move(popup_pos)
            self.camera_setup_popup.show()
        except Exception:
            pass

    def _ensure_live_scene_mask(self):
        points = self.annotation_polygons.get("zones", {}).get("crowd_space", [])
        if isinstance(points, list) and len(points) >= 3:
            return True

        if self.latest_frame_qimage is None or self.latest_frame_qimage.isNull() or self.latest_frame_size is None:
            QMessageBox.information(
                self,
                "Live Scene Mask Required",
                "Please open the camera and wait for a preview frame before starting the live session."
            )
            return False

        QMessageBox.information(
            self,
            "Live Scene Mask Required",
            "Please define the live Crowd Region of Interest (ROI) before starting the session.\n\n"
            "This mask is temporary and will be removed when the session ends."
        )

        try:
            dialog = LiveSceneMaskingDialog(self.latest_frame_qimage.copy(), parent=self)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return False
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Live Scene Masking Error",
                f"CrowdTune could not open the live scene masking workflow.\n\nDetails: {exc}"
            )
            return False

        points = dialog.points()
        if not isinstance(points, list) or len(points) < 3:
            QMessageBox.information(
                self,
                "Live Scene Mask Required",
                "Live analysis cannot continue until a valid Crowd Space ROI mask is saved."
            )
            return False

        self.annotation_polygons = {
            "zones": {"crowd_space": points},
            "boundaries": {},
            "ignore_regions": {},
        }
        self.annotation_mode = False
        self._render_current_frame()
        return True

    def load_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Video Files (*.mp4 *.avi)"
        )
        if not path:
            return

        ok, error_text = self._validate_video_file(path)
        if not ok:
            QMessageBox.critical(self, "Invalid Video File", error_text)
            return

        self.video_panel.set_mode(1)
        self.loaded_video_path = path

        if not self._ensure_video_file_scene_mask(path):
            self.loaded_video_path = None
            return

        self._load_annotation_profile_for_video(path)
        self.start_source(VideoFileSource(path))
    def start_source(self, source):
        # Preserve any transient annotations created during preview when
        # starting a new analytics session.
        self.stop_source(preserve_annotations=True)
        self.video_panel.set_live_indicator_active(not source.is_seekable())
        self.has_stability_signal = False
        self.has_forecast_signal = False
        self.stability_widget.reset()
        self.forecast_widget.reset()

        if source.is_seekable() and hasattr(source, "path"):
            self.loaded_video_path = source.path
            self._load_annotation_profile_for_video(source.path)
        else:
            # Live camera: keep any existing transient annotation_polygons created
            # during preview (do not clear them). Annotation UI should be disabled
            # once analytics start.
            self.loaded_video_path = None
            self.video_panel.set_annotation_mode(False)

        self.thread = CameraThread(source)
        self.live_start_ts = time.monotonic() if not source.is_seekable() else None

        # Initialize session recording buffers for live sessions
        self._session_records = []
        self._session_frame_count = 0

        print(f"[UI] start_source: created CameraThread for source.seekable={source.is_seekable() if hasattr(source,'is_seekable') else False}")

        self.thread.set_overlay_enabled(self.video_panel.is_overlay_enabled())
        self.thread.set_tracking_overlay_enabled(self.video_panel.is_tracking_overlay_enabled())

        # ---------- FRAME ----------
        self.thread.frame_signal.connect(self.update_frame)
        self.thread.playback_position_signal.connect(self.on_playback_position)
        self.thread.runtime_metrics_signal.connect(self.on_runtime_metrics_update)

        # ---------- ANALYTICS ----------
        self.thread.cci_signal.connect(self.on_cci_update)
        self.thread.risk_signal.connect(self.on_risk_update)

        self.thread.stability_signal.connect(self.on_stability_update)
        self.thread.forecast_signal.connect(self.on_forecast_update)
        self.thread.skc_signal.connect(self.on_skc_update)

        # If we have transient live annotations (from preview) and a valid frame size,
        # apply them to the analytics thread as a SceneProfile so masks are used.
        try:
            if not source.is_seekable() and any(self.annotation_polygons.values()):
                # prefer explicit latest frame size from preview, otherwise query source capture
                if self.latest_frame_size is not None:
                    fw, fh = self.latest_frame_size
                else:
                    fw = None
                    fh = None
                    try:
                        cap = getattr(source, "cap", None)
                        if cap is not None:
                            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            if w > 0 and h > 0:
                                fw, fh = w, h
                    except Exception:
                        fw, fh = None, None

                if fw is not None and fh is not None:
                    profile = SceneProfile(
                        name="live_transient",
                        frame_width=int(fw),
                        frame_height=int(fh),
                        zones=self.annotation_polygons.get("zones", {}),
                        boundaries=self.annotation_polygons.get("boundaries", {}),
                        ignore_regions=self.annotation_polygons.get("ignore_regions", {}),
                        semantic_weights={"crowd_space": 1.0},
                        risk_modifiers={"wall_proximity_weight": 1.0, "boundary_exclusion_weight": 1.0},
                        notes={"source": "live_transient"},
                    )
                    self.thread.scene_profile = profile
                    print(f"[UI] start_source: attached scene_profile name={profile.name} size=({profile.frame_width},{profile.frame_height})")
                    # Pre-compute a spatial analysis mask and attach it so the
                    # analytics thread does not run on the whole frame if the
                    # first frame arrives before run-time mask computation.
                    try:
                        mask = profile.get_spatial_analysis_mask((int(fh), int(fw), 3))
                        self.thread.spatial_analysis_mask = mask
                        print(f"[UI] start_source: precomputed spatial_analysis_mask shape={mask.shape if hasattr(mask,'shape') else None}")
                    except Exception:
                        pass
        except Exception:
            pass

        if source.is_seekable():
            self.video_panel.set_duration_seconds(source.get_duration_seconds())
            self.video_panel.set_current_position_seconds(0)
        else:
            self.video_panel.set_duration_seconds(0)
            self.video_panel.set_current_position_seconds(0)

        self.ui_update_timer.start()
        self.source_active = True
        self.thread.start()
        # Disable live annotation UI while analytics are running

    def stop_source(self, preserve_annotations: bool = False):
        self.source_active = False
        # Stop any lightweight preview thread if running
        if self.preview_thread is not None and getattr(self.preview_thread, "isRunning", lambda: False)():
            try:
                self.preview_thread.stop()
                self.preview_thread.wait(300)
            except Exception:
                pass
            self.preview_thread = None
            self.preview_active = False
        # Also release preview source if present (no longer needed)
        if getattr(self, "preview_source", None) is not None:
            try:
                self.preview_source.release()
            except Exception:
                pass
            self.preview_source = None

        # Hide the badge when stopped and disable Start Session until preview reopened
        try:
            self.video_panel.live_indicator_badge.setVisible(False)
            self.video_panel.start_camera_btn.setEnabled(False)
        except Exception:
            pass
        # Close camera setup popup if open
        try:
            if hasattr(self, "camera_setup_popup") and self.camera_setup_popup is not None:
                self.camera_setup_popup.close()
                self.camera_setup_popup = None
        except Exception:
            pass
        # Live camera masks are temporary: clear them when the session/preview ends.
        if not preserve_annotations and self.loaded_video_path is None:
            self.annotation_polygons = {"zones": {}, "boundaries": {}, "ignore_regions": {}}
            self.annotation_mode = False
        # Show stop feedback immediately before any blocking thread shutdown.
        self.blackout()
        QApplication.processEvents()

        if self.thread and self.thread.isRunning():
            self.thread.stop()
            # Avoid long UI freeze if capture/read teardown stalls.
            self.thread.wait(300)
        self.ui_update_timer.stop()
        self.thread = None
        self.video_panel.set_live_indicator_active(False)
        self.video_panel.set_duration_seconds(0)
        self.video_panel.set_current_position_seconds(0)
        self.video_panel.set_live_elapsed_seconds(0)
        self.video_panel.live_status_label.setText(
            "FPS: 00.0   |   Features: 000   |   Processing: 00 ms"
        )
        self.latest_cci = 0.0
        self.latest_risk = 0.0
        self.latest_projected_cci = 0.0
        self.latest_slope = 0.0
        self.latest_stability = 1.0
        self.latest_stability_trend = 0.0
        self.has_stability_signal = False
        self.has_forecast_signal = False
        self.latest_regime_phase = "Unknown"
        self.latest_forecast_direction = "Stable"
        self.latest_risk_data = {}
        self.smooth_cci = None
        self.smooth_risk = None
        self.smooth_projected_cci = None
        self.smooth_slope = None
        self.smooth_stability = None
        self.smooth_stability_trend = 0.0
        self.risk_band_state = "Low"
        self._ui_values_dirty = False
        self.cci_widget.reset()
        self.risk_widget.reset()
        self.advisory_widget.reset()
        self.stability_widget.reset()
        self.forecast_widget.reset()
        self.ternary_widget.reset()
        self.regime_map_widget.reset()
        self.s_value_label.setText("S: 0.000")
        self.k_value_label.setText("K: 0.000")
        self.c_value_label.setText("C: 0.000")
        # Disable live annotation controls after stopping

    def on_overlay_toggled(self, enabled: bool):
        if self.thread and self.thread.isRunning():
            self.thread.set_overlay_enabled(enabled)

    def on_tracking_overlay_toggled(self, enabled: bool):
        if self.thread and self.thread.isRunning():
            self.thread.set_tracking_overlay_enabled(enabled)

    def play_video(self):
        if self.thread and self.thread.isRunning():
            self.thread.resume_playback()

    def pause_video(self):
        if self.thread and self.thread.isRunning():
            self.thread.pause_playback()

    def seek_video(self, seconds: int):
        if self.thread and self.thread.isRunning():
            self.thread.seek_seconds(seconds)

    def on_playback_position(self, position_seconds: int, duration_seconds: int):
        self.video_panel.set_duration_seconds(duration_seconds)
        self.video_panel.set_current_position_seconds(position_seconds)
        # Enable export when playback reaches the natural end of the video
        try:
            if hasattr(self.video_panel, "export_results_btn") and duration_seconds and int(position_seconds) >= int(duration_seconds):
                self.video_panel.export_results_btn.setEnabled(True)
        except Exception:
            pass

    def on_runtime_metrics_update(self, fps: float, feature_count: int, processing_ms: float):
        self.video_panel.live_status_label.setText(
            f"LIVE     FPS: {fps:4.1f}     Features: {feature_count:3d}     Processing: {processing_ms:4.0f} ms"
        )
        if self.live_start_ts is not None:
            elapsed = int(max(0.0, time.monotonic() - self.live_start_ts))
            self.video_panel.set_live_elapsed_seconds(elapsed)

    def on_cci_update(self, cci_value: float):
        self.latest_cci = float(cci_value)
        self.smooth_cci = self._ema(self.smooth_cci, self.latest_cci)
        self._ui_values_dirty = True

    def on_risk_update(self, risk_data: dict):
        self.latest_risk = float(risk_data.get("risk", 0.0))
        self.latest_risk_data = dict(risk_data)
        self.latest_regime_phase = self._infer_regime_phase(risk_data)
        self.smooth_risk = self._ema(self.smooth_risk, self.latest_risk)
        self._update_risk_band_hysteresis(self.smooth_risk if self.smooth_risk is not None else self.latest_risk)
        self._ui_values_dirty = True

    def on_forecast_update(self, forecast_data: dict):
        self.has_forecast_signal = True
        self.latest_projected_cci = float(forecast_data.get("projected_cci", 0.0))
        self.latest_slope = float(forecast_data.get("slope", 0.0))
        self.latest_forecast_direction = self._infer_forecast_direction(
            self.latest_slope,
            bool(forecast_data.get("stable", False))
        )
        self.smooth_projected_cci = self._ema(self.smooth_projected_cci, self.latest_projected_cci)
        self.smooth_slope = self._ema(self.smooth_slope, self.latest_slope)
        self._ui_values_dirty = True

    def on_stability_update(self, stability_value: float):
        self.has_stability_signal = True
        value = float(stability_value)
        self.latest_stability_trend = value - self.latest_stability
        self.latest_stability = value
        prev_stability = self.smooth_stability
        self.smooth_stability = self._ema(self.smooth_stability, value)
        if prev_stability is None:
            self.smooth_stability_trend = 0.0
        else:
            self.smooth_stability_trend = self.smooth_stability - prev_stability
        self._ui_values_dirty = True

    def on_annotation_mode_toggled(self, enabled: bool):
        self.annotation_mode = bool(enabled)
        self._render_current_frame()

    def on_annotation_zone_changed(self, zone_name: str):
        if zone_name in ANNOTATION_ZONE_GROUPS:
            self.annotation_zone = zone_name
            self._render_current_frame()

    def on_annotation_point_clicked(self, widget_x: int, widget_y: int):
        # Allow annotation when annotation mode is active and a preview/frame exists.
        if not self.annotation_mode or self.latest_frame_size is None:
            return
        # For video files require loaded_video_path; for live preview loaded_video_path is None
        if self.loaded_video_path is None and not getattr(self, "preview_active", False):
            return
        mapped = self.video_panel.map_display_to_frame_point(widget_x, widget_y, self.latest_frame_size)
        if mapped is None:
            return
        group = ANNOTATION_ZONE_GROUPS[self.annotation_zone]
        polygon = self.annotation_polygons[group].setdefault(self.annotation_zone, [])
        polygon.append([int(mapped[0]), int(mapped[1])])
        self._render_current_frame()

    def undo_annotation_point(self):
        group = ANNOTATION_ZONE_GROUPS[self.annotation_zone]
        polygon = self.annotation_polygons[group].get(self.annotation_zone)
        if polygon:
            polygon.pop()
            if not polygon:
                self.annotation_polygons[group].pop(self.annotation_zone, None)
            self._render_current_frame()

    def clear_annotation_zone(self):
        group = ANNOTATION_ZONE_GROUPS[self.annotation_zone]
        self.annotation_polygons[group].pop(self.annotation_zone, None)
        self._render_current_frame()

    def save_annotation_profile(self):
        if not self.loaded_video_path or self.latest_frame_size is None:
            return
        frame_w, frame_h = self.latest_frame_size
        video_path = Path(self.loaded_video_path)
        profiles_dir = Path(__file__).resolve().parent / "scene_profiles"
        profiles_dir.mkdir(parents=True, exist_ok=True)
        profile_path = profiles_dir / f"{video_path.stem}.json"
        payload = {
            "profile_name": video_path.stem,
            "frame_size": {"width": frame_w, "height": frame_h},
            "zones": self.annotation_polygons["zones"],
            "boundaries": self.annotation_polygons["boundaries"],
            "ignore_regions": self.annotation_polygons["ignore_regions"],
            "semantic_weights": {
                "crowd_space": 1.0
            },
            "risk_modifiers": {
                "wall_proximity_weight": 1.3,
                "boundary_exclusion_weight": 1.0
            },
            "notes": {
                "flow_direction": "unspecified",
                "scene_type": "annotated_from_interface",
                "comment": "Profile created from the CrowdTune annotation mode."
            }
        }
        profile_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if self.thread and self.thread.isRunning() and hasattr(self.thread, "scene_profile"):
            self.thread.scene_profile = SceneProfile.from_json(profile_path)

    def _ensure_video_file_scene_mask(self, video_path: str):
        if self._has_valid_scene_profile(video_path):
            return True

        QMessageBox.information(
            self,
            "Scene Profile Required",
            "No saved scene masking profile was found for this video.\n\n"
            "Please define the Crowd Region of Interest (ROI) before playback analysis begins."
        )

        try:
            dialog = SceneMaskingDialog(
                video_path,
                self._scene_profiles_dir(),
                parent=self,
            )
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return False
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Scene Masking Error",
                f"CrowdTune could not open the scene masking workflow.\n\nDetails: {exc}"
            )
            return False

        if not self._has_valid_scene_profile(video_path):
            QMessageBox.information(
                self,
                "Scene Profile Required",
                "Playback analysis cannot continue until a valid Crowd Space ROI mask is saved."
            )
            return False
        return True
    def _load_annotation_profile_for_video(self, video_path: str):
        self.annotation_polygons = {"zones": {}, "boundaries": {}, "ignore_regions": {}}
        profile_path = Path(__file__).resolve().parent / "scene_profiles" / f"{Path(video_path).stem}.json"
        if not profile_path.exists():
            return
        profile = SceneProfile.from_json(profile_path)
        self.annotation_polygons = {
            "zones": profile.export_annotation_group("zones"),
            "boundaries": profile.export_annotation_group("boundaries"),
            "ignore_regions": profile.export_annotation_group("ignore_regions"),
        }
    def export_session(self):
        # Export recorded session data (CSV + PNG plots). User chooses base folder
        # and supplies a name for the session folder.
        records = getattr(self, "_session_records", None)
        if not records:
            QMessageBox.information(self, "Export Session", "No session data available to export.")
            return

        # default to the project 'results' directory
        default_base = str(Path.cwd() / "results")
        Path(default_base).mkdir(parents=True, exist_ok=True)
        base = QFileDialog.getExistingDirectory(self, "Select Destination Folder", default_base)
        if not base:
            return

        name, ok = QInputDialog.getText(self, "Session Folder Name", "Enter a name for the session folder:")
        if not ok or not name.strip():
            QMessageBox.information(self, "Export Session", "Export cancelled: no folder name provided.")
            return

        safe_name = re.sub(r"[^A-Za-z0-9._ -]+", "_", name).strip("._ -") or f"session_{int(time.time())}"
        target = Path(base) / safe_name
        if target.exists():
            resp = QMessageBox.question(self, "Overwrite?", f"Folder '{safe_name}' already exists. Overwrite contents?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if resp != QMessageBox.StandardButton.Yes:
                return
        target.mkdir(parents=True, exist_ok=True)

        # Write CSV
        csv_path = target / "session_signals.csv"
        fieldnames = ["index", "timestamp", "elapsed", "cci", "risk", "s", "k", "c", "stability", "projected_cci", "slope", "regime_phase", "forecast_dir"]
        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                for r in records:
                    row = {k: r.get(k) for k in fieldnames}
                    writer.writerow(row)
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to write CSV: {e}")
            return

        # Create PNG plots (try matplotlib)
        try:
            import matplotlib.pyplot as plt
        except Exception:
            QMessageBox.information(self, "Export Session", "matplotlib is required to create plots. CSV exported, but install matplotlib to save PNG graphs.")
            QMessageBox.information(self, "Export Complete", f"Exported CSV to: {csv_path}")
            return

        # prepare x axis: prefer elapsed seconds if available, otherwise use frame index
        elapsed_available = any(r.get("elapsed") is not None for r in records)
        if elapsed_available:
            x = [r.get("elapsed") for r in records]
            x_label = "Seconds"
        else:
            x = [int(r.get("index", i)) for i, r in enumerate(records)]
            x_label = "Frame"

        # Combined multi-panel figure (S, K, C, CCI, SI, Projected CCI)
        try:
            s_vals = [r.get("s") for r in records]
            k_vals = [r.get("k") for r in records]
            c_vals = [r.get("c") for r in records]
            cci_vals = [r.get("cci") for r in records]
            risk_vals = [r.get("risk") for r in records]
            stab_vals = [r.get("stability") for r in records]
            proj_vals = [r.get("projected_cci") for r in records]

            fig, axes = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
            fig.suptitle("CrowdTune Per-Session Metric Time Series")

            axes[0].plot(x, s_vals, color="#1f77b4")
            axes[0].set_ylabel("S")
            axes[0].set_title("Spatial Constraint S(t)")
            axes[0].grid(True)

            axes[1].plot(x, k_vals, color="#ff7f0e")
            axes[1].set_ylabel("K")
            axes[1].set_title("Kinematic Activity K(t)")
            axes[1].grid(True)

            axes[2].plot(x, c_vals, color="#2ca02c")
            axes[2].set_ylabel("C")
            axes[2].set_title("Collective Coherence C(t)")
            axes[2].grid(True)

            axes[3].plot(x, cci_vals, color="#d62728")
            axes[3].set_ylabel("CCI")
            axes[3].set_title("Crowd Constraint Index CCI(t)")
            axes[3].grid(True)

            axes[4].plot(x, stab_vals, color="#9467bd")
            axes[4].set_ylabel("SI")
            axes[4].set_title("Stability Index SI(t)")
            axes[4].grid(True)

            axes[5].plot(x, proj_vals, color="#8c564b")
            axes[5].set_ylabel("Projected_CCI")
            axes[5].set_title("Forecast / Projected CCI(t)")
            axes[5].set_xlabel(x_label)
            axes[5].grid(True)

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            all_png = target / "session_all_metrics.png"
            fig.savefig(str(all_png))
            plt.close(fig)
        except Exception as e:
            # fall back: already wrote CSV
            print(f"[Export] plotting failed: {e}")

        QMessageBox.information(self, "Export Complete", f"Session exported to: {target}")
    def export_validation_results(self):
        # Copy validation worker outputs (stored in temp dir) into a user-chosen folder.
        out_dir = getattr(self, "validation_output_dir", None)
        if out_dir is None or not Path(out_dir).exists():
            QMessageBox.information(self, "Export Validation", "No validation outputs available to export.")
            return

        base = QFileDialog.getExistingDirectory(self, "Select Destination Folder")
        if not base:
            return

        # default name from video file stem if available
        default_name = None
        try:
            if self.validation_video_path:
                default_name = Path(self.validation_video_path).stem
        except Exception:
            default_name = None

        name, ok = QInputDialog.getText(self, "Export Folder Name", "Enter folder name for validation results:", text=(default_name or "validation_results"))
        if not ok or not name.strip():
            QMessageBox.information(self, "Export Validation", "Export cancelled: no folder name provided.")
            return

        safe_name = re.sub(r"[^A-Za-z0-9._ -]+", "_", name).strip("._ -") or f"validation_{int(time.time())}"
        target = Path(base) / safe_name
        if target.exists():
            resp = QMessageBox.question(self, "Overwrite?", f"Folder '{safe_name}' already exists. Overwrite contents?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if resp != QMessageBox.StandardButton.Yes:
                return
        target.mkdir(parents=True, exist_ok=True)

        # copy files from out_dir to target
        try:
            src = Path(out_dir)
            # keep the timeseries PNG so we can embed it in the report PDF
            exclude_names = {"validation_metrics.csv"}

            # attempt to copy all files except excluded names
            for item in src.iterdir():
                try:
                    if item.name.lower() in {n.lower() for n in exclude_names}:
                        # skip excluded
                        continue
                    dest = target / item.name
                    if item.is_dir():
                        shutil.copytree(str(item), str(dest), dirs_exist_ok=True)
                    else:
                        shutil.copy2(str(item), str(dest))
                except Exception:
                    # non-fatal: continue copying other files
                    pass

            # attempt to generate a simple PDF report that includes the title, timestamp,
            # the timeseries graph (if present) and a text summary from the Analysis card.
            try:
                try:
                    from PIL import Image, ImageDraw, ImageFont
                except Exception:
                    Image = None

                # assemble textual summary by stripping HTML from the validation summary card
                def _strip_html(s):
                    return re.sub(r"<[^>]+>", "", s or "").strip()

                title_text = None
                try:
                    if self.validation_video_path:
                        # use stem to remove file extension (e.g., .mp4)
                        title_text = Path(self.validation_video_path).stem
                except Exception:
                    title_text = None

                dom = _strip_html(getattr(self.validation_summary_card, "dom_regime", QLabel()).text())
                trans = _strip_html(getattr(self.validation_summary_card, "transitions", QLabel()).text())
                evidence = _strip_html(getattr(self.validation_summary_card, "evidence", QLabel()).text())
                interp = _strip_html(getattr(self.validation_summary_card, "interpretation", QLabel()).text())
                conf = _strip_html(getattr(self.validation_summary_card, "confidence", QLabel()).text())

                summary_lines = [dom, trans, evidence, interp, conf]

                timeseries_src = src / "timeseries.png"
                timeseries_copy = target / "timeseries.png"
                if timeseries_src.exists():
                    try:
                        shutil.copy2(str(timeseries_src), str(timeseries_copy))
                    except Exception:
                        pass

                # generate PDF with a cleaner, professional layout
                pdf_created = False
                pdf_path = target / f"{safe_name}_report.pdf"

                # helper: split evidence into bullet lines
                def _evidence_bullets(evidence_text):
                    lines = []
                    if not evidence_text:
                        return lines
                    # First, try to extract dash-prefixed bullets like "- Mean S: ..."
                    try:
                        matches = re.findall(r"-\s*([A-Za-z].*?)(?=(?:-\s*[A-Za-z])|$)", evidence_text, flags=re.S)
                    except Exception:
                        matches = []
                    if matches:
                        for m in matches:
                            p = m.strip()
                            p = re.sub(r"^Key Evidence[:\-\s]*", "", p, flags=re.I)
                            if p:
                                lines.append(p)
                        return lines
                    # Fallback: split on newlines and loosely on leading dashes
                    for part in re.split(r"\n|\\r\\n", evidence_text):
                        p = part.strip()
                        if not p:
                            continue
                        p = re.sub(r"^Key Evidence[:\-\s]*", "", p, flags=re.I)
                        if p.startswith("- "):
                            lines.append(p[2:].strip())
                        else:
                            lines.append(p)
                    return lines

                # helper: remove repeated leading label prefixes from values
                def _strip_label_prefix(value, label):
                    try:
                        s = (value or '').strip()
                        if not s:
                            return s
                        pattern = r"^\s*" + re.escape(label) + r"\s*:??\s*"
                        # remove repeated occurrences of the label (case-insensitive)
                        while re.match(pattern, s, flags=re.I):
                            s = re.sub(pattern, "", s, flags=re.I)
                        # remove any leftover leading colons, dashes or whitespace
                        s = re.sub(r"^[\s:,-]+", "", s)
                        return s.strip()
                    except Exception:
                        return value or ''

                # Try ReportLab Platypus first (preferred) — Paragraphs, Spacers, KeepTogether
                try:
                    try:
                        from reportlab.lib.pagesizes import letter
                        from reportlab.lib.units import cm
                        from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Image as RLImage, PageBreak, KeepTogether, ListFlowable, ListItem
                        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                        from reportlab.lib import colors
                    except Exception:
                        raise

                    page_size = letter
                    left_margin = 3.17 * cm
                    right_margin = 3.17 * cm
                    top_margin = 2.54 * cm
                    bottom_margin = 2.54 * cm

                    styles = getSampleStyleSheet()
                    title_style = ParagraphStyle('ReportTitle', parent=styles['Title'], alignment=1, fontName='Helvetica-Bold', fontSize=18, leading=22)
                    subtitle_style = ParagraphStyle('Subtitle', parent=styles['Normal'], alignment=1, fontSize=10, textColor=colors.grey, leading=12)
                    section_heading = ParagraphStyle('SectionHeading', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=12, leading=14, spaceAfter=8)
                    # body text: 10 pt with comfortable leading (14-16)
                    body_style = ParagraphStyle('Body', parent=styles['Normal'], fontName='Helvetica', fontSize=10, leading=15, alignment=0, spaceAfter=8)
                    # smaller label style for the bolded keys but with spacing after
                    label_style = ParagraphStyle('Label', parent=body_style, fontName='Helvetica-Bold', fontSize=10, leading=15, spaceAfter=6)
                    # interpretation paragraph: slightly larger leading for readability
                    interp_style = ParagraphStyle('Interpretation', parent=body_style, fontName='Helvetica', fontSize=10, leading=16, spaceBefore=6, spaceAfter=12)
                    # list item style
                    list_item_style = ParagraphStyle('ListItem', parent=body_style, leftIndent=6, spaceAfter=6)

                    doc = BaseDocTemplate(str(pdf_path), pagesize=page_size, leftMargin=left_margin, rightMargin=right_margin, topMargin=top_margin, bottomMargin=bottom_margin)
                    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='normal')

                    def _footer(canvas, doc):
                        canvas.saveState()
                        footer_text = f"Page {doc.page}"
                        canvas.setFont('Helvetica', 9)
                        canvas.setFillColor(colors.grey)
                        canvas.drawCentredString((page_size[0]) / 2.0, 0.5 * cm, footer_text)
                        canvas.restoreState()

                    doc.addPageTemplates([PageTemplate(id='Report', frames=[frame], onPage=_footer)])

                    story = []
                    # Header
                    story.append(Paragraph(title_text or 'CrowdTune Validation Report', title_style))
                    story.append(Paragraph(time.strftime('%Y-%m-%d %H:%M:%S'), subtitle_style))
                    story.append(Spacer(1, 12))

                    # Representative graph (use most of width)
                    if timeseries_copy.exists():
                        try:
                            img = RLImage(str(timeseries_copy))
                            target_w = doc.width * 0.85
                            if getattr(img, 'drawWidth', None) is None or img.drawWidth == 0:
                                img.drawWidth = target_w
                                img.drawHeight = img.imageHeight * (target_w / img.imageWidth) if getattr(img, 'imageWidth', None) else target_w * 0.6
                            else:
                                # preserve aspect ratio
                                orig_w = img.drawWidth
                                orig_h = img.drawHeight
                                img.drawWidth = target_w
                                img.drawHeight = orig_h * (target_w / orig_w)
                            story.append(img)
                        except Exception:
                            story.append(Paragraph('Timeseries image unavailable', body_style))
                    else:
                        story.append(Paragraph('Timeseries image unavailable', body_style))

                    story.append(Spacer(1, 12))

                    # Validation Summary
                    story.append(Paragraph('Validation Summary', section_heading))

                    summary_flow = []
                    summary_flow.append(Paragraph(f'<b>Dominant Regime:</b> {(_strip_label_prefix(dom, "Dominant Regime") or "-")}', body_style))
                    summary_flow.append(Spacer(1, 12))
                    summary_flow.append(Paragraph(f'<b>Transition Observed:</b> {(_strip_label_prefix(trans, "Transition Observed") or "-")}', body_style))
                    summary_flow.append(Spacer(1, 12))

                    # Key Evidence as bullets
                    evidence_items = _evidence_bullets(evidence)
                    if evidence_items:
                        summary_flow.append(Paragraph('Key Evidence:', label_style))
                        summary_flow.append(Spacer(1, 12))
                        for ev in evidence_items:
                            summary_flow.append(Paragraph(f'• {ev}', list_item_style))
                        summary_flow.append(Spacer(1, 12))
                        # add one extra blank line above the Interpretation section
                        summary_flow.append(Spacer(1, 6))
                    else:
                        summary_flow.append(Paragraph('Key Evidence: -', body_style))
                        summary_flow.append(Spacer(1, 8))

                    summary_flow.append(Paragraph('Interpretation:', label_style))
                    summary_flow.append(Spacer(1, 12))
                    summary_flow.append(Paragraph(_strip_label_prefix(interp, 'Interpretation') or '-', interp_style))
                    summary_flow.append(Spacer(1, 12))
                    summary_flow.append(Paragraph(f'Confidence: <b>{(_strip_label_prefix(conf, "Confidence") or "-")}</b>', label_style))

                    # Add summary into the story (allow pagination rather than forcing KeepTogether)
                    story.append(Spacer(1, 12))
                    for _elem in summary_flow:
                        story.append(_elem)
                    story.append(Spacer(1, 12))

                    # Page break and Signal & Index Analysis
                    story.append(PageBreak())
                    story.append(Paragraph('Signal & Index Analysis', section_heading))
                    story.append(Spacer(1, 6))
                    if timeseries_copy.exists():
                        try:
                            img2 = RLImage(str(timeseries_copy))
                            target_w2 = doc.width
                            if getattr(img2, 'drawWidth', None) is None or img2.drawWidth == 0:
                                img2.drawWidth = target_w2
                                img2.drawHeight = img2.imageHeight * (target_w2 / img2.imageWidth) if getattr(img2, 'imageWidth', None) else target_w2 * 0.6
                            else:
                                orig_w = img2.drawWidth
                                orig_h = img2.drawHeight
                                img2.drawWidth = target_w2
                                img2.drawHeight = orig_h * (target_w2 / orig_w)
                            story.append(img2)
                        except Exception:
                            story.append(Paragraph('Timeseries image unavailable', body_style))
                    else:
                        story.append(Paragraph('Timeseries image unavailable', body_style))

                    story.append(Spacer(1, 8))
                    story.append(Paragraph('Interpretation Notes', section_heading))
                    story.append(Paragraph(interp or '-', body_style))

                    doc.build(story)
                    pdf_created = True
                except Exception:
                    pdf_created = False

                try:
                    if Image is not None:
                        try:
                            # page size: standard letter-like proportions at higher DPI
                            page_w = 1200
                            page_h = 1600
                            # margins: top/bottom = 2.54 cm (1 in -> 72 pts), left/right = 3.17 cm
                            margin_tb_pts = 72.0
                            margin_lr_pts = (3.17 / 2.54) * 72.0
                            margin_tb = int(round(margin_tb_pts))
                            margin_lr = int(round(margin_lr_pts))
                            bg = Image.new("RGB", (page_w, page_h), color=(255, 255, 255))
                            draw = ImageDraw.Draw(bg)

                            # fonts (use slightly larger body font and add spacing)
                            try:
                                font_title = ImageFont.truetype("arial.ttf", 28)
                                font_ts = ImageFont.truetype("arial.ttf", 12)
                                font_body = ImageFont.truetype("arial.ttf", 11)
                            except Exception:
                                font_title = ImageFont.load_default()
                                font_ts = ImageFont.load_default()
                                font_body = ImageFont.load_default()

                            # title (centered)
                            y = margin_tb
                            title_line = title_text or "Analysis Report"
                            tw, th = draw.textsize(title_line, font=font_title)
                            draw.text(((page_w - tw) / 2, y), title_line, fill=(0, 0, 0), font=font_title)
                            y += th + 8

                            # timestamp (centered)
                            ts_line = time.strftime("%Y-%m-%d %H:%M:%S")
                            tw, th = draw.textsize(ts_line, font=font_ts)
                            draw.text(((page_w - tw) / 2, y), ts_line, fill=(0, 0, 0), font=font_ts)
                            y += th + 18

                            # paste timeseries image centered — use ~80% of usable width
                            if timeseries_copy.exists():
                                try:
                                    g = Image.open(str(timeseries_copy)).convert("RGB")
                                    usable_w = page_w - 2 * margin_lr
                                    target_w = int(usable_w * 0.80)
                                    w, h = g.size
                                    new_w = min(target_w, w)
                                    new_h = int(h * (new_w / w))
                                    g = g.resize((new_w, new_h), Image.Resampling.LANCZOS)
                                    bg.paste(g, ((page_w - new_w) // 2, y))
                                    y += new_h + 24
                                except Exception:
                                    pass

                            # render summary: justify by simple wrapping and left-aligned within margins
                            body_x = margin_lr
                            usable_w = page_w - 2 * margin_lr
                            # build structured blocks: label-value pairs, bullets, and paragraphs
                            text_blocks = []
                            if dom:
                                text_blocks.append(('label_value', 'Dominant Regime', _strip_label_prefix(dom, 'Dominant Regime')))
                            if trans:
                                text_blocks.append(('label_value', 'Transition Observed', _strip_label_prefix(trans, 'Transition Observed')))
                            # evidence -> bullet list
                            evidence_bullets = _evidence_bullets(evidence)
                            if evidence_bullets:
                                text_blocks.append(('label', 'Key Evidence:'))
                                for b in evidence_bullets:
                                    text_blocks.append(('bullet', b))
                            if interp:
                                text_blocks.append(('label', 'Interpretation:'))
                                text_blocks.append(('paragraph', _strip_label_prefix(interp, 'Interpretation')))
                            if conf:
                                text_blocks.append(('label_value', 'Confidence', _strip_label_prefix(conf, 'Confidence')))

                            # draw each block with wrapping — compute max chars from usable width
                            try:
                                avg_char_w = font_body.getsize('M')[0]
                                if avg_char_w <= 0:
                                    raise Exception()
                                max_chars = max(40, int(usable_w / avg_char_w))
                            except Exception:
                                max_chars = 90

                            # prepare a bold font for labels if available
                            try:
                                font_body_bold = ImageFont.truetype("arialbd.ttf", getattr(font_body, 'size', 11))
                            except Exception:
                                font_body_bold = font_body

                            for block in text_blocks:
                                import textwrap
                                btype = block[0]
                                try:
                                    ascent, descent = font_body.getmetrics()
                                    line_h = ascent + descent
                                except Exception:
                                    line_h = font_body.getsize('M')[1]

                                if btype == 'label_value':
                                    # add a blank line above Interpretation label_value blocks
                                    if str(block[1]).strip().lower().startswith('interpretation'):
                                        y += line_h
                                    label = f"{block[1]}: "
                                    value = block[2] or ''
                                    wrapped = textwrap.wrap(value, width=max_chars)
                                    # label on first line (bold), value continues; subsequent lines indented
                                    label_w = font_body_bold.getsize(label)[0] if hasattr(font_body_bold, 'getsize') else 0
                                    if wrapped:
                                        draw.text((body_x, y), label, fill=(0, 0, 0), font=font_body_bold)
                                        draw.text((body_x + label_w + 6, y), wrapped[0], fill=(0, 0, 0), font=font_body)
                                        y += line_h + 10
                                        for line in wrapped[1:]:
                                            draw.text((body_x + label_w + 6, y), line, fill=(0, 0, 0), font=font_body)
                                            y += line_h + 10
                                    else:
                                        draw.text((body_x, y), label + value, fill=(0, 0, 0), font=font_body_bold)
                                        y += line_h + 10
                                    # add one extra blank line after the label_value section
                                    y += line_h

                                elif btype == 'label':
                                    label = block[1]
                                    draw.text((body_x, y), label, fill=(0, 0, 0), font=font_body_bold)
                                    y += line_h + 10
                                    # add one extra blank line after a label (section separator)
                                    y += line_h

                                elif btype == 'paragraph':
                                    paragraph = block[1] if len(block) > 1 else ''
                                    wrapped = textwrap.wrap(paragraph, width=max_chars)
                                    for line in wrapped:
                                        draw.text((body_x, y), line, fill=(0, 0, 0), font=font_body)
                                        y += line_h + 10
                                    y += line_h

                                elif btype == 'bullet':
                                    bullet = f"• {block[1]}"
                                    wrapped = textwrap.wrap(bullet, width=max_chars - 4)
                                    for i, line in enumerate(wrapped):
                                        indent_x = body_x + 12
                                        draw.text((indent_x, y), line, fill=(0, 0, 0), font=font_body)
                                        y += line_h + 10
                                    y += 6

                                else:
                                    # fallback: plain paragraph
                                    text = block[1] if len(block) > 1 else str(block)
                                    wrapped = textwrap.fill(text, width=max_chars)
                                    for line in wrapped.splitlines():
                                        draw.text((body_x, y), line, fill=(0, 0, 0), font=font_body)
                                        y += line_h + 8
                                    y += 12

                            bg.convert("RGB").save(str(pdf_path), "PDF", resolution=100.0)
                            pdf_created = True
                        except Exception:
                            pdf_created = False
                except Exception:
                    pdf_created = False

                if not pdf_created:
                    # fallback: use matplotlib to compose a simple PDF with margins and bullets
                    try:
                        import matplotlib.pyplot as plt
                        import matplotlib.image as mpimg
                        from matplotlib.backends.backend_pdf import PdfPages

                        with PdfPages(str(pdf_path)) as pdf:
                            fig_w, fig_h = 8.5, 11
                            fig = plt.figure(figsize=(fig_w, fig_h))

                            # apply margins: left/right = 3.17 cm, top/bottom = 2.54 cm
                            left_margin_in = 3.17 / 2.54
                            top_margin_in = 2.54 / 2.54
                            bottom_margin_in = top_margin_in

                            left = left_margin_in / fig_w
                            right = 1.0 - left_margin_in / fig_w
                            bottom = bottom_margin_in / fig_h
                            top = 1.0 - top_margin_in / fig_h

                            # reserve a little extra room for the suptitle so it doesn't collide
                            top = min(top, 0.96)
                            plt.subplots_adjust(left=left, right=right, top=top, bottom=bottom)

                            # place title and timestamp below the reserved top area
                            y_title = min(0.98, top + 0.02)
                            fig.suptitle(title_text or "Analysis Report", fontsize=18, y=y_title)
                            fig.text(0.5, y_title - 0.04, time.strftime("%Y-%m-%d %H:%M:%S"), ha='center')

                            # image panel centered within margins — place image above text area
                            avail_w = right - left
                            avail_h = top - bottom
                            # allocate a text area at the bottom (20% of available height)
                            text_area_height = max(0.12 * avail_h, min(0.25 * avail_h, 0.25 * avail_h))
                            # image gets remaining vertical space above the text area (with a small gap)
                            gap = 0.01
                            img_h = max(0.30 * avail_h, avail_h - text_area_height - gap)
                            img_w = avail_w * 0.80
                            img_x = left + (avail_w - img_w) / 2.0
                            img_y = bottom + text_area_height + gap
                            ax_img = fig.add_axes([img_x, img_y, img_w, img_h])
                            ax_img.set_axis_off()
                            ax_img.set_zorder(3)
                            if timeseries_copy.exists():
                                try:
                                    img = mpimg.imread(str(timeseries_copy))
                                    ax_img.imshow(img)
                                except Exception:
                                    ax_img.text(0.5, 0.5, 'Timeseries image unavailable', ha='center', va='center')
                            else:
                                ax_img.text(0.5, 0.5, 'Timeseries image unavailable', ha='center', va='center')

                            # summary text area: place at bottom and ensure it does not overlap image
                            ax_text = fig.add_axes([left, bottom, right - left, text_area_height])
                            ax_text.set_axis_off()
                            try:
                                ax_text.patch.set_alpha(0.0)
                            except Exception:
                                pass
                            ax_text.set_zorder(1)
                            lines = []
                            if dom:
                                lines.append(dom)
                            if trans:
                                lines.append(trans)
                            ev_bs = _evidence_bullets(evidence)
                            if ev_bs:
                                lines.append('Key Evidence:')
                                for b in ev_bs:
                                    lines.append(f'• {b}')
                            if interp:
                                lines.append(interp)
                            if conf:
                                lines.append(conf)

                            import textwrap
                            try:
                                # estimate chars per line (use a slightly smaller value for wrapping)
                                approx_chars = max(40, int((right - left) * fig_w * 10))
                            except Exception:
                                approx_chars = 90

                            # build structured blocks to allow bullets and label/value formatting
                            blocks = []
                            if dom:
                                blocks.append(('label_value', 'Dominant Regime', _strip_label_prefix(dom, 'Dominant Regime')))
                            if trans:
                                blocks.append(('label_value', 'Transition Observed', _strip_label_prefix(trans, 'Transition Observed')))
                            ev_bs = _evidence_bullets(evidence)
                            if ev_bs:
                                blocks.append(('label', 'Key Evidence:'))
                                for b in ev_bs:
                                    blocks.append(('bullet', b))
                            if interp:
                                blocks.append(('label', 'Interpretation:'))
                                blocks.append(('paragraph', _strip_label_prefix(interp, 'Interpretation')))
                            if conf:
                                blocks.append(('label_value', 'Confidence', _strip_label_prefix(conf, 'Confidence')))

                            # estimate number of lines to compute spacing
                            est_lines = 0
                            for b in blocks:
                                btype = b[0]
                                if btype == 'label':
                                    est_lines += 1
                                elif btype == 'label_value':
                                    wrapped = textwrap.wrap(b[2] or '', width=approx_chars)
                                    est_lines += max(1, len(wrapped))
                                elif btype == 'paragraph':
                                    wrapped = textwrap.wrap(b[1] or '', width=approx_chars)
                                    est_lines += max(1, len(wrapped))
                                elif btype == 'bullet':
                                    wrapped = textwrap.wrap(b[1], width=approx_chars - 4)
                                    est_lines += max(1, len(wrapped))
                                else:
                                    est_lines += 1

                            n_lines = max(1, est_lines)
                            # reasonable line height to balance compactness and readability
                            line_h = min(0.055, 0.95 / n_lines)
                            y = 0.98

                            for b in blocks:
                                btype = b[0]
                                if btype == 'label':
                                    ax_text.text(0.0, y, b[1], va='top', ha='left', transform=ax_text.transAxes, fontsize=9, fontweight='bold')
                                    y -= line_h
                                    # extra one-line gap after label
                                    y -= line_h
                                elif btype == 'label_value':
                                    label = f"{b[1]}:"
                                    value = b[2] or ''
                                    wrapped = textwrap.wrap(value, width=max(20, int(approx_chars * 0.9)))
                                    # if this is the Interpretation block, add a one-line gap above it
                                    if str(b[1]).strip().lower().startswith('interpretation'):
                                        y -= line_h

                                    try:
                                        # render the label first and measure its width so we can place the value immediately after
                                        label_text_obj = ax_text.text(0.0, y, label, va='top', ha='left', transform=ax_text.transAxes, fontsize=9, fontweight='bold')
                                        fig.canvas.draw()
                                        renderer = fig.canvas.get_renderer()
                                        bbox = label_text_obj.get_window_extent(renderer=renderer)
                                        axes_x1, _ = ax_text.transAxes.inverted().transform((bbox.x1, bbox.y0))
                                        padding = 0.005
                                        value_x = axes_x1 + padding

                                        if wrapped:
                                            # recompute wrapping to fit remaining axes width after the label using pixel metrics
                                            try:
                                                # axes and label bounding boxes in display (pixel) coords
                                                axes_bbox = ax_text.get_window_extent(renderer=renderer)
                                                label_bbox = label_text_obj.get_window_extent(renderer=renderer)
                                                padding_px = 2
                                                avail_px = max(10, axes_bbox.x1 - label_bbox.x1 - padding_px)
                                                # average character width in pixels for current font
                                                fp = label_text_obj.get_fontproperties()
                                                avg_char_px = renderer.get_text_width_height_descent('M', fp, ismath=False)[0]
                                                # slightly increase wrap width to better fill to the right margin
                                                wrap_chars = max(20, int(avail_px / max(1.0, avg_char_px) * 1.05) + 2)
                                                wrapped = textwrap.wrap(value, width=wrap_chars)
                                            except Exception:
                                                pass
                                            ax_text.text(value_x, y, wrapped[0], va='top', ha='left', transform=ax_text.transAxes, fontsize=9)
                                            y -= line_h
                                            for line in wrapped[1:]:
                                                ax_text.text(value_x, y, line, va='top', ha='left', transform=ax_text.transAxes, fontsize=9)
                                                y -= line_h
                                        else:
                                            y -= line_h

                                        # one-line gap after a label_value block
                                        y -= line_h
                                    except Exception:
                                        # fallback: draw label then value with a modest fixed offset
                                        ax_text.text(0.0, y, label, va='top', ha='left', transform=ax_text.transAxes, fontsize=9, fontweight='bold')
                                        small_offset = 0.03
                                        if wrapped:
                                            ax_text.text(small_offset, y, wrapped[0], va='top', ha='left', transform=ax_text.transAxes, fontsize=9)
                                            y -= line_h
                                            for line in wrapped[1:]:
                                                ax_text.text(small_offset, y, line, va='top', ha='left', transform=ax_text.transAxes, fontsize=9)
                                                y -= line_h
                                        else:
                                            y -= line_h
                                        y -= line_h
                                elif btype == 'paragraph':
                                    wrapped = textwrap.wrap(b[1] or '', width=max(20, approx_chars))
                                    if wrapped:
                                        for line in wrapped:
                                            ax_text.text(0.0, y, line, va='top', ha='left', transform=ax_text.transAxes, fontsize=9)
                                            y -= line_h
                                    y -= line_h
                                elif btype == 'bullet':
                                    # use bullet-prefixed lines, wrapped and indented
                                    wrapped = textwrap.wrap(b[1], width=max(20, int(approx_chars * 0.8)))
                                    bullet_x = 0.02
                                    indent_x = 0.06
                                    if wrapped:
                                        ax_text.text(indent_x, y, f'• {wrapped[0]}', va='top', ha='left', transform=ax_text.transAxes, fontsize=9)
                                        y -= line_h
                                        for line in wrapped[1:]:
                                            ax_text.text(indent_x, y, line, va='top', ha='left', transform=ax_text.transAxes, fontsize=9)
                                            y -= line_h
                                    else:
                                        ax_text.text(indent_x, y, '• ', va='top', ha='left', transform=ax_text.transAxes, fontsize=9)
                                        y -= line_h
                                else:
                                    ax_text.text(0.0, y, str(b), va='top', ha='left', transform=ax_text.transAxes, fontsize=9)
                                    y -= line_h

                            # If there were no blocks, render a single placeholder line
                            if not blocks:
                                ax_text.text(0.0, 1.0, 'No summary available.', va='top', ha='left', transform=ax_text.transAxes, fontsize=9)

                            pdf.savefig(fig)
                            plt.close(fig)
                            pdf_created = True
                    except Exception:
                        pdf_created = False

                if not pdf_created:
                    print(f"[Export] PDF report creation failed for {pdf_path}")
            except Exception:
                # best-effort: do not prevent export if report generation fails
                pass

            # remove temporary output directory after successful copy (best-effort)
            try:
                shutil.rmtree(str(src))
            except Exception:
                pass
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export validation results: {e}")
            return
        # disable export button and clear reference to temp outputs
        try:
            if hasattr(self.video_panel, "validation_export_btn"):
                self.video_panel.validation_export_btn.setEnabled(False)
        except Exception:
            pass
        self.validation_output_dir = None

        QMessageBox.information(self, "Export Complete", f"Validation results exported to: {target}")
    def _update_advisory(self):
        self.advisory_widget.update_advisory(
            self.smooth_cci if self.smooth_cci is not None else self.latest_cci,
            self.smooth_risk if self.smooth_risk is not None else self.latest_risk,
            self.smooth_projected_cci if self.smooth_projected_cci is not None else self.latest_projected_cci,
            self.smooth_slope if self.smooth_slope is not None else self.latest_slope,
            self.smooth_stability if self.smooth_stability is not None else self.latest_stability,
            self.smooth_stability_trend,
            self.latest_regime_phase,
            self.latest_forecast_direction
        )

    def _flush_smoothed_ui(self):
        if not self._ui_values_dirty:
            return

        cci_value = self.smooth_cci if self.smooth_cci is not None else self.latest_cci
        risk_value = self.smooth_risk if self.smooth_risk is not None else self.latest_risk
        projected_value = (
            self.smooth_projected_cci
            if self.smooth_projected_cci is not None
            else self.latest_projected_cci
        )
        slope_value = self.smooth_slope if self.smooth_slope is not None else self.latest_slope
        stability_value = self.smooth_stability if self.smooth_stability is not None else self.latest_stability

        # CCI
        self.cci_widget.update_cci(cci_value)

        # Risk dict with smoothed numeric fields + original phase components.
        risk_display = dict(self.latest_risk_data) if self.latest_risk_data else {}
        risk_display["risk"] = risk_value
        risk_display["cci_norm"] = cci_value
        risk_display["risk_state_label"] = self.risk_band_state
        risk_display["risk_state_color"] = self._risk_state_color(self.risk_band_state)
        self.risk_widget.update_risk(risk_display)

        # Stability and forecast
        if self.has_stability_signal:
            self.stability_widget.update_si(stability_value)
        if self.has_forecast_signal:
            forecast_display = {
                "projected_cci": projected_value,
                "current_cci": cci_value,
                "slope": slope_value,
                "stable": (self.latest_forecast_direction == "Stable"),
            }
            self.forecast_widget.update_forecast(forecast_display)

        # Advisory
        self._update_advisory()
        self._ui_values_dirty = False

    def _ema(self, prev: float, new_value: float):
        if prev is None:
            return float(new_value)
        new_value = float(new_value)
        prev = float(prev)
        alpha = self.ema_fast_alpha if abs(new_value - prev) >= self.ema_fast_delta else self.ema_alpha
        return (alpha * new_value) + ((1.0 - alpha) * prev)

    def _update_risk_band_hysteresis(self, risk_value: float):
        low_th = RISK.low_threshold
        crit_th = RISK.critical_threshold
        m = self.hysteresis_margin

        if self.risk_band_state == "Low":
            if risk_value >= (low_th + m):
                self.risk_band_state = "Elevated"
        elif self.risk_band_state == "Elevated":
            if risk_value >= (crit_th + m):
                self.risk_band_state = "Critical"
            elif risk_value <= (low_th - m):
                self.risk_band_state = "Low"
        elif self.risk_band_state == "Critical":
            if risk_value <= (crit_th - m):
                self.risk_band_state = "Elevated"

    @staticmethod
    def _risk_state_color(state: str) -> str:
        if state == "Low":
            return "green"
        if state == "Critical":
            return "red"
        return "amber"

    @staticmethod
    def _infer_regime_phase(risk_data: dict) -> str:
        r_gas = risk_data.get("R_gas")
        r_fluid = risk_data.get("R_fluid")
        r_granular = risk_data.get("R_granular")
        if None in (r_gas, r_fluid, r_granular):
            return "Unknown"
        phase_map = {
            "Gas": float(r_gas),
            "Fluid": float(r_fluid),
            "Granular": float(r_granular),
        }
        return max(phase_map, key=phase_map.get)

    @staticmethod
    def _infer_forecast_direction(slope: float, stable_flag: bool) -> str:
        if stable_flag:
            return "Stable"
        threshold = 0.01
        if slope > threshold:
            return "Escalating"
        if slope < -threshold:
            return "Dissipating"
        return "Stable"

    def on_skc_update(self, s_value: float, k_value: float, c_value: float):
        self.ternary_widget.update_skc(s_value, k_value, c_value)
        self.regime_map_widget.update_state(s_value, c_value)
        self.s_value_label.setText(f"Spatial (S): {s_value:.3f}")
        self.k_value_label.setText(f"Kinematic (K): {k_value:.3f}")
        self.c_value_label.setText(f"Coherence (C): {c_value:.3f}")
        # record latest SKC for session export
        try:
            self.latest_s = float(s_value)
            self.latest_k = float(k_value)
            self.latest_c = float(c_value)
        except Exception:
            pass

    # --------- UI Updates ---------
    def _draw_annotation_overlay(self, pixmap: QPixmap):
        if not self.annotation_mode:
            return pixmap
        working = QPixmap(pixmap)
        painter = QPainter(working)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        for polygons in self.annotation_polygons.values():
            for zone_name, points in polygons.items():
                if not points:
                    continue
                bgr = ZONE_COLORS.get(zone_name, (160, 160, 160))
                color = QColor(bgr[2], bgr[1], bgr[0], 220)
                fill = QColor(bgr[2], bgr[1], bgr[0], 48)
                pen_width = 3 if zone_name == self.annotation_zone else 2
                painter.setPen(QPen(color, pen_width))
                painter.setBrush(fill)
                polygon = QPolygon([QPoint(int(x), int(y)) for x, y in points])
                if len(points) >= 3:
                    painter.drawPolygon(polygon)
                else:
                    painter.drawPolyline(polygon)
                for x, y in points:
                    painter.setBrush(color)
                    painter.drawEllipse(QPoint(int(x), int(y)), 4, 4)
        painter.end()
        return working

    def _render_current_frame(self):
        if self.latest_frame_qimage is None:
            return
        pix = QPixmap.fromImage(self.latest_frame_qimage)
        pix = self._draw_annotation_overlay(pix)
        pix = pix.scaled(
            self.video_panel.video_display.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation
        )
        self.video_panel.video_display.setPixmap(pix)
    def update_frame(self, qimg):
        # Accept frames for display if either full source analytics is active
        # or a preview-only camera has been opened.
        if not (self.source_active or getattr(self, "preview_active", False)):
            return
        self.latest_frame_qimage = qimg.copy()
        self.latest_frame_size = (qimg.width(), qimg.height())
        # Record per-frame signals for session export (if recording enabled)
        if self.source_active and getattr(self, "_session_records", None) is not None:
            try:
                elapsed = None
                if self.live_start_ts is not None:
                    elapsed = time.monotonic() - self.live_start_ts
                rec = {
                    "index": int(getattr(self, "_session_frame_count", 0)),
                    "timestamp": float(time.time()),
                    "elapsed": float(elapsed) if elapsed is not None else None,
                    "cci": float(getattr(self, "latest_cci", 0.0)),
                    "risk": float(getattr(self, "latest_risk", 0.0)),
                    "s": float(getattr(self, "latest_s", 0.0)),
                    "k": float(getattr(self, "latest_k", 0.0)),
                    "c": float(getattr(self, "latest_c", 0.0)),
                    "stability": float(getattr(self, "latest_stability", 0.0)),
                    "projected_cci": float(getattr(self, "latest_projected_cci", 0.0)),
                    "slope": float(getattr(self, "latest_slope", 0.0)),
                    "regime_phase": getattr(self, "latest_regime_phase", "Unknown"),
                    "forecast_dir": getattr(self, "latest_forecast_direction", "Stable"),
                }
                self._session_records.append(rec)
                self._session_frame_count = int(getattr(self, "_session_frame_count", 0)) + 1
            except Exception:
                pass
        self._render_current_frame()

    def blackout(self):
        self.latest_frame_qimage = None
        self.latest_frame_size = None
        img = QImage(640, 480, QImage.Format.Format_RGB888)
        img.fill(Qt.GlobalColor.black)
        self.video_panel.video_display.setPixmap(QPixmap.fromImage(img))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    icon_path = Path(__file__).resolve().parent / "CrowdTune Logo- white.ico"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    ui = CrowdTuneUI()
    ui.showMaximized()
    sys.exit(app.exec())


















































































