from collections import deque
import math
import time

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QTransform
from PyQt6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

TRANSITION_THRESHOLD = 0.4


class RiskIndexWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(180)

        self.risk = 0.0
        self.cci = 0.0
        self.risk_state_label = None
        self.risk_state_color = None
        self.history_window_seconds = 10.0
        self.risk_history = deque()

        self.phase_label = QLabel("Regime: ---")
        self.phase_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.phase_label.setStyleSheet(
            """
            font-size: 14px;
            color: #bbbbbb;
            """
        )
        self.confidence_label = QLabel("Confidence: 0.00")
        self.confidence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.confidence_label.setStyleSheet(
            """
            font-size: 12px;
            color: #9e9e9e;
            """
        )

        layout = QVBoxLayout()
        layout.addWidget(self.phase_label)
        layout.addWidget(self.confidence_label)
        layout.addSpacing(10)
        layout.addStretch()
        self.setLayout(layout)

        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred
        )

    def update_risk(self, data: dict):
        self.risk = max(0.0, min(1.0, float(data.get("risk", 0.0))))
        self.cci = max(0.0, min(1.0, float(data.get("cci_norm", 0.0))))
        self.risk_state_label = data.get("risk_state_label", None)
        self.risk_state_color = data.get("risk_state_color", None)
        self._append_risk_history(self.risk)

        r_gas = data.get("R_gas", None)
        r_fluid = data.get("R_fluid", None)
        r_granular = data.get("R_granular", None)

        if None not in (r_gas, r_fluid, r_granular):
            self.update_phase(r_gas, r_fluid, r_granular)

        self.update()

    def reset(self):
        self.risk = 0.0
        self.cci = 0.0
        self.risk_state_label = None
        self.risk_state_color = None
        self.risk_history.clear()
        self.phase_label.setText("Regime: ---")
        self.confidence_label.setText("Confidence: 0.00")
        self.update()

    def update_phase(self, r_gas, r_fluid, r_granular):
        values = {
            "Gas": float(r_gas),
            "Fluid": float(r_fluid),
            "Granular": float(r_granular),
        }

        dominant = max(values, key=values.get)
        confidence = max(0.0, min(1.0, values[dominant]))

        if confidence < TRANSITION_THRESHOLD:
            self.phase_label.setText("Regime: Transitional")
            self.confidence_label.setText("Confidence: --")
        else:
            self.phase_label.setText(f"Regime: {dominant}")
            self.confidence_label.setText(f"Confidence: {confidence:.2f}")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect().adjusted(20, 72, -20, -20)
        radius = min(rect.width(), rect.height()) // 2

        center_x = rect.center().x()
        avail_mid_y = (rect.top() + rect.bottom()) / 2.0
        center_y = int(avail_mid_y + max(0.0, (radius - 76.0) / 2.0))
        center_y = min(center_y + 16, rect.bottom() - 58)
        center_y = max(center_y, rect.top() + radius)
        center = rect.center()
        center.setX(center_x)
        center.setY(center_y)

        base_pen = QPen(QColor(55, 55, 55), 12)
        base_pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        painter.setPen(base_pen)
        painter.drawArc(
            center.x() - radius,
            center.y() - radius,
            radius * 2,
            radius * 2,
            180 * 16,
            -180 * 16
        )

        ranges = [
            (0.00, 0.25, QColor(0, 145, 88, 150)),
            (0.25, 0.45, QColor(175, 135, 0, 150)),
            (0.45, 0.65, QColor(175, 95, 0, 150)),
            (0.65, 0.85, QColor(175, 55, 55, 150)),
            (0.85, 1.00, QColor(115, 0, 0, 150)),
        ]
        for r0, r1, color in ranges:
            seg_pen = QPen(color, 12)
            seg_pen.setCapStyle(Qt.PenCapStyle.FlatCap)
            painter.setPen(seg_pen)
            start_angle = int((180.0 - r0 * 180.0) * 16.0)
            span_angle = int(-(r1 - r0) * 180.0 * 16.0)
            painter.drawArc(
                center.x() - radius,
                center.y() - radius,
                radius * 2,
                radius * 2,
                start_angle,
                span_angle
            )

        tick_values = [0.25, 0.45, 0.65, 0.85]
        tick_pen = QPen(QColor(180, 180, 180, 80), 4)
        painter.setPen(tick_pen)
        for t in tick_values:
            theta = self._risk_to_theta(t)
            x_inner = center.x() + int(math.cos(theta) * (radius - 8))
            y_inner = center.y() - int(math.sin(theta) * (radius - 8))
            x_outer = center.x() + int(math.cos(theta) * (radius + 4))
            y_outer = center.y() - int(math.sin(theta) * (radius + 4))
            painter.drawLine(x_inner, y_inner, x_outer, y_outer)

        history = list(self.risk_history)
        if len(history) > 1:
            now = time.monotonic()
            for ts, risk_value in history:
                age = max(0.0, now - ts)
                fade = max(0.0, min(1.0, 1.0 - age / self.history_window_seconds))
                theta = self._risk_to_theta(risk_value)
                x = center.x() + int(math.cos(theta) * radius)
                y = center.y() - int(math.sin(theta) * radius)

                ghost_color = self._risk_zone_color(risk_value)
                ghost_color.setAlpha(int(25 + 80 * fade))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(ghost_color)
                r = 2 + int(2 * fade)
                painter.drawEllipse(x - r, y - r, 2 * r, 2 * r)

        marker_theta = self._risk_to_theta(self.risk)
        marker_x = center.x() + int(math.cos(marker_theta) * radius)
        marker_y = center.y() - int(math.sin(marker_theta) * radius)
        painter.setPen(QPen(QColor(20, 20, 20), 1))
        painter.setBrush(QColor(240, 240, 240))

        transform = QTransform()
        transform.translate(marker_x, marker_y)
        transform.rotate(-math.degrees(marker_theta))
        painter.setTransform(transform, True)
        painter.drawRect(-8, -4, 16, 6)
        painter.resetTransform()

        value_width = max(120, min(radius * 2, 200))
        value_x = center.x() - value_width // 2
        value_y = center.y() - max(34, int(radius * 0.28))
        value_text_y = value_y + 7
        value_height = 42
        separator_gap = 6
        separator_y = value_y + value_height + separator_gap
        separator_width = max(70, min(int(radius * 0.95), 120))
        separator_x0 = center.x() - separator_width // 2
        separator_x1 = center.x() + separator_width // 2

        painter.setPen(self._risk_zone_color(self.risk))
        painter.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        painter.drawText(
            value_x,
            value_text_y,
            value_width,
            value_height,
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
            f"{self.risk:.2f}"
        )

        separator_pen = QPen(QColor(210, 210, 210, 90), 1)
        separator_pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        painter.setPen(separator_pen)
        painter.drawLine(separator_x0, separator_y, separator_x1, separator_y)

        state_text = self._risk_zone_label(self.risk)
        painter.setPen(self._state_label_color(self.risk))
        painter.setFont(QFont("Arial", 15, QFont.Weight.Bold))
        painter.drawText(
            value_x,
            separator_y + separator_gap,
            value_width,
            30,
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
            state_text
        )

    def _append_risk_history(self, risk_value: float):
        now = time.monotonic()
        self.risk_history.append((now, risk_value))
        cutoff = now - self.history_window_seconds
        while self.risk_history and self.risk_history[0][0] < cutoff:
            self.risk_history.popleft()

    def _risk_to_theta(self, risk_value: float):
        return math.pi * (1.0 - max(0.0, min(1.0, risk_value)))

    def _risk_zone_color(self, risk_value: float):
        r = max(0.0, min(1.0, risk_value))
        if r < 0.25:
            return QColor(0, 145, 88)
        if r < 0.45:
            return QColor(175, 135, 0)
        if r < 0.65:
            return QColor(175, 95, 0)
        if r < 0.85:
            return QColor(175, 55, 55)
        return QColor(115, 0, 0)

    def _risk_zone_label(self, risk_value: float):
        if self.risk_state_label:
            return self.risk_state_label
        r = max(0.0, min(1.0, risk_value))
        if r < 0.25:
            return "Low"
        if r < 0.45:
            return "Elevated"
        if r < 0.65:
            return "Constrained"
        if r < 0.85:
            return "Critical"
        return "Extreme"

    def _state_label_color(self, risk_value: float):
        if self.risk_state_color == "green":
            return QColor(0, 145, 88)
        if self.risk_state_color == "amber":
            return QColor(175, 135, 0)
        if self.risk_state_color == "red":
            return QColor(175, 55, 55)
        return self._risk_zone_color(risk_value)








