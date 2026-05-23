from collections import deque

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget


class ForecastSparkline(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._values = deque(maxlen=48)
        self._line_color = QColor(140, 146, 155)
        self.setMinimumHeight(32)

    def append_value(self, value: float):
        self._values.append(max(0.0, min(1.0, float(value))))
        self.update()

    def set_line_color(self, color: QColor):
        self._line_color = QColor(color)
        self.update()

    def reset(self):
        self._values.clear()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), Qt.GlobalColor.transparent)

        if len(self._values) < 2:
            return

        left = 2
        top = 3
        plot_w = max(1, self.width() - 4)
        plot_h = max(1, self.height() - 6)

        path = QPainterPath()
        values = list(self._values)
        n = len(values)
        for idx, value in enumerate(values):
            x = left + (idx / max(1, n - 1)) * plot_w
            y = top + (1.0 - value) * plot_h
            if idx == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)

        pen = QPen(QColor(self._line_color.red(), self._line_color.green(), self._line_color.blue(), 200))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawPath(path)


class ForecastWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(120)

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(4)

        self.title = QLabel("10s Projection")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        self.title.setStyleSheet("color: #f1f3f4;")

        self.line = QFrame()
        self.line.setFrameShape(QFrame.Shape.HLine)
        self.line.setStyleSheet("color: #333333;")

        self.sparkline = ForecastSparkline()

        self.value_label = QLabel("0.00")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.value_label.setFont(QFont("Arial", 42, QFont.Weight.Bold))
        self.value_label.setStyleSheet("color: #9aa0a6;")

        self.delta_label = QLabel("Delta: +0.00")
        self.delta_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.delta_label.setFont(QFont("Arial", 10))
        self.delta_label.setStyleSheet("color: #9aa0a6;")

        self.trend_label = QLabel("Stable")
        self.trend_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.trend_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.trend_label.setStyleSheet("color: #9aa0a6;")

        layout.addWidget(self.title)
        layout.addWidget(self.line)
        layout.addWidget(self.sparkline)
        layout.addWidget(self.value_label)
        layout.addWidget(self.delta_label)
        layout.addWidget(self.trend_label)
        self.setLayout(layout)

    def update_forecast(self, data: dict):
        current = data.get("current_cci", None)
        projected = max(0.0, min(1.0, float(data.get("projected_cci", 0.0))))
        slope = float(data.get("slope", 0.0))
        stable = bool(data.get("stable", False))

        self.value_label.setText(f"{projected:.2f}")
        self.sparkline.append_value(projected)

        delta = 0.0 if current is None else projected - float(current)
        self.delta_label.setText(f"Delta: {delta:+.2f}")

        threshold = 0.01
        if stable or abs(slope) < threshold:
            trend_text = "Stable"
            trend_color = QColor(140, 146, 155)
        elif slope > 0:
            trend_text = "Escalating"
            trend_color = QColor(243, 156, 18)
        else:
            trend_text = "Dissipating"
            trend_color = QColor(46, 204, 113)

        self.trend_label.setText(trend_text)
        color_css = f"rgb({trend_color.red()}, {trend_color.green()}, {trend_color.blue()})"
        self.trend_label.setStyleSheet(f"color: {color_css}; font-size: 12px; font-weight: 700;")
        self.value_label.setStyleSheet(f"color: {color_css};")
        if trend_text == "Stable":
            self.delta_label.setStyleSheet("color: #a6adb4;")
        else:
            self.delta_label.setStyleSheet(f"color: {color_css};")
        self.sparkline.set_line_color(trend_color)

    def reset(self):
        self.sparkline.reset()
        self.value_label.setText("0.00")
        self.value_label.setStyleSheet("color: #9aa0a6;")
        self.delta_label.setText("Delta: +0.00")
        self.delta_label.setStyleSheet("color: #9aa0a6;")
        self.trend_label.setText("Awaiting Signal")
        self.trend_label.setStyleSheet("color: #9aa0a6; font-size: 12px; font-weight: 700;")

