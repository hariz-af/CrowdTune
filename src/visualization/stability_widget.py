from PyQt6.QtCharts import QChart, QChartView, QLineSeries, QScatterSeries, QValueAxis
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget


class StabilityZoneStrip(QWidget):
    def __init__(self):
        super().__init__()
        self._value = 0.0
        self.setFixedWidth(14)
        self.setMinimumHeight(120)

    def set_value(self, value: float):
        self._value = max(0.0, min(1.0, float(value)))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        w = self.width()
        h = self.height()
        x = 2
        bar_w = max(4, w - 4)

        # y=1.0 at top, y=0.0 at bottom
        def y_from_value(v: float) -> int:
            return int(round((1.0 - v) * (h - 1)))

        y_top = y_from_value(1.0)
        y_07 = y_from_value(0.7)
        y_05 = y_from_value(0.5)
        y_bot = y_from_value(0.0)

        painter.fillRect(x, y_top, bar_w, max(1, y_07 - y_top), QColor(46, 204, 113, 75))
        painter.fillRect(x, y_07, bar_w, max(1, y_05 - y_07), QColor(243, 156, 18, 75))
        painter.fillRect(x, y_05, bar_w, max(1, y_bot - y_05 + 1), QColor(231, 76, 60, 75))

        border_pen = QPen(QColor(160, 160, 160, 120), 1)
        painter.setPen(border_pen)
        painter.drawRect(x, 0, bar_w, h - 1)
        painter.drawLine(x, y_07, x + bar_w, y_07)
        painter.drawLine(x, y_05, x + bar_w, y_05)

        marker_y = y_from_value(self._value)
        painter.setPen(QPen(QColor(240, 240, 240), 2))
        painter.drawLine(x - 1, marker_y, x + bar_w + 1, marker_y)


class StabilityWidget(QWidget):
    def __init__(self, max_points=200):
        super().__init__()

        self.max_points = max_points
        self.sample_period_s = 0.12
        self.delta_window_points = max(2, int(5.0 / self.sample_period_s))

        self.series = QLineSeries()
        self.series.setColor(QColor(80, 200, 255))
        self.series.setPen(QPen(QColor(80, 200, 255), 1))

        self.current_marker = QScatterSeries()
        self.current_marker.setColor(QColor(120, 240, 255))
        self.current_marker.setBorderColor(QColor(20, 20, 20))
        self.current_marker.setMarkerSize(9.0)

        self.chart = QChart()
        self.chart.setTitle("Crowd Stability Index (SI)")
        self.chart.legend().hide()
        self.chart.setBackgroundBrush(QColor(20, 20, 20))

        self.axis_x = QValueAxis()
        self.axis_x.setRange(0, self.max_points)
        self.axis_x.setLabelFormat("%d")
        self.axis_x.setTitleText("Time")

        self.axis_y = QValueAxis()
        self.axis_y.setRange(0.0, 1.0)
        self.axis_y.setLabelFormat("%.2f")
        self.axis_y.setTitleText("Stability")

        self.chart.addAxis(self.axis_x, Qt.AlignmentFlag.AlignBottom)
        self.chart.addAxis(self.axis_y, Qt.AlignmentFlag.AlignLeft)

        self.chart.addSeries(self.series)
        self.series.attachAxis(self.axis_x)
        self.series.attachAxis(self.axis_y)

        self.chart.addSeries(self.current_marker)
        self.current_marker.attachAxis(self.axis_x)
        self.current_marker.attachAxis(self.axis_y)

        self.view = QChartView(self.chart)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.zone_strip = StabilityZoneStrip()

        self.current_value_label = QLabel("SI: 0.00")
        self.current_value_label.setStyleSheet("color: #9ddff2; font-size: 12px; font-weight: 700;")
        self.delta_label = QLabel("Delta (5s): +0.00")
        self.delta_label.setStyleSheet("color: #9aa0a6; font-size: 12px;")

        info_row = QHBoxLayout()
        # Match chart block inset so info text aligns with the dark chart edge.
        info_row.setContentsMargins(17, 0, 30, 0)
        info_row.addStretch()
        info_row.addWidget(self.current_value_label)
        info_row.addSpacing(12)
        info_row.addWidget(self.delta_label)

        chart_row = QHBoxLayout()
        chart_row.setContentsMargins(0, 0, 0, 0)
        chart_row.setSpacing(8)
        chart_row.addWidget(self.view, 1)
        chart_row.addWidget(self.zone_strip, 0, Qt.AlignmentFlag.AlignVCenter)

        layout = QVBoxLayout()
        layout.addLayout(info_row)
        layout.addLayout(chart_row)
        self.setLayout(layout)

        self.t = 0

    def update_si(self, si_value: float):
        si_value = max(0.0, min(1.0, float(si_value)))
        self.series.append(self.t, si_value)
        self.t += 1

        if self.series.count() > self.max_points:
            self.series.remove(0)

        if self.t <= self.max_points:
            self.axis_x.setRange(0, self.max_points)
        else:
            self.axis_x.setRange(self.t - self.max_points, self.t)
        self.zone_strip.set_value(si_value)

        self.current_marker.clear()
        self.current_marker.append(self.t - 1, si_value)

        self.current_value_label.setText(f"SI: {si_value:.2f}")

        points = self.series.points()
        if len(points) > self.delta_window_points:
            prev_value = points[-self.delta_window_points].y()
        elif points:
            prev_value = points[0].y()
        else:
            prev_value = si_value

        delta = si_value - prev_value
        self.delta_label.setText(f"Delta (5s): {delta:+.2f}")
        if delta > 0.01:
            self.delta_label.setStyleSheet("color: #2ecc71; font-size: 12px; font-weight: 700;")
        elif delta < -0.01:
            self.delta_label.setStyleSheet("color: #e74c3c; font-size: 12px; font-weight: 700;")
        else:
            self.delta_label.setStyleSheet("color: #9aa0a6; font-size: 12px;")

    def reset(self):
        self.series.clear()
        self.current_marker.clear()
        self.t = 0
        self.axis_x.setRange(0, self.max_points)
        self.zone_strip.set_value(0.0)
        self.current_value_label.setText("SI: --")
        self.delta_label.setText("Delta (5s): --")
        self.delta_label.setStyleSheet("color: #9aa0a6; font-size: 12px;")

