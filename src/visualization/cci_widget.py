# cci_widget.py
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QColor, QFont, QPen
from PyQt6.QtCore import Qt, QRect


class CCIWidget(QWidget):
    def __init__(self, parent=None, alpha=0.45):  # tuned for LBP-based S
        super().__init__(parent)

        self.alpha = alpha

        self.raw_cci = 0.0
        self.cci_norm = 0.0
        self.label = "Free Flow"
        self.color = QColor(80, 200, 120)

        self.setMinimumSize(220, 220)

    def update_cci(self, cci_value: float):
        self.raw_cci = float(cci_value)
        self.cci_norm = max(0.0, min(self.raw_cci, 1.0))
        self.label, self.color = self._interpret(self.cci_norm)
        self.update()

    def reset(self):
        self.raw_cci = 0.0
        self.cci_norm = 0.0
        self.label, self.color = self._interpret(0.0)
        self.update()

    def _interpret(self, cci_norm):
        if cci_norm < 0.18:
            return "Free Flow", QColor(0, 200, 120)
        if cci_norm < 0.35:
            return "Dense", QColor(255, 200, 0)
        if cci_norm < 0.55:
            return "Constrained", QColor(255, 140, 0)
        if cci_norm < 0.78:
            return "Critical", QColor(255, 80, 80)
        return "Jammed", QColor(180, 0, 0)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect()
        size = min(rect.width(), rect.height()) - 20
        cx = rect.center().x()
        cy = rect.center().y()
        radius = size // 2
        ring_rect = QRect(cx - radius, cy - radius, 2 * radius, 2 * radius)

        bg_pen = QPen(QColor(60, 60, 60), 12)
        painter.setPen(bg_pen)
        painter.drawArc(ring_rect, 0, 360 * 16)

        fg_pen = QPen(self.color, 14)
        painter.setPen(fg_pen)
        span_angle = int(360 * self.cci_norm)
        painter.drawArc(ring_rect, -90 * 16, -span_angle * 16)

        # Boundary tick at bottom end of the gauge.
        tick_pen = QPen(QColor(200, 200, 200, 130), 1)
        painter.setPen(tick_pen)
        x_outer = cx
        y_outer = cy + radius + 4
        x_inner = cx
        y_inner = cy + radius - 8
        painter.drawLine(x_outer, y_outer, x_inner, y_inner)

        # Keep both texts centered relative to the ring geometry.
        painter.setPen(Qt.GlobalColor.white)
        value_rect = QRect(
            ring_rect.left(),
            ring_rect.top() + int(ring_rect.height() * 0.33),
            ring_rect.width(),
            int(ring_rect.height() * 0.24),
        )
        painter.setFont(QFont("Inter", 26, QFont.Weight.Bold))
        painter.drawText(value_rect, Qt.AlignmentFlag.AlignCenter, f"{self.cci_norm:.2f}")

        label_rect = QRect(
            ring_rect.left(),
            ring_rect.top() + int(ring_rect.height() * 0.57),
            ring_rect.width(),
            int(ring_rect.height() * 0.14),
        )
        painter.setFont(QFont("Inter", 11))
        painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, self.label)
