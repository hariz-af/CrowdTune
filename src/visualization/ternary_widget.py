import math
from collections import deque

from PyQt6.QtWidgets import QWidget, QVBoxLayout

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class TernaryPlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._regime_overlay_enabled = False
        self._regime_overlay_points = self._build_regime_intensity_field(resolution=95)

        self.figure = Figure(facecolor="#1E1E1E", edgecolor="#1E1E1E")
        self.figure.patch.set_linewidth(0)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("background-color: #1E1E1E; border: none;")
        self.ax = self.figure.add_subplot(111)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

        self._draw_base()
        self._point_artist = None
        self._point_ring_artist = None
        self._ghost_artist = None
        self._trail_points = deque(maxlen=24)
        self._ema_alpha = 0.22
        self._s_smooth = None
        self._k_smooth = None
        self._c_smooth = None
        self.update_skc(0.33, 0.33, 0.34)

    def _draw_base(self):
        self.ax.clear()
        self.ax.set_facecolor("#1E1E1E")
        self.ax.grid(False)
        self.ax.set_axis_off()
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_aspect("equal", adjustable="box")

        h = math.sqrt(3) / 2.0

        if self._regime_overlay_enabled:
            self._draw_regime_intensity_field()
        self._draw_ternary_grid()

        # Triangle vertices: K=(0,0), C=(1,0), S=(0.5,h)
        self.ax.plot([0.0, 1.0], [0.0, 0.0], color="#c0c0c0", linewidth=2.4, zorder=3)
        self.ax.plot([0.0, 0.5], [0.0, h], color="#c0c0c0", linewidth=2.4, zorder=3)
        self.ax.plot([1.0, 0.5], [0.0, h], color="#c0c0c0", linewidth=2.4, zorder=3)

        self.ax.text(0.5, h + 0.06, "S", color="white", ha="center", va="bottom", fontsize=11, zorder=4)
        self.ax.text(-0.05, -0.03, "K", color="white", ha="right", va="top", fontsize=11, zorder=4)
        self.ax.text(1.05, -0.03, "C", color="white", ha="left", va="top", fontsize=11, zorder=4)

        # Reduce outer padding so the triangle appears about 5% larger.
        self.ax.set_xlim(-0.03, 1.03)
        self.ax.set_ylim(-0.03, h + 0.03)

        for spine in self.ax.spines.values():
            spine.set_visible(False)

    def _build_regime_intensity_field(self, resolution: int = 95):
        points = []
        gas_rgb = (46 / 255.0, 204 / 255.0, 113 / 255.0)       # green
        fluid_rgb = (243 / 255.0, 156 / 255.0, 18 / 255.0)     # orange
        granular_rgb = (231 / 255.0, 76 / 255.0, 60 / 255.0)   # red

        for i in range(resolution + 1):
            s_value = i / float(resolution)
            for j in range(resolution + 1):
                c_value = j / float(resolution)
                k_value = 1.0 - s_value - c_value
                if k_value < 0.0:
                    continue

                # Regime tendencies from S,C only (no hard argmax regions).
                t_gas = (1.0 - s_value) ** 2
                t_fluid = 1.5 * s_value * (1.0 - s_value) * c_value
                t_granular = s_value * (1.0 - c_value)
                t_sum = t_gas + t_fluid + t_granular
                if t_sum <= 1e-12:
                    continue

                r_gas = t_gas / t_sum
                r_fluid = t_fluid / t_sum
                r_granular = t_granular / t_sum

                red = (
                    r_gas * gas_rgb[0]
                    + r_fluid * fluid_rgb[0]
                    + r_granular * granular_rgb[0]
                )
                green = (
                    r_gas * gas_rgb[1]
                    + r_fluid * fluid_rgb[1]
                    + r_granular * granular_rgb[1]
                )
                blue = (
                    r_gas * gas_rgb[2]
                    + r_fluid * fluid_rgb[2]
                    + r_granular * granular_rgb[2]
                )
                # Keep it subtle so grid/triangle/point remain dominant.
                color = (red, green, blue, 0.16)

                x, y = self._ternary_to_xy(s_value, k_value, c_value)
                points.append((x, y, color))
        return points

    def _draw_regime_intensity_field(self):
        if not self._regime_overlay_points:
            return
        xs = [p[0] for p in self._regime_overlay_points]
        ys = [p[1] for p in self._regime_overlay_points]
        cs = [p[2] for p in self._regime_overlay_points]
        self.ax.scatter(xs, ys, s=8, c=cs, marker="s", linewidths=0, zorder=1)

    def set_regime_overlay_enabled(self, enabled: bool):
        self._regime_overlay_enabled = bool(enabled)
        self._draw_base()
        self.canvas.draw_idle()

    def _draw_ternary_grid(self):
        levels = (0.2, 0.4, 0.6, 0.8)
        style = {"color": "#666666", "linewidth": 0.5, "alpha": 0.25, "zorder": 2}

        for v in levels:
            # Constant S = v
            x1, y1 = self._ternary_to_xy(v, 1.0 - v, 0.0)
            x2, y2 = self._ternary_to_xy(v, 0.0, 1.0 - v)
            self.ax.plot([x1, x2], [y1, y2], **style)

            # Constant K = v
            x1, y1 = self._ternary_to_xy(0.0, v, 1.0 - v)
            x2, y2 = self._ternary_to_xy(1.0 - v, v, 0.0)
            self.ax.plot([x1, x2], [y1, y2], **style)

            # Constant C = v
            x1, y1 = self._ternary_to_xy(0.0, 1.0 - v, v)
            x2, y2 = self._ternary_to_xy(1.0 - v, 0.0, v)
            self.ax.plot([x1, x2], [y1, y2], **style)

    @staticmethod
    def _ternary_to_xy(s_value: float, k_value: float, c_value: float):
        _ = k_value
        x = 0.5 * (2.0 * c_value + s_value)
        y = (math.sqrt(3) / 2.0) * s_value
        return x, y

    def reset(self):
        self._trail_points.clear()
        self._s_smooth = None
        self._k_smooth = None
        self._c_smooth = None
        self._point_artist = None
        self._point_ring_artist = None
        self._ghost_artist = None
        self._draw_base()
        self.update_skc(0.33, 0.33, 0.34)

    def update_skc(self, s_value: float, k_value: float, c_value: float):
        s = max(0.0, float(s_value))
        k = max(0.0, float(k_value))
        c = max(0.0, float(c_value))

        total = s + k + c
        if total > 0:
            s /= total
            k /= total
            c /= total
        else:
            s, k, c = 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0

        # Smooth S/K/C before plotting to reduce high-frequency point jitter.
        if self._s_smooth is None:
            self._s_smooth, self._k_smooth, self._c_smooth = s, k, c
        else:
            a = self._ema_alpha
            self._s_smooth = a * s + (1.0 - a) * self._s_smooth
            self._k_smooth = a * k + (1.0 - a) * self._k_smooth
            self._c_smooth = a * c + (1.0 - a) * self._c_smooth

            norm = self._s_smooth + self._k_smooth + self._c_smooth
            if norm > 1e-9:
                self._s_smooth /= norm
                self._k_smooth /= norm
                self._c_smooth /= norm

        x = 0.5 * (2.0 * self._c_smooth + self._s_smooth)
        y = (math.sqrt(3) / 2.0) * self._s_smooth

        self._trail_points.append((x, y))

        # Fading trajectory ghost (oldest to newest).
        if len(self._trail_points) >= 2:
            xs = [p[0] for p in self._trail_points]
            ys = [p[1] for p in self._trail_points]
            alphas = [
                0.08 + 0.42 * (i / max(1, len(self._trail_points) - 1))
                for i in range(len(self._trail_points))
            ]
            colors = [(0.0, 1.0, 1.0, a) for a in alphas]

            if self._ghost_artist is None:
                self._ghost_artist = self.ax.scatter(
                    xs, ys, s=24, c=colors, edgecolors="none", zorder=4
                )
            else:
                self._ghost_artist.set_offsets(list(zip(xs, ys)))
                self._ghost_artist.set_facecolors(colors)

        if self._point_ring_artist is None:
            self._point_ring_artist = self.ax.scatter(
                [x], [y], s=96, c="#101010", edgecolors="none", zorder=5
            )
        else:
            self._point_ring_artist.set_offsets([[x, y]])

        if self._point_artist is None:
            self._point_artist = self.ax.scatter(
                [x], [y], s=64, c="#00ffff", edgecolors="none", zorder=6
            )
        else:
            self._point_artist.set_offsets([[x, y]])

        self.canvas.draw_idle()





