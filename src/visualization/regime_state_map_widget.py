import numpy as np
from collections import deque

from PyQt6.QtWidgets import QVBoxLayout, QWidget

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class RegimeStateMapWidget(QWidget):
    """
    2D regime map in model space:
        x-axis -> S (spatial constraint)
        y-axis -> C (coherence)

    Background is precomputed once using dominant tendency:
        Tg  = (1 - S)^2
        Tf  = 1.5 * S * (1 - S) * C
        Tgr = S * (1 - C)
    """

    def __init__(self, parent=None, resolution: int = 140):
        super().__init__(parent)
        self.resolution = int(max(40, resolution))
        self._trail_points = deque(maxlen=24)
        self._ema_alpha = 0.22
        self._s_smooth = None
        self._c_smooth = None

        self.figure = Figure(facecolor="#1E1E1E", edgecolor="#1E1E1E")
        self.figure.patch.set_linewidth(0)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("background-color: #1E1E1E; border: none;")
        self.ax = self.figure.add_subplot(111)
        self.figure.subplots_adjust(left=0.20, right=0.84, bottom=0.27, top=0.94)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

        self._point_artist = None
        self._ghost_artist = None
        self._build_static_map()
        self.update_state(0.0, 0.0)

    def _build_static_map(self):
        self.ax.clear()
        self.ax.set_facecolor("#1E1E1E")

        s_axis = np.linspace(0.0, 1.0, self.resolution)
        c_axis = np.linspace(0.0, 1.0, self.resolution)
        s_grid, c_grid = np.meshgrid(s_axis, c_axis)

        t_gas = (1.0 - s_grid) ** 2
        t_fluid = 1.5 * s_grid * (1.0 - s_grid) * c_grid
        t_granular = s_grid * (1.0 - c_grid)
        dominant = np.argmax(np.stack([t_gas, t_fluid, t_granular], axis=0), axis=0)

        # Gas, Fluid, Granular colors
        palette = np.array(
            [
                [46, 204, 113],   # green
                [243, 156, 18],   # orange
                [231, 76, 60],    # red
            ],
            dtype=np.float32,
        ) / 255.0
        rgb = palette[dominant]
        alpha = np.full((self.resolution, self.resolution, 1), 0.198, dtype=np.float32)
        rgba = np.concatenate([rgb, alpha], axis=2)

        self.ax.imshow(
            rgba,
            origin="lower",
            extent=(0.0, 1.0, 0.0, 1.0),
            interpolation="nearest",
            zorder=1,
            aspect="auto",
        )

        self.ax.set_xlim(0.0, 1.0)
        self.ax.set_ylim(0.0, 1.0)
        self.ax.set_xlabel("Spatial", color="#d0d0d0", fontsize=9, labelpad=4)
        self.ax.set_ylabel("Coherence", color="#d0d0d0", fontsize=9, labelpad=5)
        self.ax.tick_params(colors="#9aa0a6", labelsize=8)
        self.ax.grid(color="#666666", alpha=0.20, linewidth=0.5)

        for spine in self.ax.spines.values():
            spine.set_color("#a0a0a0")
            spine.set_linewidth(0.8)

        self._draw_regime_labels(dominant, s_grid, c_grid)

    def _draw_regime_labels(self, dominant, s_grid, c_grid):
        label_defs = (
            (0, "Gas", "#9be5bd"),
            (1, "Fluid", "#ffd28a"),
            (2, "Granular", "#ffada4"),
        )
        for idx, label, color in label_defs:
            mask = dominant == idx
            if not np.any(mask):
                continue
            s_center = float(np.mean(s_grid[mask]))
            c_center = float(np.mean(c_grid[mask]))
            self.ax.text(
                s_center,
                c_center,
                label,
                color=color,
                fontsize=8,
                fontweight="bold",
                ha="center",
                va="center",
                alpha=0.85,
                zorder=2.2,
            )

    def reset(self):
        self._trail_points.clear()
        self._s_smooth = None
        self._c_smooth = None
        self._point_artist = None
        self._ghost_artist = None
        self._build_static_map()
        self.update_state(0.0, 0.0)

    def update_state(self, s_value: float, c_value: float):
        s = float(np.clip(s_value, 0.0, 1.0))
        c = float(np.clip(c_value, 0.0, 1.0))

        # Same EMA smoothing behavior as ternary point.
        if self._s_smooth is None:
            self._s_smooth = s
            self._c_smooth = c
        else:
            a = self._ema_alpha
            self._s_smooth = a * s + (1.0 - a) * self._s_smooth
            self._c_smooth = a * c + (1.0 - a) * self._c_smooth

        self._trail_points.append((self._s_smooth, self._c_smooth))

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
                    xs, ys, s=24, c=colors, edgecolors="none", zorder=2.5
                )
            else:
                self._ghost_artist.set_offsets(list(zip(xs, ys)))
                self._ghost_artist.set_facecolors(colors)

        if self._point_artist is None:
            self._point_artist = self.ax.scatter(
                [self._s_smooth],
                [self._c_smooth],
                s=80,
                c="#00ffff",
                edgecolors="#101010",
                linewidths=1.2,
                zorder=3,
            )
        else:
            self._point_artist.set_offsets([[self._s_smooth, self._c_smooth]])

        self.canvas.draw_idle()





