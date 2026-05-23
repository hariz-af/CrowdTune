import cv2
import numpy as np

from tune_config import DENSITY


class LBPDensity:
    """
    Sobel-based spatial density estimator.

    Current active output:
    - density_map: normalized Sobel magnitude map for heatmap overlay
    - S_sobel: scalar spatial score (mean normalized Sobel magnitude)
    """

    def __init__(
        self,
        radius=DENSITY.radius,
        n_points=DENSITY.n_points,
        grid_size=DENSITY.grid_size,
        eps=DENSITY.eps,
    ):
        self.eps = eps
        self.global_scale = None

    def compute_sobel_map_and_score(self, frame, analysis_mask=None):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, DENSITY.blur_kernel, 0)

        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=DENSITY.sobel_kernel_size)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=DENSITY.sobel_kernel_size)

        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        p95 = float(np.percentile(magnitude, 95))

        if self.global_scale is None:
            self.global_scale = p95
        else:
            self.global_scale = max(self.global_scale, p95)

        magnitude = magnitude / (self.global_scale + self.eps)
        magnitude = np.clip(magnitude, 0.0, 1.0)

        if analysis_mask is not None:
            mask_bool = analysis_mask > 0
            masked_magnitude = np.zeros_like(magnitude)
            masked_magnitude[mask_bool] = magnitude[mask_bool]
            magnitude = masked_magnitude
            if np.any(mask_bool):
                s_sobel = float(np.mean(magnitude[mask_bool]))
            else:
                s_sobel = 0.0
        else:
            s_sobel = float(np.mean(magnitude))

        return magnitude, s_sobel

    def compute(self, frame, analysis_mask=None):
        density_map, s_sobel = self.compute_sobel_map_and_score(frame, analysis_mask=analysis_mask)
        return density_map, s_sobel

    def density_heatmap(self, density_map, frame_shape):
        heat = cv2.resize(
            density_map,
            (frame_shape[1], frame_shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
        heat = np.uint8(255 * np.clip(heat, 0.0, 1.0))
        return cv2.applyColorMap(heat, cv2.COLORMAP_JET)
