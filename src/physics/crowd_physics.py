import cv2
import numpy as np
from collections import deque

from tune_config import FUSION, MOTION, REGIME


class CrowdPhysicsAnalyzer:
    """
    Crowd Motion Analyzer (Kinematic Layer)

    Computes:
        K : normalized kinematic intensity (mean displacement)
        C : directional coherence (alignment measure)

    Does NOT compute:
        S (spatial constraint)
        CCI (fusion index)
    """

    def __init__(
        self,
        max_corners=MOTION.max_corners,
        quality=MOTION.quality,
        min_distance=MOTION.min_distance,
        k_ref_window=MOTION.k_ref_window,
        c_ema_alpha=MOTION.c_ema_alpha,
    ):
        self.max_corners = max_corners
        self.quality = quality
        self.min_distance = min_distance
        self.k_ref_window = k_ref_window
        self.c_ema_alpha = c_ema_alpha

        self.prev_gray = None
        self.prev_pts = None
        self.k_median_history = deque(maxlen=k_ref_window)
        self.c_smoothed = None

        self.lk_params = dict(
            winSize=MOTION.lk_win_size,
            maxLevel=MOTION.lk_max_level,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                MOTION.lk_criteria_count,
                MOTION.lk_criteria_eps,
            ),
        )

    def update(self, frame, analysis_mask=None):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = self._normalize_mask(analysis_mask, gray.shape)

        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_pts = self._detect_points(gray, mask)
            return self._empty_metrics()

        if self.prev_pts is None or len(self.prev_pts) == 0:
            self.prev_pts = self._detect_points(gray, mask)
            self.prev_gray = gray
            return self._empty_metrics()

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            self.prev_pts,
            None,
            **self.lk_params,
        )

        if next_pts is None or status is None:
            self.prev_pts = self._detect_points(gray, mask)
            self.prev_gray = gray
            return self._empty_metrics()

        good_prev = self.prev_pts[status.flatten() == 1]
        good_next = next_pts[status.flatten() == 1]

        if mask is not None and len(good_next) > 0:
            valid_prev = self._points_in_mask(good_prev, mask)
            valid_next = self._points_in_mask(good_next, mask)
            valid_mask = valid_prev & valid_next
            good_prev = good_prev[valid_mask]
            good_next = good_next[valid_mask]

        if len(good_next) == 0:
            self.prev_pts = self._detect_points(gray, mask)
            self.prev_gray = gray
            return self._empty_metrics()

        displacements = (good_next - good_prev).reshape(-1, 2)
        magnitudes = np.linalg.norm(displacements, axis=1)

        k_median = float(np.median(magnitudes))
        self.k_median_history.append(k_median)

        if len(self.k_median_history) < 5:
            K = np.clip(k_median / 3.0, 0.0, 1.0)
        else:
            k_ref = float(np.median(self.k_median_history))
            K = k_median / (k_median + k_ref + 1e-6)
            K = np.clip(K, 0.0, 1.0)

        if len(displacements) > 1:
            angles = np.arctan2(displacements[:, 1], displacements[:, 0])
            angle_std = np.std(angles)
            C_raw = 1.0 - (angle_std / np.pi)
            C_raw = np.clip(C_raw, 0.0, 1.0)
        else:
            C_raw = 0.0

        alpha = float(np.clip(self.c_ema_alpha, 0.0, 1.0))
        if self.c_smoothed is None:
            self.c_smoothed = C_raw
        else:
            self.c_smoothed = alpha * C_raw + (1.0 - alpha) * self.c_smoothed
        C = float(np.clip(self.c_smoothed, 0.0, 1.0))

        self.prev_gray = gray
        self.prev_pts = good_next.reshape(-1, 1, 2)

        return {
            "K": K,
            "C": C,
            "track_prev": good_prev.reshape(-1, 2),
            "track_next": good_next.reshape(-1, 2),
        }

    def _detect_points(self, gray, analysis_mask=None):
        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_corners,
            qualityLevel=self.quality,
            minDistance=self.min_distance,
            mask=analysis_mask,
        )
        if pts is None:
            return None
        return np.float32(pts)

    def _normalize_mask(self, analysis_mask, frame_shape):
        if analysis_mask is None:
            return None
        if analysis_mask.shape[:2] != frame_shape:
            return None
        if analysis_mask.dtype != np.uint8:
            return analysis_mask.astype(np.uint8)
        return analysis_mask

    def _points_in_mask(self, points, mask):
        coords = np.round(points.reshape(-1, 2)).astype(int)
        xs = np.clip(coords[:, 0], 0, mask.shape[1] - 1)
        ys = np.clip(coords[:, 1], 0, mask.shape[0] - 1)
        return mask[ys, xs] > 0

    def _empty_metrics(self):
        return {
            "K": 0.0,
            "C": 0.0,
            "track_prev": None,
            "track_next": None,
        }

    def draw(self, frame, metrics):
        text = f"K: {metrics['K']:.3f}  C: {metrics['C']:.3f}"
        cv2.putText(
            frame,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 50, 255),
            2,
        )


def compute_regime_tendency(S, C):
    """
    Continuous regime tendency mapping.

    Inputs:
        S : spatial constraint [0,1]
        C : coherence [0,1]

    Returns:
        (R_gas, R_fluid, R_granular)
    """

    S = max(0.0, min(1.0, float(S)))
    C = max(0.0, min(1.0, float(C)))

    T_gas = REGIME.gas_weight * ((1 - S) ** 2)
    T_fluid = REGIME.fluid_weight * S * (1 - S) * C
    T_granular = REGIME.granular_weight * S * (1 - C)

    T_sum = T_gas + T_fluid + T_granular
    if T_sum < 1e-12:
        return 0.0, 0.0, 0.0

    return (
        T_gas / T_sum,
        T_fluid / T_sum,
        T_granular / T_sum,
    )


def compute_K(prev_frame, frame, motion_engine, metrics=None):
    if metrics is None:
        metrics = motion_engine.update(frame)
    return float(metrics.get("K", 0.0))


def compute_C(prev_frame, frame, motion_engine, metrics=None):
    if metrics is None:
        metrics = motion_engine.update(frame)
    return float(metrics.get("C", 0.0))


def compute_CCI(S, K, C):
    cci = float(S) * (
        FUSION.k_weight * float(K)
        + FUSION.incoherence_weight * (1.0 - float(C))
    )
    return float(np.clip(cci, 0.0, 1.0))
