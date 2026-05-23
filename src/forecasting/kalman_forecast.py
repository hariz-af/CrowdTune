# kalman_forecast.py

"""
Kalman-based CCI short-horizon forecaster.

This module is intentionally standalone for now. It does not change the active
CrowdTune forecast pipeline until the pipeline explicitly selects it.
"""

import numpy as np
from collections import deque


class CCIKalmanForecaster:
    """
    Standard constant-velocity Kalman Filter for Crowd Constraint Index (CCI).

    State:
        x = [CCI, CCI_velocity]^T

    Measurement:
        z = measured CCI

    Forecast:
        projected_cci = filtered_cci + velocity * horizon
    """

    def __init__(
        self,
        horizon: float = 10.0,
        fps: float = 10.0,
        process_var: float = 0.01,
        measurement_var: float = 0.02,
        initial_uncertainty: float = 0.12,
        initial_cci: float = 0.0,
    ):
        self.horizon = float(horizon)
        self.fps = max(1e-6, float(fps))
        self.dt = 1.0 / self.fps

        self.process_var = float(process_var)
        self.measurement_var = float(measurement_var)
        self.initial_uncertainty = float(initial_uncertainty)

        # Constant-velocity state transition
        self.F = np.array(
            [
                [1.0, self.dt],
                [0.0, 1.0],
            ],
            dtype=float,
        )

        # Only CCI is directly observed
        self.H = np.array([[1.0, 0.0]], dtype=float)
        self.I = np.eye(2, dtype=float)

        self.R = np.array([[self.measurement_var]], dtype=float)
        self.Q = self._build_process_noise()

        self.reset(initial_cci)

    def reset(self, initial_cci: float = 0.0):
        self.x = np.array(
            [
                [self._clamp01(initial_cci)],
                [0.0],
            ],
            dtype=float,
        )

        self.P = np.array(
            [
                [self.initial_uncertainty, 0.0],
                [0.0, self.initial_uncertainty],
            ],
            dtype=float,
        )

        self.initialized = False

    def update(self, cci_value: float):
        measured_cci = self._clamp01(cci_value)

        if not self.initialized:
            self.x[0, 0] = measured_cci
            self.x[1, 0] = 0.0
            self.initialized = True
            return measured_cci, 0.0

        self._predict()
        self._correct(measured_cci)

        projected_cci = self.forecast(self.horizon)
        slope = float(self.x[1, 0])

        return projected_cci, slope

    def forecast(self, horizon_seconds: float | None = None):
        horizon = self.horizon if horizon_seconds is None else float(horizon_seconds)
        projected = float(self.x[0, 0] + self.x[1, 0] * horizon)
        return self._clamp01(projected)

    def filtered_state(self):
        return {
            "filtered_cci": self._clamp01(float(self.x[0, 0])),
            "velocity": float(self.x[1, 0]),
        }

    def _predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def _correct(self, measured_cci: float):
        z = np.array([[measured_cci]], dtype=float)

        residual = z - (self.H @ self.x)
        residual_cov = self.H @ self.P @ self.H.T + self.R

        K = self.P @ self.H.T @ np.linalg.inv(residual_cov)

        self.x = self.x + K @ residual
        self.P = (self.I - K @ self.H) @ self.P

    def _build_process_noise(self):
        dt = self.dt

        return self.process_var * np.array(
            [
                [dt**4 / 4.0, dt**3 / 2.0],
                [dt**3 / 2.0, dt**2],
            ],
            dtype=float,
        )

    @staticmethod
    def _clamp01(value: float):
        return max(0.0, min(1.0, float(value)))


class AdaptiveCCIKalmanForecaster(CCIKalmanForecaster):
    """
    Improved Adaptive Kalman Filter with:

    1. Velocity damping
    2. Better R adaptation using recent CCI variance
    3. Regime-aware growth using S(1 - C)
    4. Multi-horizon forecasting (3s, 5s, 10s)

    Designed for robust 10-second CrowdTune forecasting.
    """

    def __init__(
        self,
        horizon: float = 10.0,
        fps: float = 10.0,
        process_var: float = 0.01,
        measurement_var: float = 0.02,
        initial_uncertainty: float = 0.12,
        initial_cci: float = 0.0,
        adapt_alpha: float = 0.35,
        velocity_damping: float = 0.92,
        q_min_scale: float = 0.3,
        q_max_scale: float = 5.0,
        r_min_scale: float = 0.8,
        r_max_scale: float = 3.0,
        smoothing: float = 0.75,
        variance_window: int = 20,
    ):
        super().__init__(
            horizon=horizon,
            fps=fps,
            process_var=process_var,
            measurement_var=measurement_var,
            initial_uncertainty=initial_uncertainty,
            initial_cci=initial_cci,
        )

        self.base_process_var = float(process_var)
        self.base_measurement_var = float(measurement_var)

        self.adapt_alpha = float(adapt_alpha)
        self.velocity_damping = float(velocity_damping)

        self.q_min_scale = float(q_min_scale)
        self.q_max_scale = float(q_max_scale)
        self.r_min_scale = float(r_min_scale)
        self.r_max_scale = float(r_max_scale)
        self.smoothing = float(smoothing)

        self.current_process_var = float(process_var)
        self.current_measurement_var = float(measurement_var)

        # For better measurement-noise estimation
        self.recent_cci = deque(maxlen=int(max(5, variance_window)))

        self.last_regime_metric = 0.0
        self.last_multi = {}

    @staticmethod
    def _regime_metric(S, C):
        """
        Regime metric:
            g = S * (1 - C)

        High when:
        - spatial constraint is high
        - coherence collapses

        Indicates granular instability tendency.
        """
        try:
            s = float(S) if S is not None else 0.0
            c = float(C) if C is not None else 1.0
            return max(0.0, min(1.0, s * (1.0 - c)))
        except Exception:
            return 0.0

    def _recent_measurement_variance(self):
        """
        Better R adaptation:
        Use short-term variance of observed CCI.

        High variance => noisy measurements => increase R
        """
        if len(self.recent_cci) < 5:
            return 0.0

        values = np.array(self.recent_cci, dtype=float)
        return float(np.var(values))

    def _adapt_variances(self, S=None, C=None, measured_cci=None):
        g = self._regime_metric(S, C)
        self.last_regime_metric = g

        if measured_cci is not None:
            self.recent_cci.append(float(measured_cci))

        cci_var = self._recent_measurement_variance()

        # --------------------------
        # Adaptive Q (process noise)
        # --------------------------
        # Higher granular tendency => higher Q
        target_q_scale = (
            self.q_min_scale
            + (self.q_max_scale - self.q_min_scale) * g
        )

        # --------------------------
        # Adaptive R (measurement noise)
        # --------------------------
        # Higher recent CCI variance => noisier observation => higher R
        normalized_var = min(1.0, cci_var * 20.0)

        target_r_scale = (
            self.r_min_scale
            + (self.r_max_scale - self.r_min_scale) * normalized_var
        )

        target_process = max(
            1e-9,
            self.base_process_var * target_q_scale,
        )

        target_measure = max(
            1e-9,
            self.base_measurement_var * target_r_scale,
        )

        # Smooth transitions (avoid sudden jumps)
        s = self.smoothing

        self.current_process_var = (
            s * self.current_process_var
            + (1.0 - s) * target_process
        )

        self.current_measurement_var = (
            s * self.current_measurement_var
            + (1.0 - s) * target_measure
        )

        self.process_var = float(self.current_process_var)
        self.measurement_var = float(self.current_measurement_var)

        self.R = np.array([[self.measurement_var]], dtype=float)
        self.Q = self._build_process_noise()

    def update(self, cci_value: float, S=None, C=None):
        measured_cci = self._clamp01(cci_value)

        # Update adaptive Q and R
        self._adapt_variances(
            S=S,
            C=C,
            measured_cci=measured_cci,
        )

        # Regime-aware growth
        g = self._regime_metric(S, C)
        growth = self.adapt_alpha * g

        if not self.initialized:
            self.x[0, 0] = measured_cci
            self.x[1, 0] = 0.0
            self.initialized = True
            self.last_multi = self.multi_forecast([3, 5, 10])
            return measured_cci, 0.0

        # Standard Kalman predict
        self._predict()

        # -------------------------------------------------
        # FIX 1: Velocity damping + controlled growth blend
        # -------------------------------------------------
        # Prevent infinite acceleration and unstable 10s forecast
        self.x[1, 0] = (
            self.velocity_damping * self.x[1, 0]
            + growth
        )

        # Standard Kalman correction
        self._correct(measured_cci)

        projected_cci = self.forecast(self.horizon)
        slope = float(self.x[1, 0])

        self.last_multi = self.multi_forecast([3, 5, 10])

        return projected_cci, slope

    def multi_forecast(self, horizons):
        output = {}

        for h in horizons:
            try:
                output[float(h)] = self.forecast(float(h))
            except Exception:
                output[float(h)] = None

        return output

