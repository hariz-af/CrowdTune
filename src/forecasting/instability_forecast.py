# instability_forecast.py

import numpy as np
from collections import deque

from kalman_forecast import CCIKalmanForecaster, AdaptiveCCIKalmanForecaster

class InstabilityForecast:
    def __init__(self, window_size=30, horizon=3.0, fps=10):
        """
        window_size: number of past frames to compute slope
        horizon: projection horizon in seconds
        fps: frames per second of signal update
        """
        self.window_size = window_size
        self.horizon = horizon
        self.fps = fps

        self.cci_buffer = deque(maxlen=window_size)

    def reset(self, initial_cci=0.0):
        self.cci_buffer.clear()

    def update(self, cci_value):
        """
        Add latest CCI value and compute forecast.
        """
        self.cci_buffer.append(cci_value)

        if len(self.cci_buffer) < 2:
            return cci_value, 0.0  # Not enough data

        return self._compute_forecast()

    def _compute_forecast(self):
        cci_array = np.array(self.cci_buffer)

        # Time axis in seconds
        t = np.arange(len(cci_array)) / self.fps

        # Linear regression slope (first-order trend)
        slope, _ = np.polyfit(t, cci_array, 1)

        # Projection
        current_cci = cci_array[-1]
        projected_cci = current_cci + slope * self.horizon

        return projected_cci, slope

def create_forecaster(config):
    """
    Build the configured forecast engine.

    config.model:
        "linear" -> existing regression forecaster
        "kalman" -> constant-velocity Kalman CCI forecaster
    """
    model = str(getattr(config, "model", "linear")).strip().lower()

    if model == "kalman":
        # Use the adaptive Kalman forecaster by default for improved
        # longer-horizon robustness while preserving compatibility.
        return AdaptiveCCIKalmanForecaster(
            horizon=getattr(config, "horizon_seconds", 3.0),
            fps=getattr(config, "signal_fps", 10),
            process_var=getattr(config, "kalman_process_var", 0.01),
            measurement_var=getattr(config, "kalman_measurement_var", 0.02),
            initial_uncertainty=getattr(config, "kalman_initial_uncertainty", 0.12),
            adapt_alpha=getattr(config, "kalman_adapt_alpha", 0.5),
            q_min_scale=getattr(config, "kalman_q_min_scale", 0.2),
            q_max_scale=getattr(config, "kalman_q_max_scale", 5.0),
            r_min_scale=getattr(config, "kalman_r_min_scale", 0.5),
            r_max_scale=getattr(config, "kalman_r_max_scale", 2.0),
            smoothing=getattr(config, "kalman_adapt_smoothing", 0.75),
        )

    return InstabilityForecast(
        window_size=getattr(config, "window_size", 30),
        horizon=getattr(config, "horizon_seconds", 3.0),
        fps=getattr(config, "signal_fps", 10),
    )

