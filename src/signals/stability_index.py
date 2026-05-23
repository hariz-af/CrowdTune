# stability_index.py
import numpy as np
from collections import deque

from tune_config import STABILITY

class StabilityIndex:
    def __init__(self, window=STABILITY.window):
        self.window = window
        self.history = deque(maxlen=window)

    def update(self, value: float) -> float:
        self.history.append(value)

        if len(self.history) < 2:
            return 1.0

        arr = np.array(self.history)

        volatility = np.std(arr)

        k = STABILITY.sensitivity_k  # sensitivity constant

        instability_norm = volatility / (volatility + k)

        SI = 1.0 - instability_norm

        return float(np.clip(SI, 0.0, 1.0))


    def mean(self):
        return np.mean(self.history) if self.history else 1.0


def compute_SI(rolling_buffer, k=STABILITY.sensitivity_k):
    """
    Compute SI from a rolling buffer of CCI values.
    Mirrors StabilityIndex.update() math for validation pipelines.
    """
    if len(rolling_buffer) < 2:
        return 1.0

    arr = np.asarray(rolling_buffer, dtype=np.float32)
    volatility = float(np.std(arr))
    instability_norm = volatility / (volatility + float(k))
    si = 1.0 - instability_norm
    return float(np.clip(si, 0.0, 1.0))

