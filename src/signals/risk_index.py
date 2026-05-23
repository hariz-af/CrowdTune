# risk_index.py
import numpy as np

from tune_config import RISK

class RiskIndex:
    """
    Crowd Risk Index
    Purely structural — based only on Crowd Constraint Index (CCI)
    """
    # ------------------ Constants (LOCAL, LOCKED) ------------------
    CCI_EXPONENT  = RISK.cci_exponent      # structural dominance
    EPS = RISK.eps

    def __init__(self):
        self.last = {
            "risk": 0.0,
            "cci_norm": 0.0,
        }

    def update(self, cci_raw):
        cci_norm = cci_raw

        risk = (cci_norm ** self.CCI_EXPONENT)
        risk = float(np.clip(risk, 0.0, 1.0))

        self.last = {
            "risk": risk,
            "cci_norm": cci_norm
        }
        return self.last
