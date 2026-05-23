from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from tune_config import ADVISORY


class AdvisoryDecisionEngine:
    """
    Interpretable rule-based advisory engine with light hysteresis.
    """

    def __init__(self):
        self.current_situation = "Stable Conditions"
        self._pending_situation = None
        self._pending_count = 0
        self._switch_count_required = ADVISORY.switch_count_required

    def decide(
        self,
        cci: float,
        risk: float,
        projected_cci: float,
        stability_index: float,
        stability_trend: float,
        regime_phase: str,
        forecast_direction: str,
    ) -> dict:
        cci = self._clamp01(cci)
        risk = self._clamp01(risk)
        projected_cci = self._clamp01(projected_cci)
        stability_index = self._clamp01(stability_index)
        regime_phase = (regime_phase or "Unknown").strip()
        forecast_direction = (forecast_direction or "Stable").strip()

        candidate = self._rule_candidate(
            cci,
            risk,
            projected_cci,
            stability_index,
            stability_trend,
            regime_phase,
            forecast_direction,
        )

        # Hysteresis: avoid rapid advisory flips unless the new state is Critical.
        if candidate["situation"] != self.current_situation:
            if candidate["status"] == "Critical":
                self.current_situation = candidate["situation"]
                self._pending_situation = None
                self._pending_count = 0
            else:
                if self._pending_situation == candidate["situation"]:
                    self._pending_count += 1
                else:
                    self._pending_situation = candidate["situation"]
                    self._pending_count = 1

                if self._pending_count >= self._switch_count_required:
                    self.current_situation = candidate["situation"]
                    self._pending_situation = None
                    self._pending_count = 0
        else:
            self._pending_situation = None
            self._pending_count = 0

        # Build final output for the active state (possibly held by hysteresis).
        out = self._state_output(
            self.current_situation,
            cci,
            risk,
            projected_cci,
            stability_index,
            stability_trend,
            regime_phase,
            forecast_direction,
        )
        return out

    def _rule_candidate(
        self,
        cci: float,
        risk: float,
        projected_cci: float,
        stability_index: float,
        stability_trend: float,
        regime_phase: str,
        forecast_direction: str,
    ) -> dict:
        # 1) Critical Crowd Pressure
        if (
            (risk >= ADVISORY.critical_risk_granular_threshold and regime_phase == "Granular")
            or projected_cci >= ADVISORY.critical_projected_cci_threshold
            or (cci >= ADVISORY.critical_cci_threshold and stability_index <= ADVISORY.critical_stability_threshold)
        ):
            return {"status": "Critical", "situation": "Critical Crowd Pressure"}

        # 2) Dissipating Situation
        if (
            forecast_direction == "Dissipating"
            and projected_cci <= cci - ADVISORY.dissipating_projection_margin
            and stability_trend >= ADVISORY.dissipating_stability_trend_threshold
        ):
            return {"status": "Watch", "situation": "Dissipating Situation"}

        # 3) Early Instability
        if (
            cci >= ADVISORY.early_instability_cci_threshold
            and stability_index <= ADVISORY.early_instability_stability_threshold
            and stability_trend <= ADVISORY.early_instability_trend_threshold
        ):
            return {"status": "Escalating", "situation": "Early Instability"}

        # 4) Increasing Density
        if (
            (forecast_direction == "Escalating" and projected_cci >= cci + ADVISORY.increasing_density_projection_margin)
            or (cci >= ADVISORY.increasing_density_cci_threshold and risk >= ADVISORY.increasing_density_risk_threshold and stability_trend < 0.0)
        ):
            return {"status": "Watch", "situation": "Increasing Density"}

        # 5) Stable Conditions
        return {"status": "Stable", "situation": "Safe Conditions"}

    def _state_output(
        self,
        situation: str,
        cci: float,
        risk: float,
        projected_cci: float,
        stability_index: float,
        stability_trend: float,
        regime_phase: str,
        forecast_direction: str,
    ) -> dict:
        key_signals = {
            "Risk Index": round(risk, 3),
            "CCI": round(cci, 3),
            "Projected CCI": round(projected_cci, 3),
            "Stability Index": round(stability_index, 3),
            "Regime Phase": regime_phase,
        }

        if situation == "Critical Crowd Pressure":
            status = "Critical"
            reason = (
                "Elevated risk, high projected CCI and granular regime indicators."
            )
            action = "Reduce inflow and deploy crowd control at bottlenecks immediately."
            situation_text = "Crowd pressure is critical with near-term compression risk."
            confidence = self._confidence_from_agreement(
                conditions=[
                    risk >= 0.75,
                    projected_cci >= ADVISORY.critical_projected_cci_threshold,
                    regime_phase == "Granular",
                    stability_index <= 0.60,
                ],
                base=0.62,
            )
        elif situation == "Early Instability":
            status = "Escalating"
            reason = (
                "Moderate-to-high CCI with declining stability indicates early dynamic disorder."
            )
            action = "Increase observation density and prepare local flow correction measures."
            situation_text = "Signs of instability are emerging in the crowd movement pattern."
            confidence = self._confidence_from_agreement(
                conditions=[
                    cci >= ADVISORY.early_instability_cci_threshold,
                    stability_index <= ADVISORY.early_instability_stability_threshold,
                    stability_trend <= ADVISORY.early_instability_trend_threshold,
                    forecast_direction == "Escalating",
                ],
                base=0.52,
            )
        elif situation == "Increasing Density":
            status = "Watch"
            reason = (
                "Current CCI is moderate and forecast indicates continued growth."
            )
            action = "Increase observation and prepare flow diversion if trend persists."
            situation_text = "Density is increasing and may escalate if current trend continues."
            confidence = self._confidence_from_agreement(
                conditions=[
                    projected_cci >= cci + ADVISORY.increasing_density_projection_margin,
                    forecast_direction == "Escalating",
                    risk >= 0.45,
                    cci >= 0.40,
                ],
                base=0.50,
            )
        elif situation == "Dissipating Situation":
            status = "Watch"
            reason = (
                "Projected CCI is below current CCI and stability is improving."
            )
            action = "Maintain controls and monitoring for sustained normalization."
            situation_text = "Crowd pressure is easing and dynamics are becoming more stable."
            confidence = self._confidence_from_agreement(
                conditions=[
                    projected_cci < cci,
                    forecast_direction == "Dissipating",
                    stability_trend >= ADVISORY.dissipating_stability_trend_threshold,
                    risk < 0.55,
                ],
                base=0.48,
            )
        else:
            status = "Stable"
            reason = (
                "CCI, risk and forecast signals remain within controlled bounds with no strong instability cue."
            )
            action = "Continue monitoring."
            situation_text = "Crowd currently remains stable."
            confidence = self._confidence_from_agreement(
                conditions=[
                    cci < 0.45,
                    risk < 0.50,
                    forecast_direction in ("Stable", "Dissipating"),
                    stability_index >= 0.60,
                ],
                base=0.50,
            )

        return {
            "status": status,
            "situation": situation_text,
            "key_signals": key_signals,
            "reason": reason,
            "action": action,
            "confidence": round(confidence, 3),
            "state_name": situation,
        }

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _confidence_from_agreement(conditions, base: float) -> float:
        agree = sum(1 for c in conditions if c)
        score = base + (ADVISORY.confidence_step * agree)
        return max(0.0, min(1.0, score))


class AdvisoryWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(140)
        self.engine = AdvisoryDecisionEngine()
        self.last_output = None

        self.status_label = QLabel("Status: Awaiting Signal")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #d0d0d0; font-size: 14px; font-weight: 700;")

        self.situation_label = QLabel("Situation: Unknown")
        self.situation_label.setWordWrap(True)
        self.situation_label.setStyleSheet("color: #d0d0d0; font-size: 13px;")

        self.reason_label = QLabel("Reason: No crowd signal detected yet.")
        self.reason_label.setWordWrap(True)
        self.reason_label.setStyleSheet("color: #cfd8dc; font-size: 13px;")

        self.action_label = QLabel("Action: Unknown - awaiting advisory output.")
        self.action_label.setWordWrap(True)
        self.action_label.setStyleSheet("color: #cfd8dc; font-size: 13px;")

        self.confidence_label = QLabel("Confidence: 0.00")
        self.confidence_label.setWordWrap(True)
        self.confidence_label.setStyleSheet("color: #b0bec5; font-size: 11px;")

        layout = QVBoxLayout(self)
        layout.addWidget(self.status_label)
        layout.addWidget(self.situation_label)
        layout.addWidget(self.reason_label)
        layout.addWidget(self.action_label)
        layout.addWidget(self.confidence_label)
        layout.addStretch()

    def reset(self):
        self.engine = AdvisoryDecisionEngine()
        self.last_output = None
        self.status_label.setText("Status: Awaiting Signal")
        self.status_label.setStyleSheet("color: #d0d0d0; font-size: 14px; font-weight: 700;")
        self.situation_label.setText("Situation: Unknown")
        self.reason_label.setText("Reason: No crowd signal detected.")
        self.action_label.setText("Action: Unknown - awaiting advisory output.")
        self.confidence_label.setText("Confidence: 0.00")

    def update_advisory(
        self,
        cci: float,
        risk: float,
        projected_cci: float,
        slope: float,
        stability_value: float,
        stability_trend: float,
        regime_phase: str,
        forecast_direction: str,
    ) -> dict:
        out = self.engine.decide(
            cci=cci,
            risk=risk,
            projected_cci=projected_cci,
            stability_index=stability_value,
            stability_trend=stability_trend,
            regime_phase=regime_phase,
            forecast_direction=forecast_direction,
        )
        self.last_output = out

        self.status_label.setText(f"Status: {out['status']} ({out['state_name']})")
        self.status_label.setStyleSheet(
            f"color: {self._status_color(out['status'])}; font-size: 14px; font-weight: 700;"
        )
        self.situation_label.setText(f"Situation: {out['situation']}")
        self.reason_label.setText(f"Reason: {out['reason']}")
        self.action_label.setText(f"Action: {out['action']}")
        self.confidence_label.setText(f"Confidence: {out['confidence']:.2f}")

        return out

    @staticmethod
    def _status_color(status: str) -> str:
        if status == "Critical":
            return "#ff6b6b"
        if status == "Escalating":
            return "#ffb74d"
        if status == "Watch":
            return "#ffd180"
        return "#81c784"



