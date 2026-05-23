from dataclasses import dataclass


# -----------------------------------------------------------------------------
# Crowd Motion (K and C)
# These parameters affect the optical-flow / Shi-Tomasi motion model.
# Changing them affects:
# - K (kinematic activity)
# - C (collective coherence)
# - tracked feature density and responsiveness
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class MotionConfig:
    max_corners: int = 900
    quality: float = 0.001
    min_distance: int = 4
    k_ref_window: int = 90
    c_ema_alpha: float = 0.20

    lk_win_size: tuple[int, int] = (21, 21)
    lk_max_level: int = 3
    lk_criteria_count: int = 30
    lk_criteria_eps: float = 0.01


# -----------------------------------------------------------------------------
# Spatial Constraint (S)
# These parameters affect the Sobel/LBP-derived spatial score and heatmap.
# Changing them affects:
# - S (spatial constraint)
# - density heatmap appearance
# - downstream CCI / Risk / regime behavior because S feeds all of them
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class DensityConfig:
    radius: int = 1
    n_points: int = 8
    grid_size: tuple[int, int] = (20, 20)
    eps: float = 1e-6

    blur_kernel: tuple[int, int] = (3, 3)
    sobel_kernel_size: int = 3
    compute_every_n_frames: int = 4

    s_min: float = 0.12
    s_max: float = 0.60
    s_gamma: float = 0.8


# -----------------------------------------------------------------------------
# Structural Fusion -> CCI
# These weights affect how S, K, and C combine into CCI.
# Changing them affects:
# - CCI directly
# - Risk, SI, forecast, advisory downstream
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class FusionConfig:
    k_weight: float = 0.4
    incoherence_weight: float = 0.6


# -----------------------------------------------------------------------------
# Regime Tendency Model (Gas / Fluid / Granular)
# These weights affect the regime tendency equations in raw S-C model space.
# Changing them affects:
# - regime phase output
# - regime confidence
# - advisory logic when it references the phase
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class RegimeConfig:
    gas_weight: float = 1.0
    fluid_weight: float = 1.5
    granular_weight: float = 1.0

    smooth_alpha: float = 0.15
    initial_gas: float = 0.33
    initial_fluid: float = 0.33
    initial_granular: float = 0.34


# -----------------------------------------------------------------------------
# Risk Index
# These parameters affect the mapping from CCI -> Risk and the dashboard state
# hysteresis used for Low / Elevated / Critical transitions.
# Changing them affects:
# - numeric risk value
# - risk band state transitions in the UI
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class RiskConfig:
    cci_exponent: float = 1.4
    eps: float = 1e-6

    low_threshold: float = 0.35
    critical_threshold: float = 0.70
    hysteresis_margin: float = 0.03


# -----------------------------------------------------------------------------
# Stability Index (SI)
# These parameters affect the volatility-based SI computation.
# Changing them affects:
# - SI responsiveness
# - how quickly instability appears in the stability chart
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class StabilityConfig:
    window: int = 30
    sensitivity_k: float = 0.1


# -----------------------------------------------------------------------------
# Forecast
# These parameters affect the short-horizon instability projection.
# Changing them affects:
# - projected CCI
# - slope sensitivity
# - stable/escalating/dissipating classification
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ForecastConfig:
    # Forecast model selector for later integration.
    # "linear" keeps the current regression-based projection.
    # "kalman" will use the optional CCIKalmanForecaster once wired into the pipeline.
    model: str = "kalman"

    # Linear forecast settings.
    # window_size controls how many recent CCI samples are used for the regression slope.
    window_size: int = 30
    horizon_seconds: float = 10.0
    signal_fps: int = 10

    # Kalman forecast settings.
    # process_var higher = reacts faster to real CCI trend changes, but can be twitchier.
    # measurement_var higher = trusts raw CCI less, producing smoother but slower forecasts.
    # initial_uncertainty higher = adapts faster during startup/source changes.
    kalman_process_var: float = 0.002
    kalman_measurement_var: float = 0.01
    kalman_process_var: float = 0.01
    kalman_measure_var: float = 0.02
    kalman_measurement_var: float = 0.02
    kalman_initial_uncertainty: float = 0.12

    stable_si_threshold: float = 0.85
    stable_slope_threshold: float = 0.01


# -----------------------------------------------------------------------------
# Playback / Runtime
# These parameters affect responsiveness, pacing, and startup behavior.
# Changing them affects:
# - file playback smoothness
# - analytics rate in file mode
# - warm-up period before SI/forecast emissions
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class PlaybackConfig:
    target_fps_fallback: float = 30.0
    file_analytics_stride: int = 2
    live_loop_sleep_seconds: float = 0.001
    paused_sleep_seconds: float = 0.03
    warmup_seconds: float = 2.0


# -----------------------------------------------------------------------------
# Dashboard UI smoothing
# These parameters affect only display smoothing, not the raw analytics.
# Changing them affects:
# - gauge/label flicker
# - perceived latency in the dashboard
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class UISmoothingConfig:
    refresh_ms: int = 100
    ema_alpha: float = 0.32
    ema_fast_alpha: float = 0.62
    ema_fast_delta: float = 0.10


# -----------------------------------------------------------------------------
# Advisory Engine
# These are the rule thresholds for the advisory decision model.
# Changing them affects:
# - Stable / Watch / Escalating / Critical decisions
# - advisory text transitions and confidence scoring
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class AdvisoryConfig:
    switch_count_required: int = 2

    critical_risk_granular_threshold: float = 0.75
    critical_projected_cci_threshold: float = 0.80
    critical_cci_threshold: float = 0.68
    critical_stability_threshold: float = 0.55

    dissipating_projection_margin: float = 0.03
    dissipating_stability_trend_threshold: float = 0.003

    early_instability_cci_threshold: float = 0.45
    early_instability_stability_threshold: float = 0.62
    early_instability_trend_threshold: float = -0.004

    increasing_density_projection_margin: float = 0.03
    increasing_density_cci_threshold: float = 0.42
    increasing_density_risk_threshold: float = 0.48

    confidence_step: float = 0.10


MOTION = MotionConfig()
DENSITY = DensityConfig()
FUSION = FusionConfig()
REGIME = RegimeConfig()
RISK = RiskConfig()
STABILITY = StabilityConfig()
FORECAST = ForecastConfig()
PLAYBACK = PlaybackConfig()
UI_SMOOTHING = UISmoothingConfig()
ADVISORY = AdvisoryConfig()

