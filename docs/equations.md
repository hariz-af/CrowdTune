# CrowdTune Equations

Physics-Inspired Structural Crowd Instability Modeling

---

## Introduction

CrowdTune models crowd instability as a continuous structural process using interpretable physical crowd signals rather than density estimation or black-box anomaly classification.

The framework analyzes instability through the interaction between:

- Spatial Constraint (\(S\))
- Kinematic Activity (\(K\))
- Collective Coherence (\(C\))

These signals are synthesized into interpretable instability indices, regime tendencies, and short-horizon forecasting outputs.

---

# Signal Definitions

CrowdTune uses three normalized physical crowd signals:

| Signal | Description |
|---|---|
| \(S\) | Spatial Constraint |
| \(K\) | Kinematic Activity |
| \(C\) | Collective Coherence |

All signals are normalized within:

```math
[0,1]
```

where:

- \(0\) represents minimal signal intensity
- \(1\) represents maximal observed signal intensity

---

# Crowd Constraint Index (CCI)

The Crowd Constraint Index models structural crowd instability using the interaction between:

- spatial compression,
- motion activity,
- and coherence degradation.

```math
CCI(t) = S(0.4K + 0.6(1 - C))
```

Where:

- higher \(S\) increases structural instability sensitivity under constrained environments
- higher \(K\) increases motion intensity and turbulence-like activity
- lower \(C\) reflects coherence breakdown and fragmented movement

High CCI values indicate:

- increasing crowd compression
- fragmented movement organization
- elevated structural instability

---

# Stability Index (SI)

The Stability Index measures short-term temporal consistency and fluctuation behavior of crowd instability over time.

Higher SI values indicate:

- stable crowd behavior
- lower temporal volatility
- smoother instability evolution

Lower SI values indicate:

- fluctuating instability conditions
- unstable temporal dynamics
- susceptibility to rapid crowd-state transitions

Within CrowdTune:

- CCI models structural instability
- SI models temporal volatility of that instability

---

# Risk Index (RI)

The Risk Index provides an operational interpretation of escalating crowd instability conditions.

Unlike the Crowd Constraint Index (CCI), which models structural crowd behavior, the Risk Index functions as an operational urgency-oriented abstraction layer for monitoring and decision support.

```math
Risk = (CCI_{norm})^\gamma
```

Where:

- \(CCI_{norm}\) represents normalized Crowd Constraint Index values
- \(\gamma\) controls structural escalation sensitivity

Higher RI values correspond to:

- increasing operational crowd risk
- escalating instability conditions
- elevated preventive monitoring urgency

The Risk Index is intentionally designed as an operational metric rather than a physical crowd law.

---

# Crowd Physics Regime Tendencies

CrowdTune interprets crowd behavior through continuous physics-inspired structural tendencies:

- Gas-like behavior
- Fluid-like behavior
- Granular-like behavior

These tendencies are not treated as discrete classifications, but as continuous structural crowd states.

---

## Gas-Like Regime

```math
T_{gas} = (1 - S)^2
```

Characteristics:

- low spatial constraint
- large movement freedom
- sparse interactions
- unconstrained motion

Gas-like behavior is primarily governed by available movement space rather than coherence or motion intensity.

---

## Fluid-Like Regime

```math
T_{fluid} = 1.5S(1 - S)C
```

Characteristics:

- moderate spatial constraint
- coordinated collective movement
- directional alignment
- organized crowd flow

The fluid regime exists within an intermediate density window where movement remains coherent and dynamically stable.

Higher coherence strengthens fluid-like crowd behavior.

---

## Granular-Like Regime

```math
T_{granular} = S(1 - C)
```

Characteristics:

- high spatial compression
- coherence collapse
- mechanically constrained movement
- force-dominant interactions

Granular-like behavior emerges when spatial constraint increases while collective coherence degrades.

This regime represents the strongest structural instability tendency within CrowdTune.

---

# Regime Transition Interpretation

CrowdTune interprets crowd instability as a gradual structural transition process:

```math
Gas \rightarrow Fluid \rightarrow Granular
```

This progression corresponds to:

- increasing spatial compression
- reduced movement freedom
- stronger crowd interaction coupling
- coherence destabilization
- mechanically constrained motion emergence

Abrupt increases in spatial constraint combined with coherence degradation may contribute to dangerous instability escalation.

---

# Forecasting Model

CrowdTune includes a short-horizon instability forecasting layer using adaptive Kalman-based temporal modeling.

The forecasting module estimates the near-future evolution of structural crowd instability using Crowd Constraint Index (CCI) dynamics.

General forecasting form:

```math
\hat{CCI}_{t+1} = f(CCI_t, \Delta CCI_t)
```

Where:

- \(\hat{CCI}_{t+1}\) represents projected instability
- \(CCI_t\) represents current structural instability
- \(\Delta CCI_t\) represents temporal instability variation

The forecasting layer supports:

- short-term instability projection
- escalation trend estimation
- preventive crowd monitoring
- early-warning analysis

Within CrowdTune:

- CCI represents current structural instability
- SI represents temporal fluctuation stability
- Forecasting represents directional instability evolution

---

# Structural Interpretation Philosophy

CrowdTune models crowd instability as:

- a continuous structural process,
rather than:
- a simple density problem,
- binary anomaly classification,
- or post-event detection system.

The framework emphasizes:

- interpretability
- transparency
- preventive monitoring
- physics-inspired crowd reasoning

instead of purely black-box predictive outputs.

The central principle of CrowdTune is:

> Crowd instability emerges through the interaction between spatial constraint, motion behavior, and collective coherence over time.