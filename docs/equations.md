# CrowdTune Equations

---

## Signal Definitions

CrowdTune models crowd instability using three normalized crowd signals:

| Signal | Description |
|---|---|
| \(S\) | Spatial Constraint |
| \(K\) | Kinematic Activity |
| \(C\) | Collective Coherence |

All signals are normalized within:

```math
[0,1]
```

---

## Crowd Constraint Index (CCI)

The Crowd Constraint Index models instability as a structural interaction between environmental restriction, motion activity, and coherence degradation.

```math
CCI(t) = S(0.4K + 0.6(1 - C))
```

Where:

- higher \(S\) increases instability sensitivity under constrained environments
- higher \(K\) increases crowd motion intensity
- lower \(C\) reflects coherence degradation and synchronization breakdown

---

## Stability Index (SI)

The Stability Index represents the overall crowd stability state derived from the Crowd Constraint Index.

```math
SI(t) = 1 - CCI(t)
```

Higher SI values indicate more stable and coordinated crowd conditions.

---

## Risk Index (RI)

The Risk Index estimates operational crowd risk levels using regime-aware interpretation of crowd instability behavior.

```math
RI(t) \propto CCI(t)
```

Higher RI values correspond to elevated instability escalation potential under constrained crowd conditions.

---

## Forecasting Model

CrowdTune applies adaptive temporal forecasting to estimate short-horizon instability evolution using Crowd Constraint Index trends.

General forecasting form:

```math
\hat{CCI}_{t+1} = f(CCI_t, \Delta CCI_t)
```

where:

- \(\hat{CCI}_{t+1}\) represents predicted instability
- \(\Delta CCI_t\) represents temporal instability variation

---

## Signal Normalization

## Regime Interpretation

CrowdTune interprets crowd behavior through continuous crowd-regime tendencies:

| Regime | Characteristics |
|---|---|
| Gas-like | Free-flowing, low constraint |
| Fluid-like | Coordinated directional movement |
| Granular-like | Compression-dominant instability |

Regime transitions emerge from structural interaction between \(S\), \(K\), and \(C\).