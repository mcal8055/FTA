# Step 1 Findings — Targets (DEV set, n=365)

Metric replica (`fta/metric.py`) verified against hand-computed scaled-MSE. EDA below is **dev-only**.

## Key results
- **Make rate (dev):** 0.671.
- **Targets are nearly uncorrelated** (Pearson |r| ≤ 0.14; angle~depth −0.14, angle~L/R +0.10, depth~L/R −0.05)
  → three largely independent regression problems; train per-target models.
- **Angle is strongly shooter-specific.** Per-player mean angle: P0005 41.2°, P0001 42.6°, P0002 44.5°,
  P0003 47.9°, **P0004 52.4°**; within-player SD only 1.4–3.5°. Predicting angle therefore risks
  collapsing into shooter identification → motivates the Scheme A vs B contrast.
- **Heteroscedasticity is severe** (miss/made variance ratio): **depth 6.4×**, L/R 4.2×, angle 1.3×.
  Model error will be dominated by misses; **depth is the hardest target**. Report errors stratified by made/miss.
- Best shooter (P0003: 90% make) has the tightest depth/L/R dispersion and ~48° arc — consistent with an
  arc sweet-spot, but **n=5 ⇒ anecdotal**.

## Distributions (dev)
| target | unit | mean | sd | min | median | max |
|---|---|---|---|---|---|---|
| angle | deg | 45.6 | 4.7 | 29.5 | 45.0 | 58.6 |
| depth | in | 9.8 | 5.2 | −10.2 | 10.2 | 25.0 |
| left_right | in | −0.9 | 3.8 | −13.0 | −0.8 | 10.2 |

## Noise floor — deferred to Step 2 (with reason)
The ball is tracked through ~the whole flight (hold → release → apex ≈13 ft → ~10 ft hoop crossing →
post-outcome bounce). So (a) the **ball is near-perfect leakage** (firmly exclude from features), and
(b) a trustworthy noise floor requires **flight segmentation** (drop hold + post-contact frames) — the
same release/flight machinery built in Step 2. A naive single-parabola fit over the raw arc fails
(reconstructed-vs-provided entry-angle corr ≈ 0.08) because it ingests held-ball and bounce frames.
Floor will be computed as the parabola-fit residual on the clean flight once the segmenter exists.
