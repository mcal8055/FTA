# FTA — Free-Throw Biomechanics: Predicting Shot Landing from Body Motion

Retrospective study of the **SPLxUTSPAN 2026** data challenge: predict three continuous
free-throw landing outcomes — **entry angle, depth, left/right** — from markerless motion-capture
**body keypoints only**.

## Bottom line
Free-throw outcomes are governed by **real, legible biomechanics** — but each shooter implements them
through a **different kinetic chain**, so the prediction problem is **irreducibly per-player**, not
global. Only **entry angle** carries a shooter-invariant signal that transfers to unseen shooters;
**depth** and **left/right** do not — and *correct* physics explains *why* the wall is there without
moving it. This is exactly why the competition winner's **per-player** similarity-weighted model
succeeded where a shared physical law cannot.

## What we learned

**About the problem (science):**
- **Each target has a distinct, identifiable mechanism.** Entry angle ← release *direction* (wrist
  speed + knee extension). Depth ← *kinetic-chain force generation* (proximal→distal timing &
  summation-of-speed, COM forward thrust — "force, not position"). Left/right ← *late wrist-snap
  geometry*. A clean double dissociation confirms it: kinetic-chain features help depth, multi-temporal
  features help left/right — each helps the target its physics predicts.
- **The mechanisms are real but per-player.** Across all 5 shooters, **zero** features are shared among
  their top-3 SHAP drivers for any target; adding force features *raises* depth's between-player
  variance share (0.155 → 0.234). There is no common law to learn, so depth/left-right fail to transfer
  out-of-shooter (Scheme B skill −0.49 / −0.19, CIs below 0).
- **The binding constraint is shooters (n=5), not model class or features.** MiniRocket, full-series
  models, a differentiable ballistic decoder, and correct kinetic-chain physics all fail to transfer —
  because 5 people cannot reveal a shared mechanism.

**About the method (how we worked):**
- **A plausible narrative can still be wrong.** v2's "noise floor proves depth is irreducible" was a
  fit artifact (the ball is the *noisiest* tracked object, not near-perfect info); "speed ⊥ depth
  (0.05)" was off ~5× (actually +0.27). Recompute load-bearing numbers; check premises aren't inverted.
- **Use the physics that matches where the information is.** Ballistic (projectile) physics is exact
  but the body can't observe the launch state at 60 fps; the signal lives in *kinetic-chain* physics.

**Full methods, per-phase results, and the v1→v4 progression** (with every headline number, the
corrected v2 claims, and the per-player-mechanism finding) are in the single canonical write-up:
**[`reports/REPORT.md`](reports/REPORT.md)**.

## Methodology (rigor-first)
Pre-registered hold-out lock (anti-HARKing), leakage-free release detection, dual cross-validation
(stratified + leave-one-player-out), bootstrap CIs, SHAP + mixed-effects interpretation, and honest
negative results — including those that overturned our own earlier claims. See
[`reports/REPORT.md`](reports/REPORT.md) and the frozen [`reports/preregistration.md`](reports/preregistration.md).

## Models
- **Baselines** — global mean; per-player mean; player-identity one-hot.
- **Linear** — Ridge, Lasso, ElasticNet (α by CV).
- **Trees** — HistGradientBoosting (squared & absolute loss), RandomForest.
- **Full series** — MiniRocket (5k random convolutional kernels) + RidgeCV.
- **Physics** — analytic *differentiable ballistic decoder*; privileged-information (LUPI) body→launch-state
  regressor feeding that decoder (ball used as a training-only teacher).
- **Stacked (competition split)** — per-target NNLS blend over {RandomForest on player-centered residuals,
  per-player HGB, Huber, player-mean}.
- **Interpretation** — SHAP (on HGB), statsmodels MixedLM (ICC); KMeans clustering with a permutation null.

HGB is the primary single-target regressor (best transfer + hold-out); the stacked blend is the best
same-player competition model.

## Performance
*Metric legend — **skill** = 1 − MSE_model/MSE_baseline (higher better; 0 = ties the baseline). **scaled-MSE** =
MSE on MinMax-scaled targets (angle [30,60], depth [−12,30], LR [−16,16]); lower better.*

**(1) Out-of-shooter transfer — Scheme B (leave-one-player-out), skill vs per-player-mean (HGB).** The scientific headline.
| target | v1 base | best v4 | verdict |
|---|---|---|---|
| **angle** | **+0.34** | **+0.43** (Ridge+temporal) | transfers ✅ |
| depth | −0.54 | −0.49 (HGB+temporal) | no transfer |
| left_right | −0.23 | −0.19 (HGB+temporal) | no transfer |

**(2) Sealed hold-out (93 shots, HGB) — native-unit RMSE and skill vs global mean [95% CI].**
| target | RMSE | skill vs global [CI] |
|---|---|---|
| **angle** | **3.13°** (vs 5.59° baseline) | **+0.69 [0.59, 0.80]** ✅ |
| depth | 5.48 in | +0.12 [−0.08, 0.27] |
| left_right | 4.09 in | −0.00 [−0.19, 0.12] |

**(3) Official competition split (same 5 players, 345 train / 113 test) — scaled-MSE (lower better).**
| model | overall | angle | depth | left_right |
|---|---|---|---|---|
| global mean | 0.0194 | 0.0268 | 0.0149 | 0.0166 |
| per-player mean | 0.0125 | 0.0079 | 0.0132 | 0.0165 |
| Ridge (v1) | 0.0116 | 0.0079 | 0.0115 | 0.0155 |
| HGB | 0.0127 | 0.0096 | 0.0147 | 0.0139 |
| **stacked blend (v3)** | **0.0109** | 0.0075 | 0.0119 | 0.0133 |
| winner *(best of ~2500 subs)* | 0.0061 | — | — | — |

The winner's 0.0061 is the minimum over ~2500 public-leaderboard submissions (a downward-biased order
statistic); our 0.0109 is a single sealed-hold-out estimate — see `reports/REPORT.md` §3.5.

## Layout
- `fta/` — pipeline: config, loader, release detection, features, tensors, CV, models, metric
- `scripts/` — split, EDA, dataset build, experiments, interpretation, clustering, final/competition eval, ROCKET;
  `scripts/v3/` (corrected floor, multi-frame launch state, LUPI decoder) and `scripts/v4/` (kinetic-chain + multi-temporal physics)
- `reports/` — `REPORT.md` (the canonical write-up) + `preregistration.md` (frozen)
- `outputs/` — metrics + figures (large caches gitignored)
- `splits/` — sealed hold-out + competition map

Data (`SPL-Open-Data/`, competition CSVs) is gitignored. Run scripts with `PYTHONPATH=. python3 scripts/<name>.py`.

## Release history (complete — narrated in [`reports/REPORT.md`](reports/REPORT.md) §3.4)
- **`v1.0`** — body-only regression + competition benchmark; angle transferable (+0.69), depth/LR not; identity dominates.
- **`v2.0`** — ballistic physics diagnosis. *(Its noise-floor and speed↔depth claims were later corrected in v3.)*
- **`v3.0`** — corrected noise floor + speed claim; honest same-player ML = **0.010901** sMSE (beats prior 0.01164); privileged-info ballistic decoder adds nothing.
- **`v4.0`** — correct physics (kinetic chain + multi-temporal): explains the per-player transfer wall but does not move it. **Project conclusion.**
