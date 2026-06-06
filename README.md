# FTA — Free-Throw Biomechanics: Predicting Shot Landing from Body Motion

Rigorous, retrospective study of the **SPLxUTSPAN 2026** data challenge: predict three continuous
free-throw landing outcomes — **entry angle, depth, left/right** — from markerless motion-capture
**body keypoints only**.

## Headline findings (v1)
- **Entry angle is predictable and transferable** to unseen shooters (hold-out skill vs global mean
  **+0.69 [0.59, 0.80]**, R² ≈ 0.64); drivers = wrist speed + knee extension at release.
- **Depth & left/right are the hard targets.** Fingertip launch-vector + hoop-relative aim features
  rescued left/right (R² −0.14 → +0.15); depth stays near 0.
- **Shooter identity dominates** — unsupervised clustering recovers all 5 shooters (ARI 1.0); models
  only tie a per-player-mean baseline.
- **Apples-to-apples on the official split:** our best ≈ **0.0116** sMSE vs winner **0.006136** (~1.9×).
- **Full-series MiniRocket did not beat 37 scalars** and both overfit identity out-of-shooter →
  **bottleneck is data (5 shooters / 345 shots), not model class.**

## Methodology (rigor-first)
Pre-registered hold-out lock (anti-HARKing), leakage-free release detection, dual cross-validation
(stratified + leave-one-player-out), bootstrap CIs, SHAP + mixed-effects interpretation, and honest
negative results. See [`reports/REPORT.md`](reports/REPORT.md), [`reports/preregistration.md`](reports/preregistration.md),
and [`reports/STATUS.md`](reports/STATUS.md).

## Layout
- `fta/` — pipeline: config, loader, release detection, features, tensors, CV, models, metric
- `scripts/` — split, EDA, dataset build, experiments, interpretation, clustering, final/competition eval, ROCKET
- `reports/` — write-up, pre-registration, per-step findings, research briefs
- `outputs/` — metrics + figures (large caches gitignored)
- `splits/` — sealed hold-out + competition map

Data (`SPL-Open-Data/`, competition CSVs) is gitignored. Run scripts with `PYTHONPATH=. python3 scripts/<name>.py`.

## Roadmap
- **v1** — body-only regression study + competition benchmark (this release).
- **v2** — physics-informed approach: differentiable ballistic decoder (body motion → launch state →
  analytic projectile → outcomes). See [`reports/research_physics_informed.md`](reports/research_physics_informed.md).
