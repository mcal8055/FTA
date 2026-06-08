# FTA — Free-Throw Biomechanics: Predicting Shot Landing from Body Motion

Rigorous, retrospective study of the **SPLxUTSPAN 2026** data challenge: predict three continuous
free-throw landing outcomes — **entry angle, depth, left/right** — from markerless motion-capture
**body keypoints only**.

## Headline findings (v4 — correct physics, out-of-shooter transfer)
- **The transfer hypothesis is REFUTED.** Implementing the winner's *correct* physics — kinetic chain
  (proximal→distal peak-speed sequencing, summation-of-speed, COM forward thrust) + multi-temporal
  sampling — did **not** rescue **depth** or **left_right** out-of-shooter (Scheme B, leave-one-player-out).
  Best depth transfer skill vs per-player-mean = **−0.49** (HGB+temporal), best left_right = **−0.19**;
  both 95% CIs lie entirely below 0 — still worse than predicting the held-out player's average. Adding
  physics to Ridge made transfer dramatically *worse* (depth −2.0 → −8.8, left_right −0.43 → −5.3).
- **The one real win is narrow:** multi-temporal geometry lifts the target that *already* transferred —
  **angle** skill **+0.29 → +0.43** (Ridge+temporal). Kinetic-chain features helped no target's transfer.
- **The physics is the right *explanation*, not a transfer lever.** SHAP confirms "force, not position":
  COM forward thrust (`com_fwd_vel_m350`, #9) and ankle→knee timing (`lag_ankle_knee_ms`, #1) are top
  depth drivers (physics = 67% of depth importance). But between-player residual share *rose* for depth
  (0.155 → 0.234): the force features are themselves per-player signatures. With only 5 shooters, each
  controlling each target through a different segment (0 features shared across players' top-3), there is
  no common mechanism to carry across the hold-out. See [`reports/findings_v4.md`](reports/findings_v4.md).

## Headline findings (v1)
- **Entry angle is predictable and transferable** to unseen shooters (hold-out skill vs global mean
  **+0.69 [0.59, 0.80]**, R² ≈ 0.64); drivers = wrist speed + knee extension at release.
- **Depth & left/right are the hard targets.** Fingertip launch-vector + hoop-relative aim features
  rescued left/right (R² −0.14 → +0.15); depth stays near 0.
- **Shooter identity dominates** — unsupervised clustering recovers all 5 shooters (ARI 1.0); models
  only tie a per-player-mean baseline.
- **Apples-to-apples on the official split:** v3 best **0.010901** sMSE (beats prior **0.01164**)
  vs winner **0.006136** (1.78×). Per target: angle 0.00749 / depth 0.01192 / left_right 0.01330.
  Gain is **better ML on the legal same-player split** — a per-target NNLS-stacked blend exploiting
  player identity + RandomForest body-residual structure; **not** physics. The privileged-information
  ballistic decoder (LUPI) added nothing (depth 0.01329, worse than body-only). See
  [`reports/findings_v3.md`](reports/findings_v3.md).
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
