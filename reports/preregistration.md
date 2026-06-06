# Pre-Registration — Free-Throw Biomechanics Regression

**Date:** 2026-06-06 · **Author:** Josh McAlister · **Seed:** 1561737

> Written **before** feature engineering and modeling. Its purpose is to separate
> **confirmatory** claims (committed here, tested once on the locked hold-out) from
> **exploratory** findings (generated during analysis, reported as hypothesis-generating).

## Honesty disclosure (anti-HARKing)
We inspected **aggregate** patterns of the full dataset before writing this (overall/per-player make
rates, entry-angle→make trend, landing dispersion). Therefore the earlier hypotheses H1–H5
(`freethrow_hypotheses.md`) are **exploratory, not confirmatory**. This document re-states only what we
commit to as confirmatory, and locks a test set that was *not* used to form any hypothesis at the
shot level.

## Locked hold-out (do not open during development)
- File: `splits/holdout_split.json` (seed 1561737, 20% stratified by player).
- 2025-12-18 session only: **365 dev / 93 test** (P0001 70/18, P0002 74/19, P0003 72/18, P0004 71/18, P0005 78/20).
- Dev/test target means & SDs verified similar at creation. **Test ids are scored exactly once, at the end.**
- 2024-08-28 session is **not** in this split — OOD probe only.

## Task & primary endpoints (pre-specified)
- Predict 3 continuous targets from **body keypoints only**: angle (`entry_angle`), depth (`landing_y`),
  left_right (`landing_x`). Ball + outcomes are leakage.
- **Primary metric:** per-target **RMSE (native units)** and **skill score vs the per-player-mean baseline**,
  each with 95% bootstrap CIs. **Secondary:** official MinMax-scaled MSE (leaderboard comparability).
- **Primary CV:** Scheme A (stratified shot-level) for leaderboard-comparable performance; Scheme B
  (leave-one-player-out) for the generalization claim. The **A–B gap** is a pre-specified reported quantity.

## Pre-specified feature families (confirmatory set)
Extracted from the body skeleton, event-aligned to a body-detected release, real-time differentiation:
1. release-instant kinematics (elbow/knee/trunk angles, wrist height)
2. joint angles & range of motion
3. velocities at release (wrist vertical, elbow angular, hand-path speed)
4. tempo / timing (dip→release, leg–arm coordination)
5. release geometry (height, hand-path azimuth/elevation)
6. asymmetry / balance (stance width, COM sway)
7. smoothness (normalized jerk / spectral arc length)

Finger-derived features are a **separate exploratory block** (2025-only; break OOD comparability).

## Confirmatory hypotheses (committed; tested on hold-out)
- **C1.** Body-motion features predict the targets better than the **global-mean** baseline
  (skill score > 0, CI excludes 0) on the locked test set, for at least angle and depth.
- **C2.** **Release-geometry** features (family 5) rank among the top predictors for **depth** and
  **left_right** (top-quartile importance by both SHAP and mixed-model effect, FDR-controlled).
- **C3 (key scientific claim).** Movement signal is **transferable / within-shooter**: the movement-only
  model beats the **per-player-mean** baseline under **Scheme B** and/or under within-player centering
  (skill score > 0, CI excludes 0). Failure here means the model learned identity, not biomechanics.

## Explicitly exploratory (reported as hypothesis-generating, not confirmatory)
Specific per-joint importance rankings; entry-angle optimum (H1); landing-variance/precision (H2);
clustering modes & miss taxonomy (Step 6b); per-player descriptive differences (n=5 → anecdotal);
deep-learning models; OOD transfer to the 2024 session.

## Analysis decisions fixed in advance
- All per-fold transforms (scaling, player-mean centering, release detector) fit on **train fold only**.
- Multiplicity: **Benjamini–Hochberg FDR** on inferential feature claims.
- Report the **target noise floor** (ball-trajectory-fit residuals) with every model number; do not claim
  improvement below it.
- Between-shooter statements are capped as anecdotal (n=5 players).
