# v2 — Data Catalog & Speed/Depth Investigation

## Information available to us (per shot)
- **Timestamped frames:** `time` in ms, ~60 fps, ~240 frames. Velocities/accelerations of any keypoint
  are derivable by finite differences — but **fast events (the release flick, ~50 ms) are undersampled.**
- **Ball xyz (feet):** full flight (hold → apex ≈13 ft → rim crossing → bounce). Encodes the true launch
  speed and trajectory. **Leakage for prediction (cannot be a feature) — used here only as ground truth.**
- **69 body keypoints xyz (feet)** incl. fingers (2025). The only legal model inputs.
- **Targets:** angle (deg), depth (in), left_right (in).

## Can we derive launch speed from the timestamped data?
- **True launch speed (from ball) = 22.7 ft/s, very consistent** (IQR 21.8–23.7; shooters vary speed ±~1).
- **Body pose cannot recover it.** Fingertip launch-speed estimators all give ~9 ft/s and correlate only
  **0.24** with the ball truth — despike / raw finite-diff / Savgol-derivative all agree. The markerless
  hand model smooths the fast flick; better differentiation does not help. Speed is **not observable**
  from body pose at 60 fps.

## But speed isn't the depth lever anyway (overturns the earlier hypothesis)
- **True launch speed → depth correlation = +0.05 (≈0).** Speed is nearly constant, so it can't explain
  depth variation.
- Even the **ball's own crossing → depth correlation ≈ 0.25** (from `noise_floor`). So **depth is
  noise-dominated** — the "theoretical" target is itself noisy and its variance is concentrated in
  erratic misses (made/miss variance ratio 6.4×). No velocity engineering recovers it.

## Reframe (productive)
- **Depth is intrinsically hard for everyone** (incl. the winner); not a fixable observability gap.
- **The winner's edge is almost certainly in angle + left/right**, which are **direction-driven** and
  **direction IS observable** from pose (crude physics already gives angle corr 0.72). Plausible winner
  decomposition: angle ~0.003, left/right ~0.005, depth ~0.010 → avg ~0.006.
- **Our headroom is on the learnable targets (angle, left/right), via better launch-DIRECTION modeling —
  not on depth, and not via speed recovery.**

## Next step
Pivot v2 from "recover speed for depth" to **sharpening angle & left/right through better launch-direction
estimation** (direction is observable; a denoised, possibly forward-kinematics-based release-direction
should beat single-instant finite differences). Keep depth honest as a noise-limited target.

## Direction-refinement test — negative (signal already saturated)
Added 10 refined launch-direction features (denoised fingertip-velocity direction, hand-path displacement
direction, arm/forearm pointing → lateral aim + elevation vs the rim line) and tested on DEV Scheme-A CV
(`scripts/direction_eval.py`):
- **RidgeCV:** direction features *hurt* every target (it shrinks all coefficients, can't drop noise).
- **LassoCV** (zeros out noise features): only sub-noise gains (angle −0.0002, depth −0.0017, left_right
  −0.0008; all within the ~0.005 seed SD).
- The existing 37 features already include `aim_error_lateral`, `launch_elevation`, `release_handpath_azimuth`
  — so the refinements are noisier duplicates. **Direction signal is saturated.**
- Incidental real finding: **Lasso > Ridge** as base model (angle 0.0071 vs 0.0076, depth 0.0172 vs 0.0203).

**Bottom line:** the principled v2 levers (full-series representation, physics decoding, direction
refinement) are exhausted; our ceiling is ~0.011–0.012 vs the winner's 0.0061, with depth noise-limited.
The remaining gap is not reachable through these routes.
