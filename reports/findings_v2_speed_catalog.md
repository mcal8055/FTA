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
