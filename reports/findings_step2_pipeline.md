# Step 2 Findings — Data Pipeline & Release Detection

## Release detection (leakage-free) — validated
- **Handedness is mixed** and detected per shot: dev split is 294 right- / 71 left-handed.
- **Body-only detector = peak elbow extension angular velocity** (the shooting "snap") in the upward
  swing, calibrated offset +3 frames. Chosen over peak-wrist-velocity, which fires ~30 frames early for
  slow ball-raisers (e.g. P0002).
- **Validation vs ball-separation reference (dev, n=365):** median |error| = **1 frame**, mean 1.9,
  p90 = 2, **96% within ≤3 frames**. Per-player ≥89% within 3 (P0003 99%). Meets the <3-frame target.
- The detector uses **only body keypoints** → safe to use at feature time; the ball is then dropped.

## Ball is full-flight leakage
The 2025 ball is tracked through nearly the whole arc (hold → release → apex ≈13 ft → ~10 ft hoop
crossing → post-outcome bounce). It essentially encodes the answer, so it is **excluded from all features**.

## Noise floor — estimable but loose (honest negative)
Using release-segmented clean flight (release → first post-apex z-minimum), robust (outlier-rejected)
projectile fit:
- **Ball measurement noise:** parabola-fit residual median **4.1 in** (mean 8.3 in) — about one ball
  width; consistent with the README's "ball tracking is noisy" warning.
- **Targets are NOT cleanly recoverable from the ball** by projectile physics: entry-angle recon corr
  **0.42** (RMSE 7.8°), depth corr 0.25, left_right corr 0.33.
- **Implication:** a *tight* irreducible floor cannot be established from the noisy ball; the targets
  carry substantial measurement noise and/or provider-side modeling. We therefore benchmark models
  **empirically against the per-player-mean baseline and the locked test set**, and treat the physics
  reconstruction RMSEs (angle ≈8°, depth ≈5 in, L/R ≈3.6 in) as a loose sanity reference, not a hard floor.

## Marker set
`CORE_MARKERS` (config): 19 reliable skeleton points (shoulders/elbows/wrists/hips/knees/ankles/toes/heels
+ nose/neck/mid-hip). Fingers excluded by default (2025-only); available as an optional block.
