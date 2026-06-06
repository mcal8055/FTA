# v2 Findings — Physics-Informed Analysis (measured-launch baseline)

## What we did
Built an analytic ballistic forward model (`fta/physics.py`) and a no-training baseline: estimate the
release state (position + velocity) from shooting-hand fingertip tracking, then apply exact projectile
physics to predict angle/depth/left-right (`scripts/physics_baseline.py`, DEV / Scheme-A CV).

## Decisive finding: launch SPEED is unobservable at 60 fps
- **Measured fingertip launch speed ≈ 9 ft/s; a 13-ft free throw requires ≈ 22 ft/s.** Under-measured
  ~2.5× — the explosive release flick is undersampled/smoothed at 60 fps.
- Consequently **0%** of raw-measured trajectories even reach rim height; ~81% only after scaling v0 ×3.5.

## Mechanistic decomposition of the targets (the key insight)
Separating launch **direction** (observable) from **speed** (not), with a global speed scale + per-fold
affine calibration:

| target | governed by | physics corr(pred,true) | verdict |
|---|---|---|---|
| **angle** | launch direction | **+0.72** | observable; physics recovers it |
| left_right | lateral direction | +0.08 | weak (single-instant direction is noisy) |
| **depth** | launch speed | **−0.10** | NOT observable from instantaneous velocity |

This explains our v1 pattern from first principles: **angle is predictable because it depends on a
direction the pose captures; depth is the wall because it depends on launch speed, which 60 fps
markerless pose does not capture.**

## Implication for the physics-informed model
A differentiable ballistic *decoder* fed measured velocity will not overcome this — the bottleneck is
**information in the inputs, not model capacity**. The only route to depth is inferring launch speed
*indirectly* from the broader movement pattern (leg drive, follow-through amplitude, tempo) — i.e.,
ordinary learning, which our v1 features already attempted (depth R² ≈ 0). The winner's lower overall
error (0.0061) suggests *some* depth/LR signal is inferable that we haven't captured, but it is not
recoverable via direct launch-velocity physics.

## Honest conclusion
The physics-informed approach is a **major explanatory win** (mechanistic account of the depth wall) but
is **unlikely to beat v1 on prediction**, because depth is data-limited (speed unobservable), not
model-limited. Recommended framing for v2: present this as a physics-informed *analysis/diagnosis*, not a
leaderboard play. A full learned differentiable-decoder remains optional (test whether end-to-end
physics-constrained learning improves angle/LR generalization), with depth expectations managed.
