# Physics-Informed ML for Free-Throw Outcome Prediction — Focused Brief

*Scope: applied to our problem (body motion → angle/depth/left-right), small data (345 shots / 5 shooters),
depth as the weak target. Quick brief, not a full review.*

## The key idea for our problem: a differentiable ballistic decoder
Our three targets are **not arbitrary regression outputs — they are the output of a known physical
process**: the ball leaves the fingertips with a release **state** (position `p0`, velocity `v0`, and
backspin), then follows projectile motion to where it crosses the 10-ft rim plane (rim known:
JSON ~(41.67,0,10)). Angle/depth/left-right are a closed-form, **differentiable** function of `(p0, v0)`.

So instead of `body-motion → targets` (a generic regressor that must relearn ballistics from 345 samples),
use:

```
body-motion  --(learned)-->  launch state (p0, v0[, spin])  --(FIXED analytic ballistics)-->  angle/depth/LR
```

Train end-to-end with the loss on the real targets; gradients flow through the analytic decoder (the
descending z=10 crossing is a closed-form quadratic root → analytic gradients). Optionally add an
**auxiliary loss** supervising the launch state against the *measured* fingertip kinematics (we can
observe `p0`/`v0` from tracking) — a semi-supervised intermediate.

## Why this is the right family here
- **Regularization by physics → attacks our exact failure mode.** The ballistic decoder is
  shooter-invariant; only launch-state estimation is learned. This shrinks the hypothesis space to
  physically realizable maps, which should reduce the identity-overfitting that sank MiniRocket/ridge on
  leave-one-player-out (n=345/5 shooters).
- **Targets depth directly.** Depth = front/back distance is a deterministic function of launch speed +
  angle + release height. Pushing a predicted launch state through exact ballistics means depth error
  comes only from launch-state error — not from a generic model trying to learn a nonlinear speed→depth
  surface from few samples.
- **Interpretable & mechanistic** (the 40% write-up loves this).

## Method taxonomy (which fit us)
| family | fit | note |
|---|---|---|
| **Physics-as-decoder / differentiable forward model** | **best** | our recommended route (above) |
| **Residual / hybrid (physics + ML correction)** | strong | physics gives base prediction; ML learns residual (drag, spin, tracking bias) |
| Physics-guided loss (soft penalty for violating dynamics) | ok | softer than a hard decoder; less leverage here |
| PINNs (solve PDE/ODE by penalizing residuals in the loss) | **poor fit** | PINNs solve differential equations / forecast fields; our task is extrinsic regression with a *known closed-form* map — use the map directly, don't re-derive it |
| Neural ODE rollout | overkill | ballistics is analytic; no need to integrate a learned ODE |

## Recommended FIRST experiment (cheap, decisive): measured-launch ballistic baseline
Before any training, compute the launch state from **observed fingertip kinematics** at release
(`p0`, `v0` we already extract), run the **analytic projectile map** to predict angle/depth/LR, and score
on train-CV. This is a no-learning baseline that answers the pivotal question:

- If it predicts depth/LR **well** → the physics path is promising; build the learned decoder.
- If it predicts **poorly** → the bottleneck is **launch-velocity estimation noise** (depth depends on
  launch *speed*, the hardest quantity to estimate from finite-differenced pose — the same noise that
  limited our ball reconstruction, ~4 in). That would bound how far physics-informed ML can go and is
  itself a key, honest finding.

## Caveats / risks
- **Velocity-noise bottleneck (main risk):** small launch-speed error → large depth error. Test via the
  baseline above first.
- **Drag + Magnus (backspin):** pure projectile ignores ~small aero effects over ~13 ft; include a simple
  drag/Magnus term if depth precision demands it (free throws have meaningful backspin).
- **Release-state observability:** fingertip distal markers are noisy; smoothing/var-reduction of `v0`
  matters more than model capacity.

## Relation to what we already built
We already extract a crude launch vector (fingertip velocity) and hoop-relative aim — but we feed them as
*features to a generic regressor*. The physics-informed upgrade is to **estimate the full launch state and
push it through real ballistics**, so the model only has to get the (low-dim, physical) launch state right.

## Sources
- Hybrid model + PINN for ballistic prediction (embeds ballistic constraints, limited data) —
  ScienceDirect S0952197625029963.
- Differentiable solver regularizes decoder/latent toward true dynamics — arXiv 2505.14595.
- Physics-Informed Tracking: autoencoder with embedded differentiable physics for trajectories — arXiv 2604.16895.
- CNN-LSTM basketball: arm-joint kinematics → forward kinematics → fingertip velocity → projectile physics
  (release angles ~46–47°, matches our ~45–48° finding) — PMC8486516.
- PINNs intuitive guide (taxonomy) — Towards Data Science.
