# v3 — Re-investigation: corrected physics floor, multi-frame launch state, and a verified new competition number

This round re-opened two v2 conclusions that turned out to be wrong, rebuilt the physics substrate
correctly, and re-scored everything on the **official same-player competition split** (345 train /
113 test, reproduced via `scripts/competition_eval.py`'s target-match mapping). Headline:

> **Verified competition-split scaled-MSE = 0.010901** (per target: angle **0.00749** / depth
> **0.01192** / left_right **0.01330**). **Beats the prior 0.01164**; **1.78×** the winner's 0.006136.
> The gain is **better ML on the legal same-player split — not physics.**

All numbers below are measured, leakage-audited (PASS), and reproduced by an independent re-run.
Code: `scripts/v3/` (`physics_v3.py`, `foundation_eval.py`, `exp_a_physics_decoder.py`,
`exp_b_geom_features.py`, `exp_c_ml_ceiling.py`). Outputs: `outputs/v3_*.json`.

---

## 1. Two v2 claims were wrong

**(a) The "noise floor" was a fit artifact, not a floor.** v2's `noise_floor.py` reported that even
the ball reconstructs depth at only corr 0.25, and concluded depth is intrinsically noise-dominated.
But the SPL README states the cameras were optimized for the **body**, and ball tracking "might be
noisy" — so the ball is the *noisiest* object in the dataset, and "even the ball can't recover it"
does not bound a body model. Refitting a gravity-constrained parabola to the **clean airborne arc**
(z ≥ 10.8 ft, pre-rim-contact; median fit residual **0.081 ft**) and extrapolating the undisturbed
10-ft crossing recovers the official targets far better:

| target | v2 "floor" corr | v3 clean joint reconstruction (corr) |
|---|---|---|
| entry angle | 0.42 | **0.566** |
| depth | 0.25 | **0.413** |
| left_right | 0.33 | **0.540** |

The "floor" moved when the estimator improved — so it was never a floor. **Depth is recoverable**
(`depth_recoverable = true`). On the clean made/swish subset the depth *correlation* drops (0.27) but
its scaled-MSE is tiny (0.005): depth variance lives in the misses, not in label noise.

**(b) "Speed is constant, so depth is unobservable" is a non-sequitur, and one half was numerically
wrong.** Launch-speed *magnitude* really is tightly controlled (median **22.16 ft/s**, CV **4.7%**) —
that half reproduces. But v2's companion claim corr(speed, depth) = 0.05 is wrong: it reproduces at
**+0.275**. And constancy never implied irrelevance — depth is a **joint** function of (speed,
launch elevation, release position), which the marginal correlations understated. This claim lived
only in v2 prose with no committed code; v3 computes it (`outputs/v3_foundation.json`).

---

## 2. Why the physics-informed approaches still didn't win

We built the physics correctly this time — a **multi-frame** body launch-state estimator
(`physics_v3.body_launch_fit`, windowed hand/forearm fit, not v2's single-frame finite difference)
and a privileged-information (LUPI) decoder. They did **not** improve prediction, and the foundation
analysis says exactly why: **the body cannot observe the launch state well enough.**

Body→ball observability (corr of body-estimated vs ball-truth launch components):

| component | body can observe it? |
|---|---|
| launch elevation | 0.376 (weak) |
| release_y / release_x | 0.352 / 0.257 (weak) |
| launch azimuth | −0.075 (no) |
| launch speed | −0.335 (no — even sign-inverted) |
| release_z | −0.021 (no) |

- **Exp A (LUPI ballistic decoder):** total **0.011902**, depth **0.01329** — *worse* than body-only.
  Pushing a poorly-observed launch state through exact physics buys nothing.
- **Exp B (joint-geometry features):** total **0.011886**. The body ballistic-crossing features were
  all-NaN — a short-window markerless pose velocity never yields a descending rim crossing, which is
  the same observability wall in another form.

So the v2 *diagnosis* (direction observable, speed not) was right; the hoped-for *predictive* payoff
from physics is blocked because 60 fps markerless pose can't pin the end-effector launch vector.

---

## 3. Where the win actually came from (Exp C)

The leaderboard is a **same-player** task; prior work over-invested in leave-one-player-out transfer.
Applying strong, well-tuned conventional ML on the *right* split — using only the 37 leakage-free
body features + legal player identity — beats the prior:

- Candidate zoo: per-player-mean, Ridge/Huber (player one-hot **and** player-centered residual),
  HistGradientBoosting (squared & absolute loss), RandomForest, per-player separate models.
- Per-target **NNLS-stacked blend** over train out-of-fold predictions; blend-vs-single chosen by
  CV-within-train; test touched once.

**Verified result 0.010901** (angle 0.00749 / depth 0.01192 / left_right 0.01330). Decomposition of
the 0.00485 gain over the per-player-mean baseline: **left_right 0.00317**, depth 0.00124, angle
0.00044. Final left_right/angle predictors are RandomForest-on-player-centered-residual heavy
(weights 0.66 / 0.74); depth is Huber-centered (0.42) + per-player HGB (0.31) + player-mean (0.22).
The gain is nonlinear body-residual structure exploited within known shooters — **not** physics.

---

## 4. Leakage audit — PASS

1. **No ball at test.** Exp C uses only the 37 columns of `features_2025.parquet` (zero `ball_*`
   columns); the physics-suspicious names (`launch_speed`, `launch_elevation`, `dist_to_rim`,
   `aim_error_lateral`) are computed in `fta/features.py:187-211` from fingertip *player* keypoints +
   the static rim constant — body-only. Ball is training-only LUPI supervision in Exp A.
2. **Split exact.** `competition_eval.py` mapping → train 345 / test 113, all 5 players in both, zero
   shot_id overlap.
3. **Nothing fit on test.** Scalers, per-player means, model params, and NNLS blend weights are all
   fit on train via KFold-within-train OOF; test targets read only at final scoring.
4. **No target leakage.** Max |feature–target corr| = 0.766 (`wrist_jerk_norm` vs angle), a
   legitimate biomechanical predictor — nothing trivially encodes a target.

---

## 5. Honest standing & remaining gap

- **Improved, modestly:** 0.010901 vs prior 0.01164 (−6.3% relative), still **1.78×** the winner.
- The remaining gap is concentrated in **left_right (0.0133)** and **depth (0.0119)**; angle (0.0075)
  is near its recoverable ceiling.
- Both corrected facts (recoverable noise floor, speed↔depth = 0.27) say the signal *exists*; the
  blocker is that the body channel that carries it (end-effector launch direction/speed) is
  under-sampled by 60 fps markerless pose, and there are only **5 shooters / 345 shots** to learn the
  per-shot deviations. Closing the gap to the winner most likely needs sharper per-shot direction
  recovery for left_right and cleaner release-velocity for depth than this capture allows — or the
  winner simply extracted more per-shot structure from the same data than our feature set does, which
  remains an open, testable question (a direct per-target ablation against the public leaderboard
  would localize it).

**Bottom line:** v2's "depth is a noise wall" was overstated and rested on a fit artifact plus a wrong
correlation; depth and left_right do carry recoverable signal, and exploiting the same-player metric
honestly buys a real (if modest) improvement. Physics remains a strong *explanation* of the structure
but not, at this capture resolution, a predictive lever.
