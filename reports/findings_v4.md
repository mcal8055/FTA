# v4 — Did the *correct* physics improve the metrics that measure understanding?

The competition winner's post-mortem revealed that we had used the wrong physics: **ballistic**
(projectile) physics, when the signal lives in **kinetic-chain** (force-generation) physics and in
**when** each target is biomechanically committed. This round implements those principles — without
copying his leaderboard-tuned numbers — and asks the question that actually matters here: not MSE
(a weak, miss-dominated metric on 113 shots), but whether correct physics improves our recorded
**explanatory** metrics (R², Spearman, skill) and, above all, **out-of-shooter transfer (Scheme B,
leave-one-player-out)** — the one metric that tests whether we've learned *mechanism* rather than
*identity*.

**New features** (`scripts/v4/kinetic_chain.py`, `scripts/v4/temporal.py`; 50 + 54 features):
- **Kinetic chain:** ankle→knee→hip→trunk→shoulder→elbow→wrist→fingertip peak-speed magnitudes and
  timings, proximal-to-distal sequencing lags, summation-of-speed ratios, COM forward-thrust velocity.
- **Multi-temporal:** mechanism-bearing kinematics sampled at fixed pre-release offsets
  (−0.50 … 0.0 s) — no winner frame indices imported, no train/test frame selection (leakage-free).

Evaluated against the v1 37-feature baseline across four sets (base, +kinetic, +temporal, +both) under
both CV schemes, with bootstrap CIs. Base reproduces the v1 prior (Scheme B HGB depth −0.578 vs v1
−0.539; left_right −0.196 vs −0.228), confirming the harness is faithful.

---

## 1. The headline: the transfer hypothesis is REFUTED

Under leave-one-player-out, kinetic-chain and temporal physics do **not** rescue out-of-shooter
transfer for depth or left_right. (Skill vs per-player-mean; HGB = the well-regularized, fair read.)

| target | base | +kinetic | +temporal | +both | v1 prior |
|---|---|---|---|---|---|
| **depth** | −0.578 | −1.070 | **−0.493** | −0.713 | −0.539 |
| **left_right** | −0.196 | −0.213 | **−0.191** | −0.224 | −0.228 |
| **angle** | +0.171 | +0.055 | **+0.173** | +0.132 | +0.29/+0.34 |

All depth and left_right transfer skills have 95% CIs **entirely below zero** — still worse than
predicting the held-out player's average. The best physics can do is make depth *a hair less bad*
(+temporal), nowhere near parity. Kinetic-chain features **hurt** transfer on every target, including
diluting angle (+0.171 → +0.055). With Ridge the overfitting is louder: depth craters −3.52 → −8.84,
left_right −0.72 → −5.32 once 100+ physics features hit 4 training shooters.

**The one real win is narrow:** multi-temporal geometry lifts the target that *already* transferred —
**angle**, Ridge skill **+0.29 → +0.43** (Spearman 0.561 → 0.648). Earlier, leaner temporal sampling
of elbow/wrist geometry sharpens the launch-direction signal that is genuinely shooter-invariant.

## 2. Same-shooter (Scheme A): a clean mechanistic double dissociation

Where identity is legitimately exploitable, the physics nudges each weak target in exactly the
direction its biomechanics predict — small, but mechanistically clean (HGB):

- **Depth improves with KINETIC-CHAIN, not temporal:** R² −0.023 → **+0.034** (+both, ΔR² +0.057),
  Spearman 0.270 → 0.335, skill −0.064 → −0.005 (≈ parity with player-mean). Matches *depth is
  force/COM-momentum driven*.
- **Left_right improves with TEMPORAL, not kinetic:** R² 0.111 → **0.133** (+temporal), Spearman
  0.362 → 0.391, skill +0.114 → **+0.136** (CI [+0.049, +0.222], excludes 0). Matches *left_right is
  committed late, at the wrist snap*.
- **Angle is saturated** (base R² 0.695, Spearman 0.839); physics adds nothing.

The **double dissociation** — kinetic→depth, temporal→left_right, each family helping the target its
physics predicts — is the strongest confirmation that these features encode the *right mechanism*.
Caveat: gains are small, HGB-only (Ridge degrades from collinearity at ~140 features on 365 shots),
and only the left_right temporal gain clears its CI.

## 3. Why physics is the right *explanation* but a poor *transfer lever*

The interpretation phase resolves the apparent paradox (physics-dominated, mechanistically correct, yet
non-transferable):

- **SHAP is physics-dominated:** physics features = 70.4% (angle), 66.7% (depth), 55.6% (left_right) of
  total HGB importance. Depth loads on kinetic-chain **timing/summation** (`lag_ankle_knee_ms` #1,
  `sos_hip_knee`, `relspeed_fingertip`); left_right on near-release wrist/elbow geometry; angle on
  mid-motion distal speed. The features the model uses are the ones the physics predicts.
- **"Force not position" is directionally confirmed but secondary:** COM forward thrust carries the
  largest single thrust share of depth importance (11.4%; `com_fwd_vel_m350` ranks #9), but the
  *dominant* depth drivers are proximal-to-distal **timing and summation-of-speed**, not COM momentum
  per se.
- **The killer finding — per-player mechanism is real, and universal mechanism is not.** Across all
  three targets, **zero features are shared by all five players' top-3 SHAP lists** (13–15 distinct
  features span them). For depth alone: **P0001** is driven by COM forward velocity, **P0003** by
  shoulder angular velocity / knee speed, **P0005** by elbow-extension velocity — *different body
  segments, same target*. There is no single shooter-invariant law to learn.
- **ICC corroborates:** adding physics *reduces* the between-player residual share for angle
  (0.028 → 0.019) and left_right (0.296 → 0.199) — invariant signal — but *raises* it for depth
  (0.155 → 0.234) while out-of-fold R² goes negative (0.146 → −0.027). For depth the force features are
  themselves **per-player signatures**: they add identity-aligned variance, which is exactly why
  pooled physics *degrades* depth transfer in Scheme B.

## 4. What this means

This closes the project's central question. *"Have we used physics to its fullest?"* — now, yes, and
the answer is more interesting than a wall:

> **Free-throw outcomes are governed by real, identifiable biomechanics — kinetic-chain force
> generation for depth, late wrist geometry for left_right — but each shooter implements them through a
> different kinetic-chain solution. The mechanism is physical and legible; it is just not shared across
> shooters. So the prediction problem is irreducibly per-player, and no amount of correct physics,
> pooled across shooters, can transfer.**

This is *precisely why the winning approach was per-player similarity-weighted local regression*, not a
shared physical model: he matched each shot to biomechanically similar shots from the **same** shooter,
sidestepping the absence of a universal law. Our v4 result is the mechanistic explanation of his
empirical choice — and of v1's "identity dominates," v2's depth wall, and v3's finding that ML
exploiting same-player structure helps while physics decoding does not.

**Limitations.** n = 5 shooters caps Scheme B precision (wide CIs, large per-player swings); the
per-player-mechanism claim, though striking, rests on 5 players and should be read as strong evidence,
not proof. But the *direction* — no positive depth/left_right transfer anywhere, across both models and
all feature sets — is robust.

**Closing.** Physics done correctly was a superb *explanation* and a poor *transfer lever*: it told us
exactly why the wall is there (the depth-bearing kinetics are per-player) without moving it. That is a
complete, honest scientific ending — not "we couldn't crack it," but "we now know the structure of why
it doesn't generalize, and it isn't noise or model capacity — it's biomechanical individuality."
