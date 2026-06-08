# Predicting Free-Throw Landing Outcomes from Body Motion
### A rigorous, retrospective study on the SPLxUTSPAN 2026 data (v1 → v4)

**Bottom line.** From a shooter's body keypoints alone, **entry angle is predictable and transferable to
unseen shooters** (hold-out skill vs global mean **+0.69, 95% CI [0.59, 0.80]**; RMSE 3.1° vs 5.6°).
**Depth and left/right are not predictable out-of-shooter** (transfer skill below 0). The reason is the
central finding of this study: free-throw outcomes are governed by **real, identifiable biomechanics**,
but **each shooter implements them through a different kinetic chain**, so the prediction problem is
**irreducibly per-player**, not global. *Correct* physics explains precisely **why** the wall is there
without moving it — which is exactly why the competition winner's per-player model succeeded where a
shared physical law cannot.

This single report supersedes the per-step and per-version working notes; the full progression is
preserved in git under tags `v1.0`–`v4.0`. The pre-registration is kept frozen and separate
(`reports/preregistration.md`).

---

## 1. Task & data
- **Goal (Kaggle SPLxUTSPAN 2026):** predict three continuous landing outcomes — **angle, depth,
  left/right** — from markerless motion-capture body keypoints. Metric: MSE on MinMax-scaled targets.
- **Data:** 2025-12-18 session = 458 free throws, 5 shooters, 60 fps (the competition set; the CSV stores
  the full 240-frame time series per channel, so all entrants had the full trajectory). 2024-08-28 = 125
  shots, P0001 only, 30 fps, no fingers (OOD probe only).
- **Leakage control:** the ball is tracked through the flight and encodes the answer → **excluded from
  all features**. Features are body keypoints only. The ball is used *only* as a training-time teacher
  in the v3 privileged-information experiment (never at test).

## 2. Methods (TRIPOD-style summary)
- **Pre-registration & hold-out lock** (`reports/preregistration.md`): 20% stratified-by-player test set
  (365 dev / 93 test) sealed before modeling; confirmatory claims fixed in advance; H1–H5 declared
  exploratory.
- **Release detection** (leakage-free): handedness + body-only **elbow-extension-snap** detector,
  validated vs ball separation (median **1-frame** error, 96% ≤3). Keypoints Hampel-despiked.
- **Features:** (v1) 37 body-only scalar kinematics around release; (v3) a **multi-frame** body
  launch-state estimator (windowed hand/forearm fit, not single-frame) + ball-derived teacher state;
  (v4) **kinetic-chain** force-transfer features (proximal→distal peak-speed sequencing,
  summation-of-speed, COM forward thrust) and **multi-temporal** sampling at fixed pre-release offsets.
  All fps-correct, 0% missing.
- **Two CV schemes:** **A** = stratified shot-level (same shooters, leaderboard-comparable, 5×5);
  **B** = leave-one-player-out (new shooter). All transforms fit train-fold-only.
- **Models:** baseline ladder (global → per-player mean → player-id), Ridge/Lasso/HGB/RF, a
  differentiable ballistic decoder (v2/v3), and per-target stacked blends (v3). 3 single-target regressors.
- **Metrics:** RMSE/MAE, R², Spearman, scaled-MSE, **skill vs per-player-mean**, bootstrap CIs; SHAP +
  statsmodels MixedLM (ICC) with FDR; clustering with a permutation null.

## 3. Results

### 3.1 Targets & the noise floor (a v2 claim, corrected in v3)
Targets are near-uncorrelated (|r|≤0.14) → three separate problems. Severe heteroscedasticity: miss/made
variance ratio **6.4× (depth)**, 4.2× (L/R), 1.3× (angle) — error is dominated by misses.
**Correction:** v2 reported that even the ball reconstructs depth only at corr 0.25 and concluded depth
is irreducibly noisy. That was a **fit artifact** — the ball is the *noisiest* tracked object (cameras
were optimized for the body), not "near-perfect information." A clean gravity-parabola fit to the
airborne arc (residual 0.08 ft) recovers the official targets far better: **angle 0.57, depth 0.41,
left/right 0.54**. The floor moved when the estimator improved, so it was never a floor — depth is
recoverable in principle; it is not noise.

### 3.2 Prediction (the headline)
| | Scheme A skill | **Scheme B skill (new shooter)** | Hold-out skill vs global [CI] |
|---|---|---|---|
| **angle** | ~0 (ties player-mean) | **+0.29 / +0.34** | **+0.69 [0.59, 0.80]** ✅ |
| depth | ~0 | −0.5 to −2.0 | +0.12 [−0.08, 0.27] |
| left_right | ~0 | −0.25 to −0.4 | −0.00 [−0.19, 0.12] |

- **Identity dominates:** feature models only tie the per-player-mean baseline under Scheme A; clustering
  recovers players exactly (ARI 1.00, null p95 0.007).
- **Only entry angle transfers** to unseen shooters; depth & left/right show no generalizable signal,
  confirmed on the sealed hold-out.

### 3.3 Which movements matter — and why depth/LR don't transfer
Entry-angle **ICC = 0.78** (a stable shooter trait); robust drivers (SHAP ∩ FDR mixed-model): **peak
wrist speed** and **knee extension at release** — leg drive + hand speed set launch direction.

The v4 physics features sharpen the mechanism per target and produce a clean **double dissociation**
(same-shooter, HGB): **kinetic-chain features help depth** (R² −0.02 → +0.03), **multi-temporal features
help left/right** (skill +0.114 → +0.136, CI excludes 0) — each family helps the target its physics
predicts. SHAP is physics-dominated (67% of depth importance), and "force, not position" is confirmed:
ankle→knee timing (`lag_ankle_knee_ms`, #1) and COM forward thrust are top depth drivers.

**But the mechanisms are per-player.** Across all 5 shooters, **zero features are shared** among their
top-3 drivers for any target; depth's between-player residual share *rises* when physics is added
(0.155 → 0.234) — the force features are per-player signatures. P0001's depth is driven by COM forward
velocity, P0003's by shoulder/knee speed, P0005's by elbow extension: *different segments, same target*.
With 5 shooters there is **no shared mechanism to transfer** — which is why correct physics, pooled
across shooters, leaves depth/LR transfer negative or worse (best Scheme B skill −0.49 / −0.19).

### 3.4 The physics arc (v2 → v3 → v4)
- **v2 (ballistic diagnosis):** an analytic projectile decoder showed launch *speed* is unobservable at
  60 fps (measured ~9 ft/s vs true ~22.7) and direction-driven targets (angle) are observable while
  speed-driven depth is not. *Corrected later:* speed magnitude is ~constant (CV 4.7%) but
  corr(speed, depth) is **+0.275**, not the asserted ~0.05; depth is a *joint* trajectory function, not a
  speed problem.
- **v3 (corrected floor + same-player ML):** with the floor corrected, a privileged-information ballistic
  decoder (ball as a training-only teacher) **adds nothing** — the body cannot observe the launch state
  (speed corr −0.34). The competition gain came instead from honest ML on the *legal same-player split*.
- **v4 (correct physics):** kinetic-chain + temporal features are the right *explanation* (§3.3) but a
  poor *transfer lever* — the per-player-mechanism finding is the conclusion of the study.

### 3.5 Competition split (same-player) and reading the leaderboard honestly
The official split is a random split of the **same 5 players** (345/113), so player identity is a legal,
powerful signal. A per-target NNLS-stacked blend (RandomForest on player-centered residuals + per-player
HGB) reaches **0.010901** scaled-MSE — beating our prior **0.01164** — driven mostly by left/right.
Leakage audited (PASS). This is **1.78×** the winner's 0.006148, but the two numbers are not the same
kind of estimate: the winner's score is the **best of ~2500 submissions** against a ~50-shot public set
(an order statistic, biased low), whereas ours is a single sealed-hold-out estimate. We deprioritized
scaled-MSE throughout: it is weak and miss-dominated on 113 shots.

### 3.6 Does the full time series help? (No — shooters are the bottleneck)
**MiniRocket** on the full release-aligned series ties 37 scalars under Scheme A (0.0118 vs 0.0121) and
**overfits identity badly** under Scheme B (~2× worse than the per-shooter mean). A 10k-feature model
memorizes who is shooting. **The binding constraint is shooters (n=5 / 345 shots), not model class or
feature richness** — a data-hungry transformer would do worse, not better (confirmed empirically).

### 3.7 Form, miss modes, OOD (brief)
Tighter form tracks higher make rate (corr −0.42, n=5, directional). Misses cluster into 4 modes (right;
long-left; short-flat-left; high-short) with **depth the dominant miss axis**. The frozen 2025 model runs
on 30 fps 2024 P0001 data (angle RMSE 2.9°) but doesn't beat that shooter's own mean.

## 4. Answers to the challenge's questions
1. **Most consistent joint movements in good shooters?** The angle-determining mechanics — **wrist speed
   and knee extension at release**; tighter overall form tracks higher make rate.
2. **How does release affect entry angle?** Strongly — release mechanics predict entry angle out-of-shooter
   (skill +0.3–0.7). Entry angle is the one outcome body motion reliably explains.
3. **Predict the shot before the ball leaves the hand?** **For entry angle, yes.** For depth & left/right,
   **no** — they are governed by per-shooter kinetics with no shared, transferable signature.
4. **Biomechanical patterns → repeatable form?** Compact, low-dispersion mechanics associate with higher
   accuracy; depth/LR scatter is per-player force/timing, the hard, largely un-poolable part.

## 5. What we learned
**Science:** each target has a distinct, legible mechanism (angle←direction, depth←kinetic-chain force,
LR←late wrist geometry); the mechanisms are real but **per-player**, so only angle transfers; the binding
constraint is the number of shooters, not models or features.
**Method:** a plausible, internally-consistent narrative can still be wrong (the noise-floor artifact; a
correlation off by ~5×) — recompute load-bearing numbers and check premises aren't inverted; use the
physics that matches where the information is (kinetic-chain, not ballistic); don't hunt for one lever
(the winner stacked five small validated gains); match effort to the metric and read leaderboard scores
as order statistics, not unbiased estimates.

## 6. Limitations
- **n=5 shooters:** Scheme B, ICC, and the per-player-mechanism claim are directional/strong-evidence,
  not proof; between-shooter and form-vs-accuracy claims are anecdotal.
- Provider targets are "theoretical" values; interpretation is associational, not causal; heteroscedastic
  depth inflates RMSE; within-session serial dependence is modeled only via a sensitivity split.

## 7. Reproducibility
Fixed seed (1561737); tagged history `v1.0`–`v4.0`; `fta/` modules + `scripts/` (incl. `scripts/v3`,
`scripts/v4`); artifacts regenerate into `outputs/` (gitignored). Hold-out scored once; competition test
touched once.
