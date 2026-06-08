# Predicting Free-Throw Landing Outcomes from Body Motion
### A rigorous, retrospective study on the SPLxUTSPAN 2026 data

**Bottom line.** From a shooter's body keypoints alone, **entry angle is predictable and the signal is
transferable to unseen shooters** (hold-out skill vs global mean **+0.69, 95% CI [0.59, 0.80]**;
RMSE 3.1° vs 5.6°). **Depth and left/right are essentially not predictable** from body motion
(hold-out skill CIs span 0). Much of what looks like "skill" is **shooter identity**: each athlete's
motion is so distinctive that unsupervised clustering recovers all five players perfectly (ARI 1.00),
and feature models only *tie* a "predict this shooter's average" baseline.

**v4 update — the "correct physics" did not move the transfer wall (`reports/findings_v4.md`).**
We implemented the competition winner's principles without leakage — kinetic chain (ankle→knee→hip→…→
fingertip peak-speed sequencing, summation-of-speed ratios, COM forward thrust) and multi-temporal
sampling of the depth/angle/left_right levers at fixed pre-release offsets — and tested the falsifiable
claim that *force-flow / timing* features generalize across shooters where v1's *position* features did
not. **Refuted for depth and left_right:** under leave-one-player-out (Scheme B), best transfer skill vs
per-player-mean is **−0.49** (depth) and **−0.19** (left_right), CIs entirely below 0; adding physics to
Ridge made transfer far worse. The lone gain is **angle +0.29 → +0.43** (multi-temporal). SHAP confirms
the physics is the right *explanation* (COM forward thrust + ankle→knee timing are top depth drivers,
"force not position"), but ICC shows the depth-bearing kinetics are themselves per-player signatures
(between-player residual share rose 0.155 → 0.234), so with 5 shooters there is no shared mechanism to
transfer. Physics told us *why* the wall exists; it did not move it.

---

## 1. Task & data
- **Goal (Kaggle SPLxUTSPAN 2026):** predict three continuous landing outcomes — **angle, depth,
  left/right** — from markerless motion-capture body keypoints. Metric: MSE on MinMax-scaled targets.
- **Data:** 2025-12-18 session = 458 free throws, 5 shooters, 60 fps (the competition set). 2024-08-28
  = 125 shots, P0001 only, 30 fps, no finger tracking (used as OOD probe only).
- **Leakage control:** the ball is tracked through nearly the whole flight and so encodes the answer —
  **excluded from all features**, along with the outcomes. Features are body keypoints only.

## 2. Methods (TRIPOD-style summary)
- **Pre-registration & hold-out lock** (`reports/preregistration.md`): 20% stratified-by-player test set
  (`splits/holdout_split.json`, 365 dev / 93 test) sealed before modeling; H1–H5 declared exploratory
  (data had been inspected); confirmatory claims C1/C3 fixed in advance.
- **Release detection** (leakage-free): per-shot handedness + body-only **elbow-extension-snap** detector,
  validated vs ball separation (median **1-frame** error, 96% within ≤3). Keypoints despiked (Hampel)
  for tracking glitches.
- **Features:** 28 body-only scalar kinematics around release (release pose, joint angles/ROM, velocities,
  tempo, release geometry in a per-shot body frame, asymmetry, smoothness). Real-time derivatives
  (fps-correct). 0% missing.
- **Two CV schemes:** **A** = stratified shot-level (same shooters, leaderboard-comparable, 5×5 repeats);
  **B** = leave-one-player-out (generalization to a new shooter). All transforms fit train-fold-only.
- **Models:** baseline ladder (global mean → per-player mean → player-id) then Ridge/Lasso/ElasticNet
  and HistGradientBoosting; 3 single-target regressors.
- **Metrics:** native-unit RMSE/MAE, R², Spearman, scaled-MSE, and **skill vs per-player-mean**, with
  mean±SD across seeds and **bootstrap CIs**. Interpretation via SHAP + statsmodels MixedLM (ICC) with
  FDR; descriptive clustering with a permutation null.

## 3. Results

### 3.1 Targets & noise floor
Targets are near-uncorrelated (|r|≤0.14) → three separate problems. Severe heteroscedasticity: miss/made
variance ratio **6.4× (depth)**, 4.2× (L/R), 1.3× (angle) — errors dominated by misses (Fig 4). The ball
is too noisy (~4 in fit residual) to reconstruct the targets by physics (angle corr 0.42, depth 0.25), so
a tight irreducible floor isn't estimable; we benchmark empirically.

### 3.2 Prediction (the headline)
| | Scheme A skill | **Scheme B skill (new shooter)** | Hold-out skill vs global [CI] |
|---|---|---|---|
| **angle** | ~0 (ties player-mean) | **+0.29 / +0.34** | **+0.69 [0.59, 0.80]** ✅ |
| depth | ~0 | −0.5 to −2.0 | +0.12 [−0.08, 0.27] |
| left_right | ~0 | −0.25 to −0.4 | −0.00 [−0.19, 0.12] |

- **Identity dominates** (Fig 1, Fig 3): feature models only tie the per-player-mean baseline under
  Scheme A; clustering recovers players exactly (ARI 1.00, null p95 0.007).
- **Only entry angle transfers** to unseen shooters (Scheme B skill > 0; hold-out CI excludes 0).
- Depth & left/right show **no generalizable signal** — confirmed on the sealed hold-out.

### 3.3 Which movements matter (Fig 2)
Entry-angle **ICC = 0.78** (mostly a stable shooter trait). Robust drivers (SHAP ∩ FDR-significant
mixed-model): **peak wrist speed** and **knee extension at release** — leg drive + hand speed set launch,
hence entry angle. Depth ICC 0.08 and L/R ICC 0.02 with ~no significant features → not mechanically
readable.

### 3.4 Form consistency & miss modes
- **Tighter form → better shooting** (corr −0.42 form-dispersion vs make rate; best shooter P0003 has the
  most compact form). n=5 → anecdotal but directional.
- **Miss taxonomy** (4 modes): right; long-and-left; short-flat-left; high-and-short — depth is the
  dominant miss axis.

### 3.5 OOD probe (2024 P0001, 30 fps, no fingers)
Frozen 2025 model runs cleanly on 30 fps data (angle RMSE 2.9°), but doesn't beat P0001's own mean —
again, a consistent shooter's personal average is hard to beat.

## 4. Answers to the newsletter's questions
1. **Most consistent joint movements in good shooters?** The angle-determining mechanics — **wrist speed
   and knee extension at release** — are the consistent, predictable part; tighter overall form tracks
   higher make rate.
2. **How does release affect entry angle?** Strongly: release mechanics predict entry angle out-of-shooter
   (skill +0.3–0.7). Entry angle is the one outcome body motion reliably explains.
3. **Predict the shot before the ball leaves the hand?** **For entry angle, yes** (from pre/at-release
   body features). **For depth & left/right, no** — they aren't determined by readable body kinematics.
4. **Biomechanical patterns → repeatable form?** Compact, low-dispersion mechanics associate with higher
   accuracy; depth/L-R scatter is the hard, largely unmodeled part.

## 4b. Does opening up the full time series help? (No — data is the bottleneck)
We tested whether richer representation closes the gap, using **MiniRocket** (random convolutional
kernels + RidgeCV) on the **full release-aligned multivariate series** (all ~70 keypoints, mid-hip
centered, left-handers mirrored) — train-split CV only, identical folds to the scalar model.

| model | Scheme A (same shooters) | Scheme B (new shooter) |
|---|---|---|
| MiniRocket (full series) | 0.01177 | 0.0416 |
| 37 scalar features (ridge) | 0.01212 | 0.0490 |
| per-player-mean baseline | 0.01254 | **0.0227** |

- Full series beats 37 scalars by only ~3% (within noise) and barely beats the per-shooter mean.
- **Both models overfit shooter identity badly** — leave-one-player-out is ~2× *worse* than predicting
  the mean. With 5 shooters, a 10k-feature model memorizes who is shooting.
- **Conclusion: the binding constraint is data (5 players / 345 shots), not model class or feature
  richness.** This also implies a data-hungry transformer (e.g. PatchTST) would do worse here, not
  better — confirmed empirically, not just argued. Added fingertip launch-vector + hoop-relative aim
  features rescued left/right (R² −0.14 → +0.15) and brought our official-split error to ~0.0116
  (~1.9× the winner's 0.006136); the remaining gap is concentrated in depth and is not closed by
  full-series modeling.

## 5. Limitations / threats to validity
- **n=5 shooters:** Scheme B and ICC are directional, not precise; between-shooter claims are anecdotal.
- **Targets carry measurement noise** (provider "theoretical" values; ball noisy) → a ceiling on depth/LR
  predictability we cannot fully separate from model limitation.
- Heteroscedastic depth inflates RMSE; within-session serial dependence un-modeled beyond a sensitivity
  split. Interpretation is associational, not causal.

## 6. Reproducibility
Fixed seed (1561737); `git` history per step; `fta/` modules + `scripts/`; artifacts in `outputs/`
(features, experiments, interpretation, clustering, final_eval, figures). Hold-out scored once.
