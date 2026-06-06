# Steps 4-5 Findings — Baselines, Dual CV, Models (DEV, n=365)

Metric = official scaled-MSE (lower better) + skill score vs the **per-player-mean** baseline
(>0 means beats "just predict this shooter's average"). Scheme A = stratified shot-level (same
shooters, leaderboard-comparable); Scheme B = leave-one-player-out (new shooter). Linear models on
28 body features, StandardScaler, 5×5 repeated folds for A.

## Baseline ladder (scaled-MSE)
| model | A: sMSE | A: skill | B: sMSE | B: skill |
|---|---|---|---|---|
| B0 global mean | 0.0180 | −0.93 | 0.0216 | 0 |
| **B1 per-player mean** | **0.0118** | 0 | 0.0216 | 0 |
| B2 player one-hot | 0.0118 | 0 | 0.0216 | 0 |
| ridge (features) | 0.0120 | −0.02 | 0.0312 | −0.72 |
| hgb (features) | 0.0131 | −0.12 | 0.0217 | −0.14 |
| ridge feat + player id | 0.0118 | −0.01 | 0.0347 | −0.93 |
| ridge within-player | 0.0118 | 0.00 | 0.0372 | −1.01 |

## Headline results
1. **Identity dominates (Scheme A).** Body-feature models only **tie** the per-player-mean baseline
   (skill ≈ 0). On same-shooter data, knowing *who* shoots is as predictive as their measured motion.
2. **Only entry angle transfers to new shooters (Scheme B).** Per-target skill vs global mean:

   | target | A skill (ridge) | **B skill (ridge / hgb)** | B RMSE |
   |---|---|---|---|
   | angle | −0.07 | **+0.29 / +0.34** | ~4.5–4.7° |
   | depth | +0.00 | −2.03 / −0.54 | 6.6–9.2 in |
   | left_right | +0.01 | −0.43 / −0.25 | 4.3–4.6 in |

   Body kinematics predict a **new** shooter's entry angle (transferable, physically sensible:
   release mechanics → launch → entry angle). Depth & left/right **do not transfer** (negative skill).
3. **The A–B gap** for angle (RMSE 2.5° → 4.5°) quantifies how much apparent accuracy is shooter-specific
   memorization vs transferable mechanics — but angle still beats the global mean out-of-shooter.
4. **HGB generalizes best** out-of-shooter (angle B skill +0.34); linear models overfit shooter-specific
   structure (worse B skill).

## Interpretation (pre-registered claims)
- **C1** (features beat global mean for angle & depth): **angle YES** (esp. Scheme B); depth/LR ~tie in A,
  fail in B → partially supported.
- **C3** (transferable/within-shooter movement signal): **supported for angle only** (Scheme B skill > 0);
  depth/LR show no transferable signal — the key honest negative result.
- Consistent with the noise floor: even the (noisy) ball poorly reconstructs depth/LR, so body motion
  predicting them precisely was unlikely. Much depth/LR variance appears to be noise/unmodeled factors.

## Caveats
- n=5 players ⇒ Scheme B has 5 folds; treat magnitudes as directional (per-player breakdown in
  outputs/experiments.json). Final claims await the locked hold-out (Step 7).
- Heteroscedasticity (depth miss/made var 6.4×) inflates depth RMSE; report stratified at write-up.
