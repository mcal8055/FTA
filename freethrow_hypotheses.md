# Free-Throw Dataset — Hypothesis Generation

*Source: SPL Open-Data `basketball/freethrow` · 583 shots · 6 player-sessions · ball + 67 body markers tracked in 3D*
*Method: hypothesis-generation skill framework (observation-grounded → competing hypotheses → predictions → tests → falsification → quality check)*

---

## 1. The phenomenon

Free-throw outcome (made/missed) varies substantially in this dataset, and we want to explain **what drives a make vs. a miss** — using both the shot-level metadata (entry angle, landing location) and the motion-capture layer (release biomechanics).

## 2. Grounding observations (from real metadata only — nothing derived/guessed)

| Observation | Value |
|---|---|
| Overall make rate | **66.7%** (389/583) |
| Per-player make rate range | **43.9% (P0005) → 90.0% (P0003)** |
| P0001 across two sessions (16 mo apart) | 70.4% (2024) vs 69.3% (2025) — **stable** |
| Entry angle, made vs missed (mean) | 45.7° vs 44.2° |
| Make rate by entry-angle bin | rises 46% (38–40°) → ~76–80% (44–50°), softens after |
| Landing dispersion, made vs missed | x SD **2.58 vs 5.50**, y SD **3.03 vs 7.60** (makes far tighter) |
| Landing y, made vs missed (mean) | 10.1 vs 8.8 |

**⚠️ Data-quality flag (must handle before any biomechanics work):** sampling rate is **not uniform** — 458 shots at 60 Hz, 125 at 30 Hz. The 30 Hz shots are *exactly* the 2024-08-28 / P0001 session. All files have a fixed 240 frames, so that session spans ~8 s of real time while the others span ~4 s. Any kinematic feature (velocity, smoothness, release timing) computed naively will be biased across sessions. Treat sampling rate as a confound / resample before comparing.

---

## 3. Competing & complementary hypotheses

### H1 — Entry-angle optimality (inverted-U)
**Mechanism:** The rim presents a larger *effective aperture* to the ball within an optimal entry-angle band; too flat and the ball meets the front/back rim, too steep and horizontal-error tolerance shrinks. Make probability should be a **concave function of entry angle**, peaking around ~45–50°.
**Evidence:** Binned make rate climbs from ~46% to ~76–80% then softens.
**Prediction:** Logistic regression of `made ~ entry_angle + entry_angle²` yields a **negative quadratic term**; fitted optimum in 45–50°.
**Falsified if:** relationship is flat or monotonic (no significant quadratic term).

### H2 — Precision beats accuracy (landing variance)
**Mechanism:** Skilled shooting is dominated by **low trial-to-trial variance**, not a different average aim point. Makes should cluster tightly around the rim center; misses scatter.
**Evidence:** Landing SD for makes is roughly **half** that of misses on both axes.
**Prediction:** Per-player landing-location **variance** correlates with miss rate more strongly than per-player mean landing offset does.
**Falsified if:** mean offset explains misses but variance does not.

### H3 — Shooter identity dominates (large, stable individual differences)
**Mechanism:** Stable technique/skill differences across people set a baseline make rate that swamps within-session shot-to-shot variation.
**Evidence:** 44%→90% spread across players; P0001 nearly identical across two sessions 16 months apart.
**Prediction:** A mixed-effects model shows a large **between-player random-effect variance**; player ID is the strongest single predictor.
**Falsified if:** between-player variance is small once entry angle / landing are controlled.

### H4 — Release-mechanics signature (needs the tracking layer)
**Mechanism:** Consistent kinematics of the shooting arm (elbow/wrist path, release height, motion smoothness) produce consistent ball launch → makes. Makes should show **tighter, more repeatable** release kinematics than misses, within a player.
**Evidence:** Not yet tested — requires validated release-event detection on the motion capture (deliberately *not* faked here).
**Prediction:** Within-player, variance of release-window joint trajectories is lower on makes than misses.
**Falsified if:** release kinematics are indistinguishable between makes and misses within player.
**Prerequisite:** resolve the 30/60 Hz confound; build a defensible release-detection method first.

### H5 — Angle effect is confounded by shooter (Simpson's-paradox guard)
**Mechanism:** The apparent entry-angle→make relationship (H1) could be **between-shooter** — better shooters happen to favor different angles — rather than a within-shooter causal effect.
**Prediction:** The H1 quadratic effect **persists within players** (player-centered entry angle), not only across them.
**Falsified if:** the angle effect vanishes once player is controlled → H1 is an artifact of H3.

---

## 4. Suggested priority & test order

1. **H1 + H5 together** — cheap, metadata-only; logistic regression with player controls. Settles whether entry angle is a real lever.
2. **H2** — metadata-only; per-player variance vs make rate.
3. **H3** — metadata-only; mixed model, quantify how much outcome is "who's shooting."
4. **H4** — the deep one; only after the tracking-quality EDA and a validated release detector (and after fixing the sampling-rate confound).

## 5. Quality check (skill criteria)

All five are **testable** and **falsifiable** with this dataset. H1/H2/H3/H5 need only the clean metadata (no fabrication risk). H4 is the highest-value but highest-effort, gated on real biomechanics processing. H5 exists specifically to keep H1 honest.

## 6. Notes / deviations from the skill

- The skill mandates AI-generated schematics via a `scientific-schematics` skill that isn't installed, and a 50+ citation LaTeX report. Both are inappropriate for an internal data-exploration task, so I produced grounded markdown instead. I can render figures (matplotlib: angle→make curve, landing scatter by outcome, per-player bars) on request.
