# Project Status — checkpoint before physics-informed ML

## Where we are
Complete, rigorous body-only regression study (angle / depth / left_right) on the SPLxUTSPAN
free-throw data. Pipeline in `fta/`, scripts in `scripts/`, artifacts in `outputs/`, write-up in
`reports/REPORT.md`. Hold-out sealed; train-CV used for model selection; no test-set fishing.

## Headline results
- **Angle is predictable & transferable** (hold-out skill vs global mean +0.69 [0.59,0.80]; R²≈0.64).
- **Depth & left/right are the wall.** Fingertip launch-vector + hoop-relative aim features rescued
  left/right (R² −0.14 → +0.15); depth R² ≈ 0.
- **Identity dominates** (clustering recovers all 5 shooters, ARI 1.0; models only tie per-player mean).
- **Apples-to-apples on the official competition split:** our best ≈ **0.0116** sMSE vs **winner 0.006136**
  (~1.9×).
- **Full-series MiniRocket did not beat 37 scalars** (Scheme A 0.0118 vs 0.0121) and both overfit
  identity on LOPO → **bottleneck is data (5 shooters / 345 shots), not model/representation.**
  Implies PatchTST/transformers would do worse here.

## Why physics-informed ML is the next direction
The remaining gap is concentrated in **depth**, which is governed by **launch speed/position** — a
quantity with exact physics: release state (position + velocity) → ballistic trajectory → where the ball
crosses the 10-ft rim plane (rim known: CSV (5.25,-25,10); JSON ~(41.67,0,10)). Instead of regressing
targets directly, a physics-informed model could **predict the launch state from body motion, then apply
the analytic projectile map** as a fixed decoder — injecting the known dynamics as structure. This may
(a) improve depth, (b) reduce overfitting by constraining the hypothesis space (helpful at n=345), and
(c) yield interpretable, mechanistic predictions. Next step: research physics-informed / physics-guided
ML approaches and assess fit for this problem.

## Reproduce
`PYTHONPATH=. python3 scripts/{make_split,target_eda,build_dataset,run_experiments,interpret,cluster_analysis,final_eval,competition_eval,rocket_eval}.py`
