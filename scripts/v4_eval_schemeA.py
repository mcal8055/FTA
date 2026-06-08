"""v4 Scheme A evaluation: do kinetic-chain / temporal physics features improve
the metrics that measure UNDERSTANDING (RMSE, R2, Spearman, skill vs per-player-mean)
under stratified same-shooter CV?

Four feature sets merged on shot_id:
  base        = features_2025.parquet (37 v1 features)
  base+kinetic
  base+temporal
  base+both

Models: Ridge and HGB (project factories). Per-target standardization + fitting
strictly train-only within each fold (handled by make_feature_predictor pipeline).
Repeated stratified Scheme A (5 seeds x 5 folds). Full metric_suite + bootstrap CI
on the per-target skill score vs per-player-mean.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from fta.config import OUTPUTS_DIR, TARGETS
from fta.cv import scheme_a_folds
from fta.metric import metric_suite
from fta.models import make_feature_predictor, model_factories, _run_folds

NON_FEATURES = {"shot_id", "player", "split", "made", "hand_right", *TARGETS}
SEEDS = range(5)
N_SPLITS = 5


def build_sets():
    base = pd.read_parquet(OUTPUTS_DIR / "features_2025.parquet")
    kin = pd.read_parquet(OUTPUTS_DIR / "v4_kinetic.parquet")
    tmp = pd.read_parquet(OUTPUTS_DIR / "v4_temporal.parquet")

    base_feats = [c for c in base.columns if c not in NON_FEATURES]
    kin_feats = [c for c in kin.columns if c not in NON_FEATURES and c not in base.columns]
    tmp_feats = [c for c in tmp.columns if c not in NON_FEATURES and c not in base.columns]

    # Merge physics-only columns onto base by shot_id (keeps base meta: split, player, targets)
    merged = base.merge(kin[["shot_id", *kin_feats]], on="shot_id", how="inner")
    merged = merged.merge(tmp[["shot_id", *tmp_feats]], on="shot_id", how="inner")
    assert len(merged) == len(base), (len(merged), len(base))

    dev = merged[merged.split == "dev"].reset_index(drop=True)
    sets = {
        "base":          base_feats,
        "base+kinetic":  base_feats + kin_feats,
        "base+temporal": base_feats + tmp_feats,
        "base+both":     base_feats + kin_feats + tmp_feats,
    }
    return dev, sets, {"base": len(base_feats), "kinetic": len(kin_feats),
                       "temporal": len(tmp_feats)}


def eval_set(dev, feature_cols, predictor):
    """Repeated Scheme A. Aggregate per-target metrics mean±SD across seeds.
    Also pool OOF preds from the first seed for a stable bootstrap-CI on skill."""
    per_seed = []
    pooled = None  # (yt, yp, yb) from seed 0 for bootstrap CI
    for seed in SEEDS:
        folds = list(scheme_a_folds(dev, N_SPLITS, seed))
        yt, yp, yb = _run_folds(dev, feature_cols, predictor, folds)
        per_seed.append(metric_suite(yt, yp, baseline=yb))
        if seed == 0:
            pooled = (yt, yp, yb)

    out = {}
    for t in TARGETS:
        rmse = [r[t]["rmse"] for r in per_seed]
        r2 = [r[t]["r2"] for r in per_seed]
        sp = [r[t]["spearman"] for r in per_seed]
        sk = [r[t]["skill"] for r in per_seed]
        smse = [r[t]["scaled_mse"] for r in per_seed]
        out[t] = {
            "rmse": float(np.mean(rmse)), "rmse_sd": float(np.std(rmse)),
            "r2": float(np.mean(r2)), "r2_sd": float(np.std(r2)),
            "spearman": float(np.mean(sp)), "spearman_sd": float(np.std(sp)),
            "skill_vs_playermean": float(np.mean(sk)), "skill_sd": float(np.std(sk)),
            "scaled_mse": float(np.mean(smse)),
        }
    # bootstrap CI on skill (per target) using pooled seed-0 OOF preds.
    # skill = 1 - MSE_model/MSE_baseline; resample shot indices jointly so model and
    # baseline see the same resampled shots, then take the percentile CI of the ratio.
    yt, yp, yb = pooled
    for t in TARGETS:
        rng = np.random.default_rng(0)
        m = len(yt[t]); vals = []
        ytt = np.asarray(yt[t]); ypt = np.asarray(yp[t]); ybt = np.asarray(yb[t])
        for _ in range(2000):
            idx = rng.integers(0, m, m)
            mse_m = np.mean((ypt[idx] - ytt[idx]) ** 2)
            mse_b = np.mean((ybt[idx] - ytt[idx]) ** 2)
            vals.append(1 - mse_m / mse_b if mse_b > 0 else np.nan)
        lo, hi = np.nanpercentile(vals, [2.5, 97.5])
        out[t]["skill_ci"] = [float(lo), float(hi)]
    return out


def main():
    dev, sets, counts = build_sets()
    print(f"dev shots={len(dev)}  feature counts: {counts}")
    facto = model_factories()
    models = {
        "ridge": make_feature_predictor(facto["ridge"]),
        "hgb": make_feature_predictor(facto["hgb"]),
    }

    results = {"meta": {"scheme": "A", "n_dev": len(dev),
                        "seeds": list(SEEDS), "n_splits": N_SPLITS,
                        "feature_counts": counts,
                        "baseline_for_skill": "per_player_mean"}}
    for mname, pred in models.items():
        results[mname] = {}
        for sname, cols in sets.items():
            results[mname][sname] = eval_set(dev, cols, pred)
            r = results[mname][sname]
            print(f"\n[{mname}] {sname} ({len(cols)} feats)")
            for t in TARGETS:
                print(f"  {t:11s} RMSE={r[t]['rmse']:.3f}  R2={r[t]['r2']:+.3f}  "
                      f"rho={r[t]['spearman']:+.3f}  skill={r[t]['skill_vs_playermean']:+.3f} "
                      f"CI[{r[t]['skill_ci'][0]:+.3f},{r[t]['skill_ci'][1]:+.3f}]  "
                      f"sMSE={r[t]['scaled_mse']:.4f}")

    with open(OUTPUTS_DIR / "v4_eval_schemeA.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nWrote {OUTPUTS_DIR / 'v4_eval_schemeA.json'}")


if __name__ == "__main__":
    main()
