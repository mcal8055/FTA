"""Steps 4-5 — baseline ladder + models under both CV schemes (DEV only).

Reports the official scaled-MSE plus the native-unit metric suite with mean±SD across
Scheme-A seeds, the Scheme A vs B gap, and skill score vs the per-player-mean baseline.
Writes outputs/experiments.json. Hold-out test is NOT touched here.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from fta.config import OUTPUTS_DIR, TARGETS
from fta.models import (evaluate_scheme_a, evaluate_scheme_b, make_feature_predictor,
                        model_factories, pred_global_mean, pred_player_mean,
                        pred_player_onehot)

NON_FEATURES = {"shot_id", "player", "split", "made", *TARGETS}


def agg_a(runs):
    """Mean±SD across seeds of overall scaled_mse + mean_skill, and per-target rmse."""
    smse = [r["overall"]["scaled_mse"] for r in runs]
    skill = [r["overall"].get("mean_skill", np.nan) for r in runs]
    out = {"scaled_mse_mean": float(np.mean(smse)), "scaled_mse_sd": float(np.std(smse)),
           "mean_skill_mean": float(np.nanmean(skill)), "mean_skill_sd": float(np.nanstd(skill))}
    for t in TARGETS:
        rmse = [r[t]["rmse"] for r in runs]
        sk = [r[t].get("skill", np.nan) for r in runs]
        out[t] = {"rmse_mean": float(np.mean(rmse)), "rmse_sd": float(np.std(rmse)),
                  "skill_mean": float(np.nanmean(sk))}
    return out


def main():
    df = pd.read_parquet(OUTPUTS_DIR / "features_2025.parquet")
    dev = df[df.split == "dev"].reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in NON_FEATURES]
    print(f"dev shots={len(dev)} features={len(feature_cols)}")

    facto = model_factories()
    # baseline ladder + models. (predictor, label)
    ladder = [
        ("B0_global_mean", pred_global_mean),
        ("B1_player_mean", pred_player_mean),
        ("B2_player_onehot", pred_player_onehot),
        ("M_ridge_feat", make_feature_predictor(facto["ridge"])),
        ("M_lasso_feat", make_feature_predictor(facto["lasso"])),
        ("M_enet_feat", make_feature_predictor(facto["elasticnet"])),
        ("M_hgb_feat", make_feature_predictor(facto["hgb"])),
        ("M_ridge_feat+id", make_feature_predictor(facto["ridge"], use_player=True)),
        ("M_ridge_within_player", make_feature_predictor(facto["ridge"], center_by_player=True)),
    ]

    results = {}
    print(f"\n{'model':26s}{'A:sMSE':>10s}{'A:skill':>9s}{'B:sMSE':>10s}{'B:skill':>9s}")
    for name, pred in ladder:
        a = agg_a(evaluate_scheme_a(dev, feature_cols, pred))
        b_overall, b_pp = evaluate_scheme_b(dev, feature_cols, pred)
        results[name] = {
            "scheme_a": a,
            "scheme_b": {"scaled_mse": b_overall["overall"]["scaled_mse"],
                         "mean_skill": b_overall["overall"].get("mean_skill"),
                         "per_target": {t: {"rmse": b_overall[t]["rmse"],
                                            "skill": b_overall[t].get("skill")} for t in TARGETS},
                         "per_player_skill": {p: b_pp[p]["overall"].get("mean_skill")
                                              for p in b_pp}},
        }
        print(f"{name:26s}{a['scaled_mse_mean']:10.4f}{a['mean_skill_mean']:9.3f}"
              f"{b_overall['overall']['scaled_mse']:10.4f}"
              f"{b_overall['overall'].get('mean_skill', float('nan')):9.3f}")

    # headline: best feature model A-B gap, per-target detail
    print("\nPer-target RMSE (native units) — Scheme A mean±SD | Scheme B:")
    for name in ("M_ridge_feat", "M_hgb_feat", "M_ridge_feat+id", "M_ridge_within_player"):
        a = results[name]["scheme_a"]; b = results[name]["scheme_b"]["per_target"]
        print(f"  {name}")
        for t in TARGETS:
            print(f"    {t:11s} A={a[t]['rmse_mean']:.2f}±{a[t]['rmse_sd']:.2f} "
                  f"(skill {a[t]['skill_mean']:+.3f})   B={b[t]['rmse']:.2f} (skill {b[t]['skill']:+.3f})")

    OUTPUTS_DIR.mkdir(exist_ok=True)
    with open(OUTPUTS_DIR / "experiments.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nWrote {OUTPUTS_DIR / 'experiments.json'}")


if __name__ == "__main__":
    main()
