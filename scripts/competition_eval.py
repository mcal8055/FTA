"""Apples-to-apples: our model on the OFFICIAL competition train/test split.

The competition CSV is the same 2025 body-only time series as our JSON (verified:
train rows map 1-1 to JSON shots by exact targets). So we attach the competition
split to our existing features and evaluate on the official 113-shot test set with the
official MinMax-scaled MSE. Compares to the winner's 0.006136.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from fta.config import OUTPUTS_DIR, TARGETS, TARGET_UNITS
from fta.loader import list_shots, load_metadata
from fta.metric import metric_suite, scaled_mse, scaled_mse_per_target
from fta.models import (make_feature_predictor, model_factories, pred_global_mean,
                        pred_player_mean)

WINNER = 0.006136
NON = {"shot_id", "player", "split", "csplit", "made", *TARGETS}


def main():
    # competition split: train = target-matched 345; test = complement 113
    tgt_key = {}
    for s in list_shots():
        m = load_metadata(s.path)
        tgt_key[(round(m["angle"], 2), round(m["depth"], 2), round(m["left_right"], 2))] = s.shot_id
    tr = pd.read_csv("spl-utspan-data-challenge-2026/train.csv")
    train_ids = {tgt_key[(round(r.angle, 2), round(r.depth, 2), round(r.left_right, 2))]
                 for _, r in tr.iterrows()}

    df = pd.read_parquet(OUTPUTS_DIR / "features_2025.parquet")
    df["csplit"] = np.where(df.shot_id.isin(train_ids), "train", "test")
    trn = df[df.csplit == "train"].reset_index(drop=True)
    ten = df[df.csplit == "test"].reset_index(drop=True)
    feats = [c for c in df.columns if c not in NON]
    print(f"OFFICIAL competition split: train={len(trn)} test={len(ten)}  features={len(feats)}")

    yt = {t: ten[t].to_numpy(float) for t in TARGETS}
    facto = model_factories()
    preds = {
        "global_mean": pred_global_mean(trn, ten, feats),
        "player_mean": pred_player_mean(trn, ten, feats),
        "ridge": make_feature_predictor(facto["ridge"])(trn, ten, feats),
        "hgb": make_feature_predictor(facto["hgb"])(trn, ten, feats),
    }

    print(f"\n{'model':14s}{'sMSE':>10s}   per-target scaled-MSE (ang/dep/lr)")
    results = {}
    for n, yp in preds.items():
        sm = scaled_mse(yt, yp)
        per = scaled_mse_per_target(yt, yp)
        results[n] = {"scaled_mse": sm, "per_target_scaled": per}
        print(f"{n:14s}{sm:10.6f}   " + " ".join(f"{per[t]:.4f}" for t in TARGETS))

    best = min(results, key=lambda k: results[k]["scaled_mse"])
    print(f"\nbest = {best}: {results[best]['scaled_mse']:.6f}   winner = {WINNER:.6f}  "
          f"(ratio {results[best]['scaled_mse']/WINNER:.2f}x)")

    print("\nNative-unit RMSE on official test (HGB):")
    ms = metric_suite(yt, preds["hgb"])
    for t in TARGETS:
        print(f"  {t:11s} RMSE={ms[t]['rmse']:.2f}{TARGET_UNITS[t]}  R2={ms[t]['r2']:+.3f}")

    json.dump({"winner": WINNER, "results": results}, open(OUTPUTS_DIR / "competition_eval.json", "w"), indent=2)
    print(f"\nWrote {OUTPUTS_DIR / 'competition_eval.json'}")


if __name__ == "__main__":
    main()
