"""Step 7 — FINAL evaluation on the locked hold-out (scored ONCE) + 2024 OOD probe.

Trains final models on 2025 dev, evaluates on the sealed 2025 test set, and reports the
metric suite with bootstrap CIs on skill scores vs both baselines — testing the
pre-registered confirmatory claims C1/C3. Then applies the frozen model to the 2024
P0001 session (different day, 30 fps, no fingers) as an out-of-distribution probe.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

from fta.config import (OUTPUTS_DIR, SESSION_2024, TARGETS, TARGET_UNITS)
from fta.features import extract_features
from fta.loader import list_shots, load_metadata, load_tracking
from fta.metric import metric_suite
from fta.models import (make_feature_predictor, model_factories, pred_global_mean,
                        pred_player_mean)

warnings.filterwarnings("ignore")
NON = {"shot_id", "player", "split", "made", *TARGETS}


def skill_ci(yt, yp, yb, n=2000, seed=0):
    """Bootstrap 95% CI of skill = 1 - MSE_model/MSE_baseline (per target)."""
    rng = np.random.default_rng(seed)
    m = len(yt[TARGETS[0]])
    boot = {t: [] for t in TARGETS}
    for _ in range(n):
        idx = rng.integers(0, m, m)
        for t in TARGETS:
            e = (yp[t][idx] - yt[t][idx]) ** 2
            b = (yb[t][idx] - yt[t][idx]) ** 2
            boot[t].append(1 - e.mean() / b.mean())
    return {t: (round(float(np.percentile(boot[t], 2.5)), 3),
               round(float(np.percentile(boot[t], 97.5)), 3)) for t in TARGETS}


def main():
    df = pd.read_parquet(OUTPUTS_DIR / "features_2025.parquet")
    dev = df[df.split == "dev"].reset_index(drop=True)
    test = df[df.split == "test"].reset_index(drop=True)
    feats = [c for c in df.columns if c not in NON]
    facto = model_factories()
    print(f"FINAL hold-out eval: train dev={len(dev)} -> test={len(test)} (scored once)")

    yt = {t: test[t].to_numpy(float) for t in TARGETS}
    preds = {
        "global_mean": pred_global_mean(dev, test, feats),
        "player_mean": pred_player_mean(dev, test, feats),
        "ridge": make_feature_predictor(facto["ridge"])(dev, test, feats),
        "hgb": make_feature_predictor(facto["hgb"])(dev, test, feats),
    }
    out = {"test": {}, "ood_2024": {}}
    print(f"\n{'model':14s}{'sMSE':>9s}  per-target RMSE (native)")
    for name, yp in preds.items():
        ms = metric_suite(yt, yp, baseline=preds["player_mean"])
        rmse = "  ".join(f"{t[:3]}={ms[t]['rmse']:.2f}{TARGET_UNITS[t]}" for t in TARGETS)
        out["test"][name] = {"scaled_mse": ms["overall"]["scaled_mse"],
                             "per_target": {t: {"rmse": ms[t]["rmse"], "skill_vs_player": ms[t].get("skill")}
                                            for t in TARGETS}}
        print(f"{name:14s}{ms['overall']['scaled_mse']:9.4f}  {rmse}")

    # confirmatory claims: skill of hgb vs global-mean and vs player-mean, with CIs
    print("\nConfirmatory test (HGB) — skill vs baselines [95% bootstrap CI]:")
    for base_name in ("global_mean", "player_mean"):
        ci = skill_ci(yt, preds["hgb"], preds[base_name])
        ms = metric_suite(yt, preds["hgb"], baseline=preds[base_name])
        out["test"][f"hgb_skill_vs_{base_name}"] = {
            t: {"skill": round(ms[t]["skill"], 3), "ci95": ci[t]} for t in TARGETS}
        print(f"  vs {base_name}:")
        for t in TARGETS:
            sig = "" if ci[t][0] <= 0 <= ci[t][1] else "  *CI excludes 0*"
            print(f"    {t:11s} skill={ms[t]['skill']:+.3f}  CI={ci[t]}{sig}")

    # ---- OOD probe: 2024 P0001 ----
    rows = []
    for s in list_shots(SESSION_2024):
        tk = load_tracking(s.path); m = load_metadata(s.path)
        f = extract_features(tk); f["player"] = s.player
        for t in TARGETS:
            f[t] = m[t]
        rows.append(f)
    ood = pd.DataFrame(rows)
    yt_o = {t: ood[t].to_numpy(float) for t in TARGETS}
    ridge_pred = make_feature_predictor(facto["hgb"])(dev, ood, feats)
    # baseline for OOD: P0001's 2025 dev mean (best available prior for same person)
    p1 = dev[dev.player == "P0001"]
    base_o = {t: np.full(len(ood), p1[t].mean()) for t in TARGETS}
    ms_o = metric_suite(yt_o, ridge_pred, baseline=base_o)
    print(f"\nOOD probe 2024 P0001 (n={len(ood)}, 30fps, no fingers) — frozen 2025 HGB:")
    for t in TARGETS:
        print(f"  {t:11s} RMSE={ms_o[t]['rmse']:.2f}{TARGET_UNITS[t]}  "
              f"skill vs P0001-mean={ms_o[t].get('skill'):+.3f}")
        out["ood_2024"][t] = {"rmse": ms_o[t]["rmse"], "skill_vs_p0001_mean": ms_o[t].get("skill")}

    with open(OUTPUTS_DIR / "final_eval.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {OUTPUTS_DIR / 'final_eval.json'}")


if __name__ == "__main__":
    main()
