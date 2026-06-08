"""v4 Scheme-B (leave-one-player-out) evaluation of physics feature families.

Central question: do kinetic-chain and/or temporal physics features move DEPTH and
LEFT_RIGHT *transfer* skill (vs per-player-mean) from v1's NEGATIVE values toward/above 0,
i.e. generalize to UNSEEN shooters, where v1's pose-POSITION features failed?

Four feature sets: base, base+kinetic, base+temporal, base+both.
All standardization / model fitting strictly train-only within each LOPO fold
(handled by fta.models.make_feature_predictor + evaluate_scheme_b).

Two skill baselines per target:
  - skill_vs_playermean : 1 - MSE_model / MSE(per-player train mean)  [the headline transfer test]
  - skill_vs_globalmean : 1 - MSE_model / MSE(global train mean)
Bootstrap 95% CIs on per-target skill (vs player-mean) by resampling shots.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from fta.config import TARGETS, OUTPUTS_DIR
from fta.cv import scheme_b_folds
from fta.metric import metric_suite, bootstrap_ci
from fta.models import (
    make_feature_predictor, model_factories,
    pred_player_mean, pred_global_mean,
)

# hand_right (handedness sign) is a legitimate shooter-invariant feature and was part
# of the v1 37-feature baseline, so we KEEP it in `base`. We drop the per-set duplicate
# copies from the kinetic/temporal frames (handled below) so it appears once.
META = {"shot_id", "player", "split", "made", "angle", "depth", "left_right",
        "participant_id", "trial_id", "result", "handedness"}
KIN_TMP_META = META | {"hand_right"}  # avoid duplicating hand_right across merges


def feat_cols(df):
    return [c for c in df.columns if c not in META]


def load_sets():
    base = pd.read_parquet(OUTPUTS_DIR / "features_2025.parquet")
    kin = pd.read_parquet(OUTPUTS_DIR / "v4_kinetic.parquet")
    tmp = pd.read_parquet(OUTPUTS_DIR / "v4_temporal.parquet")

    bcols = feat_cols(base)  # keeps hand_right (v1 baseline = 37 feats)
    kcols = [c for c in kin.columns if c not in KIN_TMP_META]
    tcols = [c for c in tmp.columns if c not in KIN_TMP_META]

    # merge all feature families onto one frame keyed by shot_id
    key = ["shot_id", "player"] + TARGETS
    merged = base[key + bcols].merge(
        kin[["shot_id"] + kcols], on="shot_id", validate="1:1").merge(
        tmp[["shot_id"] + tcols], on="shot_id", validate="1:1")
    assert len(merged) == len(base) == 458, len(merged)
    assert merged[bcols + kcols + tcols].isna().sum().sum() == 0

    sets = {
        "base": bcols,
        "base+kinetic": bcols + kcols,
        "base+temporal": bcols + tcols,
        "base+both": bcols + kcols + tcols,
    }
    return merged, sets, {"base": len(bcols), "kinetic": len(kcols), "temporal": len(tcols)}


def run_folds(df, cols, predictor):
    """OOF predictions over LOPO folds; also player-mean and global-mean baselines."""
    n = len(df)
    yp = {t: np.full(n, np.nan) for t in TARGETS}
    yb_pm = {t: np.full(n, np.nan) for t in TARGETS}
    yb_gm = {t: np.full(n, np.nan) for t in TARGETS}
    for tr, te, _ in scheme_b_folds(df):
        train, test = df.iloc[tr], df.iloc[te]
        pr = predictor(train, test, cols)
        bpm = pred_player_mean(train, test, cols)
        bgm = pred_global_mean(train, test, cols)
        for t in TARGETS:
            yp[t][te] = pr[t]
            yb_pm[t][te] = bpm[t]
            yb_gm[t][te] = bgm[t]
    yt = {t: df[t].to_numpy(float) for t in TARGETS}
    return yt, yp, yb_pm, yb_gm


def per_target_record(yt, yp, yb_pm, yb_gm):
    """Full metric suite (vs player-mean) + skill vs global-mean + bootstrap CI on
    skill_vs_playermean per target."""
    suite_pm = metric_suite(yt, yp, baseline=yb_pm)   # 'skill' = skill_vs_playermean
    out = {}
    for t in TARGETS:
        rec = dict(suite_pm[t])
        rec["skill_vs_playermean"] = rec.pop("skill")
        # skill vs global mean
        mse_m = float(np.mean((yp[t] - yt[t]) ** 2))
        mse_gm = float(np.mean((yb_gm[t] - yt[t]) ** 2))
        rec["skill_vs_globalmean"] = 1 - mse_m / mse_gm if mse_gm > 0 else float("nan")

        # bootstrap CI on skill_vs_playermean (resample shots)
        def skill_pm(a, b, _t=t):
            err = np.asarray(b[_t]) - np.asarray(a["yt"][_t])
            base = np.asarray(a["yb"][_t]) - np.asarray(a["yt"][_t])
            mb = np.mean(base ** 2)
            return 1 - np.mean(err ** 2) / mb if mb > 0 else np.nan
        # pack so bootstrap_ci resamples consistent indices
        a_pack = {"_": np.zeros(len(yt[t]))}
        rng = np.random.default_rng(0)
        m = len(yt[t])
        vals = []
        for _ in range(2000):
            idx = rng.integers(0, m, m)
            err = yp[t][idx] - yt[t][idx]
            base = yb_pm[t][idx] - yt[t][idx]
            mb = np.mean(base ** 2)
            vals.append(1 - np.mean(err ** 2) / mb if mb > 0 else np.nan)
        lo, hi = np.nanpercentile(vals, [2.5, 97.5])
        rec["skill_vs_playermean_ci95"] = [float(lo), float(hi)]
        out[t] = rec
    return out, suite_pm["overall"]


def per_player_skill(df, yt, yp, yb_pm):
    pl = df["player"].to_numpy()
    res = {}
    for t in TARGETS:
        d = {}
        for p in sorted(df["player"].unique()):
            mk = pl == p
            mse_m = float(np.mean((yp[t][mk] - yt[t][mk]) ** 2))
            mse_b = float(np.mean((yb_pm[t][mk] - yt[t][mk]) ** 2))
            d[p] = 1 - mse_m / mse_b if mse_b > 0 else float("nan")
        res[t] = d
    return res


def main():
    merged, sets, counts = load_sets()
    factories = model_factories()
    models = {"ridge": factories["ridge"], "hgb": factories["hgb"]}

    v1_baseline = {  # from outputs/experiments.json scheme_b (v1 37-feat baseline)
        "ridge": {"angle": 0.2858, "depth": -2.0249, "left_right": -0.4336},
        "hgb": {"angle": 0.3352, "depth": -0.5389, "left_right": -0.2277},
    }

    result = {
        "scheme": "B",
        "description": "Leave-one-player-out (train 4 players, test held-out 5th, rotate all 5). "
                       "skill_vs_playermean is the headline transfer metric.",
        "n_shots": int(len(merged)),
        "players": sorted(merged["player"].unique().tolist()),
        "feature_counts": counts,
        "v1_baseline_schemeB_skill_vs_playermean": v1_baseline,
        "models": {},
    }

    for mname, fac in models.items():
        predictor = make_feature_predictor(fac, use_player=False, center_by_player=False)
        result["models"][mname] = {}
        for sname, cols in sets.items():
            yt, yp, yb_pm, yb_gm = run_folds(merged, cols, predictor)
            per_t, overall = per_target_record(yt, yp, yb_pm, yb_gm)
            pps = per_player_skill(merged, yt, yp, yb_pm)
            result["models"][mname][sname] = {
                "n_features": len(cols),
                "overall": overall,
                "per_target": per_t,
                "per_player_skill_vs_playermean": pps,
            }
            print(f"[{mname}] {sname:14s} "
                  f"angle skillPM={per_t['angle']['skill_vs_playermean']:+.3f} "
                  f"depth={per_t['depth']['skill_vs_playermean']:+.3f} "
                  f"LR={per_t['left_right']['skill_vs_playermean']:+.3f}")

    out_path = OUTPUTS_DIR / "v4_eval_schemeB.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print("WROTE", out_path)
    return result


if __name__ == "__main__":
    main()
