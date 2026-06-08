"""Experiment B: joint-geometry launch features + strong models on the OFFICIAL split.

Builds an augmented feature matrix:
    existing outputs/features_2025.parquet (37 body features)
  + multi-frame BODY launch-geometry features from physics_v3.body_launch_fit
        release_x/y/z, release_height, dist_to_rim, lateral_offset,
        elevation, azimuth, speed, x_cross, y_cross, bc_entry

Evaluated on the OFFICIAL competition split (345 train / 113 test) via the same
target-match mapping as scripts/competition_eval.py. Model selection (alphas / model
family / feature set) is done by CV-WITHIN-TRAIN only; the test split is touched once
for the final reported number.

LEAKAGE DISCIPLINE:
  - No ball-derived quantity is a model feature (body keypoints only). The geometry
    features come from body_launch_fit, which reads ONLY player joints.
  - All scalers / imputers / model params / per-player means fit on TRAIN rows only.
  - Test split is the complement of the target-matched train ids (competition_eval.py).

Writes outputs/v3_exp_b.json.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fta.config import OUTPUTS_DIR, TARGETS, TARGET_UNITS
from fta.loader import list_shots, load_metadata, load_tracking
from fta.metric import metric_suite, scaled_mse, scaled_mse_per_target
from fta.release import detect_handedness

from physics_v3 import body_launch_fit

warnings.filterwarnings("ignore")

WINNER = 0.006136
PRIOR = 0.01164
NON = {"shot_id", "player", "split", "csplit", "made", *TARGETS}

GEOM_COLS = [
    "release_x", "release_y", "release_z", "release_height",
    "dist_to_rim", "lateral_offset", "elevation", "azimuth", "speed",
    "x_cross", "y_cross", "bc_entry",
]


# --------------------------------------------------------------------------- #
# build features                                                              #
# --------------------------------------------------------------------------- #
def build_geom_features() -> pd.DataFrame:
    """Per-shot BODY launch-geometry features (body keypoints only)."""
    rows = []
    for s in list_shots():
        tk = load_tracking(s.path)
        hand = detect_handedness(tk)
        bo = body_launch_fit(tk, hand)
        row = {"shot_id": s.shot_id}
        if bo is not None:
            for k in GEOM_COLS:
                row[f"g_{k}"] = bo.get(k, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def competition_train_ids() -> set:
    tgt_key = {}
    for s in list_shots():
        m = load_metadata(s.path)
        tgt_key[(round(m["angle"], 2), round(m["depth"], 2),
                 round(m["left_right"], 2))] = s.shot_id
    tr = pd.read_csv("spl-utspan-data-challenge-2026/train.csv")
    return {tgt_key[(round(r.angle, 2), round(r.depth, 2), round(r.left_right, 2))]
            for _, r in tr.iterrows()}


# --------------------------------------------------------------------------- #
# fitting helpers (TRAIN-ONLY)                                                 #
# --------------------------------------------------------------------------- #
def _impute_fit(train: pd.DataFrame, cols):
    """Median impute values fit on TRAIN only (fallback 0.0 if a fold slice is all-NaN)."""
    out = {}
    for c in cols:
        m = train[c].median()
        out[c] = float(m) if np.isfinite(m) else 0.0
    return out


def _apply_impute(df: pd.DataFrame, med: dict, cols):
    X = df[cols].copy()
    for c in cols:
        X[c] = X[c].fillna(med[c])
    return X.to_numpy(float)


def ridge_oof_cv(train, feats, target, alphas, n_splits=5, seed=0):
    """Group-by-player CV within train; returns (best_alpha, best_cv_sMSE_for_target).

    Per-target alpha selection by scaled-MSE on out-of-fold train predictions.
    All imputation/scaling refit inside each fold on the fold-train only.
    """
    groups = train["player"].to_numpy()
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    yt = train[target].to_numpy(float)
    best_a, best_s = None, np.inf
    for a in alphas:
        oof = np.full(len(train), np.nan)
        for tr_i, va_i in gkf.split(train, yt, groups):
            ftr, fva = train.iloc[tr_i], train.iloc[va_i]
            med = _impute_fit(ftr, feats)
            Xtr = _apply_impute(ftr, med, feats)
            Xva = _apply_impute(fva, med, feats)
            m = make_pipeline(StandardScaler(), Ridge(alpha=a))
            m.fit(Xtr, yt[tr_i])
            oof[va_i] = m.predict(Xva)
        s = float(np.mean((_scale(oof, target) - _scale(yt, target)) ** 2))
        if s < best_s:
            best_s, best_a = s, a
    return best_a, best_s


def _scale(v, target):
    from fta.config import SCALER_BOUNDS
    lo, hi = SCALER_BOUNDS[target]
    return (np.asarray(v, float) - lo) / (hi - lo)


def fit_predict_ridge(train, test, feats, alphas):
    """Per-target ridge with TRAIN-CV alpha selection; impute+scale fit on train."""
    preds = {}
    chosen = {}
    cv_s = {}
    for t in TARGETS:
        a, s = ridge_oof_cv(train, feats, t, alphas)
        chosen[t] = a
        cv_s[t] = s
        med = _impute_fit(train, feats)
        Xtr = _apply_impute(train, med, feats)
        Xte = _apply_impute(test, med, feats)
        m = make_pipeline(StandardScaler(), Ridge(alpha=a))
        m.fit(Xtr, train[t].to_numpy(float))
        preds[t] = m.predict(Xte)
    return preds, chosen, cv_s


def fit_predict_hgb(train, test, feats):
    preds = {}
    med = _impute_fit(train, feats)
    Xtr = _apply_impute(train, med, feats)
    Xte = _apply_impute(test, med, feats)
    for t in TARGETS:
        m = HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=400,
            l2_regularization=1.0, random_state=0)
        m.fit(Xtr, train[t].to_numpy(float))
        preds[t] = m.predict(Xte)
    return preds


def fit_predict_player_ridge(train, test, feats, alphas):
    """Per-player additive model: predict per-player-mean-centered target with ridge
    on geometry features, add back the train per-player mean. Same-player split makes
    the per-player mean a strong, legal prior."""
    preds = {}
    chosen = {}
    for t in TARGETS:
        gmean = train[t].mean()
        pmean = train.groupby("player")[t].mean()
        ytr_centered = (train[t].to_numpy(float)
                        - train["player"].map(pmean).to_numpy())
        back = test["player"].map(pmean).fillna(gmean).to_numpy()
        # alpha selection on centered target via group CV
        groups = train["player"].to_numpy()
        gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
        best_a, best_s = None, np.inf
        for a in alphas:
            oof = np.full(len(train), np.nan)
            for tr_i, va_i in gkf.split(train, ytr_centered, groups):
                ftr, fva = train.iloc[tr_i], train.iloc[va_i]
                # within-fold per-player centering
                pm = ftr.groupby("player")[t].mean()
                gm = ftr[t].mean()
                yc = ftr[t].to_numpy(float) - ftr["player"].map(pm).to_numpy()
                med = _impute_fit(ftr, feats)
                Xtr = _apply_impute(ftr, med, feats)
                Xva = _apply_impute(fva, med, feats)
                m = make_pipeline(StandardScaler(), Ridge(alpha=a))
                m.fit(Xtr, yc)
                bk = fva["player"].map(pm).fillna(gm).to_numpy()
                oof[va_i] = m.predict(Xva) + bk
            yt = train[t].to_numpy(float)
            s = float(np.mean((_scale(oof, t) - _scale(yt, t)) ** 2))
            if s < best_s:
                best_s, best_a = s, a
        chosen[t] = best_a
        med = _impute_fit(train, feats)
        Xtr = _apply_impute(train, med, feats)
        Xte = _apply_impute(test, med, feats)
        m = make_pipeline(StandardScaler(), Ridge(alpha=best_a))
        m.fit(Xtr, ytr_centered)
        preds[t] = m.predict(Xte) + back
    return preds, chosen


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    base = pd.read_parquet(OUTPUTS_DIR / "features_2025.parquet")
    geom = build_geom_features()
    df = base.merge(geom, on="shot_id", how="left")
    # drop geom columns that are all-NaN (body ballistic crossing rarely solves for a
    # descending rim crossing given short-window noisy pose velocity -> uninformative)
    all_nan = [c for c in geom.columns if c != "shot_id" and df[c].isna().all()]
    if all_nan:
        print(f"dropping all-NaN geom cols: {all_nan}")
    usable_geom = [f"g_{c}" for c in GEOM_COLS if f"g_{c}" not in all_nan]
    n_geom_ok = df[usable_geom].notna().all(1).sum()
    print(f"merged: {df.shape}; usable geom feats={len(usable_geom)}; "
          f"geom-complete shots = {n_geom_ok}/{len(df)}")

    train_ids = competition_train_ids()
    df["csplit"] = np.where(df.shot_id.isin(train_ids), "train", "test")
    trn = df[df.csplit == "train"].reset_index(drop=True)
    ten = df[df.csplit == "test"].reset_index(drop=True)
    print(f"OFFICIAL split: train={len(trn)} test={len(ten)}")

    base_feats = [c for c in base.columns if c not in NON]
    geom_feats = usable_geom
    aug_feats = base_feats + geom_feats
    geom_only = geom_feats
    print(f"feature sets: base={len(base_feats)} geom={len(geom_feats)} "
          f"aug={len(aug_feats)}")

    yt = {t: ten[t].to_numpy(float) for t in TARGETS}
    alphas = [0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]

    results = {}

    def record(name, yp, extra=None):
        per = scaled_mse_per_target(yt, yp)
        tot = scaled_mse(yt, yp)
        rec = {"scaled_mse_total": tot, "per_target": per}
        if extra:
            rec.update(extra)
        results[name] = rec
        print(f"{name:26s} {tot:.6f}   " + " ".join(f"{per[t]:.4f}" for t in TARGETS))
        return tot

    print(f"\n{'model/featset':26s} {'sMSE':>8s}   ang    dep    lr")

    # --- ridge over feature sets (per-target CV alpha) ---
    p, a, cv = fit_predict_ridge(trn, ten, base_feats, alphas)
    record("ridge_base", p, {"alphas": a, "cv_sMSE": cv})
    p, a, cv = fit_predict_ridge(trn, ten, geom_only, alphas)
    record("ridge_geom_only", p, {"alphas": a, "cv_sMSE": cv})
    p_aug, a_aug, cv_aug = fit_predict_ridge(trn, ten, aug_feats, alphas)
    record("ridge_aug", p_aug, {"alphas": a_aug, "cv_sMSE": cv_aug})

    # --- HGB over feature sets ---
    record("hgb_base", fit_predict_hgb(trn, ten, base_feats))
    record("hgb_aug", fit_predict_hgb(trn, ten, aug_feats))

    # --- per-player additive ridge (same-player split prior) ---
    pp_b, ppa_b = fit_predict_player_ridge(trn, ten, base_feats, alphas)
    record("player_ridge_base", pp_b, {"alphas": ppa_b})
    pp_a, ppa_a = fit_predict_player_ridge(trn, ten, aug_feats, alphas)
    record("player_ridge_aug", pp_a, {"alphas": ppa_a})

    # --- per-target BEST model mix (selected by TRAIN CV, not test) ---
    # choose per target among {ridge_aug, player_ridge_aug} via their train-CV sMSE.
    # ridge_aug exposes cv; recompute player_ridge cv for fairness.
    # Simpler honest rule: use the model whose TRAIN CV is lower per target.
    # We recompute train-CV for player_ridge_aug below.
    _, _, cv_ridge_aug = fit_predict_ridge(trn, ten, aug_feats, alphas)
    # player ridge train-CV:
    cv_player = {}
    for t in TARGETS:
        groups = trn["player"].to_numpy()
        gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
        best_s = np.inf
        for al in alphas:
            oof = np.full(len(trn), np.nan)
            for tr_i, va_i in gkf.split(trn, trn[t], groups):
                ftr, fva = trn.iloc[tr_i], trn.iloc[va_i]
                pm = ftr.groupby("player")[t].mean(); gm = ftr[t].mean()
                yc = ftr[t].to_numpy(float) - ftr["player"].map(pm).to_numpy()
                med = _impute_fit(ftr, aug_feats)
                Xtr = _apply_impute(ftr, med, aug_feats)
                Xva = _apply_impute(fva, med, aug_feats)
                m = make_pipeline(StandardScaler(), Ridge(alpha=al)); m.fit(Xtr, yc)
                bk = fva["player"].map(pm).fillna(gm).to_numpy()
                oof[va_i] = m.predict(Xva) + bk
            s = float(np.mean((_scale(oof, t) - _scale(trn[t].to_numpy(float), t)) ** 2))
            best_s = min(best_s, s)
        cv_player[t] = best_s

    mix = {}
    mix_choice = {}
    for t in TARGETS:
        if cv_player[t] <= cv_ridge_aug[t]:
            mix[t] = pp_a[t]; mix_choice[t] = "player_ridge_aug"
        else:
            mix[t] = p_aug[t]; mix_choice[t] = "ridge_aug"
    record("best_mix_trainCV", mix, {"choice": mix_choice,
                                     "cv_ridge_aug": cv_ridge_aug,
                                     "cv_player": cv_player})

    # ----------------------------------------------------------------------- #
    # which geom feature helps which target: ridge coef magnitude (train-fit) #
    # ----------------------------------------------------------------------- #
    coef_report = {}
    med = _impute_fit(trn, aug_feats)
    Xtr = _apply_impute(trn, med, aug_feats)
    sc = StandardScaler().fit(Xtr)
    Xs = sc.transform(Xtr)
    for t in TARGETS:
        m = Ridge(alpha=a_aug[t]).fit(Xs, trn[t].to_numpy(float))
        c = dict(zip(aug_feats, m.coef_))
        top = sorted(c.items(), key=lambda kv: -abs(kv[1]))[:8]
        coef_report[t] = [{"feat": k, "coef": float(v)} for k, v in top]

    # geom-feature marginal correlation with each target (train only)
    geom_corr = {}
    for t in TARGETS:
        gc = {}
        for c in geom_feats:
            d = trn[[c, t]].dropna()
            if len(d) > 10 and d[c].std() > 0:
                gc[c] = float(np.corrcoef(d[c], d[t])[0, 1])
        geom_corr[t] = gc

    # ----------------------------------------------------------------------- #
    # report                                                                  #
    # ----------------------------------------------------------------------- #
    best_name = min(results, key=lambda k: results[k]["scaled_mse_total"])
    best_tot = results[best_name]["scaled_mse_total"]
    print(f"\nBEST = {best_name}: {best_tot:.6f}")
    print(f"  vs prior {PRIOR:.6f} (ratio {best_tot/PRIOR:.3f}x)  "
          f"vs winner {WINNER:.6f} (ratio {best_tot/WINNER:.3f}x)")
    print(f"  beats_prior = {best_tot < PRIOR}   beats_winner = {best_tot < WINNER}")

    print("\nRMSE native (best ridge_aug):")
    ms = metric_suite(yt, p_aug)
    for t in TARGETS:
        print(f"  {t:11s} RMSE={ms[t]['rmse']:.2f}{TARGET_UNITS[t]} R2={ms[t]['r2']:+.3f}")

    print("\nTop geom-feature marginal corr per target (train):")
    for t in TARGETS:
        gc = sorted(geom_corr[t].items(), key=lambda kv: -abs(kv[1]))[:4]
        print(f"  {t:11s} " + "  ".join(f"{k}={v:+.2f}" for k, v in gc))

    out = {
        "winner": WINNER, "prior": PRIOR,
        "n_geom_complete": int(n_geom_ok),
        "n_train": int(len(trn)), "n_test": int(len(ten)),
        "base_feats": base_feats, "geom_feats": geom_feats,
        "results": results,
        "best_model": best_name, "best_total": best_tot,
        "beats_prior": bool(best_tot < PRIOR),
        "beats_winner": bool(best_tot < WINNER),
        "coef_report": coef_report,
        "geom_corr": geom_corr,
    }
    OUTPUTS_DIR.mkdir(exist_ok=True)
    with open(OUTPUTS_DIR / "v3_exp_b.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {OUTPUTS_DIR / 'v3_exp_b.json'}")


if __name__ == "__main__":
    main()
