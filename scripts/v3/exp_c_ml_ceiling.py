"""Experiment C — Honest ML ceiling on the SAME-PLAYER competition split.

Question: how far does strong, well-tuned conventional ML get using ONLY the
existing 37 body-only features (outputs/features_2025.parquet) + player identity,
on the OFFICIAL competition split (345 train / 113 test, a random split of the same
5 players)? This isolates "just better ML on the same-player task" from "needs new
physics", versus prior Ridge 0.011636 and winner 0.006136.

NO new physics. NO ball features (the 37 features are leakage-free body-only;
verified fta/features.py line 3). Player identity is a LEGAL signal here.

Leakage discipline:
  - Test split is touched ONCE, for the final number only.
  - ALL fitting (scalers, target means, per-player means, model params, blend
    weights, hyperparameters) happens on TRAIN only, via K-fold CV WITHIN train.
  - The competition split is reproduced exactly via competition_eval.py's mapping
    (target-match train.csv rows to JSON shots -> train_ids; test = complement).
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fta.config import OUTPUTS_DIR, TARGETS, TARGET_UNITS
from fta.loader import list_shots, load_metadata
from fta.metric import metric_suite, scaled_mse, scaled_mse_per_target

warnings.filterwarnings("ignore")

WINNER = 0.006136
PRIOR = 0.01164
NON = {"shot_id", "player", "split", "csplit", "made", *TARGETS}
SEED = 1561737


# --------------------------------------------------------------------------- #
# Competition split (exactly as scripts/competition_eval.py)                   #
# --------------------------------------------------------------------------- #
def load_competition_df():
    tgt_key = {}
    for s in list_shots():
        m = load_metadata(s.path)
        tgt_key[(round(m["angle"], 2), round(m["depth"], 2), round(m["left_right"], 2))] = s.shot_id
    tr = pd.read_csv("spl-utspan-data-challenge-2026/train.csv")
    train_ids = {tgt_key[(round(r.angle, 2), round(r.depth, 2), round(r.left_right, 2))]
                 for _, r in tr.iterrows()}
    df = pd.read_parquet(OUTPUTS_DIR / "features_2025.parquet")
    df["csplit"] = np.where(df.shot_id.isin(train_ids), "train", "test")
    return df, sorted([c for c in df.columns if c not in NON])


# --------------------------------------------------------------------------- #
# Building blocks. Each "model" is a function (trn, tst, feats) -> {t: yhat}.   #
# They fit ONLY on trn. Player-centering and scaling are fit on trn only.       #
# --------------------------------------------------------------------------- #
def _design(df, feats, players, use_player):
    X = df[feats].to_numpy(float)
    if use_player:
        oh = np.array([[1.0 if p == pl else 0.0 for pl in players] for p in df["player"]])
        X = np.hstack([X, oh])
    return X


def _player_means(trn, t):
    g = trn[t].mean()
    pm = trn.groupby("player")[t].mean()
    return pm, g


def predict_player_mean(trn, tst, feats):
    out = {}
    for t in TARGETS:
        pm, g = _player_means(trn, t)
        out[t] = tst["player"].map(pm).fillna(g).to_numpy()
    return out


def make_centered_regressor(est_factory, use_player=False):
    """Subtract per-player train mean (within-player residual), fit estimator on
    residual, add player mean back. All fit on trn only."""
    def predict(trn, tst, feats):
        players = sorted(trn["player"].unique())
        Xtr = _design(trn, feats, players, use_player)
        Xte = _design(tst, feats, players, use_player)
        out = {}
        for t in TARGETS:
            pm, g = _player_means(trn, t)
            ytr = trn[t].to_numpy(float) - trn["player"].map(pm).to_numpy()
            back = tst["player"].map(pm).fillna(g).to_numpy()
            model = make_pipeline(StandardScaler(), est_factory())
            model.fit(Xtr, ytr)
            out[t] = model.predict(Xte) + back
        return out
    return predict


def make_plain_regressor(est_factory, use_player=True):
    def predict(trn, tst, feats):
        players = sorted(trn["player"].unique())
        Xtr = _design(trn, feats, players, use_player)
        Xte = _design(tst, feats, players, use_player)
        out = {}
        for t in TARGETS:
            ytr = trn[t].to_numpy(float)
            model = make_pipeline(StandardScaler(), est_factory())
            model.fit(Xtr, ytr)
            out[t] = model.predict(Xte)
        return out
    return predict


def make_per_player_regressor(est_factory):
    """Fit a SEPARATE model per player on that player's residual-vs-its-mean.
    Falls back to centered-global if a player has too few rows."""
    def predict(trn, tst, feats):
        out = {t: np.zeros(len(tst)) for t in TARGETS}
        players = sorted(trn["player"].unique())
        glob = make_centered_regressor(est_factory, use_player=False)
        gpred = glob(trn, tst, feats)
        for t in TARGETS:
            out[t] = gpred[t].copy()
        for p in players:
            sub = trn[trn.player == p]
            te_mask = (tst.player == p).to_numpy()
            if te_mask.sum() == 0:
                continue
            if len(sub) < 30:
                continue  # keep global-centered fallback
            Xtr = sub[feats].to_numpy(float)
            Xte = tst[te_mask][feats].to_numpy(float)
            for t in TARGETS:
                ytr = sub[t].to_numpy(float)
                mu = ytr.mean()
                model = make_pipeline(StandardScaler(), est_factory())
                model.fit(Xtr, ytr - mu)
                out[t][te_mask] = model.predict(Xte) + mu
        return out
    return predict


# Estimator factories ------------------------------------------------------- #
def f_ridge(alpha):
    return lambda: Ridge(alpha=alpha)


def f_huber(alpha, eps):
    return lambda: HuberRegressor(alpha=alpha, epsilon=eps, max_iter=2000)


def f_hgb(depth, lr, n, l2, leaf, loss="squared_error"):
    return lambda: HistGradientBoostingRegressor(
        max_depth=depth, learning_rate=lr, max_iter=n, l2_regularization=l2,
        min_samples_leaf=leaf, loss=loss, random_state=0)


def f_rf(n, depth, leaf):
    return lambda: RandomForestRegressor(
        n_estimators=n, max_depth=depth, min_samples_leaf=leaf,
        n_jobs=-1, random_state=0)


# --------------------------------------------------------------------------- #
# CV-within-train scorer: per-target scaled-MSE, leakage-safe (fit per fold).   #
# --------------------------------------------------------------------------- #
def cv_score_per_target(trn, feats, model_fn, n_splits=5, seeds=(0, 1, 2)):
    """Out-of-fold per-target scaled-MSE averaged over repeated K-fold, all within
    train. Returns {target: mean_scaled_mse}. Stratify-ish via plain KFold on the
    shuffled frame (players are interleaved)."""
    per_runs = {t: [] for t in TARGETS}
    for seed in seeds:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        oof = {t: np.full(len(trn), np.nan) for t in TARGETS}
        idx = trn.reset_index(drop=True)
        for tr_i, te_i in kf.split(idx):
            a, b = idx.iloc[tr_i], idx.iloc[te_i]
            pr = model_fn(a, b, feats)
            for t in TARGETS:
                oof[t][te_i] = pr[t]
        yt = {t: idx[t].to_numpy(float) for t in TARGETS}
        per = scaled_mse_per_target(yt, oof)
        for t in TARGETS:
            per_runs[t].append(per[t])
    return {t: float(np.mean(per_runs[t])) for t in TARGETS}


# --------------------------------------------------------------------------- #
def main():
    df, feats = load_competition_df()
    trn = df[df.csplit == "train"].reset_index(drop=True)
    tst = df[df.csplit == "test"].reset_index(drop=True)
    print(f"OFFICIAL split: train={len(trn)} test={len(tst)} feats={len(feats)}")
    yt_test = {t: tst[t].to_numpy(float) for t in TARGETS}

    # --- candidate model zoo (each is a model_fn) -------------------------- #
    candidates = {
        "player_mean": predict_player_mean,
        # plain (player one-hot appended), various learners
        "ridge_oh_a10": make_plain_regressor(f_ridge(10.0), use_player=True),
        "ridge_oh_a30": make_plain_regressor(f_ridge(30.0), use_player=True),
        "ridge_oh_a60": make_plain_regressor(f_ridge(60.0), use_player=True),
        "huber_oh": make_plain_regressor(f_huber(0.01, 1.35), use_player=True),
        # player-centered (subtract player mean, model the residual)
        "ridge_ctr_a10": make_centered_regressor(f_ridge(10.0)),
        "ridge_ctr_a30": make_centered_regressor(f_ridge(30.0)),
        "ridge_ctr_a60": make_centered_regressor(f_ridge(60.0)),
        "huber_ctr": make_centered_regressor(f_huber(0.01, 1.35)),
        # gradient boosting
        "hgb_ctr_d3": make_centered_regressor(f_hgb(3, 0.05, 300, 1.0, 20)),
        "hgb_ctr_d2": make_centered_regressor(f_hgb(2, 0.03, 500, 2.0, 30)),
        "hgb_oh_d3": make_plain_regressor(f_hgb(3, 0.05, 300, 1.0, 20), use_player=True),
        "hgb_ctr_huberloss": make_centered_regressor(
            f_hgb(3, 0.05, 300, 1.0, 20, loss="absolute_error")),
        # random forest
        "rf_ctr": make_centered_regressor(f_rf(400, 6, 5)),
        "rf_oh": make_plain_regressor(f_rf(400, 6, 5), use_player=True),
        # per-player separate models
        "perplayer_ridge": make_per_player_regressor(f_ridge(20.0)),
        "perplayer_hgb": make_per_player_regressor(f_hgb(3, 0.05, 200, 1.0, 20)),
    }

    # --- score every candidate by CV-within-train (per target) ------------- #
    print("\n== CV-within-train per-target scaled-MSE (model selection) ==")
    cv = {}
    for name, fn in candidates.items():
        cv[name] = cv_score_per_target(trn, feats, fn)
        print(f"  {name:22s} ang={cv[name]['angle']:.5f} dep={cv[name]['depth']:.5f} "
              f"lr={cv[name]['left_right']:.5f} tot={np.mean(list(cv[name].values())):.5f}")

    # --- per-target winner by CV (the honest "pick best model per target") -- #
    best_per_target = {t: min(cv, key=lambda n: cv[n][t]) for t in TARGETS}
    print("\nCV-selected best model per target:")
    for t in TARGETS:
        print(f"  {t:11s} -> {best_per_target[t]} (cv {cv[best_per_target[t]][t]:.5f})")

    # --- TUNE the per-target blend weights by CV (stacking, train-only) ----- #
    # For each target, blend the OOF preds of a small set of strong, diverse
    # models with non-negative weights chosen to minimise CV scaled-MSE. Weights
    # are fit on train OOF only, then applied to the test preds of the SAME models.
    blend_pool = ["ridge_ctr_a30", "huber_ctr", "hgb_ctr_d3", "hgb_ctr_d2",
                  "rf_ctr", "perplayer_hgb", "player_mean"]

    # Build train-OOF matrix (one consistent KFold) and test preds for the pool.
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    idx = trn.reset_index(drop=True)
    oof = {n: {t: np.full(len(idx), np.nan) for t in TARGETS} for n in blend_pool}
    for tr_i, te_i in kf.split(idx):
        a, b = idx.iloc[tr_i], idx.iloc[te_i]
        for n in blend_pool:
            pr = candidates[n](a, b, feats)
            for t in TARGETS:
                oof[n][t][te_i] = pr[t]
    test_pred = {n: candidates[n](trn, tst, feats) for n in blend_pool}
    ytr_full = {t: idx[t].to_numpy(float) for t in TARGETS}

    def fit_blend_weights(t):
        """Non-negative least squares on scaled targets -> minimise scaled-MSE.
        Closed form via NNLS on the OOF prediction matrix."""
        from scipy.optimize import nnls
        from fta.metric import minmax_scale
        M = np.column_stack([minmax_scale(oof[n][t], t) for n in blend_pool])
        y = minmax_scale(ytr_full[t], t)
        w, _ = nnls(M, y)
        if w.sum() == 0:
            w = np.ones(len(blend_pool)) / len(blend_pool)
        return w

    blend_w = {t: fit_blend_weights(t) for t in TARGETS}
    # CV scaled-MSE of the blend (still train-only OOF):
    from fta.metric import minmax_scale
    blend_oof = {}
    for t in TARGETS:
        M = np.column_stack([oof[n][t] for n in blend_pool])
        blend_oof[t] = M @ blend_w[t]
    blend_cv = scaled_mse_per_target(ytr_full, blend_oof)
    print("\nStacked blend CV per-target:", {t: round(blend_cv[t], 5) for t in TARGETS},
          "tot", round(np.mean(list(blend_cv.values())), 5))
    print("blend weights:")
    for t in TARGETS:
        print(f"  {t:11s}", {blend_pool[i]: round(float(blend_w[t][i]), 3)
                             for i in range(len(blend_pool)) if blend_w[t][i] > 1e-3})

    # ================= FINAL TEST EVALUATION (touch test ONCE) ============== #
    # Build the final predictor: per-target, take the better of (CV-best single
    # model) vs (stacked blend), decided by CV — then apply to test.
    final_test = {}
    final_choice = {}
    for t in TARGETS:
        single_cv = cv[best_per_target[t]][t]
        if blend_cv[t] <= single_cv:
            M = np.column_stack([test_pred[n][t] for n in blend_pool])
            final_test[t] = M @ blend_w[t]
            final_choice[t] = f"blend(cv={blend_cv[t]:.5f})"
        else:
            final_test[t] = candidates[best_per_target[t]](trn, tst, feats)[t]
            final_choice[t] = f"{best_per_target[t]}(cv={single_cv:.5f})"

    print("\n== FINAL per-target choice (by CV) ==")
    for t in TARGETS:
        print(f"  {t:11s} -> {final_choice[t]}")

    final_per = scaled_mse_per_target(yt_test, final_test)
    final_total = scaled_mse(yt_test, final_test)
    print(f"\n=== FINAL TEST scaled-MSE ===")
    for t in TARGETS:
        print(f"  {t:11s} {final_per[t]:.6f}")
    print(f"  TOTAL       {final_total:.6f}")
    print(f"  prior(Ridge)={PRIOR:.6f}  winner={WINNER:.6f}  "
          f"ratio_to_winner={final_total/WINNER:.2f}x")
    print(f"  beats_prior={final_total < PRIOR}  beats_winner={final_total < WINNER}")

    # Also report: the single best WHOLE-model on test (for reference/sanity),
    # plus native-unit RMSE/R2 of the final.
    ms = metric_suite(yt_test, final_test)
    print("\nFinal native-unit:")
    for t in TARGETS:
        print(f"  {t:11s} RMSE={ms[t]['rmse']:.2f}{TARGET_UNITS[t]} "
              f"R2={ms[t]['r2']:+.3f} spearman={ms[t]['spearman']:+.3f}")

    # whole-model test scores (reference only; selection was by CV)
    whole_test = {}
    for name, fn in candidates.items():
        p = fn(trn, tst, feats)
        whole_test[name] = scaled_mse_per_target(yt_test, p)
        whole_test[name]["total"] = scaled_mse(yt_test, p)

    out = {
        "winner": WINNER, "prior": PRIOR,
        "n_train": len(trn), "n_test": len(tst), "n_features": len(feats),
        "cv_within_train": cv,
        "cv_best_per_target": {t: best_per_target[t] for t in TARGETS},
        "blend_pool": blend_pool,
        "blend_weights": {t: {blend_pool[i]: float(blend_w[t][i])
                              for i in range(len(blend_pool))} for t in TARGETS},
        "blend_cv_per_target": blend_cv,
        "final_choice_per_target": final_choice,
        "final_test_per_target": final_per,
        "final_test_total": final_total,
        "ratio_to_winner": final_total / WINNER,
        "beats_prior": bool(final_total < PRIOR),
        "beats_winner": bool(final_total < WINNER),
        "final_native": {t: {k: ms[t][k] for k in ("rmse", "r2", "spearman")}
                         for t in TARGETS},
        "whole_model_test_reference": whole_test,
    }
    json.dump(out, open(OUTPUTS_DIR / "v3_exp_c.json", "w"), indent=2)
    print(f"\nWrote {OUTPUTS_DIR / 'v3_exp_c.json'}")
    return out


if __name__ == "__main__":
    main()
