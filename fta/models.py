"""Baseline ladder, predictors, and the leakage-safe CV evaluator.

A *predictor* is a callable (train_df, test_df, feature_cols) -> {target: pred array}.
All fitting (means, scalers, regressors) happens on train_df ONLY. The evaluator
accumulates out-of-fold predictions, co-computes the per-player-mean baseline on the
same folds (for skill scores), and returns the metric suite.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import TARGETS
from .cv import scheme_a_folds, scheme_b_folds
from .metric import metric_suite


# --- predictors --------------------------------------------------------------
def pred_global_mean(train, test, feature_cols):
    return {t: np.full(len(test), train[t].mean()) for t in TARGETS}


def pred_player_mean(train, test, feature_cols):
    """Per-player train mean; fall back to global mean for unseen players (Scheme B)."""
    out = {}
    for t in TARGETS:
        gmean = train[t].mean()
        pmean = train.groupby("player")[t].mean()
        out[t] = test["player"].map(pmean).fillna(gmean).to_numpy()
    return out


def pred_player_onehot(train, test, feature_cols):
    """Linear model on one-hot player id only (sanity ~ player-mean)."""
    from sklearn.linear_model import LinearRegression
    players = sorted(train["player"].unique())
    def oh(df):
        return np.array([[1.0 if p == pl else 0.0 for pl in players] for p in df["player"]])
    Xtr, Xte = oh(train), oh(test)
    out = {}
    for t in TARGETS:
        m = LinearRegression().fit(Xtr, train[t])
        out[t] = m.predict(Xte)
    return out


def make_feature_predictor(estimator_factory, use_player=False, center_by_player=False):
    """Feature-based predictor. estimator_factory() -> fresh sklearn regressor.
    use_player: append one-hot player to features.
    center_by_player: subtract train per-player target mean (within-player signal);
                      added back at predict (global mean for unseen player)."""
    def predictor(train, test, feature_cols):
        players = sorted(train["player"].unique())
        def design(df):
            X = df[feature_cols].to_numpy(float)
            if use_player:
                oh = np.array([[1.0 if p == pl else 0.0 for pl in players] for p in df["player"]])
                X = np.hstack([X, oh])
            return X
        Xtr, Xte = design(train), design(test)
        out = {}
        for t in TARGETS:
            ytr = train[t].to_numpy(float)
            if center_by_player:
                pmean = train.groupby("player")[t].mean()
                gmean = train[t].mean()
                ytr = ytr - train["player"].map(pmean).to_numpy()
                back = test["player"].map(pmean).fillna(gmean).to_numpy()
            else:
                back = 0.0
            model = make_pipeline(StandardScaler(), estimator_factory())
            model.fit(Xtr, ytr)
            out[t] = model.predict(Xte) + back
        return out
    return predictor


# --- evaluator ---------------------------------------------------------------
def _run_folds(df, feature_cols, predictor, folds):
    """Accumulate OOF predictions (+ player-mean baseline) over a set of folds that
    together cover every row exactly once."""
    n = len(df)
    yp = {t: np.full(n, np.nan) for t in TARGETS}
    yb = {t: np.full(n, np.nan) for t in TARGETS}
    for tr, te, _ in folds:
        train, test = df.iloc[tr], df.iloc[te]
        pr = predictor(train, test, feature_cols)
        bl = pred_player_mean(train, test, feature_cols)
        for t in TARGETS:
            yp[t][te] = pr[t]
            yb[t][te] = bl[t]
    yt = {t: df[t].to_numpy(float) for t in TARGETS}
    return yt, yp, yb


def evaluate_scheme_a(df, feature_cols, predictor, seeds=range(5), n_splits=5):
    """Repeated stratified CV; returns per-seed metric suites for mean±SD aggregation."""
    runs = []
    for seed in seeds:
        folds = list(scheme_a_folds(df, n_splits, seed))
        yt, yp, yb = _run_folds(df, feature_cols, predictor, folds)
        runs.append(metric_suite(yt, yp, baseline=yb))
    return runs


def evaluate_scheme_b(df, feature_cols, predictor):
    """Leave-one-player-out; returns overall metric suite + per-player fold metrics."""
    folds = list(scheme_b_folds(df))
    yt, yp, yb = _run_folds(df, feature_cols, predictor, folds)
    overall = metric_suite(yt, yp, baseline=yb)
    per_player = {}
    pl = df["player"].to_numpy()
    for p in sorted(df["player"].unique()):
        mask = pl == p
        per_player[p] = metric_suite(
            {t: yt[t][mask] for t in TARGETS},
            {t: yp[t][mask] for t in TARGETS},
            baseline={t: yb[t][mask] for t in TARGETS})
    return overall, per_player


# --- model factories ---------------------------------------------------------
def model_factories():
    return {
        "ridge": lambda: Ridge(alpha=10.0),
        "lasso": lambda: Lasso(alpha=0.05, max_iter=5000),
        "elasticnet": lambda: ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=5000),
        "hgb": lambda: HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0,
            random_state=0),
    }
