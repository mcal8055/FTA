"""Cross-validation schemes (the heart of the honest evaluation).

Scheme A — stratified-by-player shot-level K-fold: matches the Kaggle test set
           (same 5 players), so it estimates leaderboard performance; identity is
           legitimately exploitable. Repeated with multiple seeds.
Scheme B — leave-one-player-out (GroupKFold by player): estimates generalization to a
           NEW shooter. The A-B gap quantifies identity-memorization vs transferable signal.

Both yield (train_idx, test_idx, fold_label) over a DataFrame's positional indices.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def scheme_a_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = 0):
    """Stratified by player (keeps the 5-player mix in every fold)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y = df["player"].to_numpy()
    for k, (tr, te) in enumerate(skf.split(np.zeros(len(df)), y)):
        yield tr, te, f"A.s{seed}.f{k}"


def scheme_a_repeated(df: pd.DataFrame, n_splits: int = 5, seeds=range(5)):
    for seed in seeds:
        yield from scheme_a_folds(df, n_splits, seed)


def scheme_b_folds(df: pd.DataFrame):
    """Leave-one-player-out."""
    players = sorted(df["player"].unique())
    idx = np.arange(len(df))
    pl = df["player"].to_numpy()
    for p in players:
        te = idx[pl == p]
        tr = idx[pl != p]
        yield tr, te, f"B.{p}"


def time_ordered_split(df: pd.DataFrame, train_frac: float = 0.7):
    """Sensitivity check: within each player, train on early trials, test on late
    (serial-dependence / session-drift probe). trial_id sort = chronological."""
    tr_parts, te_parts = [], []
    df = df.reset_index(drop=True)
    for p, g in df.groupby("player"):
        g = g.sort_values("shot_id")        # P000X/T#### sorts chronologically
        cut = int(round(len(g) * train_frac))
        tr_parts.append(g.index[:cut].to_numpy())
        te_parts.append(g.index[cut:].to_numpy())
    return np.concatenate(tr_parts), np.concatenate(te_parts)
