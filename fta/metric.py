"""Official competition metric: MSE on MinMax[0,1]-scaled targets.

The Kaggle evaluation applies a MinMax scaler with FIXED published bounds (not fit
to data), then averages MSE across the three targets. We replicate it exactly so
local numbers match the leaderboard. Values outside the bounds scale outside [0,1];
the competition solution file is assumed within-bounds, so we DO NOT clip by default
(clip=True available for sensitivity).
"""
from __future__ import annotations

import numpy as np

from .config import SCALER_BOUNDS, TARGETS


def minmax_scale(values, target: str, clip: bool = False):
    lo, hi = SCALER_BOUNDS[target]
    scaled = (np.asarray(values, float) - lo) / (hi - lo)
    return np.clip(scaled, 0.0, 1.0) if clip else scaled


def scaled_mse_per_target(y_true: dict, y_pred: dict, clip: bool = False) -> dict:
    """y_true/y_pred: {target -> array}. Returns {target -> scaled MSE}."""
    out = {}
    for t in TARGETS:
        st = minmax_scale(y_true[t], t, clip)
        sp = minmax_scale(y_pred[t], t, clip)
        out[t] = float(np.mean((st - sp) ** 2))
    return out


def scaled_mse(y_true: dict, y_pred: dict, clip: bool = False) -> float:
    """Official averaged metric (lower is better)."""
    per = scaled_mse_per_target(y_true, y_pred, clip)
    return float(np.mean([per[t] for t in TARGETS]))


def metric_suite(y_true: dict, y_pred: dict, baseline: dict | None = None) -> dict:
    """Per-target RMSE/MAE/medAE (native units), R2, Spearman, scaled-MSE; plus
    skill score vs an optional baseline (1 - MSE/MSE_baseline, native units).
    Returns {target -> {...}, 'overall': {scaled_mse, mean_skill}}.
    """
    from scipy.stats import spearmanr
    out = {}
    skills = []
    for t in TARGETS:
        yt = np.asarray(y_true[t], float)
        yp = np.asarray(y_pred[t], float)
        err = yp - yt
        mse = float(np.mean(err ** 2))
        ss_tot = float(np.sum((yt - yt.mean()) ** 2))
        r2 = 1 - np.sum(err ** 2) / ss_tot if ss_tot > 0 else np.nan
        sp = np.nan if np.ptp(yp) == 0 or np.ptp(yt) == 0 else float(spearmanr(yp, yt).statistic)
        rec = {
            "rmse": float(np.sqrt(mse)),
            "mae": float(np.mean(np.abs(err))),
            "medae": float(np.median(np.abs(err))),
            "r2": float(r2),
            "spearman": sp,
            "scaled_mse": scaled_mse_per_target(y_true, y_pred)[t],
        }
        if baseline is not None:
            mse_b = float(np.mean((np.asarray(baseline[t], float) - yt) ** 2))
            rec["skill"] = 1 - mse / mse_b if mse_b > 0 else np.nan
            skills.append(rec["skill"])
        out[t] = rec
    out["overall"] = {"scaled_mse": scaled_mse(y_true, y_pred)}
    if skills:
        out["overall"]["mean_skill"] = float(np.mean(skills))
    return out


def bootstrap_ci(y_true: dict, y_pred: dict, stat_fn, n: int = 2000, seed: int = 0):
    """95% percentile bootstrap CI for a scalar stat_fn(y_true, y_pred) over shots."""
    rng = np.random.default_rng(seed)
    m = len(next(iter(y_true.values())))
    vals = []
    for _ in range(n):
        idx = rng.integers(0, m, m)
        yt = {t: np.asarray(y_true[t])[idx] for t in y_true}
        yp = {t: np.asarray(y_pred[t])[idx] for t in y_pred}
        vals.append(stat_fn(yt, yp))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def _selftest() -> None:
    # angle bounds [30,60] (range 30): a 3.0-degree error -> scaled 0.1 -> sq 0.01
    yt = {"angle": [45.0], "depth": [0.0], "left_right": [0.0]}
    yp = {"angle": [48.0], "depth": [0.0], "left_right": [0.0]}
    per = scaled_mse_per_target(yt, yp)
    assert abs(per["angle"] - 0.01) < 1e-12, per
    assert per["depth"] == 0.0 and per["left_right"] == 0.0
    assert abs(scaled_mse(yt, yp) - 0.01 / 3) < 1e-12
    # depth bounds [-12,30] (range 42): 4.2 error -> 0.1 scaled -> 0.01 sq
    yt2 = {"angle": [45.0], "depth": [10.0], "left_right": [0.0]}
    yp2 = {"angle": [45.0], "depth": [14.2], "left_right": [0.0]}
    assert abs(scaled_mse_per_target(yt2, yp2)["depth"] - 0.01) < 1e-9
    print("metric self-test PASSED")


if __name__ == "__main__":
    _selftest()
