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
