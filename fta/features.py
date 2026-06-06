"""Step 3 — body-only scalar features per shot.

Leakage-free: uses ONLY player keypoints (no ball, no outcomes). Features are computed
around the body-detected release on the native time grid (2025 = 60 fps); derivatives use
real `time` deltas so the same code is fps-correct for the 30 fps OOD probe. Sided markers
are resolved to shooting/support side; directional features use a per-shot body frame
(translate to mid-hip, face-forward via shoulder line disambiguated by nose).

extract_features(tk) -> dict[str, float]   (NaN for any feature that can't be computed)
"""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from .loader import Tracking
from .release import body_release_frame, detect_handedness

WIN_BEFORE_S = 1.0    # window start before release
WIN_AFTER_S = 0.3     # window end after release


# --- geometry helpers --------------------------------------------------------
def _angle(a, b, c):
    """Angle at b between a-b and c-b (degrees), per frame."""
    ba, bc = a - b, c - b
    cos = (ba * bc).sum(-1) / (np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def _smooth(arr, fps):
    """Savitzky-Golay smooth each column; window ~ 0.18 s, robust to short/NaN series."""
    out = arr.copy()
    n = len(arr)
    win = max(5, int(round(0.18 * fps)) | 1)   # odd
    if n < win:
        return out
    for k in range(arr.shape[1]):
        col = arr[:, k]
        if np.isnan(col).any():
            idx = np.arange(n)
            good = ~np.isnan(col)
            if good.sum() < win:
                continue
            col = np.interp(idx, idx[good], col[good])
        out[:, k] = savgol_filter(col, win, 3)
    return out


def _vel(arr, t):
    return np.gradient(arr, t, axis=0)


def _safe(x):
    return float(x) if x is not None and np.isfinite(x) else np.nan


# --- main --------------------------------------------------------------------
def extract_features(tk: Tracking, with_fingers: bool = False) -> dict:
    hand = detect_handedness(tk)
    rel = body_release_frame(tk, hand)
    t = tk.time
    fps = tk.sampling_rate or int(round(1.0 / np.median(np.diff(t))))
    f = {"hand_right": 1.0 if hand == "R" else 0.0}
    if rel is None:
        return f

    SH_s = "RIGHT_SHOULDER" if hand == "R" else "LEFT_SHOULDER"
    SH_o = "LEFT_SHOULDER" if hand == "R" else "RIGHT_SHOULDER"
    EL_s = "RIGHT_ELBOW" if hand == "R" else "LEFT_ELBOW"
    WR_s = "RIGHT_WRIST" if hand == "R" else "LEFT_WRIST"
    HIP_s = "RIGHT_HIP" if hand == "R" else "LEFT_HIP"
    HIP_o = "LEFT_HIP" if hand == "R" else "RIGHT_HIP"
    KNEE_s = "RIGHT_KNEE" if hand == "R" else "LEFT_KNEE"
    KNEE_o = "LEFT_KNEE" if hand == "R" else "RIGHT_KNEE"
    ANK_s = "RIGHT_ANKLE" if hand == "R" else "LEFT_ANKLE"
    ANK_o = "LEFT_ANKLE" if hand == "R" else "RIGHT_ANKLE"

    J = {name: _smooth(tk.joint(name), fps) for name in (
        "NOSE", "NECK", "MID_HIP", SH_s, SH_o, EL_s, WR_s, HIP_s, HIP_o,
        KNEE_s, KNEE_o, ANK_s, ANK_o)}

    # window indices
    lo = max(0, rel - int(round(WIN_BEFORE_S * fps)))
    hi = min(len(t), rel + int(round(WIN_AFTER_S * fps)) + 1)
    w = slice(lo, hi)

    # derived series
    elbow = _angle(J[SH_s], J[EL_s], J[WR_s])                       # shooting elbow
    knee_s = _angle(J[HIP_s], J[KNEE_s], J[ANK_s])
    knee_o = _angle(J[HIP_o], J[KNEE_o], J[ANK_o])
    trunk = J["NECK"] - J["MID_HIP"]
    trunk_lean = np.degrees(np.arctan2(np.linalg.norm(trunk[:, :2], axis=1), trunk[:, 2]))
    wr = J[WR_s]
    wr_v = _vel(wr, t)
    wr_speed = np.linalg.norm(wr_v, axis=1)
    elbow_v = np.gradient(elbow, t)

    # body frame at release: forward = horizontal facing (perp to shoulder line, toward nose)
    sl = (J[SH_s] - J[SH_o])[rel, :2]
    fwd = np.array([-sl[1], sl[0]])
    nose_dir = (J["NOSE"] - J["MID_HIP"])[rel, :2]
    if fwd @ nose_dir < 0:
        fwd = -fwd
    fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
    lat = np.array([-fwd[1], fwd[0]])      # +lat = shooter's left of facing

    def body_xy(vec_xy):
        return float(vec_xy @ fwd), float(vec_xy @ lat)

    # ---- features ----
    # release-instant kinematics
    f["release_elbow_angle"] = _safe(elbow[rel])
    f["release_knee_shoot"] = _safe(knee_s[rel])
    f["release_knee_support"] = _safe(knee_o[rel])
    f["release_trunk_lean"] = _safe(trunk_lean[rel])
    f["release_wrist_height"] = _safe(wr[rel, 2])
    rel_wr_rel_hip = (wr - J["MID_HIP"])[rel]
    fwd_w, lat_w = body_xy(rel_wr_rel_hip[:2])
    f["release_wrist_forward"] = fwd_w
    f["release_wrist_lateral"] = lat_w
    f["release_wrist_above_hip"] = _safe(rel_wr_rel_hip[2])

    # velocities at release
    f["release_wrist_speed"] = _safe(wr_speed[rel])
    f["release_wrist_vz"] = _safe(wr_v[rel, 2])
    f["release_elbow_ext_vel"] = _safe(elbow_v[rel])
    vh = np.hypot(wr_v[rel, 0], wr_v[rel, 1])
    f["release_handpath_elevation"] = _safe(np.degrees(np.arctan2(wr_v[rel, 2], vh)))
    f["release_handpath_azimuth"] = _safe(np.degrees(np.arctan2(*body_xy(wr_v[rel, :2])[::-1])))

    # range of motion over window
    def rng(series):
        s = series[w]
        s = s[np.isfinite(s)]
        return float(s.max() - s.min()) if s.size else np.nan
    f["rom_elbow"] = rng(elbow)
    f["rom_knee_shoot"] = rng(knee_s)
    f["rom_wrist_height"] = rng(wr[:, 2])
    f["rom_trunk_lean"] = rng(trunk_lean)
    f["min_knee_shoot"] = _safe(np.nanmin(knee_s[w])) if np.isfinite(knee_s[w]).any() else np.nan
    f["peak_wrist_speed"] = _safe(np.nanmax(wr_speed[w])) if np.isfinite(wr_speed[w]).any() else np.nan
    f["peak_elbow_ext_vel"] = _safe(np.nanmax(elbow_v[w])) if np.isfinite(elbow_v[w]).any() else np.nan

    # tempo / timing (seconds relative to release)
    def t_of(series, reducer):
        seg = series[w]
        if not np.isfinite(seg).any():
            return np.nan
        i = lo + int(reducer(np.where(np.isfinite(seg), seg, np.nan)))
        return float(t[i] - t[rel])
    f["t_deepest_knee_to_release"] = t_of(knee_s, np.nanargmin)   # dip timing
    f["t_lowest_wrist_to_release"] = t_of(wr[:, 2], np.nanargmin)
    f["t_peak_wrist_speed_to_release"] = t_of(wr_speed, np.nanargmax)

    # asymmetry / balance
    ankle_vec = (J[ANK_s] - J[ANK_o])[rel]
    f["stance_width"] = _safe(np.linalg.norm(ankle_vec[:2]))
    f["knee_asym_release"] = _safe(abs(knee_s[rel] - knee_o[rel]))
    hip = J["MID_HIP"][w][:, :2]
    hip = hip[np.isfinite(hip).all(1)]
    f["com_sway"] = float(np.linalg.norm(hip.max(0) - hip.min(0))) if len(hip) else np.nan

    # smoothness: normalized jerk of wrist path over window (lower = smoother)
    seg = wr[w]
    good = np.isfinite(seg).all(1)
    if good.sum() > 8:
        js = np.gradient(np.gradient(np.gradient(seg[good], t[w][good], axis=0),
                                     t[w][good], axis=0), t[w][good], axis=0)
        dur = t[w][good][-1] - t[w][good][0]
        path = np.linalg.norm(np.diff(seg[good], axis=0), axis=1).sum()
        f["wrist_jerk_norm"] = float(np.sqrt(np.mean((js**2).sum(1))) * dur**3 / (path + 1e-6))
    else:
        f["wrist_jerk_norm"] = np.nan

    return f
