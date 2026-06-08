"""v4 KINETIC CHAIN feature family — proximal-to-distal energy transfer.

Hypothesis (from the competition winner's post-mortem, used as PRINCIPLES not numbers):
  Force flows ground -> ankle -> knee -> hip -> trunk -> shoulder -> elbow ->
  wrist -> fingertip. Depth is FORCE-driven (whole-body forward COM thrust toward
  the rim), angle is elbow geometry, left_right is the late wrist snap. Skilled
  shooting shows SUMMATION OF SPEED: each distal segment peaks LATER and FASTER
  than the proximal one. These kinematic (speed / timing / sequencing) features are
  shooter-INVARIANT in construction, so the test is whether they TRANSFER to unseen
  shooters (Scheme B) for depth / left_right where v1 position features failed.

Leakage-free:
  * body pose ONLY (no ball xyz, no outcomes).
  * release detected from the BODY (fta.release.body_release_frame).
  * left-handers mirrored to the shooting-side chain -> handedness-invariant.
  * NO tuned frame indices; everything is anchored to the body-detected release and
    a window around it. Forward direction is the horizontal COM->rim vector (geometry,
    not target info).

Output: outputs/v4_kinetic.parquet, one row per shot:
    shot_id, player, + ~24 kinematic features.

Run:  python scripts/v4/kinetic_chain.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fta.config import OUTPUTS_DIR, RIM
from fta.loader import Tracking, despike, list_shots, load_metadata, load_tracking
from fta.release import body_release_frame, detect_handedness

# Window around release on which we look for peaks (seconds). Covers the upward
# drive (dip -> launch) and a little follow-through, long enough to contain every
# segment's peak speed without leaking outcome-specific frame choices.
WIN_BEFORE_S = 1.0
WIN_AFTER_S = 0.3

# Proximal -> distal chain on the SHOOTING side. Each entry maps a logical segment
# name to the keypoint whose linear speed we track. trunk is handled separately
# (mid-hip -> neck) for its angular contribution; its linear proxy is NECK.
CHAIN = ["ankle", "knee", "hip", "trunk", "shoulder", "elbow", "wrist", "fingertip"]


def _sided(hand: str) -> dict:
    """Resolve sided keypoint names to the shooting side (mirror left-handers)."""
    S = "RIGHT" if hand == "R" else "LEFT"
    O = "LEFT" if hand == "R" else "RIGHT"
    return {
        "ankle": f"{S}_ANKLE",
        "knee": f"{S}_KNEE",
        "hip": f"{S}_HIP",
        "trunk": "NECK",            # linear proxy for trunk top
        "shoulder": f"{S}_SHOULDER",
        "elbow": f"{S}_ELBOW",
        "wrist": f"{S}_WRIST",
        # fingertip: shooting-hand index distal (ball's last contact)
        "fingertip": f"{S}_SECOND_FINGER_DISTAL",
        # extras
        "SH_s": f"{S}_SHOULDER", "SH_o": f"{O}_SHOULDER",
        "HIP_s": f"{S}_HIP", "HIP_o": f"{O}_HIP",
        "NECK": "NECK", "MID_HIP": "MID_HIP", "NOSE": "NOSE",
        "fingertip_alt": f"{S}_THIRD_FINGER_DISTAL",
    }


def _smooth(arr: np.ndarray, fps: float) -> np.ndarray:
    """Despike + Savitzky-Golay each column (~0.18 s window), NaN-robust via interp."""
    out = arr.copy()
    n = len(arr)
    win = max(5, int(round(0.18 * fps)) | 1)
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
        col = despike(col)
        out[:, k] = savgol_filter(col, win, 3)
    return out


def _safe(x) -> float:
    return float(x) if x is not None and np.isfinite(x) else np.nan


def _speed(series: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Per-frame linear speed magnitude (ft/s) of a (n,3) trajectory."""
    v = np.gradient(series, t, axis=0)
    return np.linalg.norm(v, axis=1)


def _peak_in_window(sig: np.ndarray, lo: int, hi: int, t: np.ndarray, rel: int):
    """Return (peak_value, peak_time_rel_to_release_s, peak_abs_idx) over [lo,hi)."""
    seg = sig[lo:hi]
    if not np.isfinite(seg).any():
        return np.nan, np.nan, None
    j = int(np.nanargmax(np.where(np.isfinite(seg), seg, -np.inf)))
    i = lo + j
    return float(seg[j]), float(t[i] - t[rel]), i


def extract_kinetic(tk: Tracking) -> dict:
    """Kinetic-chain feature dict for one shot (NaN where uncomputable)."""
    hand = detect_handedness(tk)
    rel = body_release_frame(tk, hand)
    t = tk.time
    fps = tk.sampling_rate or int(round(1.0 / np.median(np.diff(t))))
    f: dict[str, float] = {"hand_right": 1.0 if hand == "R" else 0.0}
    if rel is None:
        return f

    m = _sided(hand)

    # fingertip can be missing -> fall back to middle finger, then wrist.
    fing = m["fingertip"]
    if fing not in tk.players or np.all(np.isnan(tk.joint(fing))):
        fing = m["fingertip_alt"]
    if fing not in tk.players or np.all(np.isnan(tk.joint(fing))):
        fing = m["wrist"]

    seg_kp = {seg: (fing if seg == "fingertip" else m[seg]) for seg in CHAIN}
    needed = set(seg_kp.values()) | {m["SH_s"], m["SH_o"], m["HIP_s"], m["HIP_o"],
                                     m["NECK"], m["MID_HIP"], m["NOSE"]}
    J = {name: _smooth(tk.joint(name), fps) for name in needed}

    n = len(t)
    lo = max(0, rel - int(round(WIN_BEFORE_S * fps)))
    hi = min(n, rel + int(round(WIN_AFTER_S * fps)) + 1)

    # ---- per-segment peak speed + timing ------------------------------------
    speed = {seg: _speed(J[seg_kp[seg]], t) for seg in CHAIN}
    peak_val, peak_t, peak_idx = {}, {}, {}
    for seg in CHAIN:
        pv, pt, pi = _peak_in_window(speed[seg], lo, hi, t, rel)
        peak_val[seg], peak_t[seg], peak_idx[seg] = pv, pt, pi
        f[f"peakspeed_{seg}"] = _safe(pv)
        f[f"tpeak_{seg}"] = _safe(pt)            # s relative to release
        f[f"relspeed_{seg}"] = _safe(speed[seg][rel])   # speed AT release

    # ---- SEQUENCING: consecutive proximal->distal peak lags (ms) -------------
    seq_lags_ms = []
    ordered_pairs = 0
    valid_pairs = 0
    for a, b in zip(CHAIN[:-1], CHAIN[1:]):
        ta, tb = peak_t[a], peak_t[b]
        if np.isfinite(ta) and np.isfinite(tb):
            lag = (tb - ta) * 1000.0            # +ve = distal peaks AFTER proximal (proper)
            f[f"lag_{a}_{b}_ms"] = lag
            seq_lags_ms.append(lag)
            valid_pairs += 1
            if lag > 0:
                ordered_pairs += 1
        else:
            f[f"lag_{a}_{b}_ms"] = np.nan
    # overall sequence monotonicity: fraction of adjacent pairs firing distal-after-proximal
    f["seq_monotonicity"] = float(ordered_pairs / valid_pairs) if valid_pairs else np.nan
    # total proximal->distal sweep duration (ankle peak -> fingertip peak), ms
    if np.isfinite(peak_t["ankle"]) and np.isfinite(peak_t["fingertip"]):
        f["seq_sweep_ms"] = (peak_t["fingertip"] - peak_t["ankle"]) * 1000.0
    else:
        f["seq_sweep_ms"] = np.nan

    # ---- SUMMATION OF SPEED: distal/proximal peak-speed ratios ---------------
    def ratio(distal, proximal):
        d, p = peak_val[distal], peak_val[proximal]
        if np.isfinite(d) and np.isfinite(p) and p > 1e-6:
            return float(d / p)
        return np.nan
    f["sos_knee_ankle"] = ratio("knee", "ankle")
    f["sos_hip_knee"] = ratio("hip", "knee")
    f["sos_shoulder_hip"] = ratio("shoulder", "hip")
    f["sos_elbow_shoulder"] = ratio("elbow", "shoulder")
    f["sos_wrist_elbow"] = ratio("wrist", "elbow")
    f["sos_fingertip_wrist"] = ratio("fingertip", "wrist")
    f["sos_fingertip_hip"] = ratio("fingertip", "hip")     # whole upper-chain gain

    # ---- COM (mid-hip) directed velocities: thrust proxies ------------------
    com = J["MID_HIP"]
    com_v = np.gradient(com, t, axis=0)
    # forward = horizontal unit vector from COM-at-release toward the rim (geometry only)
    to_rim = np.asarray(RIM)[:2] - com[rel, :2]
    fwd = to_rim / (np.linalg.norm(to_rim) + 1e-9)
    com_fwd = com_v[:, :2] @ fwd                  # forward (toward-rim) COM speed series
    com_up = com_v[:, 2]                           # vertical COM speed series

    pv, pt, _ = _peak_in_window(com_fwd, lo, hi, t, rel)
    f["com_fwd_peak"] = _safe(pv)                  # whole-body forward thrust (depth force)
    f["com_fwd_tpeak"] = _safe(pt)
    f["com_fwd_at_release"] = _safe(com_fwd[rel])

    pvu, ptu, _ = _peak_in_window(com_up, lo, hi, t, rel)
    f["com_up_peak"] = _safe(pvu)                  # leg-drive / vertical impulse proxy
    f["com_up_tpeak"] = _safe(ptu)
    f["com_up_at_release"] = _safe(com_up[rel])

    # ---- trunk / hip angular velocity (rotation contribution) ---------------
    # shoulder-line and hip-line yaw (heading in the horizontal plane), deg/s.
    def yaw_series(p_s, p_o):
        d = (J[p_s] - J[p_o])[:, :2]
        return np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    sh_yaw = yaw_series(m["SH_s"], m["SH_o"])
    hip_yaw = yaw_series(m["HIP_s"], m["HIP_o"])
    sh_av = np.degrees(np.gradient(sh_yaw, t))
    hip_av = np.degrees(np.gradient(hip_yaw, t))
    # peak absolute angular speed in window
    pa, _, _ = _peak_in_window(np.abs(sh_av), lo, hi, t, rel)
    pah, _, _ = _peak_in_window(np.abs(hip_av), lo, hi, t, rel)
    f["shoulder_angvel_peak"] = _safe(pa)
    f["hip_angvel_peak"] = _safe(pah)
    # separation: shoulders rotating relative to hips (x-factor velocity) at release
    f["trunk_twist_angvel_release"] = _safe(sh_av[rel] - hip_av[rel])

    return f


def build() -> pd.DataFrame:
    shots = list_shots()
    rows = []
    for sh in shots:
        tk = load_tracking(sh.path)
        meta = load_metadata(sh.path)
        feats = extract_kinetic(tk)
        rows.append({
            "shot_id": sh.shot_id,
            "player": sh.player,
            "angle": meta.get("angle"),
            "depth": meta.get("depth"),
            "left_right": meta.get("left_right"),
            **feats,
        })
    return pd.DataFrame(rows)


def main():
    df = build()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUTS_DIR / "v4_kinetic.parquet"
    df.to_parquet(out, index=False)

    id_cols = {"shot_id", "player", "angle", "depth", "left_right"}
    feat_cols = [c for c in df.columns if c not in id_cols]
    miss = df[feat_cols].isna().mean().mean()
    print(f"wrote {out}")
    print(f"rows={len(df)}  n_features={len(feat_cols)}  mean_missing_rate={miss:.4f}")
    worst = df[feat_cols].isna().mean().sort_values(ascending=False).head(5)
    print("highest-missing features:")
    print(worst.to_string())
    print("features:", feat_cols)


if __name__ == "__main__":
    main()
