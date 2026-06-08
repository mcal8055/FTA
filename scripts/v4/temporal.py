"""v4 feature family — MULTI-TEMPORAL kinematics (leakage-free, body-pose only).

HYPOTHESIS (multi-temporal commitment).  Different free-throw targets are committed
at different times in the motion, through different body segments:
  * DEPTH      — committed EARLY, via mid-jump center-of-mass forward momentum
                 (whole-body forward thrust toward the rim);
  * ANGLE      — committed MID, via shooting-elbow geometry / forearm elevation;
  * LEFT_RIGHT — committed LATE, at the wrist snap.
v1 sampled kinematics essentially at a SINGLE instant (the release frame), so it could
only see the LATE state and tended to memorize shooter identity from static positions.
Here we instead sample a compact set of mechanism-bearing scalars at MULTIPLE FIXED time
offsets BEFORE release, so the model can read the early/ mid/ late commitments directly.

LEAKAGE DISCIPLINE.
  * Body keypoints ONLY — no ball, no outcomes touch the features.
  * Offsets are FIXED in seconds relative to the body-detected release
    (fta.release.body_release_frame), NOT tuned against any leaderboard / target.
  * The per-target "which-offset-does-corr-peak" analysis at the bottom is DESCRIPTIVE
    interpretability only; it is printed, NOT used to pick features or offsets.
  * Directional quantities use a per-shot body frame (forward = horizontal facing toward
    the nose, perpendicular to the shoulder line), so left-handers and right-handers are
    mirrored consistently and no global-court coordinate leaks shooter position.

OFFSETS (seconds before release; 0.0 = release instant):
    {-0.50, -0.35, -0.20, -0.10, -0.05, 0.0}
At each offset we capture mechanism scalars:
    com_fwd_vel   — mid-hip (COM proxy) velocity toward the rim (ft/s)   [DEPTH lever]
    com_height    — mid-hip height (ft)
    com_fwd_pos   — mid-hip forward position relative to its own release-frame value (ft)
    elbow_angle   — shooting elbow flexion/extension (deg)                [ANGLE lever]
    wrist_height  — shooting-wrist height above mid-hip (ft)
    wrist_fwd     — shooting-wrist forward position relative to mid-hip (ft) [LEFT_RIGHT/late]
    knee_flex     — shooting-side knee flexion angle (deg)
    trunk_lean    — trunk lean from vertical (deg)

Plus a few CROSS-OFFSET summary scalars (still leakage-free, fixed construction):
    com_fwd_vel_peak / t (kinetic-chain proximal driver timing), elbow extension swept,
    wrist rise swept.

extract_temporal(tk) -> dict[str, float]   (NaN where a frame/segment is unavailable).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from fta.config import OUTPUTS_DIR, SPLITS_DIR, TARGETS
from fta.loader import Tracking, despike, list_shots, load_metadata, load_tracking
from fta.release import body_release_frame, detect_handedness

# Fixed multi-temporal offsets (seconds relative to release). Negative = before release.
OFFSETS_S = (-0.50, -0.35, -0.20, -0.10, -0.05, 0.0)


def _tag(off: float) -> str:
    """Stable column suffix for an offset, e.g. -0.50 -> 'm500', 0.0 -> 'm000' (ms)."""
    ms = int(round(-off * 1000))
    return f"m{ms:03d}"


# --- geometry helpers (mirror fta.features conventions) ----------------------
def _angle(a, b, c):
    ba, bc = a - b, c - b
    cos = (ba * bc).sum(-1) / (np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def _smooth(arr, fps):
    """Despike + Savitzky-Golay each column (~0.18 s window), NaN-robust."""
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


def _safe(x):
    return float(x) if x is not None and np.isfinite(x) else np.nan


# --- main extractor ----------------------------------------------------------
def extract_temporal(tk: Tracking) -> dict:
    """Multi-temporal kinematic descriptors for one shot. NaN where unavailable.

    Also returns, prefixed '_dbg_', the per-offset frame index used (negative if a
    requested offset was clamped to the first available frame) so the caller can
    report the missing/clamp rate. These _dbg_ keys are dropped before the parquet.
    """
    hand = detect_handedness(tk)
    rel = body_release_frame(tk, hand)
    t = tk.time
    fps = tk.sampling_rate or int(round(1.0 / np.median(np.diff(t))))
    f = {"hand_right": 1.0 if hand == "R" else 0.0}
    if rel is None:
        # release undetectable: emit NaNs for every feature, flag fully missing.
        for off in OFFSETS_S:
            tag = _tag(off)
            for base in ("com_fwd_vel", "com_height", "com_fwd_pos", "elbow_angle",
                         "wrist_height", "wrist_fwd", "knee_flex", "trunk_lean"):
                f[f"{base}_{tag}"] = np.nan
            f[f"_dbg_clamp_{tag}"] = 1.0
        for s in ("com_fwd_vel_peak", "t_com_fwd_vel_peak", "elbow_ext_swept",
                  "wrist_rise_swept", "com_fwd_advance"):
            f[s] = np.nan
        f["_dbg_rel_missing"] = 1.0
        return f
    f["_dbg_rel_missing"] = 0.0

    SH_s = "RIGHT_SHOULDER" if hand == "R" else "LEFT_SHOULDER"
    SH_o = "LEFT_SHOULDER" if hand == "R" else "RIGHT_SHOULDER"
    EL_s = "RIGHT_ELBOW" if hand == "R" else "LEFT_ELBOW"
    WR_s = "RIGHT_WRIST" if hand == "R" else "LEFT_WRIST"
    HIP_s = "RIGHT_HIP" if hand == "R" else "LEFT_HIP"
    KNEE_s = "RIGHT_KNEE" if hand == "R" else "LEFT_KNEE"
    ANK_s = "RIGHT_ANKLE" if hand == "R" else "LEFT_ANKLE"
    names = ["NOSE", "NECK", "MID_HIP", SH_s, SH_o, EL_s, WR_s, HIP_s, KNEE_s, ANK_s]
    J = {name: _smooth(tk.joint(name), fps) for name in names}

    # --- per-shot body frame at release: forward = facing toward nose (perp shoulders) ---
    sl = (J[SH_s] - J[SH_o])[rel, :2]
    fwd = np.array([-sl[1], sl[0]])
    nose_dir = (J["NOSE"] - J["MID_HIP"])[rel, :2]
    if fwd @ nose_dir < 0:
        fwd = -fwd
    fwd = fwd / (np.linalg.norm(fwd) + 1e-9)

    # --- derived series over the whole shot ---
    mid = J["MID_HIP"]
    mid_v = np.gradient(mid, t, axis=0)
    com_fwd_vel = mid_v[:, :2] @ fwd                      # forward (toward-rim) COM speed
    com_fwd_pos = (mid[:, :2] - mid[rel, :2]) @ fwd       # forward COM displ vs release
    com_height = mid[:, 2]
    elbow = _angle(J[SH_s], J[EL_s], J[WR_s])
    knee = _angle(J[HIP_s], J[KNEE_s], J[ANK_s])
    trunk = J["NECK"] - J["MID_HIP"]
    trunk_lean = np.degrees(np.arctan2(np.linalg.norm(trunk[:, :2], axis=1), trunk[:, 2]))
    wr = J[WR_s]
    wrist_height = (wr - mid)[:, 2]                       # wrist height above hip
    wrist_fwd = (wr[:, :2] - mid[:, :2]) @ fwd            # wrist forward of hip

    n = len(t)
    t_rel = t[rel]

    def frame_at(off: float):
        """Nearest available frame to (release + off seconds); clamp into range.
        Returns (idx, clamped_flag)."""
        target_t = t_rel + off
        i = int(np.argmin(np.abs(t - target_t)))
        clamped = False
        i0 = i
        i = int(np.clip(i, 0, n - 1))
        if i != i0:
            clamped = True
        # also flag if the requested offset fell before frame 0 of the recording
        if off < 0 and t[0] > target_t + 1e-6:
            clamped = True
        return i, clamped

    series = {
        "com_fwd_vel": com_fwd_vel,
        "com_height": com_height,
        "com_fwd_pos": com_fwd_pos,
        "elbow_angle": elbow,
        "wrist_height": wrist_height,
        "wrist_fwd": wrist_fwd,
        "knee_flex": knee,
        "trunk_lean": trunk_lean,
    }

    frames = {}
    for off in OFFSETS_S:
        tag = _tag(off)
        i, clamped = frame_at(off)
        frames[off] = i
        f[f"_dbg_clamp_{tag}"] = 1.0 if clamped else 0.0
        for base, s in series.items():
            f[f"{base}_{tag}"] = _safe(s[i])

    # --- cross-offset kinetic-chain summaries (fixed construction, leakage-free) ---
    # Window = from the earliest sampled offset frame to release.
    lo = frames[OFFSETS_S[0]]
    hi = rel
    if hi <= lo:
        lo = max(0, hi - 1)
    seg = slice(lo, hi + 1)

    cfv = com_fwd_vel[seg]
    good = np.isfinite(cfv)
    if good.any():
        k = int(np.nanargmax(np.where(good, cfv, -np.inf)))
        f["com_fwd_vel_peak"] = _safe(cfv[k])
        f["t_com_fwd_vel_peak"] = _safe(t[lo + k] - t_rel)  # sec rel release (<=0)
    else:
        f["com_fwd_vel_peak"] = np.nan
        f["t_com_fwd_vel_peak"] = np.nan

    def swept(s):
        sw = s[seg]
        sw = sw[np.isfinite(sw)]
        return float(sw.max() - sw.min()) if sw.size else np.nan

    f["elbow_ext_swept"] = swept(elbow)        # total elbow extension over the windup
    f["wrist_rise_swept"] = swept(wrist_height)
    # net forward COM advance from window start to release (whole-body thrust magnitude)
    f["com_fwd_advance"] = _safe(com_fwd_pos[lo] - com_fwd_pos[rel]) if np.isfinite(
        com_fwd_pos[lo]) and np.isfinite(com_fwd_pos[rel]) else np.nan

    return f


def build() -> None:
    split = json.load(open(SPLITS_DIR / "holdout_split.json"))
    dev, test = set(split["dev_ids"]), set(split["test_ids"])

    rows = []
    n_clamp_offset = {off: 0 for off in OFFSETS_S}
    n_rel_missing = 0
    for i, s in enumerate(list_shots(), 1):
        tk = load_tracking(s.path)
        m = load_metadata(s.path)
        feat = extract_temporal(tk)
        n_rel_missing += int(feat.pop("_dbg_rel_missing", 0.0))
        for off in OFFSETS_S:
            n_clamp_offset[off] += int(feat.pop(f"_dbg_clamp_{_tag(off)}", 0.0))
        rec = {"shot_id": s.shot_id, "player": s.player,
               "split": "dev" if s.shot_id in dev else "test" if s.shot_id in test else "?",
               "made": m["made"]}
        rec.update({t: m[t] for t in TARGETS})
        rec.update(feat)
        rows.append(rec)
        if i % 100 == 0:
            print(f"  {i} shots...")

    df = pd.DataFrame(rows)
    OUTPUTS_DIR.mkdir(exist_ok=True)
    out = OUTPUTS_DIR / "v4_temporal.parquet"
    df.to_parquet(out, index=False)

    meta_cols = ("shot_id", "player", "split", "made", *TARGETS)
    feat_cols = [c for c in df.columns if c not in meta_cols]
    print(f"\nWrote {out}")
    print(f"  shots={len(df)}  features={len(feat_cols)}  "
          f"dev={(df.split=='dev').sum()}  test={(df.split=='test').sum()}")
    print(f"  release undetected (all-NaN) shots: {n_rel_missing}/{len(df)} "
          f"({100*n_rel_missing/len(df):.1f}%)")
    print("  per-offset clamp counts (requested offset fell before frame 0):")
    for off in OFFSETS_S:
        print(f"    {off:+.2f}s: {n_clamp_offset[off]}/{len(df)} "
              f"({100*n_clamp_offset[off]/len(df):.1f}%)")
    na = df[feat_cols].isna().mean()
    bad = na[na > 0]
    print(f"  features with any NaN: {len(bad)}"
          + (f" (max NaN rate {bad.max():.3f})" if len(bad) else ""))

    # ----- DESCRIPTIVE interpretability (NOT used for feature selection) -----
    # For each target, on DEV shots only, find at which offset |corr(feature,target)|
    # peaks for the three mechanism levers (COM fwd vel = depth, elbow = angle,
    # wrist fwd = left_right). Printed to confirm the multi-temporal hypothesis.
    print("\n  [DESCRIPTIVE — train-style corr-by-offset peak; NOT used for selection]")
    dev_df = df[df.split == "dev"]
    lever_for = {
        "depth": "com_fwd_vel",
        "angle": "elbow_angle",
        "left_right": "wrist_fwd",
    }
    for tgt in TARGETS:
        y = dev_df[tgt].to_numpy()
        for lever, base in lever_for.items():
            corrs = []
            for off in OFFSETS_S:
                col = dev_df[f"{base}_{_tag(off)}"].to_numpy()
                mask = np.isfinite(col) & np.isfinite(y)
                if mask.sum() > 10 and np.ptp(col[mask]) > 0:
                    c = np.corrcoef(col[mask], y[mask])[0, 1]
                else:
                    c = np.nan
                corrs.append(c)
            ac = np.abs(np.array(corrs, float))
            if np.isfinite(ac).any():
                kbest = int(np.nanargmax(ac))
                print(f"    target={tgt:<10} lever={base:<13} "
                      f"|corr| peaks at {OFFSETS_S[kbest]:+.2f}s "
                      f"(|r|={ac[kbest]:.3f})")


if __name__ == "__main__":
    build()
