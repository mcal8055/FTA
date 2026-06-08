"""v3 Foundation analyses: real recoverable ceiling + observability + speed stats.

(A) PROPER NOISE FLOOR — clean JOINT ball reconstruction -> each target, 5-fold CV.
(B) OBSERVABILITY     — body_launch_fit components vs ball_launch_fit components.
(C) SPEED STATS       — median / IQR / CV% / corr(launch_speed, depth).

Writes outputs/v3_foundation.json. Evaluated on the full 2025 session (the recoverable
ceiling is a data property, not a model; CV is over shots, stratified by player).
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

from fta.config import OUTPUTS_DIR, TARGETS
from fta.cv import scheme_a_folds
from fta.loader import list_shots, load_metadata, load_tracking
from fta.metric import scaled_mse_per_target
from fta.release import detect_handedness

from physics_v3 import ball_launch_fit, body_launch_fit

warnings.filterwarnings("ignore")


def collect() -> pd.DataFrame:
    rows = []
    for s in list_shots():
        tk = load_tracking(s.path)
        m = load_metadata(s.path)
        hand = detect_handedness(tk)
        bf = ball_launch_fit(tk)
        bo = body_launch_fit(tk, hand)
        row = {"shot_id": s.shot_id, "player": s.player, "made": m["made"],
               "angle": m["angle"], "depth": m["depth"], "left_right": m["left_right"]}
        if bf is not None:
            row.update({f"ball_{k}": v for k, v in bf.items()})
        if bo is not None:
            row.update({f"body_{k}": v for k, v in bo.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def _cv_affine(df, feat_cols, target, n_splits=5, seed=0):
    """5-fold (stratified-by-player) CV affine map feat_cols -> target.
    Returns (cv_corr, cv_scaled_mse, predictions aligned to df index)."""
    yt = df[target].to_numpy(float)
    yp = np.full(len(df), np.nan)
    for tr, te, _ in scheme_a_folds(df, n_splits, seed):
        A = np.column_stack([df.iloc[tr][c].to_numpy(float) for c in feat_cols]
                             + [np.ones(len(tr))])
        coef, *_ = np.linalg.lstsq(A, yt[tr], rcond=None)
        B = np.column_stack([df.iloc[te][c].to_numpy(float) for c in feat_cols]
                             + [np.ones(len(te))])
        yp[te] = B @ coef
    corr = float(np.corrcoef(yp, yt)[0, 1])
    return corr, yp, yt


def analysis_A(df_all):
    """Ball-reconstruction ceiling: crossing(x,y)->depth/LR, entry_angle->angle."""
    out = {}
    # require a valid ball fit + crossing
    df = df_all.dropna(subset=["ball_x_cross", "ball_y_cross", "ball_entry_angle"]).copy()
    # drop the rare flipped-frame outliers (crossing far from rim)
    df = df[df["ball_x_cross"] > 25].reset_index(drop=True)

    def ceiling(d):
        res = {}
        ang_corr, ang_pred, _ = _cv_affine(d, ["ball_entry_angle"], "angle")
        dep_corr, dep_pred, _ = _cv_affine(d, ["ball_x_cross", "ball_y_cross"], "depth")
        lr_corr, lr_pred, _ = _cv_affine(d, ["ball_x_cross", "ball_y_cross"], "left_right")
        yt = {t: d[t].to_numpy(float) for t in TARGETS}
        yp = {"angle": ang_pred, "depth": dep_pred, "left_right": lr_pred}
        per = scaled_mse_per_target(yt, yp)
        res["corr"] = {"angle": ang_corr, "depth": dep_corr, "left_right": lr_corr}
        res["scaled_mse"] = {**{t: per[t] for t in TARGETS},
                             "total": float(np.mean([per[t] for t in TARGETS]))}
        return res

    out["all"] = ceiling(df)
    out["all"]["n"] = int(len(df))
    made = df[df["made"] == 1].reset_index(drop=True)
    out["made_only"] = ceiling(made)
    out["made_only"]["n"] = int(len(made))
    out["ball_resid_median_ft"] = float(np.nanmedian(df["ball_resid_ft"]))
    return out, df


def analysis_B(df_all):
    """Observability: body launch-state components vs ball (teacher) components."""
    pairs = {
        "speed": ("body_speed", "ball_speed"),
        "elevation": ("body_elevation", "ball_elevation"),
        "azimuth": ("body_azimuth", "ball_azimuth"),
        "release_x": ("body_release_x", "ball_release_x"),
        "release_y": ("body_release_y", "ball_release_y"),
        "release_z": ("body_release_z", "ball_release_z"),
    }
    out = {}
    for name, (bc, tc) in pairs.items():
        d = df_all.dropna(subset=[bc, tc])
        d = d[d["ball_x_cross"] > 25]
        if len(d) < 10 or np.ptp(d[bc]) == 0 or np.ptp(d[tc]) == 0:
            out[name] = float("nan")
            continue
        out[name] = float(np.corrcoef(d[bc].to_numpy(float), d[tc].to_numpy(float))[0, 1])
    return out


def analysis_C(df_all):
    d = df_all.dropna(subset=["ball_speed", "depth"])
    d = d[d["ball_x_cross"] > 25]
    sp = d["ball_speed"].to_numpy(float)
    dep = d["depth"].to_numpy(float)
    q1, med, q3 = np.percentile(sp, [25, 50, 75])
    return {
        "median": float(med),
        "iqr_lo": float(q1), "iqr_hi": float(q3),
        "cv_pct": float(100.0 * sp.std() / sp.mean()),
        "corr_speed_depth": float(np.corrcoef(sp, dep)[0, 1]),
        "n": int(len(d)),
    }


def main():
    df = collect()
    n_ball = df["ball_speed"].notna().sum()
    n_body = df["body_speed"].notna().sum()
    print(f"collected {len(df)} shots; ball-fit ok={n_ball}, body-fit ok={n_body}")

    A, dfA = analysis_A(df)
    B = analysis_B(df)
    C = analysis_C(df)

    print("\n(A) BALL ceiling (5-fold CV, all clean):")
    for t in TARGETS:
        print(f"  {t:11s} corr={A['all']['corr'][t]:+.3f}  "
              f"sMSE={A['all']['scaled_mse'][t]:.4f}")
    print(f"  TOTAL sMSE={A['all']['scaled_mse']['total']:.4f}  (n={A['all']['n']})")
    print("  made/swish-only:")
    for t in TARGETS:
        print(f"    {t:11s} corr={A['made_only']['corr'][t]:+.3f}  "
              f"sMSE={A['made_only']['scaled_mse'][t]:.4f}")

    print("\n(B) OBSERVABILITY body vs ball (corr):")
    for k, v in B.items():
        print(f"  {k:11s} {v:+.3f}")

    print("\n(C) SPEED stats:")
    print(f"  median={C['median']:.2f} ft/s  IQR=[{C['iqr_lo']:.2f},{C['iqr_hi']:.2f}]  "
          f"CV={C['cv_pct']:.1f}%  corr(speed,depth)={C['corr_speed_depth']:+.3f}")

    depth_corr = A["all"]["corr"]["depth"]
    depth_recoverable = bool(depth_corr > 0.35)
    print(f"\ndepth_recoverable = {depth_recoverable} (CV ball-depth corr={depth_corr:.3f})")

    out = {
        "n_shots": int(len(df)),
        "n_ball_fit": int(n_ball), "n_body_fit": int(n_body),
        "ball_ceiling": A,
        "observability": B,
        "speed_stats": C,
        "depth_recoverable": depth_recoverable,
        "ref_release_z_ft": 8.0,
    }
    OUTPUTS_DIR.mkdir(exist_ok=True)
    with open(OUTPUTS_DIR / "v3_foundation.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {OUTPUTS_DIR / 'v3_foundation.json'}")


if __name__ == "__main__":
    main()
