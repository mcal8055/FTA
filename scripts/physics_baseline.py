"""v2 — measured-launch ballistic baseline + velocity-observability diagnosis (DEV/train-CV).

Tests whether exact projectile physics, fed the launch state measured from fingertip
tracking, can predict the targets. Finding: launch SPEED is grossly under-measured at
60fps (~9 vs required ~22 ft/s) so raw physics fails. We then isolate DIRECTION from
SPEED: scale v0 by a single global factor k (fit on train) and apply a per-fold affine
calibration of the physics crossing to the targets, to see which targets are physically
observable (direction-driven angle/left-right) vs not (speed-driven depth).
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

from fta.config import OUTPUTS_DIR, SPLITS_DIR, TARGETS
from fta.cv import scheme_a_folds
from fta.loader import list_shots, load_metadata, load_tracking
from fta.metric import scaled_mse_per_target
from fta.physics import ballistic_crossing, launch_state
from fta.release import body_release_frame, detect_handedness

warnings.filterwarnings("ignore")


def collect(dev):
    rows = []
    for s in list_shots():
        if s.shot_id not in dev:
            continue
        tk = load_tracking(s.path); h = detect_handedness(tk)
        rel = body_release_frame(tk, h)
        ls = launch_state(tk, rel, h)
        if ls is None:
            continue
        p0, v0 = ls
        m = load_metadata(s.path)
        rows.append({"shot_id": s.shot_id, "player": s.player,
                     "p0": p0, "v0": v0, "speed": float(np.linalg.norm(v0)),
                     **{t: m[t] for t in TARGETS}})
    return pd.DataFrame(rows)


def crossings_at_scale(df, k):
    xs, ys, eas, ok = [], [], [], []
    for _, r in df.iterrows():
        cr = ballistic_crossing(r.p0, r.v0 * k)
        if cr is None:
            xs.append(np.nan); ys.append(np.nan); eas.append(np.nan); ok.append(False)
        else:
            ea, xc, yc, t = cr
            xs.append(xc); ys.append(yc); eas.append(ea); ok.append(True)
    return np.array(xs), np.array(ys), np.array(eas), np.array(ok)


def main():
    dev = set(json.load(open(SPLITS_DIR / "holdout_split.json"))["dev_ids"])
    df = collect(dev).reset_index(drop=True)
    print(f"shots={len(df)}  measured launch speed: median={df.speed.median():.1f} ft/s "
          f"(free throw needs ~22)")

    # how much scaling to even reach the rim?
    for k in (1.0, 1.5, 2.0, 2.4, 3.0):
        _, _, _, ok = crossings_at_scale(df, k)
        print(f"  v0 x{k}: {ok.mean()*100:4.0f}% trajectories reach rim height")

    # pick global k that maximizes reach, then evaluate physics+affine under Scheme A CV
    ks = np.linspace(1.5, 3.5, 11)
    reach = [crossings_at_scale(df, k)[3].mean() for k in ks]
    k = float(ks[int(np.argmax(reach))])
    print(f"\nusing global speed scale k={k:.2f} (max reach {max(reach)*100:.0f}%)")

    xs, ys, eas, ok = crossings_at_scale(df, k)
    df2 = df[ok].reset_index(drop=True)
    X = np.column_stack([xs[ok], ys[ok], np.ones(ok.sum())])
    feat = {"angle": eas[ok], "depth_xy": X, "lr_xy": X}

    # Scheme A CV: per-fold affine calibration of physics outputs -> targets
    yt = {t: df2[t].to_numpy(float) for t in TARGETS}
    yp = {t: np.full(len(df2), np.nan) for t in TARGETS}
    for tr, te, _ in scheme_a_folds(df2, 5, 0):
        # angle: linear a*entry+b
        A = np.column_stack([eas[ok][tr], np.ones(len(tr))])
        ca = np.linalg.lstsq(A, df2.iloc[tr]["angle"], rcond=None)[0]
        yp["angle"][te] = np.column_stack([eas[ok][te], np.ones(len(te))]) @ ca
        # depth, left_right: affine on crossing (x,y)
        for t in ("depth", "left_right"):
            c = np.linalg.lstsq(X[tr], df2.iloc[tr][t], rcond=None)[0]
            yp[t][te] = X[te] @ c

    per = scaled_mse_per_target(yt, yp)
    print("\nphysics(direction)+global-speed+affine, Scheme A scaled-MSE:")
    for t in TARGETS:
        r = np.corrcoef(yp[t], yt[t])[0, 1]
        print(f"  {t:11s} scaled-MSE={per[t]:.4f}  corr(pred,true)={r:+.3f}")
    print("\n(reference: ridge Scheme A per-target ~ angle 0.008, depth 0.015, left_right 0.013)")

    json.dump({"measured_speed_median": float(df.speed.median()),
               "global_k": k, "reach_frac": float(max(reach)),
               "scheme_a_scaled_mse": per},
              open(OUTPUTS_DIR / "physics_baseline.json", "w"), indent=2)
    print(f"\nWrote {OUTPUTS_DIR/'physics_baseline.json'}")


if __name__ == "__main__":
    main()
