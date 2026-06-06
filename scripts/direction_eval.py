"""v2 — refined launch-DIRECTION features, test on angle + left_right (DEV, Scheme A CV).

Direction survives the 60fps undersampling even though speed does not, so we estimate
magnitude-free release-direction cues and their aim error vs the line to the rim:
  1. denoised fingertip velocity direction (averaged over release window)
  2. hand-path displacement direction over a window (robust to per-frame jitter)
  3. arm-pointing (wrist->fingertip) and forearm (elbow->wrist) directions
Each -> signed lateral aim angle + elevation (and elevation above the direct rim line).
Evaluates ridge with vs without these added to the existing 37 features. Hold-out sealed.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fta.config import OUTPUTS_DIR, RIM, SPLITS_DIR, TARGETS
from fta.cv import scheme_a_folds
from fta.loader import despike, list_shots, load_tracking
from fta.metric import scaled_mse_per_target
from fta.release import body_release_frame, detect_handedness

warnings.filterwarnings("ignore")
RIMA = np.array(RIM)


def _aim(d, p0):
    """signed lateral angle + elevation (+ elevation above direct rim line), degrees."""
    to = RIMA - p0
    a, b = d[:2], to[:2]
    lat = np.degrees(np.arctan2(a[0] * b[1] - a[1] * b[0], a @ b))
    el = np.degrees(np.arctan2(d[2], np.linalg.norm(d[:2])))
    elh = np.degrees(np.arctan2(to[2], np.linalg.norm(to[:2])))
    return lat, el, el - elh


def direction_features(tk):
    h = detect_handedness(tk)
    rel = body_release_frame(tk, h)
    out = {}
    if rel is None:
        return out
    side = "RIGHT" if h == "R" else "LEFT"
    ftips = [f"{side}_SECOND_FINGER_DISTAL", f"{side}_THIRD_FINGER_DISTAL"]
    have = all(f in tk.players and not np.all(np.isnan(tk.joint(f))) for f in ftips)
    t = tk.time; n = len(t)
    def J(m): return np.column_stack([despike(tk.joint(m)[:, k]) for k in range(3)])
    wr = J(f"{side}_WRIST"); el_ = J(f"{side}_ELBOW")
    tip = np.mean([J(f) for f in ftips], axis=0) if have else wr
    p0 = tip[rel]
    w = slice(max(1, rel - 2), min(n, rel + 3))

    # 1. denoised velocity direction
    v = np.gradient(tip, t, axis=0)
    dv = v[w].mean(0)
    if np.linalg.norm(dv) > 1e-6:
        lat, el, elh = _aim(dv, p0)
        out["dir_vel_lateral"], out["dir_vel_elev"], out["dir_vel_elev_above"] = lat, el, elh
    # 2. hand-path displacement direction over +-4 frames
    a, b = max(0, rel - 4), min(n - 1, rel + 4)
    dp = tip[b] - tip[a]
    if np.linalg.norm(dp) > 1e-6:
        lat, el, elh = _aim(dp, p0)
        out["dir_path_lateral"], out["dir_path_elev"], out["dir_path_elev_above"] = lat, el, elh
    # 3. arm-pointing (wrist->fingertip) and forearm (elbow->wrist)
    for nm, d in (("hand", tip[rel] - wr[rel]), ("fore", wr[rel] - el_[rel])):
        if np.linalg.norm(d) > 1e-6:
            lat, el, elh = _aim(d, p0)
            out[f"dir_{nm}_lateral"], out[f"dir_{nm}_elev"] = lat, el
    return out


def main():
    dev = set(json.load(open(SPLITS_DIR / "holdout_split.json"))["dev_ids"])
    base = pd.read_parquet(OUTPUTS_DIR / "features_2025.parquet")
    base = base[base.shot_id.isin(dev)].reset_index(drop=True)

    rows = []
    for s in list_shots():
        if s.shot_id not in dev:
            continue
        f = direction_features(load_tracking(s.path)); f["shot_id"] = s.shot_id
        rows.append(f)
    newf = pd.DataFrame(rows)
    df = base.merge(newf, on="shot_id").reset_index(drop=True)
    new_cols = [c for c in newf.columns if c != "shot_id"]
    base_cols = [c for c in base.columns if c not in {"shot_id", "player", "split", "made", *TARGETS}]
    print(f"dev={len(df)}  base feats={len(base_cols)}  new dir feats={len(new_cols)} ({new_cols})")
    print(f"new-feature NaN counts: {df[new_cols].isna().sum().to_dict()}")

    def cv(cols):
        yt = {t: df[t].to_numpy(float) for t in TARGETS}
        yp = {t: np.full(len(df), np.nan) for t in TARGETS}
        accum = {t: [] for t in TARGETS}
        for seed in range(5):
            for tr, te, _ in scheme_a_folds(df, 5, seed):
                Xtr = df.iloc[tr][cols].to_numpy(float); Xte = df.iloc[te][cols].to_numpy(float)
                col_mean = np.nanmean(Xtr, 0)
                Xtr = np.where(np.isnan(Xtr), col_mean, Xtr); Xte = np.where(np.isnan(Xte), col_mean, Xte)
                for t in TARGETS:
                    m = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(Xtr, df.iloc[tr][t])
                    yp[t][te] = m.predict(Xte)
            for t in TARGETS:
                accum[t].append(scaled_mse_per_target(yt, yp)[t])
        return {t: (float(np.mean(accum[t])), float(np.std(accum[t]))) for t in TARGETS}

    print("\nScheme A scaled-MSE (mean±sd over 5 seeds):")
    b = cv(base_cols); a = cv(base_cols + new_cols)
    for t in TARGETS:
        d = a[t][0] - b[t][0]
        print(f"  {t:11s} base={b[t][0]:.4f}±{b[t][1]:.4f}  +dir={a[t][0]:.4f}±{a[t][1]:.4f}  "
              f"delta={d:+.4f} {'(better)' if d<0 else ''}")
    json.dump({"base": b, "with_direction": a}, open(OUTPUTS_DIR / "direction_eval.json", "w"), indent=2)
    print(f"\nWrote {OUTPUTS_DIR/'direction_eval.json'}")


if __name__ == "__main__":
    main()
