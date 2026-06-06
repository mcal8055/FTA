"""Step 1/2 — irreducible-noise floor on the targets (DEV only).

Now feasible because release detection lets us segment the CLEAN flight
(release -> first z-minimum after apex = before rim/floor contact). We:
  1. fit a gravity-constrained parabola z(t) + linear x(t), y(t) on the clean flight;
     the fit RESIDUAL (feet) measures ball-tracking noise.
  2. reconstruct entry angle at the descending 10-ft crossing -> compare to provided.
  3. fit an affine map crossing(x,y ft) -> (depth, left_right inches); residuals = floor.
These residuals lower-bound the error any BODY-only model can achieve, since even the
ball (near-perfect information) does not pin the provided targets exactly.
"""
from __future__ import annotations

import json

import numpy as np

from fta.config import OUTPUTS_DIR, SPLITS_DIR
from fta.loader import list_shots, load_tracking, load_metadata
from fta.release import ball_release_frame, detect_handedness

G = 32.174  # ft/s^2


def clean_flight(tk, rel):
    """Indices from release to the first z local-min after the apex (pre-contact)."""
    z = tk.ball[:, 2]
    n = len(z)
    if rel is None or rel >= n - 6:
        return None
    apex = rel + int(np.nanargmax(z[rel:]))
    end = apex
    for i in range(apex + 1, n - 1):
        if not np.isfinite(z[i]):
            break
        if z[i] <= z[i + 1]:   # z stops descending -> contact/bounce
            end = i
            break
        end = i
    idx = np.arange(rel, end + 1)
    idx = idx[np.isfinite(tk.ball[idx]).all(1)]
    return idx if len(idx) >= 8 else None


def fit_shot(tk, rel):
    idx = clean_flight(tk, rel)
    if idx is None:
        return None
    t = tk.time[idx] - tk.time[idx[0]]
    x, y, z = tk.ball[idx, 0], tk.ball[idx, 1], tk.ball[idx, 2]
    # robust: iteratively reject ball outliers (README warns ball tracking is noisy)
    keep = np.ones(len(t), bool)
    ax = ay = None; vz0 = z0 = 0.0
    for _ in range(3):
        ax = np.polyfit(t[keep], x[keep], 1); ay = np.polyfit(t[keep], y[keep], 1)
        vz0, z0 = np.polyfit(t[keep], z[keep] + 0.5 * G * t[keep] ** 2, 1)
        r_all = np.sqrt((x - np.polyval(ax, t))**2 + (y - np.polyval(ay, t))**2
                        + (z - (z0 + vz0 * t - 0.5 * G * t * t))**2)
        nk = r_all < max(2.5 * np.median(r_all), 0.3)
        if nk.sum() < 6 or (nk == keep).all():
            keep = nk if nk.sum() >= 6 else keep
            break
        keep = nk
    resid = float(np.sqrt(np.mean(r_all[keep] ** 2)))
    # descending 10-ft crossing
    a, b, c = -0.5 * G, vz0, z0 - 10.0
    disc = b * b - 4 * a * c
    if disc < 0:
        return resid, None, None, None
    tc = (-b - np.sqrt(disc)) / (2 * a)
    vzc = vz0 - G * tc
    entry = np.degrees(np.arctan2(-vzc, np.hypot(ax[0], ay[0])))
    return resid, entry, ax[0] * tc + ax[1], ay[0] * tc + ay[1]


def main():
    dev = set(json.load(open(SPLITS_DIR / "holdout_split.json"))["dev_ids"])
    resids, ea_pred, xc, yc, ea_t, dep_t, lr_t = [], [], [], [], [], [], []
    n_fit = 0
    for s in list_shots():
        if s.shot_id not in dev:
            continue
        tk = load_tracking(s.path); m = load_metadata(s.path)
        out = fit_shot(tk, ball_release_frame(tk, detect_handedness(tk)))
        if out is None:
            continue
        resid, entry, x, y = out
        resids.append(resid)
        if entry is not None and np.isfinite(entry):
            n_fit += 1
            ea_pred.append(entry); xc.append(x); yc.append(y)
            ea_t.append(m["angle"]); dep_t.append(m["depth"]); lr_t.append(m["left_right"])

    resids = np.array(resids)
    ea_pred, xc, yc = map(np.array, (ea_pred, xc, yc))
    ea_t, dep_t, lr_t = map(np.array, (ea_t, dep_t, lr_t))

    print(f"clean-flight fits: {len(resids)} shots; crossing solved: {n_fit}")
    print(f"\nball parabola-fit residual (measurement noise): "
          f"median={np.median(resids)*12:.2f} in  mean={resids.mean()*12:.2f} in")

    out = {"n_fit": int(len(resids)),
           "ball_fit_residual_in": {"median": round(float(np.median(resids))*12, 3),
                                     "mean": round(float(resids.mean())*12, 3)}}

    ae = ea_pred - ea_t
    print(f"\nENTRY ANGLE recon vs provided: RMSE={np.sqrt(np.mean(ae**2)):.2f} deg "
          f"MAE={np.mean(np.abs(ae)):.2f} corr={np.corrcoef(ea_pred, ea_t)[0,1]:.3f}")
    out["angle_floor_deg"] = {"rmse": round(float(np.sqrt(np.mean(ae**2))), 3),
                              "mae": round(float(np.mean(np.abs(ae))), 3),
                              "corr": round(float(np.corrcoef(ea_pred, ea_t)[0, 1]), 3)}

    A = np.c_[xc, yc, np.ones_like(xc)]
    for name, tgt in (("depth", dep_t), ("left_right", lr_t)):
        coef, *_ = np.linalg.lstsq(A, tgt, rcond=None)
        res = tgt - A @ coef
        rmse = float(np.sqrt(np.mean(res**2)))
        print(f"{name:11s} affine-recon RMSE={rmse:.2f} in  MAE={np.mean(np.abs(res)):.2f} in  "
              f"corr={np.corrcoef(A@coef, tgt)[0,1]:.3f}")
        out.setdefault("position_floor_in", {})[name] = {
            "rmse": round(rmse, 3), "corr": round(float(np.corrcoef(A@coef, tgt)[0, 1]), 3)}

    OUTPUTS_DIR.mkdir(exist_ok=True)
    with open(OUTPUTS_DIR / "noise_floor.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {OUTPUTS_DIR/'noise_floor.json'}")


if __name__ == "__main__":
    main()
