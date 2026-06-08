"""v3 physics substrate — corrected launch-state estimation.

Two estimators over one shot's `Tracking`:

  ball_launch_fit(tracking)        -> TEACHER / ground-truth launch state.
      Fits a gravity-constrained parabola (g = 32.174 ft/s^2) to the CLEAN airborne
      arc: the contiguous run of frames around the trajectory apex whose ball z stays
      above 10.8 ft (>= 8 frames required). This window is BEFORE the descending rim
      plane, so it never contains rim / backboard contact (which deflects the ball on
      misses). Ball outliers are rejected robustly (the SPL README warns ball tracking
      is the noisiest object). From the fit we reconstruct the descending 10 ft-plane
      crossing, entry angle there, and a launch state at a FIXED reference release
      height so launch speed/elevation are comparable across shots.

  body_launch_fit(tracking, hand)  -> STUDENT launch state from BODY POSE ONLY.
      Fits the shooting-hand end-effector trajectory (mean of fingertip distals + wrist,
      with forearm as fallback) over a multi-frame window around the body-detected
      release (fta.release.body_release_frame). A degree-2 polynomial per axis gives a
      SMOOTH release position and velocity VECTOR evaluated at the release instant — not
      a single-frame finite difference. Everything is then expressed in a hoop-relative
      frame and pushed through the analytic ballistic decoder.

Coordinate frame (verified empirically on the 2025 session, 456/458 shots):
  JSON feet frame; shooter shoots toward +x; rim center ~ (41.67, 0, 10) ft. The 2
  outlier shots have a flipped frame (release detected late); we detect the shooting
  direction per shot from the hand's own horizontal motion so the hoop-relative
  features stay sign-correct.
"""
from __future__ import annotations

import numpy as np

from fta.config import RIM
from fta.loader import Tracking, despike
from fta.physics import G, RIM_Z, ballistic_crossing
from fta.release import body_release_frame, detect_handedness

# Reference release height (ft) at which launch speed / elevation are reported, so the
# magnitude is a comparable scalar across shots rather than depending on where each
# parabola happens to be sampled. 8.0 ft ~ ball-off-fingers height for these shooters.
REF_RELEASE_Z = 8.0
Z_AIRBORNE = 10.8   # ft; clean-arc threshold (above the rim plane, pre-contact)
MIN_ARC = 8         # min clean-arc frames to attempt a fit


# --------------------------------------------------------------------------- #
# (1) BALL teacher                                                            #
# --------------------------------------------------------------------------- #
def _clean_arc_idx(tk: Tracking) -> np.ndarray | None:
    """Contiguous frame indices around the ball-z apex with z > Z_AIRBORNE."""
    z = tk.ball[:, 2]
    n = len(z)
    if n < MIN_ARC:
        return None
    finite = np.isfinite(z)
    if finite.sum() < MIN_ARC:
        return None
    apex = int(np.nanargmax(z))
    if not (np.isfinite(z[apex]) and z[apex] > Z_AIRBORNE):
        return None
    lo = apex
    while lo > 0 and np.isfinite(z[lo - 1]) and z[lo - 1] > Z_AIRBORNE:
        lo -= 1
    hi = apex
    while hi < n - 1 and np.isfinite(z[hi + 1]) and z[hi + 1] > Z_AIRBORNE:
        hi += 1
    idx = np.arange(lo, hi + 1)
    idx = idx[np.isfinite(tk.ball[idx]).all(1)]
    return idx if len(idx) >= MIN_ARC else None


def ball_launch_fit(tk: Tracking) -> dict | None:
    """Gravity-constrained parabola fit to the clean airborne arc (teacher state).

    Returns a dict (or None if no fittable arc):
      vx, vy            horizontal velocity (ft/s, constant in drag-free flight)
      vz0, z0           vertical launch params of the fitted parabola (at arc start)
      apex_z            fitted apex height (ft)
      x_cross, y_cross  reconstructed DESCENDING 10 ft-plane crossing (ft)
      entry_angle       entry angle below horizontal at the crossing (deg)
      speed             launch speed magnitude at REF_RELEASE_Z (ft/s)
      elevation         launch elevation angle at REF_RELEASE_Z (deg, above horizontal)
      azimuth           launch azimuth in the hoop frame (deg; 0 = straight at rim)
      release_x/y/z     release-point proxy: parabola position at REF_RELEASE_Z (ft)
      resid_ft          RMS gravity-parabola fit residual (ft) — ball noise proxy
      n_arc             frames used after outlier rejection
    """
    idx = _clean_arc_idx(tk)
    if idx is None:
        return None
    t0 = tk.time[idx[0]]
    t = tk.time[idx] - t0
    x, y, z = tk.ball[idx, 0], tk.ball[idx, 1], tk.ball[idx, 2]

    keep = np.ones(len(t), bool)
    ax = ay = None
    vz0 = z0 = 0.0
    for _ in range(3):
        ax = np.polyfit(t[keep], x[keep], 1)
        ay = np.polyfit(t[keep], y[keep], 1)
        vz0, z0 = np.polyfit(t[keep], z[keep] + 0.5 * G * t[keep] ** 2, 1)
        zhat = z0 + vz0 * t - 0.5 * G * t * t
        r = np.sqrt((x - np.polyval(ax, t)) ** 2
                    + (y - np.polyval(ay, t)) ** 2
                    + (z - zhat) ** 2)
        nk = r < max(2.5 * np.median(r), 0.30)
        if nk.sum() < 6 or (nk == keep).all():
            break
        keep = nk
    zhat = z0 + vz0 * t - 0.5 * G * t * t
    r = np.sqrt((x - np.polyval(ax, t)) ** 2
                + (y - np.polyval(ay, t)) ** 2
                + (z - zhat) ** 2)
    resid = float(np.sqrt(np.mean(r[keep] ** 2)))

    vx, vy = float(ax[0]), float(ay[0])
    vz0 = float(vz0); z0 = float(z0)
    apex_z = z0 + vz0 * vz0 / (2 * G) if vz0 > 0 else z0

    # descending 10 ft crossing of the FITTED parabola
    a, b, c = -0.5 * G, vz0, z0 - RIM_Z
    disc = b * b - 4 * a * c
    if disc < 0:
        x_cross = y_cross = entry = np.nan
    else:
        tc = (-b - np.sqrt(disc)) / (2 * a)
        x_cross = vx * tc + ax[1]
        y_cross = vy * tc + ay[1]
        vzc = vz0 - G * tc
        entry = np.degrees(np.arctan2(-vzc, np.hypot(vx, vy)))

    # launch state reported at a FIXED reference height (rising branch). Solve the
    # parabola for the time it last passes REF_RELEASE_Z on the way up.
    a2, b2, c2 = -0.5 * G, vz0, z0 - REF_RELEASE_Z
    disc2 = b2 * b2 - 4 * a2 * c2
    if disc2 >= 0:
        t_ref = (-b2 + np.sqrt(disc2)) / (2 * a2)   # smaller root = rising
    else:
        t_ref = 0.0
    vz_ref = vz0 - G * t_ref
    rx = vx * t_ref + ax[1]
    ry = vy * t_ref + ay[1]
    rz = z0 + vz0 * t_ref - 0.5 * G * t_ref ** 2

    vh = np.hypot(vx, vy)
    speed = float(np.sqrt(vh * vh + vz_ref * vz_ref))
    elevation = float(np.degrees(np.arctan2(vz_ref, vh)))
    # azimuth toward the rim: shooting direction is the sign of (rim - release) . v_h
    to_rim = np.array([RIM[0] - rx, RIM[1] - ry])
    sgn = np.sign(np.dot(to_rim, [vx, vy])) or 1.0
    # signed lateral angle of velocity relative to the rim bearing
    bearing = np.arctan2(to_rim[1], to_rim[0])
    vdir = np.arctan2(sgn * vy, sgn * vx)
    azimuth = float(np.degrees(np.arctan2(np.sin(vdir - bearing), np.cos(vdir - bearing))))

    return {
        "vx": vx, "vy": vy, "vz0": vz0, "z0": z0,
        "apex_z": float(apex_z),
        "x_cross": float(x_cross), "y_cross": float(y_cross),
        "entry_angle": float(entry),
        "speed": speed, "elevation": elevation, "azimuth": azimuth,
        "release_x": float(rx), "release_y": float(ry), "release_z": float(rz),
        "resid_ft": resid, "n_arc": int(keep.sum()),
    }


# --------------------------------------------------------------------------- #
# (2) BODY student                                                            #
# --------------------------------------------------------------------------- #
def _end_effector(tk: Tracking, hand: str) -> np.ndarray | None:
    """Shooting-hand end-effector (n,3): mean of available fingertip distals + wrist,
    falling back through the forearm. Despiked per axis."""
    side = "RIGHT" if hand == "R" else "LEFT"
    candidates = [
        f"{side}_SECOND_FINGER_DISTAL", f"{side}_THIRD_FINGER_DISTAL",
        f"{side}_WRIST",
    ]
    cols = []
    for j in candidates:
        if j in tk.players:
            a = tk.joint(j)
            if np.isfinite(a).all(1).mean() > 0.5:
                cols.append(a)
    if not cols:
        # forearm fallback
        for j in (f"{side}_WRIST", f"{side}_ELBOW"):
            if j in tk.players:
                a = tk.joint(j)
                if np.isfinite(a).all(1).mean() > 0.5:
                    cols.append(a)
    if not cols:
        return None
    ee = np.nanmean(np.stack(cols), axis=0)
    ee = np.column_stack([despike(ee[:, k]) for k in range(3)])
    return ee


def body_launch_fit(tk: Tracking, hand: str | None = None,
                    win: int = 5) -> dict | None:
    """Multi-frame BODY-POSE launch-state estimate (student state).

    A degree-2 polynomial is fit per axis to the shooting-hand end-effector over a
    +-`win` window around the body-detected release; release position is the poly at
    the release frame, velocity is the analytic poly derivative there (smooth, not a
    one-frame difference). Returns a dict (or None):
      release_x/y/z       smoothed release position (ft)
      release_height      release_z (ft)
      dist_to_rim         horizontal distance release->rim (ft)
      lateral_offset      signed lateral offset of release vs the rim bearing (ft)
      elevation           launch elevation (deg above horizontal)
      azimuth             launch azimuth vs rim bearing (deg)
      speed               smoothed launch-speed proxy (ft/s)
      x_cross,y_cross,bc_entry  ballistic-decoder crossing of the body launch state
    """
    if hand is None:
        hand = detect_handedness(tk)
    rel = body_release_frame(tk, hand)
    if rel is None:
        return None
    ee = _end_effector(tk, hand)
    if ee is None:
        return None
    n = len(tk.time)
    lo = max(0, rel - win)
    hi = min(n, rel + win + 1)
    sl = np.arange(lo, hi)
    sl = sl[np.isfinite(ee[sl]).all(1)]
    if len(sl) < 5:
        return None
    t = tk.time[sl]
    t_rel = tk.time[min(rel, n - 1)]
    p0 = np.empty(3)
    v0 = np.empty(3)
    for k in range(3):
        coef = np.polyfit(t - t_rel, ee[sl, k], 2)   # a*dt^2 + b*dt + c
        p0[k] = coef[2]                              # value at dt=0 (release)
        v0[k] = coef[1]                              # derivative at dt=0

    rx, ry, rz = float(p0[0]), float(p0[1]), float(p0[2])
    vx, vy, vz = float(v0[0]), float(v0[1]), float(v0[2])
    vh = np.hypot(vx, vy)
    speed = float(np.sqrt(vh * vh + vz * vz))
    elevation = float(np.degrees(np.arctan2(vz, vh)))

    to_rim = np.array([RIM[0] - rx, RIM[1] - ry])
    dist_to_rim = float(np.hypot(*to_rim))
    bearing = np.arctan2(to_rim[1], to_rim[0])
    sgn = np.sign(np.dot(to_rim, [vx, vy])) or 1.0
    vdir = np.arctan2(sgn * vy, sgn * vx)
    azimuth = float(np.degrees(np.arctan2(np.sin(vdir - bearing), np.cos(vdir - bearing))))
    # lateral offset: release displacement perpendicular to the rim bearing
    lateral_offset = float(np.sin(-bearing) * (rx - RIM[0]) + np.cos(-bearing) * (ry - RIM[1]))

    cr = ballistic_crossing(p0, v0)
    if cr is None:
        x_cross = y_cross = bc_entry = np.nan
    else:
        bc_entry, x_cross, y_cross, _ = cr

    return {
        "release_x": rx, "release_y": ry, "release_z": rz,
        "release_height": rz,
        "dist_to_rim": dist_to_rim, "lateral_offset": lateral_offset,
        "elevation": elevation, "azimuth": azimuth, "speed": speed,
        "x_cross": float(x_cross), "y_cross": float(y_cross),
        "bc_entry": float(bc_entry),
    }
