"""Analytic ballistic forward model (the differentiable decoder for v2).

Given a release state (position p0, velocity v0) in JSON feet coordinates, drag-free
projectile motion gives the descending crossing of the 10-ft rim plane and the entry
angle there. Pure numpy + closed-form (the crossing is a quadratic root) so it is
differentiable for the learned-decoder version later.
"""
from __future__ import annotations

import numpy as np

G = 32.174  # ft/s^2
RIM_Z = 10.0


def ballistic_crossing(p0, v0, g: float = G, rim_z: float = RIM_Z):
    """Return (entry_angle_deg, x_cross, y_cross, t_cross) at the DESCENDING rim-plane
    crossing, or None if the trajectory never reaches rim height.

    z(t) = p0z + v0z t - 0.5 g t^2 ;  x,y linear.  Descending root has the larger t.
    """
    p0 = np.asarray(p0, float); v0 = np.asarray(v0, float)
    a, b, c = -0.5 * g, v0[2], p0[2] - rim_z
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    t = (-b - np.sqrt(disc)) / (2 * a)        # larger root = descending
    if t <= 0:
        return None
    xc = p0[0] + v0[0] * t
    yc = p0[1] + v0[1] * t
    vz = v0[2] - g * t
    vh = np.hypot(v0[0], v0[1])
    entry = np.degrees(np.arctan2(-vz, vh))   # below horizontal
    return entry, xc, yc, t


def launch_state(tk, rel, hand, win=3):
    """Estimate release point p0 and velocity v0 from the shooting-hand fingertips
    (index+middle distal). v0 is taken at the instant of PEAK fingertip speed within a
    small window around `rel` (the actual ball-off-fingers moment), using light finite
    differences on despiked positions (NOT the heavy feature smoothing, which would
    attenuate the brief release peak). Returns (p0, v0) or None.
    """
    from .loader import despike
    side = "RIGHT" if hand == "R" else "LEFT"
    fingers = [f"{side}_SECOND_FINGER_DISTAL", f"{side}_THIRD_FINGER_DISTAL"]
    if not all(f in tk.players and not np.all(np.isnan(tk.joint(f))) for f in fingers):
        return None
    tip = np.mean([tk.joint(f) for f in fingers], axis=0)        # (n,3)
    tip = np.column_stack([despike(tip[:, k]) for k in range(3)])
    t = tk.time
    v = np.gradient(tip, t, axis=0)                              # light finite diff
    n = len(t)
    lo, hi = max(1, rel - win), min(n - 1, rel + win + 1)
    speed = np.linalg.norm(v[lo:hi], axis=1)
    k = lo + int(np.nanargmax(speed))                           # launch instant
    return tip[k], v[k]
