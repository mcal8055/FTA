"""Full multivariate time-series tensors per shot (for ROCKET / deep models).

No feature compression: every keypoint channel over a release-aligned window.
- centre x,y on mid-hip at release (removes court-position identity; keeps z = height)
- mirror left-handers to a canonical right-handed frame (negate lateral y + swap L/R
  markers) so all shooters pool
- NaN-interpolate; release-aligned crop [rel-90, rel+30] @60fps -> 120 frames
Returns X of shape (n_channels, n_timepoints).
"""
from __future__ import annotations

import numpy as np

from .loader import Tracking
from .release import body_release_frame, detect_handedness

WIN_BEFORE, WIN_AFTER = 90, 30   # frames @60fps -> 120-frame window (2.0 s)


def _marker_order(joints):
    # stable: central first, then L/R pairs interleaved so swap is clean
    central = [m for m in joints if not (m.startswith("LEFT_") or m.startswith("RIGHT_"))]
    lefts = [m for m in joints if m.startswith("LEFT_")]
    order = sorted(central)
    for L in sorted(lefts):
        R = "RIGHT_" + L[5:]
        if R in joints:
            order += [L, R]
    # any unpaired rights
    order += [m for m in sorted(joints) if m not in order]
    return order


def _interp_nan(a):
    out = a.copy()
    n = len(a)
    idx = np.arange(n)
    for k in range(a.shape[1]):
        col = a[:, k]
        good = np.isfinite(col)
        if good.sum() == 0:
            out[:, k] = 0.0
        elif good.sum() < n:
            out[:, k] = np.interp(idx, idx[good], col[good])
    return out


def build_tensor(tk: Tracking, markers=None) -> np.ndarray | None:
    hand = detect_handedness(tk)
    rel = body_release_frame(tk, hand)
    if rel is None:
        return None
    joints = list(tk.joints)
    order = markers or _marker_order(joints)

    # centre x,y on mid-hip at release
    mh = _interp_nan(tk.joint("MID_HIP"))
    cxy = mh[rel, :2]

    arrs = {}
    for m in order:
        a = _interp_nan(tk.joint(m)).copy()
        a[:, 0] -= cxy[0]
        a[:, 1] -= cxy[1]
        arrs[m] = a

    # mirror left-handers: negate lateral (y), swap L/R labels
    if hand == "L":
        swapped = {}
        for m in order:
            if m.startswith("LEFT_"):
                src = "RIGHT_" + m[5:]
            elif m.startswith("RIGHT_"):
                src = "LEFT_" + m[6:]
            else:
                src = m
            a = arrs.get(src, arrs[m]).copy()
            a[:, 1] = -a[:, 1]
            swapped[m] = a
        arrs = swapped

    # release-aligned crop with edge padding
    n = len(tk.time)
    lo, hi = rel - WIN_BEFORE, rel + WIN_AFTER
    chans = []
    for m in order:
        a = arrs[m]                       # (n,3)
        idx = np.clip(np.arange(lo, hi), 0, n - 1)
        chans.append(a[idx].T)            # (3, T)
    return np.vstack(chans)               # (3*len(order), T)
