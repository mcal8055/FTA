"""Release-event detection.

Two detectors:
  * ball_release_frame  — REFERENCE label using the ball (for offline validation only).
  * body_release_frame  — BODY-ONLY detector (no ball), used at feature time so the
    pipeline is leakage-free.

Handedness is detected per shot (players are mixed L/R). Release biomechanics: the
ball leaves the shooting hand near peak hand (wrist) upward speed, with the elbow
near extension. The body-only detector finds peak wrist vertical velocity within the
upward swing; a constant median offset (calibrated vs the ball label) aligns it.
"""
from __future__ import annotations

import numpy as np

from .loader import Tracking, despike


def detect_handedness(tk: Tracking) -> str:
    """'R' or 'L' — the wrist that stays closer to the ball through the shot."""
    dR = np.linalg.norm(tk.ball - tk.joint("RIGHT_WRIST"), axis=1)
    dL = np.linalg.norm(tk.ball - tk.joint("LEFT_WRIST"), axis=1)
    return "R" if np.nanmedian(dR) < np.nanmedian(dL) else "L"


def shooting_wrist(tk: Tracking, hand: str) -> np.ndarray:
    return tk.joint("RIGHT_WRIST" if hand == "R" else "LEFT_WRIST")


def _elbow_angle(tk: Tracking, hand: str) -> np.ndarray:
    """Angle at the shooting elbow (shoulder–elbow–wrist), degrees."""
    sh = tk.joint(f"{'RIGHT' if hand == 'R' else 'LEFT'}_SHOULDER")
    el = tk.joint(f"{'RIGHT' if hand == 'R' else 'LEFT'}_ELBOW")
    wr = shooting_wrist(tk, hand)
    ba, bc = sh - el, wr - el
    cos = (ba * bc).sum(-1) / (np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def ball_release_frame(tk: Tracking, hand: str | None = None) -> int | None:
    """Reference release: first frame the ball separates from the shooting hand
    (>0.9 ft) while rising and above 5 ft. Ball-dependent — validation only."""
    if hand is None:
        hand = detect_handedness(tk)
    w = shooting_wrist(tk, hand)
    dH = np.linalg.norm(tk.ball - w, axis=1)
    bvz = np.gradient(tk.ball[:, 2], tk.time)
    for i in range(5, len(dH) - 1):
        if dH[i] > 0.9 and bvz[i] > 0 and tk.ball[i, 2] > 5.0:
            return i
    return None


def body_release_frame(tk: Tracking, hand: str | None = None,
                       offset: int = 3) -> int | None:
    """Body-only release: peak elbow *extension angular velocity* (the shooting
    snap) within the upward swing toward the wrist apex. Far more robust than wrist
    speed (which fires early for slow ball-raisers). `offset` (calibrated vs the ball
    label) aligns the snap peak to the ball-separation convention. Default +3 frames."""
    if hand is None:
        hand = detect_handedness(tk)
    wz = despike(shooting_wrist(tk, hand)[:, 2])   # ignore boundary tracking glitches
    if np.all(np.isnan(wz)):
        return None
    n = len(wz)
    # apex = highest wrist within the interior (exclude first/last 3 frames of junk)
    interior = slice(3, n - 3)
    top = 3 + int(np.nanargmax(wz[interior]))
    if top < 6:
        return None
    elbv = despike(np.gradient(_elbow_angle(tk, hand), tk.time))   # +ve = extending
    lo, hi = max(1, top - 40), min(n, top + 6)
    seg = elbv[lo:hi]
    if seg.size == 0 or np.all(np.isnan(seg)):
        return None
    rel = lo + int(np.nanargmax(seg))     # the extension snap
    return int(np.clip(rel + offset, 0, len(wz) - 1))
