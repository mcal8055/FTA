"""Load SPL free-throw trial JSONs.

Shot grain = one JSON file = one free throw. This module handles enumeration and
metadata; full tracking-array extraction (joint -> (n_frames, 3)) is added in Step 2.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import DATA_DIR, SESSION_2025, TARGET_FIELDS


@dataclass(frozen=True)
class Shot:
    shot_id: str      # "P0001/T0001"
    player: str       # "P0001"
    trial_id: str     # "T0001"
    session: str      # date string
    path: Path


def list_shots(session: str = SESSION_2025) -> list[Shot]:
    """Enumerate shots for a session, sorted by player then trial id."""
    session_dir = DATA_DIR / session
    shots: list[Shot] = []
    for path in session_dir.glob("*/*.json"):
        player = path.parent.name              # P0001
        # filename: BB_FT_P0001_T0001.json -> trial id is the last underscore token
        trial_id = path.stem.split("_")[-1]    # T0001
        shots.append(Shot(f"{player}/{trial_id}", player, trial_id, session, path))
    shots.sort(key=lambda s: (s.player, s.trial_id))
    return shots


def load_raw(path: Path | str) -> dict:
    """Load a trial JSON verbatim (tracking ball coords may contain NaN)."""
    with open(path) as fh:
        return json.load(fh)


def load_metadata(path: Path | str) -> dict:
    """Return shot-level metadata + the three targets (no tracking arrays)."""
    d = load_raw(path)
    meta = {
        "participant_id": d.get("participant_id"),
        "trial_id": d.get("trial_id"),
        "trial_date": d.get("trial_date"),
        "sampling_rate": d.get("sampling_rate"),
        "result": d.get("result"),
        "made": int(d.get("result") == "made"),
        "n_frames": len(d.get("tracking", [])),
    }
    for field, target in TARGET_FIELDS.items():
        meta[target] = d.get(field)
    return meta


def despike(col: np.ndarray, win: int = 7, n_sigma: float = 4.0) -> np.ndarray:
    """Hampel filter: replace points far from a rolling median (tracking glitches,
    e.g. a keypoint teleporting for a few frames near the recording boundary)."""
    col = np.asarray(col, float)
    n = len(col)
    if n < win:
        return col
    h = win // 2
    out = col.copy()
    for i in range(n):
        seg = col[max(0, i - h):min(n, i + h + 1)]
        med = np.median(seg)
        mad = np.median(np.abs(seg - med)) + 1e-9
        if np.abs(col[i] - med) > n_sigma * 1.4826 * mad:
            out[i] = med
    return out


def _vec(c) -> list[float]:
    """Coerce a coord (possibly None / contains None) to a float3 with NaN for missing."""
    if c is None:
        return [np.nan, np.nan, np.nan]
    return [np.nan if v is None else float(v) for v in c]


@dataclass
class Tracking:
    time: np.ndarray                      # (n,) seconds
    ball: np.ndarray                      # (n, 3) feet, NaN where out of volume
    players: dict[str, np.ndarray]        # joint -> (n, 3) feet, NaN where missing
    sampling_rate: int
    joints: tuple[str, ...]

    def joint(self, name: str) -> np.ndarray:
        return self.players[name]


def load_tracking(path: Path | str) -> Tracking:
    """Extract time + ball + all player joints as (n_frames, 3) arrays (NaN-aware)."""
    d = load_raw(path)
    tr = d["tracking"]
    n = len(tr)
    time = np.array([fr["time"] for fr in tr], float) / 1000.0   # ms -> s
    ball = np.array([_vec(fr["data"].get("ball")) for fr in tr], float)
    joints = tuple(tr[0]["data"]["player"].keys())
    players = {
        j: np.array([_vec(fr["data"]["player"].get(j)) for fr in tr], float)
        for j in joints
    }
    return Tracking(time=time, ball=ball, players=players,
                    sampling_rate=int(d.get("sampling_rate", 0)), joints=joints)
