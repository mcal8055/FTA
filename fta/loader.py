"""Load SPL free-throw trial JSONs.

Shot grain = one JSON file = one free throw. This module handles enumeration and
metadata; full tracking-array extraction (joint -> (n_frames, 3)) is added in Step 2.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

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
