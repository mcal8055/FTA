"""Shared configuration: paths, target definitions, official scaler bounds.

Single source of truth so every module agrees on field<->target mapping and the
competition scaler. Imported across loader/metric/features/cv/models.
"""
from __future__ import annotations

from pathlib import Path

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FREETHROW_DIR = PROJECT_ROOT / "SPL-Open-Data" / "basketball" / "freethrow"
DATA_DIR = FREETHROW_DIR / "data"

SESSION_2025 = "2025-12-18"   # competition session: 458 shots, 5 players, 60 fps, fingers valid
SESSION_2024 = "2024-08-28"   # OOD probe only: 125 shots, P0001, 30 fps, no fingers

SPLITS_DIR = PROJECT_ROOT / "splits"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = PROJECT_ROOT / "reports"

# --- Targets -----------------------------------------------------------------
# JSON metadata field -> competition target name
TARGET_FIELDS = {
    "entry_angle": "angle",      # degrees
    "landing_y": "depth",        # inches, front/back
    "landing_x": "left_right",   # inches, side
}
TARGETS = ["angle", "depth", "left_right"]
TARGET_UNITS = {"angle": "deg", "depth": "in", "left_right": "in"}

# Official MinMax[0,1] scaler bounds from the Kaggle evaluation page.
SCALER_BOUNDS = {
    "angle": (30.0, 60.0),
    "depth": (-12.0, 30.0),
    "left_right": (-16.0, 16.0),
}

# --- Reproducibility ---------------------------------------------------------
RANDOM_SEED = 1561737          # fixed project seed (derived from user id)
HOLDOUT_FRAC = 0.20            # fraction of each player's shots reserved as locked test set
