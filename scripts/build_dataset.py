"""Step 3 — build the shot-level feature table for the 2025 session.

Extracts body-only features for every 2025 shot, attaches targets + the dev/test
split label, and writes outputs/features_2025.parquet. The `split` column enforces
discipline: modeling uses split=='dev'; split=='test' is touched only at final eval.

Run: PYTHONPATH=. python scripts/build_dataset.py
"""
from __future__ import annotations

import json

import pandas as pd

from fta.config import OUTPUTS_DIR, SPLITS_DIR, TARGETS
from fta.features import extract_features
from fta.loader import list_shots, load_metadata, load_tracking


def main() -> None:
    split = json.load(open(SPLITS_DIR / "holdout_split.json"))
    dev, test = set(split["dev_ids"]), set(split["test_ids"])

    rows = []
    for i, s in enumerate(list_shots(), 1):
        tk = load_tracking(s.path)
        m = load_metadata(s.path)
        feat = extract_features(tk)
        rec = {"shot_id": s.shot_id, "player": s.player,
               "split": "dev" if s.shot_id in dev else "test" if s.shot_id in test else "?",
               "made": m["made"]}
        rec.update({t: m[t] for t in TARGETS})
        rec.update(feat)
        rows.append(rec)
        if i % 100 == 0:
            print(f"  {i} shots...")

    df = pd.DataFrame(rows)
    OUTPUTS_DIR.mkdir(exist_ok=True)
    out = OUTPUTS_DIR / "features_2025.parquet"
    df.to_parquet(out, index=False)

    feat_cols = [c for c in df.columns if c not in
                 ("shot_id", "player", "split", "made", *TARGETS)]
    print(f"\nWrote {out}")
    print(f"  shots={len(df)}  features={len(feat_cols)}  "
          f"dev={ (df.split=='dev').sum() }  test={ (df.split=='test').sum() }")
    na = df[feat_cols].isna().mean()
    bad = na[na > 0]
    print(f"  features with any NaN: {len(bad)}" + (f" -> {dict(bad.round(3))}" if len(bad) else ""))


if __name__ == "__main__":
    main()
