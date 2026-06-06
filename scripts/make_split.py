"""Step 0.5 — lock an untouched hold-out test set (anti-HARKing).

Stratified by player: reserve HOLDOUT_FRAC of EACH player's shots as a test set,
fixed seed, written once to splits/holdout_split.json. The test ids must never be
opened during development; final models are scored on them exactly once.

Run: python scripts/make_split.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import date

import numpy as np

from fta.config import HOLDOUT_FRAC, RANDOM_SEED, SESSION_2025, SPLITS_DIR
from fta.loader import list_shots, load_metadata

OUT = SPLITS_DIR / "holdout_split.json"


def main() -> None:
    shots = list_shots(SESSION_2025)
    by_player: dict[str, list[str]] = defaultdict(list)
    for s in shots:
        by_player[s.player].append(s.shot_id)

    rng = np.random.default_rng(RANDOM_SEED)
    dev, test = [], []
    per_player = {}
    for player in sorted(by_player):
        ids = sorted(by_player[player])
        rng.shuffle(ids)
        n_test = max(1, round(len(ids) * HOLDOUT_FRAC))
        test_ids = sorted(ids[:n_test])
        dev_ids = sorted(ids[n_test:])
        test += test_ids
        dev += dev_ids
        per_player[player] = {"n": len(ids), "n_dev": len(dev_ids), "n_test": len(test_ids)}

    dev, test = sorted(dev), sorted(test)
    assert not (set(dev) & set(test)), "dev/test overlap!"
    assert len(set(dev) | set(test)) == len(shots), "split does not cover all shots!"

    # sanity: target distributions should look similar dev vs test (report, don't enforce)
    meta = {s.shot_id: load_metadata(s.path) for s in shots}
    def stats(ids, tgt):
        vals = np.array([meta[i][tgt] for i in ids], float)
        return {"mean": round(float(vals.mean()), 3), "sd": round(float(vals.std()), 3)}
    target_check = {
        t: {"dev": stats(dev, t), "test": stats(test, t)}
        for t in ("angle", "depth", "left_right")
    }

    SPLITS_DIR.mkdir(exist_ok=True)
    payload = {
        "created": date(2026, 6, 6).isoformat(),
        "seed": RANDOM_SEED,
        "session": SESSION_2025,
        "holdout_frac": HOLDOUT_FRAC,
        "stratify": "player",
        "n_total": len(shots),
        "n_dev": len(dev),
        "n_test": len(test),
        "per_player": per_player,
        "target_check": target_check,
        "dev_ids": dev,
        "test_ids": test,
    }
    with open(OUT, "w") as fh:
        json.dump(payload, fh, indent=2)

    print(f"Wrote {OUT}")
    print(f"  total={len(shots)}  dev={len(dev)}  test={len(test)}")
    for p, c in per_player.items():
        print(f"  {p}: n={c['n']:3d} dev={c['n_dev']:3d} test={c['n_test']:2d}")
    print("  target dev-vs-test mean/sd:")
    for t, d in target_check.items():
        print(f"    {t:11s} dev={d['dev']}  test={d['test']}")


if __name__ == "__main__":
    main()
