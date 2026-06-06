"""Step 1 — target-side EDA on the DEV set only (hold-out stays sealed).

Profiles the three regression targets: distributions, inter-target relationships,
per-player spread, and made/miss dispersion (heteroscedasticity check).
Writes outputs/target_eda.json and prints a summary.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from fta.config import OUTPUTS_DIR, TARGETS, TARGET_UNITS
from fta.loader import list_shots, load_metadata
from fta.config import SPLITS_DIR


def main() -> None:
    dev = set(json.load(open(SPLITS_DIR / "holdout_split.json"))["dev_ids"])
    rows = []
    for s in list_shots():
        if s.shot_id not in dev:
            continue
        m = load_metadata(s.path)
        m["player"] = s.player
        rows.append(m)
    df = pd.DataFrame(rows)
    print(f"DEV shots: {len(df)}  players: {sorted(df.player.unique())}")
    print(f"make rate (dev): {df.made.mean():.3f}")

    summary = {"n_dev": len(df), "make_rate": round(float(df.made.mean()), 3)}

    # 1. target distributions
    print("\n=== target distributions ===")
    dist = {}
    for t in TARGETS:
        v = df[t].astype(float)
        d = dict(mean=round(v.mean(), 3), sd=round(v.std(), 3), min=round(v.min(), 2),
                 q25=round(v.quantile(.25), 2), median=round(v.median(), 2),
                 q75=round(v.quantile(.75), 2), max=round(v.max(), 2))
        dist[t] = d
        print(f"  {t:11s}({TARGET_UNITS[t]}): {d}")
    summary["distributions"] = dist

    # 2. inter-target correlation (Pearson + Spearman)
    print("\n=== inter-target correlation (Pearson / Spearman) ===")
    cors = {}
    for i, a in enumerate(TARGETS):
        for b in TARGETS[i+1:]:
            pr = pearsonr(df[a], df[b])[0]
            sr = spearmanr(df[a], df[b])[0]
            cors[f"{a}~{b}"] = {"pearson": round(pr, 3), "spearman": round(sr, 3)}
            print(f"  {a:11s} ~ {b:11s}: r={pr:+.3f}  rho={sr:+.3f}")
    summary["inter_target_corr"] = cors

    # 3. per-player target mean/sd + make rate
    print("\n=== per-player (dev) ===")
    pp = {}
    for p, g in df.groupby("player"):
        rec = {"n": len(g), "make_rate": round(g.made.mean(), 3)}
        for t in TARGETS:
            rec[t] = {"mean": round(g[t].mean(), 2), "sd": round(g[t].std(), 2)}
        pp[p] = rec
        print(f"  {p}: n={rec['n']:3d} make={rec['make_rate']:.3f} | " +
              " ".join(f"{t}={rec[t]['mean']:+.1f}±{rec[t]['sd']:.1f}" for t in TARGETS))
    summary["per_player"] = pp

    # 4. made vs missed dispersion (heteroscedasticity)
    print("\n=== made vs missed dispersion (SD; variance ratio miss/made) ===")
    het = {}
    made_g, miss_g = df[df.made == 1], df[df.made == 0]
    for t in TARGETS:
        sd_made, sd_miss = made_g[t].std(), miss_g[t].std()
        het[t] = {"sd_made": round(sd_made, 2), "sd_miss": round(sd_miss, 2),
                  "var_ratio_miss_made": round((sd_miss**2) / (sd_made**2), 2)}
        print(f"  {t:11s}: made SD={sd_made:.2f}  miss SD={sd_miss:.2f}  "
              f"var ratio={het[t]['var_ratio_miss_made']:.2f}")
    summary["made_miss_dispersion"] = het

    OUTPUTS_DIR.mkdir(exist_ok=True)
    with open(OUTPUTS_DIR / "target_eda.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nWrote {OUTPUTS_DIR / 'target_eda.json'}")


if __name__ == "__main__":
    main()
