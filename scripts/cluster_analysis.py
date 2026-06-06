"""Step 6b — unsupervised structure (DESCRIPTIVE only; never a predictor).

1. Global clustering diagnostic: does KMeans on body features recover PLAYERS?
   ARI vs player labels with a permutation null (Texas-sharpshooter guard).
2. Within-player form consistency: per-player feature dispersion vs make rate.
3. Outcome-space miss taxonomy: cluster missed shots in (angle, depth, left_right).
Writes outputs/clustering.json (+ UMAP coords for figures).
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from fta.config import OUTPUTS_DIR, RANDOM_SEED, TARGETS

warnings.filterwarnings("ignore")
NON = {"shot_id", "player", "split", "made", *TARGETS}


def main():
    df = pd.read_parquet(OUTPUTS_DIR / "features_2025.parquet")
    dev = df[df.split == "dev"].reset_index(drop=True)
    feats = [c for c in df.columns if c not in NON]
    Xz = StandardScaler().fit_transform(dev[feats].to_numpy(float))
    players = dev["player"].to_numpy()
    out = {}

    # 1. global clustering diagnostic (k=5) — do clusters == players?
    km = KMeans(5, n_init=10, random_state=RANDOM_SEED).fit_predict(Xz)
    ari = adjusted_rand_score(players, km)
    rng = np.random.default_rng(RANDOM_SEED)
    null = [adjusted_rand_score(rng.permutation(players), km) for _ in range(500)]
    out["global_cluster_vs_player"] = {
        "ari": round(float(ari), 3),
        "null_mean": round(float(np.mean(null)), 3),
        "null_p95": round(float(np.percentile(null, 95)), 3),
        "verdict": "clusters track player identity" if ari > np.percentile(null, 95)
                   else "no identity structure",
    }
    print(f"global clustering ARI vs player = {ari:.3f} (null p95={np.percentile(null,95):.3f}) "
          f"-> {out['global_cluster_vs_player']['verdict']}")

    # 2. within-player form consistency vs make rate (descriptive, n=5)
    print("\nper-player form dispersion vs make rate:")
    pp = {}
    for p in sorted(dev["player"].unique()):
        m = players == p
        disp = float(np.mean(np.linalg.norm(Xz[m] - Xz[m].mean(0), axis=1)))  # spread in feature space
        mk = float(dev.loc[m, "made"].mean())
        pp[p] = {"form_dispersion": round(disp, 3), "make_rate": round(mk, 3)}
        print(f"  {p}: dispersion={disp:.2f}  make_rate={mk:.3f}")
    disps = [pp[p]["form_dispersion"] for p in pp]; mks = [pp[p]["make_rate"] for p in pp]
    r = float(np.corrcoef(disps, mks)[0, 1])
    out["within_player_consistency"] = {"per_player": pp,
                                        "corr_dispersion_vs_makerate": round(r, 3),
                                        "note": "n=5 players -> anecdotal"}
    print(f"  corr(dispersion, make_rate) = {r:+.3f}  (n=5, anecdotal)")

    # 3. outcome-space miss taxonomy (missed shots only)
    miss = dev[dev.made == 0]
    Y = StandardScaler().fit_transform(miss[TARGETS].to_numpy(float))
    kc = KMeans(4, n_init=10, random_state=RANDOM_SEED).fit(Y)
    lab = kc.labels_
    modes = {}
    for c in range(4):
        cen = miss[TARGETS].to_numpy(float)[lab == c].mean(0)
        modes[f"mode_{c}"] = {"n": int((lab == c).sum()),
                              **{t: round(float(cen[i]), 1) for i, t in enumerate(TARGETS)}}
    out["miss_taxonomy"] = modes
    print(f"\nmiss taxonomy ({len(miss)} misses), centroids (angle/depth/LR):")
    for k, v in modes.items():
        print(f"  {k}: n={v['n']:3d} angle={v['angle']} depth={v['depth']} lr={v['left_right']}")

    with open(OUTPUTS_DIR / "clustering.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {OUTPUTS_DIR / 'clustering.json'}")


if __name__ == "__main__":
    main()
