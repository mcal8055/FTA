"""Step 7 — figures for the write-up. Writes outputs/figures/*.png."""
from __future__ import annotations

import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from fta.config import OUTPUTS_DIR, TARGETS

warnings.filterwarnings("ignore")
FIG = OUTPUTS_DIR / "figures"
NON = {"shot_id", "player", "split", "made", *TARGETS}
PALETTE = {"P0001": "#4C72B0", "P0002": "#DD8452", "P0003": "#55A868",
           "P0004": "#C44E52", "P0005": "#8172B3"}


def fig_skill_a_vs_b():
    exp = json.load(open(OUTPUTS_DIR / "experiments.json"))
    m = exp["M_hgb_feat"]
    a = [m["scheme_a"][t]["skill_mean"] for t in TARGETS]
    b = [m["scheme_b"]["per_target"][t]["skill"] for t in TARGETS]
    x = np.arange(3); w = 0.38
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - w/2, a, w, label="Scheme A (same shooters)", color="#4C72B0")
    ax.bar(x + w/2, b, w, label="Scheme B (new shooter)", color="#C44E52")
    ax.axhline(0, color="k", lw=.8)
    ax.set_xticks(x); ax.set_xticklabels(TARGETS)
    ax.set_ylabel("skill vs baseline"); ax.set_title("HGB skill by target & CV scheme")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "1_skill_a_vs_b.png", dpi=130); plt.close(fig)


def fig_angle_shap():
    it = json.load(open(OUTPUTS_DIR / "interpretation.json"))
    top = it["angle"]["top_shap"]
    names = list(top)[::-1]; vals = [top[n] for n in names]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(names, vals, color="#55A868")
    ax.set_xlabel("mean |SHAP|  (deg)"); ax.set_title("Entry-angle drivers (HGB SHAP)")
    fig.tight_layout(); fig.savefig(FIG / "2_angle_shap.png", dpi=130); plt.close(fig)


def fig_umap_identity():
    import umap
    df = pd.read_parquet(OUTPUTS_DIR / "features_2025.parquet")
    dev = df[df.split == "dev"]
    Xz = StandardScaler().fit_transform(dev[[c for c in df.columns if c not in NON]].to_numpy(float))
    emb = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=1561737).fit_transform(Xz)
    fig, ax = plt.subplots(figsize=(6, 5))
    for p in sorted(dev["player"].unique()):
        mask = (dev["player"] == p).to_numpy()
        ax.scatter(emb[mask, 0], emb[mask, 1], s=18, alpha=.7, label=p, color=PALETTE[p])
    ax.set_title("Body-feature UMAP — colored by shooter\n(clusters = players: ARI 1.00)")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2"); ax.legend(markerscale=1.5)
    fig.tight_layout(); fig.savefig(FIG / "3_umap_identity.png", dpi=130); plt.close(fig)


def fig_miss_taxonomy():
    df = pd.read_parquet(OUTPUTS_DIR / "features_2025.parquet")
    dev = df[df.split == "dev"]
    fig, ax = plt.subplots(figsize=(6, 5))
    made = dev[dev.made == 1]; miss = dev[dev.made == 0]
    ax.scatter(made.left_right, made.depth, s=14, alpha=.35, color="#55A868", label="made")
    ax.scatter(miss.left_right, miss.depth, s=18, alpha=.7, color="#C44E52", label="miss")
    ax.axhline(0, color="grey", lw=.6); ax.axvline(0, color="grey", lw=.6)
    ax.set_xlabel("left  <-  left_right (in)  ->  right")
    ax.set_ylabel("short  <-  depth (in)  ->  long")
    ax.set_title("Landing outcomes: made (tight) vs miss (scattered)")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "4_landing_made_miss.png", dpi=130); plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    fig_skill_a_vs_b(); print("1_skill_a_vs_b.png")
    fig_angle_shap(); print("2_angle_shap.png")
    fig_miss_taxonomy(); print("4_landing_made_miss.png")
    try:
        fig_umap_identity(); print("3_umap_identity.png")
    except Exception as e:
        print(f"UMAP figure skipped: {e}")
    print(f"figures -> {FIG}")


if __name__ == "__main__":
    main()
