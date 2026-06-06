"""Step 6 — interpretation: which movements matter, and who-vs-how (DEV only).

For each target:
  * SHAP (HGB) mean|value| feature ranking.
  * statsmodels MixedLM fixed effects (standardized features) + random player intercept;
    FDR-corrected p-values; ICC from a null model = between-player variance share.
  * Collinearity-aware: report where SHAP top-ranks and mixed-model significance AGREE.
Writes outputs/interpretation.json.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import shap
import statsmodels.formula.api as smf
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.stats.multitest import multipletests

from fta.config import OUTPUTS_DIR, TARGETS

warnings.filterwarnings("ignore")
NON = {"shot_id", "player", "split", "made", *TARGETS}


def icc(df, target):
    """Between-player variance share from null model target ~ 1 + (1|player)."""
    m = smf.mixedlm(f"{target} ~ 1", df, groups=df["player"]).fit(reml=True)
    vg = float(m.cov_re.iloc[0, 0])
    ve = float(m.scale)
    return vg / (vg + ve)


def main():
    df = pd.read_parquet(OUTPUTS_DIR / "features_2025.parquet")
    dev = df[df.split == "dev"].reset_index(drop=True)
    feats = [c for c in df.columns if c not in NON]
    X = dev[feats].to_numpy(float)
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-9)
    zdf = pd.DataFrame(Xz, columns=feats); zdf["player"] = dev["player"].values

    out = {}
    for t in TARGETS:
        y = dev[t].to_numpy(float)
        # SHAP via HGB
        hgb = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05,
                                            max_iter=300, l2_regularization=1.0, random_state=0)
        hgb.fit(X, y)
        sv = shap.TreeExplainer(hgb).shap_values(X)
        shap_imp = dict(sorted(zip(feats, np.abs(sv).mean(0)), key=lambda kv: -kv[1]))

        # MixedLM fixed effects (standardized) + FDR
        zdf[t] = y
        formula = f"{t} ~ " + " + ".join(feats)
        mm = smf.mixedlm(formula, zdf, groups=zdf["player"]).fit(reml=False)
        pvals = mm.pvalues.drop(["Intercept", "Group Var"], errors="ignore")
        coefs = mm.params[pvals.index]
        rej, q, *_ = multipletests(pvals.values, alpha=0.05, method="fdr_bh")
        mm_sig = {f: {"coef": float(coefs[f]), "q": float(q[i])}
                  for i, f in enumerate(pvals.index) if rej[i]}

        icc_t = icc(dev, t)
        top_shap = list(shap_imp)[:8]
        agree = [f for f in top_shap if f in mm_sig]
        out[t] = {
            "icc_between_player": round(icc_t, 3),
            "top_shap": {f: round(shap_imp[f], 4) for f in top_shap},
            "mixedlm_fdr_significant": {f: {"coef": round(v["coef"], 3), "q": round(v["q"], 4)}
                                        for f, v in mm_sig.items()},
            "agree_shap_and_mixedlm": agree,
        }
        print(f"\n=== {t} ===  ICC(between-player)={icc_t:.3f}")
        print(f"  top SHAP: {top_shap[:6]}")
        print(f"  FDR-sig mixedLM ({len(mm_sig)}): {list(mm_sig)[:6]}")
        print(f"  AGREE (robust drivers): {agree}")

    with open(OUTPUTS_DIR / "interpretation.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {OUTPUTS_DIR / 'interpretation.json'}")


if __name__ == "__main__":
    main()
