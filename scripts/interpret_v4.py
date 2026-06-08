"""Step 6b — interpretation of the v4 PHYSICS features (kinetic-chain + multi-temporal).

Extends scripts/interpret.py. For each target (DEV split only, train-only standardization):
  (1) SHAP (HGB) mean|value| ranking on base+both -> do physics features rank top?
  (2) COM forward-thrust as a DEPTH driver ("force not position")?
  (3) ICC v1 (base only) vs v4 (base+both): does between-player variance share drop?
  (4) Per-player mechanism: per-player SHAP top features -> different drivers per player?

Writes outputs/v4_interpretation.json.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import shap
import statsmodels.formula.api as smf
from sklearn.ensemble import HistGradientBoostingRegressor

from fta.config import OUTPUTS_DIR, TARGETS

warnings.filterwarnings("ignore")

NON = {"shot_id", "player", "split", "made", "hand_right", *TARGETS}

# Physics feature-family tags (for "is it a physics feature?" classification)
KINETIC_PREFIXES = (
    "peakspeed_", "tpeak_", "relspeed_", "lag_", "seq_", "sos_",
    "com_fwd_", "com_up_", "shoulder_angvel", "hip_angvel", "trunk_twist",
)
TEMPORAL_SUFFIXES = (
    "_m500", "_m350", "_m200", "_m100", "_m050", "_m000",
)
TEMPORAL_EXTRA = ("com_fwd_vel_peak", "t_com_fwd_vel_peak", "elbow_ext_swept",
                  "wrist_rise_swept", "com_fwd_advance")

# COM forward-thrust / momentum features (the winner's "force" story for depth)
COM_THRUST = (
    "com_fwd_peak", "com_fwd_tpeak", "com_fwd_at_release",
    "com_fwd_advance", "com_fwd_vel_peak", "t_com_fwd_vel_peak",
) + tuple(f"com_fwd_vel_{s.lstrip('_')}" for s in TEMPORAL_SUFFIXES) \
  + tuple(f"com_fwd_pos_{s.lstrip('_')}" for s in TEMPORAL_SUFFIXES)


def family(f: str) -> str:
    if f.startswith(KINETIC_PREFIXES) or f in ("shoulder_angvel_peak", "hip_angvel_peak", "trunk_twist_angvel_release"):
        return "kinetic"
    if f.endswith(TEMPORAL_SUFFIXES) or f in TEMPORAL_EXTRA:
        return "temporal"
    return "v1_base"


def load_merged():
    base = pd.read_parquet(OUTPUTS_DIR / "features_2025.parquet")
    kin = pd.read_parquet(OUTPUTS_DIR / "v4_kinetic.parquet")
    tmp = pd.read_parquet(OUTPUTS_DIR / "v4_temporal.parquet")
    kin = kin[[c for c in kin.columns if c == "shot_id" or c not in base.columns]]
    tmp = tmp[[c for c in tmp.columns if c == "shot_id" or c not in base.columns]]
    df = base.merge(kin, on="shot_id").merge(tmp, on="shot_id")
    return df


def icc(df, target):
    """Between-player variance share from null model target ~ 1 + (1|player)."""
    m = smf.mixedlm(f"{target} ~ 1", df, groups=df["player"]).fit(reml=True)
    vg = float(m.cov_re.iloc[0, 0])
    ve = float(m.scale)
    return vg / (vg + ve)


def icc_residual(df, feats, target, seed=0):
    """ICC of the OUT-OF-FOLD residual after Ridge on standardized features.

    Cross-fitted (5-fold, stratified-by-player NOT used to avoid leakage of
    target; standardization + fit strictly train-only inside each fold). If
    physics features capture shooter-INVARIANT mechanism, they explain real
    between-player signal and the residual ICC drops (player identity no longer
    needed to explain the leftover). If they only TRACK identity, the residual
    is whatever they could not memorize and ICC need not drop; the honest test
    is whether the between-player share of the *out-of-fold* residual falls.
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    X = df[feats].to_numpy(float)
    y = df[target].to_numpy(float)
    oof = np.zeros_like(y)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in kf.split(X):
        sc = StandardScaler().fit(X[tr])
        m = Ridge(alpha=10.0).fit(sc.transform(X[tr]), y[tr])
        oof[te] = m.predict(sc.transform(X[te]))
    resid = y - oof
    rdf = pd.DataFrame({"r": resid, "player": df["player"].values})
    mm = smf.mixedlm("r ~ 1", rdf, groups=rdf["player"]).fit(reml=True)
    vg = float(mm.cov_re.iloc[0, 0])
    ve = float(mm.scale)
    icc_resid = vg / (vg + ve)
    # MixedLM REML estimates vg->0 once the regression has removed the small
    # between-player offset, so ICC saturates at 0 and is not a sensitive
    # discriminator. The honest between-player-residual measure is the spread of
    # per-player residual MEANS relative to the raw target's between-player std:
    # how much of the original identity offset survives the regression.
    per_player_resid_mean_std = float(rdf.groupby("player")["r"].mean().std())
    r2_oof = float(1 - np.var(resid) / (np.var(y) + 1e-12))
    return icc_resid, per_player_resid_mean_std, r2_oof


def main():
    df = load_merged()
    dev = df[df.split == "dev"].reset_index(drop=True)
    feats = [c for c in df.columns if c not in NON]
    base_feats = [c for c in feats if family(c) == "v1_base"]
    phys_feats = [c for c in feats if family(c) != "v1_base"]

    X = dev[feats].to_numpy(float)
    players = dev["player"].values
    uniq_players = sorted(np.unique(players))

    out = {
        "meta": {
            "feature_set": "base+both (v1_base + kinetic + temporal)",
            "n_dev": int(len(dev)),
            "n_features_total": len(feats),
            "n_base": len(base_feats),
            "n_physics": len(phys_feats),
            "players": uniq_players,
            "note": "DEV split only; standardization train-only; ball xyz never a feature.",
        }
    }

    for t in TARGETS:
        y = dev[t].to_numpy(float)
        hgb = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05,
                                            max_iter=300, l2_regularization=1.0,
                                            random_state=0)
        hgb.fit(X, y)
        sv = shap.TreeExplainer(hgb).shap_values(X)  # (n, p)
        imp = np.abs(sv).mean(0)
        rank = dict(sorted(zip(feats, imp), key=lambda kv: -kv[1]))

        top15 = list(rank)[:15]
        top_phys = [f for f in rank if family(f) != "v1_base"][:8]

        total_imp = imp.sum() + 1e-12
        share_phys = float(sum(imp[feats.index(f)] for f in phys_feats) / total_imp)
        share_com_thrust = float(sum(imp[feats.index(f)] for f in feats if f in COM_THRUST) / total_imp)

        # COM thrust ranking for depth story
        com_ranked = [(f, round(float(rank[f]), 4), list(rank).index(f) + 1)
                      for f in rank if f in COM_THRUST][:6]

        # ICC: v1 baseline null model + cross-fitted residual diagnostics.
        icc_null = icc(dev, t)
        raw_player_mean_std = float(pd.Series(y).groupby(players).mean().std())
        icc_resid_base, ppstd_base, r2_base = icc_residual(dev, base_feats, t)
        icc_resid_phys, ppstd_phys, r2_phys = icc_residual(dev, base_feats + phys_feats, t)
        # between-player residual offset as fraction of raw identity offset
        bp_share_base = ppstd_base / (raw_player_mean_std + 1e-12)
        bp_share_phys = ppstd_phys / (raw_player_mean_std + 1e-12)

        # ---- Per-player mechanism: per-player HGB SHAP top features ----
        per_player = {}
        for p in uniq_players:
            mask = players == p
            if mask.sum() < 25:
                per_player[p] = {"n": int(mask.sum()), "skipped": True}
                continue
            Xp, yp = X[mask], y[mask]
            hp = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05,
                                               max_iter=200, l2_regularization=1.0,
                                               random_state=0)
            hp.fit(Xp, yp)
            svp = shap.TreeExplainer(hp).shap_values(Xp)
            impp = np.abs(svp).mean(0)
            rp = sorted(zip(feats, impp), key=lambda kv: -kv[1])
            per_player[p] = {
                "n": int(mask.sum()),
                "top5": [f for f, _ in rp[:5]],
                "top5_family": [family(f) for f, _ in rp[:5]],
            }

        # Agreement: how disjoint are per-player top-3 sets?
        top3_sets = {p: set(v.get("top5", [])[:3]) for p, v in per_player.items()
                     if not v.get("skipped")}
        all_top3 = set().union(*top3_sets.values()) if top3_sets else set()
        shared_all = set.intersection(*top3_sets.values()) if top3_sets else set()
        per_player_divergence = {
            "n_distinct_features_in_any_top3": len(all_top3),
            "n_features_shared_by_all_players_top3": len(shared_all),
            "shared_by_all": sorted(shared_all),
        }

        out[t] = {
            "icc_v1_null": round(icc_null, 3),
            "raw_between_player_mean_std": round(raw_player_mean_std, 3),
            "icc_residual_after_v1_base": round(icc_resid_base, 3),
            "icc_residual_after_v1_plus_physics": round(icc_resid_phys, 3),
            "oof_r2_v1_base": round(r2_base, 3),
            "oof_r2_v1_plus_physics": round(r2_phys, 3),
            "between_player_residual_std_v1_base": round(ppstd_base, 4),
            "between_player_residual_std_v1_plus_physics": round(ppstd_phys, 4),
            "between_player_residual_share_v1_base": round(bp_share_base, 3),
            "between_player_residual_share_v1_plus_physics": round(bp_share_phys, 3),
            "between_player_share_dropped_by_physics": bool(bp_share_phys < bp_share_base),
            "shap_physics_importance_share": round(share_phys, 3),
            "shap_com_thrust_importance_share": round(share_com_thrust, 3),
            "top15_shap": {f: round(float(rank[f]), 4) for f in top15},
            "top15_families": [family(f) for f in top15],
            "top_physics_features": top_phys,
            "com_thrust_features_ranked": [
                {"feature": f, "shap": s, "overall_rank": r} for f, s, r in com_ranked
            ],
            "per_player_mechanism": per_player,
            "per_player_divergence": per_player_divergence,
        }

        print(f"\n=== {t} ===")
        print(f"  ICC null={icc_null:.3f}  raw_bp_std={raw_player_mean_std:.3f}")
        print(f"  OOF R2 base={r2_base:.3f} phys={r2_phys:.3f} | bp-resid-share base={bp_share_base:.3f} phys={bp_share_phys:.3f} dropped={bp_share_phys<bp_share_base}")
        print(f"  SHAP physics share={share_phys:.3f}  com_thrust share={share_com_thrust:.3f}")
        print(f"  top6 SHAP: {[(f, family(f)) for f in top15[:6]]}")
        print(f"  top physics: {top_phys[:5]}")
        print(f"  COM thrust ranks: {[(f, r) for f, _, r in com_ranked[:4]]}")
        print(f"  per-player top3 distinct={len(all_top3)} shared-by-all={len(shared_all)}")

    with open(OUTPUTS_DIR / "v4_interpretation.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {OUTPUTS_DIR / 'v4_interpretation.json'}")


if __name__ == "__main__":
    main()
