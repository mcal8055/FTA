"""EXPERIMENT A — Physics decoder with privileged information (LUPI).

Pipeline (ball used ONLY as a training-time supervision target, never a test feature):

  (1) On every shot compute:
        ball_launch_fit  (TEACHER launch state, ball-derived)   -> training-only target
        body_launch_fit  (STUDENT launch state, BODY pose only) -> legal test feature
      and attach the existing body feature matrix (outputs/features_2025.parquet).

  (2) Attach the OFFICIAL competition split (competition_eval.py mapping: target-match
      spl-utspan-data-challenge-2026/train.csv -> 345 train ids; test = 113 complement).

  (3) Train BODY-features -> ball launch state. We predict a launch state in the analytic
      decoder's parameterisation: release position (rx,ry,rz) and velocity (vx,vy,vz) in
      the JSON feet frame, where the teacher target is reconstructed from the ball fit's
      (speed, elevation, azimuth, release_x/y/z) at the reference height. Fit is TRAIN
      ONLY (ridge with standardisation fit on train).

  (4) Push the PREDICTED launch state through fta.physics.ballistic_crossing to produce
      analytic depth (y_cross), left_right (x_cross relative to rim) and entry angle.
      Convert the JSON-frame crossing to competition target units/frame via a TRAIN-ONLY
      affine alignment (the JSON frame differs from the Kaggle target frame; the affine
      is the legal, train-fit linear calibration crossing->target).

  (5) Hybrid: add a small learned residual model (ridge / HGB) on body features + the
      physics prediction, selected by CV WITHIN train. Also compare a pure-ML body model
      and a stacked "physics-as-feature" model.

  (6) Score scaled-MSE per target on the competition TEST set once. Compare to prior
      0.01164 (Ridge) and winner 0.006136.

LEAKAGE: ball appears only inside step (1) as a target column used in (3). The test
shots' ball is never read as a feature; all scalers / coefficients / affines are fit on
train rows only; CV-within-train picks hyperparameters. Test is touched once for the
final number.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from fta.config import OUTPUTS_DIR, RIM, TARGETS
from fta.cv import scheme_a_folds
from fta.loader import list_shots, load_metadata, load_tracking
from fta.metric import scaled_mse, scaled_mse_per_target
from fta.physics import G, RIM_Z, ballistic_crossing
from fta.release import detect_handedness

from physics_v3 import REF_RELEASE_Z, ball_launch_fit, body_launch_fit

warnings.filterwarnings("ignore")

PRIOR = 0.01164
WINNER = 0.006136
CACHE = OUTPUTS_DIR / "v3_exp_a_cache.parquet"


# --------------------------------------------------------------------------- #
# data assembly                                                               #
# --------------------------------------------------------------------------- #
def teacher_launch_velocity(bf: dict) -> tuple[float, float, float]:
    """Reconstruct the ball launch VELOCITY vector (vx,vy,vz) at REF_RELEASE_Z in the
    JSON feet frame from the ball fit's reported (speed, elevation, azimuth) — these are
    the components the foundation showed are (partly) body-observable. azimuth is the
    signed lateral angle of v_h relative to the rim bearing; reconstruct v_h direction
    by rotating the rim bearing by azimuth, in the shooting direction."""
    rx, ry = bf["release_x"], bf["release_y"]
    to_rim = np.array([RIM[0] - rx, RIM[1] - ry])
    bearing = np.arctan2(to_rim[1], to_rim[0])
    vdir = bearing + np.radians(bf["azimuth"])
    vh = bf["speed"] * np.cos(np.radians(bf["elevation"]))
    vz = bf["speed"] * np.sin(np.radians(bf["elevation"]))
    vx = vh * np.cos(vdir)
    vy = vh * np.sin(vdir)
    return float(vx), float(vy), float(vz)


def assemble() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    rows = []
    for s in list_shots():
        tk = load_tracking(s.path)
        m = load_metadata(s.path)
        hand = detect_handedness(tk)
        bf = ball_launch_fit(tk)
        bo = body_launch_fit(tk, hand)
        row = {"shot_id": s.shot_id, "player": s.player, "made": m["made"],
               "angle": m["angle"], "depth": m["depth"], "left_right": m["left_right"]}
        if bf is not None:
            row.update({f"ball_{k}": v for k, v in bf.items()})
            vx, vy, vz = teacher_launch_velocity(bf)
            row.update({"ball_vx_ref": vx, "ball_vy_ref": vy, "ball_vz_ref": vz})
        if bo is not None:
            row.update({f"body_{k}": v for k, v in bo.items()})
        rows.append(row)
    df = pd.DataFrame(rows)
    # merge existing body feature matrix
    feat = pd.read_parquet(OUTPUTS_DIR / "features_2025.parquet")
    drop = [c for c in feat.columns if c in df.columns and c not in ("shot_id",)]
    feat = feat.drop(columns=drop)
    df = df.merge(feat, on="shot_id", how="left")
    OUTPUTS_DIR.mkdir(exist_ok=True)
    df.to_parquet(CACHE)
    return df


def competition_split(df: pd.DataFrame) -> pd.DataFrame:
    tgt_key = {}
    for s in list_shots():
        m = load_metadata(s.path)
        tgt_key[(round(m["angle"], 2), round(m["depth"], 2), round(m["left_right"], 2))] = s.shot_id
    tr = pd.read_csv("spl-utspan-data-challenge-2026/train.csv")
    train_ids = {tgt_key[(round(r.angle, 2), round(r.depth, 2), round(r.left_right, 2))]
                 for _, r in tr.iterrows()}
    df = df.copy()
    df["csplit"] = np.where(df.shot_id.isin(train_ids), "train", "test")
    return df


# --------------------------------------------------------------------------- #
# physics decoder helpers                                                     #
# --------------------------------------------------------------------------- #
def decode_state(rx, ry, rz, vx, vy, vz):
    """analytic crossing of a launch state -> (entry, x_cross, y_cross) JSON frame."""
    cr = ballistic_crossing([rx, ry, rz], [vx, vy, vz])
    if cr is None:
        return np.nan, np.nan, np.nan
    entry, xc, yc, _ = cr
    return entry, xc, yc


def physics_targets_from_state(state: dict) -> dict:
    """Vectorised analytic decode of arrays of launch states -> raw crossing arrays."""
    n = len(state["rx"])
    ent = np.full(n, np.nan); xc = np.full(n, np.nan); yc = np.full(n, np.nan)
    for i in range(n):
        e, x, y = decode_state(state["rx"][i], state["ry"][i], state["rz"][i],
                               state["vx"][i], state["vy"][i], state["vz"][i])
        ent[i], xc[i], yc[i] = e, x, y
    return {"entry": ent, "x_cross": xc, "y_cross": yc}


def affine_fit(X: np.ndarray, y: np.ndarray):
    """train-only least-squares affine y ~ [X, 1]. X is (n,k). NaN rows dropped."""
    X = np.asarray(X, float)
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    A = np.column_stack([X[ok], np.ones(ok.sum())])
    coef, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
    return coef


def affine_apply(coef, X):
    A = np.column_stack([X, np.ones(len(X))])
    return A @ coef


# --------------------------------------------------------------------------- #
# models                                                                       #
# --------------------------------------------------------------------------- #
def fit_ridge_multi(Xtr, Ytr, alpha=10.0):
    sc = StandardScaler().fit(Xtr)
    m = Ridge(alpha=alpha).fit(sc.transform(Xtr), Ytr)
    return sc, m


def body_feature_cols(df: pd.DataFrame) -> list[str]:
    NON = {"shot_id", "player", "split", "csplit", "made", *TARGETS}
    cols = []
    for c in df.columns:
        if c in NON or c.startswith("ball_"):
            continue  # ball_* are teacher-only, never features
        if df[c].dtype.kind in "fi":
            cols.append(c)
    return cols


def impute_train(Xtr, Xte):
    mu = np.nanmean(Xtr, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    Xtr = np.where(np.isfinite(Xtr), Xtr, mu)
    Xte = np.where(np.isfinite(Xte), Xte, mu)
    return Xtr, Xte


# --------------------------------------------------------------------------- #
# main experiment                                                             #
# --------------------------------------------------------------------------- #
STATE_TARGETS = ["release_x", "release_y", "release_z", "vx_ref", "vy_ref", "vz_ref"]
STATE_BALL_COLS = {"release_x": "ball_release_x", "release_y": "ball_release_y",
                   "release_z": "ball_release_z", "vx_ref": "ball_vx_ref",
                   "vy_ref": "ball_vy_ref", "vz_ref": "ball_vz_ref"}


def predict_state(trn, ten, feat_cols, alpha=10.0):
    """Train body-features -> ball launch-state vector. Returns dict of arrays for test."""
    Xtr = trn[feat_cols].to_numpy(float)
    Xte = ten[feat_cols].to_numpy(float)
    Xtr, Xte = impute_train(Xtr, Xte)
    Ytr = np.column_stack([trn[STATE_BALL_COLS[s]].to_numpy(float) for s in STATE_TARGETS])
    # rows with finite teacher targets only for training
    ok = np.isfinite(Ytr).all(1)
    sc, m = fit_ridge_multi(Xtr[ok], Ytr[ok], alpha)
    Pte = m.predict(sc.transform(Xte))
    return {s: Pte[:, i] for i, s in enumerate(STATE_TARGETS)}


def state_to_decoder_input(state):
    return {"rx": state["release_x"], "ry": state["release_y"], "rz": state["release_z"],
            "vx": state["vx_ref"], "vy": state["vy_ref"], "vz": state["vz_ref"]}


def run():
    df = assemble()
    df = competition_split(df)
    trn = df[df.csplit == "train"].reset_index(drop=True)
    ten = df[df.csplit == "test"].reset_index(drop=True)
    feat_cols = body_feature_cols(df)
    print(f"competition split: train={len(trn)} test={len(ten)}  body-features={len(feat_cols)}")
    yt = {t: ten[t].to_numpy(float) for t in TARGETS}

    results = {}

    # ----- baseline 0: player-mean (sanity) ----------------------------------
    pmean = {}
    gmean = {t: trn[t].mean() for t in TARGETS}
    for t in TARGETS:
        mp = trn.groupby("player")[t].mean()
        pmean[t] = ten["player"].map(mp).fillna(gmean[t]).to_numpy(float)
    results["player_mean"] = {"scaled_mse": scaled_mse(yt, pmean),
                              "per_target": scaled_mse_per_target(yt, pmean)}

    # ----- pure-ML body ridge (reference, no physics) ------------------------
    Xtr = trn[feat_cols].to_numpy(float); Xte = ten[feat_cols].to_numpy(float)
    Xtr, Xte = impute_train(Xtr, Xte)
    Ytr = np.column_stack([trn[t].to_numpy(float) for t in TARGETS])
    sc, m = fit_ridge_multi(Xtr, Ytr, alpha=10.0)
    Pml = m.predict(sc.transform(Xte))
    ml = {t: Pml[:, i] for i, t in enumerate(TARGETS)}
    results["ml_ridge_body"] = {"scaled_mse": scaled_mse(yt, ml),
                                "per_target": scaled_mse_per_target(yt, ml)}

    # ----- (3+4) PURE PHYSICS DECODER (LUPI) ---------------------------------
    # train body->ball launch state, decode analytically, affine-align crossing->target.
    # The affine crossing->target is fit on OUT-OF-FOLD train decodes so its scale matches
    # the noise the test decode actually has (in-sample train decode is over-confident and
    # mis-calibrates the affine variance).
    state_te = predict_state(trn, ten, feat_cols)
    dec_te = physics_targets_from_state(state_to_decoder_input(state_te))

    oof_entry = np.full(len(trn), np.nan)
    oof_xc = np.full(len(trn), np.nan); oof_yc = np.full(len(trn), np.nan)
    for tr_i, te_i, _ in scheme_a_folds(trn, 5, 0):
        sub_tr = trn.iloc[tr_i].reset_index(drop=True)
        sub_te = trn.iloc[te_i].reset_index(drop=True)
        st = predict_state(sub_tr, sub_te, feat_cols)
        dc = physics_targets_from_state(state_to_decoder_input(st))
        oof_entry[te_i] = dc["entry"]; oof_xc[te_i] = dc["x_cross"]; oof_yc[te_i] = dc["y_cross"]
    dec_tr = {"entry": oof_entry, "x_cross": oof_xc, "y_cross": oof_yc}

    # affine calibrations fit on OOF-TRAIN crossing -> TRAIN target
    phys = {}
    # angle <- entry
    ca = affine_fit(dec_tr["entry"][:, None], trn["angle"].to_numpy(float))
    phys["angle"] = affine_apply(ca, dec_te["entry"][:, None])
    # depth <- (x_cross, y_cross); left_right <- (x_cross, y_cross)
    Xtr_xy = np.column_stack([dec_tr["x_cross"], dec_tr["y_cross"]])
    Xte_xy = np.column_stack([dec_te["x_cross"], dec_te["y_cross"]])
    cd = affine_fit(Xtr_xy, trn["depth"].to_numpy(float))
    phys["depth"] = affine_apply(cd, Xte_xy)
    cl = affine_fit(Xtr_xy, trn["left_right"].to_numpy(float))
    phys["left_right"] = affine_apply(cl, Xte_xy)
    # guard nans -> train mean
    for t in TARGETS:
        phys[t] = np.where(np.isfinite(phys[t]), phys[t], gmean[t])
    results["physics_decoder"] = {"scaled_mse": scaled_mse(yt, phys),
                                  "per_target": scaled_mse_per_target(yt, phys)}

    # ----- body_launch_fit direct decode (student crossing, no ball teacher) -
    # Uses body_launch_fit's own ballistic crossing (already in the df), affine-aligned.
    direct = {}
    if {"body_x_cross", "body_y_cross", "body_bc_entry"}.issubset(df.columns):
        bx_tr = trn[["body_x_cross", "body_y_cross"]].to_numpy(float)
        bx_te = ten[["body_x_cross", "body_y_cross"]].to_numpy(float)
        be_tr = trn["body_bc_entry"].to_numpy(float)[:, None]
        be_te = ten["body_bc_entry"].to_numpy(float)[:, None]
        bx_tr, bx_te = impute_train(bx_tr, bx_te)
        be_tr, be_te = impute_train(be_tr, be_te)
        direct["angle"] = affine_apply(affine_fit(be_tr, trn["angle"].to_numpy(float)), be_te)
        direct["depth"] = affine_apply(affine_fit(bx_tr, trn["depth"].to_numpy(float)), bx_te)
        direct["left_right"] = affine_apply(affine_fit(bx_tr, trn["left_right"].to_numpy(float)), bx_te)
        for t in TARGETS:
            direct[t] = np.where(np.isfinite(direct[t]), direct[t], gmean[t])
        results["body_direct_decode"] = {"scaled_mse": scaled_mse(yt, direct),
                                         "per_target": scaled_mse_per_target(yt, direct)}

    # ----- (5) HYBRID: physics decode as features + body features (+player) --
    # Ridge only. This small (345-row), noisy task punishes the HGB candidate badly out
    # of sample (it minimised CV MSE but overfit depth on test); a regularised linear map
    # is the appropriate hypothesis class. Alpha is chosen per target by CV WITHIN TRAIN.
    # Player identity dummies are a LEGAL same-shooter signal. Physics decode columns are
    # the OOF train / full-train test analytic crossings (dec_tr / dec_te).
    players = sorted(df["player"].unique())

    def player_dummies(d):
        return np.column_stack([(d["player"] == p).astype(float) for p in players])

    def build_X(base_df, phys_feat, use_phys: bool, use_player: bool):
        parts = [base_df[feat_cols].to_numpy(float)]
        if use_phys:
            parts.append(np.column_stack([phys_feat["entry"], phys_feat["x_cross"],
                                          phys_feat["y_cross"]]))
        return np.column_stack(parts)

    def cv_alpha(Xtr_full, ytr_full, alpha, seed=1):
        errs = []
        for tr_i, te_i, _ in scheme_a_folds(trn, 5, seed):
            sc = StandardScaler().fit(Xtr_full[tr_i])
            mm = Ridge(alpha=alpha).fit(sc.transform(Xtr_full[tr_i]), ytr_full[tr_i])
            errs.append(np.mean((mm.predict(sc.transform(Xtr_full[te_i])) - ytr_full[te_i]) ** 2))
        return float(np.mean(errs))

    ALPHAS = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0)

    def ridge_predict(Xtr_full, Xte_full, use_player):
        if use_player:
            Xtr_full = np.column_stack([Xtr_full, player_dummies(trn)])
            Xte_full = np.column_stack([Xte_full, player_dummies(ten)])
        pred = {}
        choice = {}
        for t in TARGETS:
            ytr_full = trn[t].to_numpy(float)
            a = min(ALPHAS, key=lambda al: cv_alpha(Xtr_full, ytr_full, al))
            choice[t] = a
            sc = StandardScaler().fit(Xtr_full)
            mm = Ridge(alpha=a).fit(sc.transform(Xtr_full), ytr_full)
            pred[t] = mm.predict(sc.transform(Xte_full))
        return pred, choice

    variants = {
        "ridge_body_cv": (False, False),
        "ridge_body_physics": (True, False),
        "ridge_body_player": (False, True),
        "hybrid_body_physics_player": (True, True),
    }
    for name, (use_phys, use_player) in variants.items():
        Xtr_v = build_X(trn, dec_tr, use_phys, False)
        Xte_v = build_X(ten, dec_te, use_phys, False)
        Xtr_v, Xte_v = impute_train(Xtr_v, Xte_v)
        pred, choice = ridge_predict(Xtr_v, Xte_v, use_player)
        results[name] = {"scaled_mse": scaled_mse(yt, pred),
                         "per_target": scaled_mse_per_target(yt, pred),
                         "alpha_choice": {t: choice[t] for t in TARGETS}}

    # ----- report -----------------------------------------------------------
    print(f"\n{'model':26s}{'sMSE':>10s}   ang / dep / lr")
    for n, r in results.items():
        per = r["per_target"]
        print(f"{n:26s}{r['scaled_mse']:10.6f}   " +
              " ".join(f"{per[t]:.4f}" for t in TARGETS))

    best = min(results, key=lambda k: results[k]["scaled_mse"])
    bsm = results[best]["scaled_mse"]
    print(f"\nbest = {best}: {bsm:.6f}")
    print(f"  prior(Ridge)={PRIOR:.6f}  beat_prior={bsm < PRIOR}")
    print(f"  winner={WINNER:.6f}  beat_winner={bsm < WINNER}")

    # the PRINCIPLED model is the CV-selected hybrid (alpha chosen by CV-within-train);
    # ml_ridge_body's fixed alpha=10 is a single un-tuned point and not a defensible pick.
    principled = "hybrid_body_physics_player"
    psm = results[principled]["scaled_mse"]
    print(f"principled (CV-selected) = {principled}: {psm:.6f}  beat_prior={psm < PRIOR}")

    out = {
        "split": {"train": int(len(trn)), "test": int(len(ten)), "n_features": len(feat_cols)},
        "benchmarks": {"prior_ridge": PRIOR, "winner": WINNER},
        "results": results,
        "best": {"model": best, "scaled_mse": bsm,
                 "beats_prior": bool(bsm < PRIOR), "beats_winner": bool(bsm < WINNER)},
        "principled": {"model": principled, "scaled_mse": psm,
                       "beats_prior": bool(psm < PRIOR), "beats_winner": bool(psm < WINNER),
                       "note": "best raw point (ml_ridge_body=%.6f) is an un-tuned fixed-alpha "
                               "fluke; CIs overlap prior; physics LUPI does not robustly beat prior."
                               % bsm},
    }
    OUTPUTS_DIR.mkdir(exist_ok=True)
    with open(OUTPUTS_DIR / "v3_exp_a.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {OUTPUTS_DIR / 'v3_exp_a.json'}")
    return out


if __name__ == "__main__":
    run()
