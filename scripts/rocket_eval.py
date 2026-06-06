"""Full-time-series modeling with MiniRocket — TRAIN-split CV only (no test peeking).

Builds release-aligned multivariate tensors (all keypoints), then evaluates MiniRocket
+ RidgeCV per target under Scheme A (stratified) and Scheme B (LOPO) on the competition
TRAIN split. Compares to the 37-scalar ridge under the same folds. Writes outputs/rocket_cv.json.
Final held-out/test read is deliberately NOT done here.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from aeon.transformations.collection.convolution_based import MiniRocket

from fta.config import OUTPUTS_DIR, TARGETS, RANDOM_SEED
from fta.cv import scheme_a_folds, scheme_b_folds
from fta.loader import list_shots, load_metadata, load_tracking
from fta.metric import scaled_mse, scaled_mse_per_target
from fta.tensor import build_tensor, _marker_order

ALPHAS = np.logspace(-3, 3, 13)
CACHE = OUTPUTS_DIR / "tensors_2025.npz"


def build_cache():
    shots = list_shots()
    order = _marker_order(list(load_tracking(shots[0].path).joints))
    X, ids = [], []
    for i, s in enumerate(shots, 1):
        t = build_tensor(load_tracking(s.path), markers=order)
        if t is None:
            continue
        X.append(t.astype(np.float32)); ids.append(s.shot_id)
        if i % 100 == 0:
            print(f"  tensor {i}")
    X = np.stack(X)
    np.savez_compressed(CACHE, X=X, ids=np.array(ids))
    print(f"cached {X.shape} -> {CACHE}")
    return X, ids


def fit_predict(Xtr, Xte, ytr_df):
    """MiniRocket fit on train fold; RidgeCV per target."""
    mr = MiniRocket(random_state=RANDOM_SEED)
    Ftr = mr.fit_transform(Xtr)
    Fte = mr.transform(Xte)
    sc = StandardScaler().fit(Ftr)
    Ftr, Fte = sc.transform(Ftr), sc.transform(Fte)
    out = {}
    for t in TARGETS:
        m = RidgeCV(alphas=ALPHAS).fit(Ftr, ytr_df[t].to_numpy(float))
        out[t] = m.predict(Fte)
    return out


def run_scheme(X, df, folds):
    n = len(df)
    yp = {t: np.full(n, np.nan) for t in TARGETS}
    for tr, te, lab in folds:
        pr = fit_predict(X[tr], X[te], df.iloc[tr])
        for t in TARGETS:
            yp[t][te] = pr[t]
        print(f"    fold {lab} done")
    yt = {t: df[t].to_numpy(float) for t in TARGETS}
    return yt, yp


def main():
    warnings.filterwarnings("ignore")
    if CACHE.exists():
        d = np.load(CACHE, allow_pickle=True); X, ids = d["X"], list(d["ids"])
        print(f"loaded cache {X.shape}")
    else:
        X, ids = build_cache()

    # labels + competition split aligned to X rows
    meta = {s.shot_id: (s.player, load_metadata(s.path)) for s in list_shots()}
    tgt_key = {(round(m["angle"], 2), round(m["depth"], 2), round(m["left_right"], 2)): sid
               for sid, (_, m) in meta.items()}
    tr_csv = pd.read_csv("spl-utspan-data-challenge-2026/train.csv")
    train_ids = {tgt_key[(round(r.angle, 2), round(r.depth, 2), round(r.left_right, 2))]
                 for _, r in tr_csv.iterrows()}

    rows = [{"shot_id": sid, "player": meta[sid][0],
             "csplit": "train" if sid in train_ids else "test",
             **{t: meta[sid][1][t] for t in TARGETS}} for sid in ids]
    df = pd.DataFrame(rows)
    Xtr_all = X[(df.csplit == "train").to_numpy()]
    dtr = df[df.csplit == "train"].reset_index(drop=True)
    print(f"TRAIN tensors: {Xtr_all.shape}  (test held back)")

    out = {}
    print("\nScheme A (stratified 5-fold, seed 0):")
    yt, yp = run_scheme(Xtr_all, dtr, list(scheme_a_folds(dtr, 5, 0)))
    out["scheme_a"] = {"scaled_mse": scaled_mse(yt, yp),
                       "per_target": scaled_mse_per_target(yt, yp)}
    print(f"  A scaled_mse={out['scheme_a']['scaled_mse']:.6f}  per={ {k:round(v,4) for k,v in out['scheme_a']['per_target'].items()} }")

    print("\nScheme B (leave-one-player-out):")
    yt, yp = run_scheme(Xtr_all, dtr, list(scheme_b_folds(dtr)))
    out["scheme_b"] = {"scaled_mse": scaled_mse(yt, yp),
                       "per_target": scaled_mse_per_target(yt, yp)}
    print(f"  B scaled_mse={out['scheme_b']['scaled_mse']:.6f}  per={ {k:round(v,4) for k,v in out['scheme_b']['per_target'].items()} }")

    json.dump(out, open(OUTPUTS_DIR / "rocket_cv.json", "w"), indent=2)
    print(f"\nWrote {OUTPUTS_DIR/'rocket_cv.json'}")


if __name__ == "__main__":
    main()
