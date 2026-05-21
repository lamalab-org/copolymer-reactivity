#!/usr/bin/env python3
"""Reproduce the PolyCarp train/test performance table from the paper.

Reproduces ``tab:train_test_voting_performance`` — per-class accuracy
(recall), precision, F1 and coverage for the *voting* model (XGBoost +
nearest-neighbour lookup, predictions retained only where the two agree) on
the train and test splits.

It does NOT retrain. The XGBoost predictions come either from the released
``artifacts/model_bundle`` (default) or, with ``--api``, from a deployed
API's ``/predict/batch`` endpoint — both serve the identical model, so the
table reproduces either way. The nearest-neighbour lookup is a deterministic
RDKit-fingerprint baseline (not the trained model); it is always computed
locally against the train-only pool the paper uses.

Usage
-----
    cd copol_prediction
    python reproduce_paper_metrics.py                       # local model bundle
    python reproduce_paper_metrics.py --api http://localhost:8000

Exits non-zero if any reproduced number deviates from the paper by > 0.005.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import precision_recall_fscore_support

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "src"))
sys.path.insert(0, SCRIPT_DIR)  # so `analysis.analyze_model` is importable

from analysis.analyze_model import (  # noqa: E402
    compute_fingerprints_for_smiles,
    compute_naive_baseline_predictions_with_similarity,
)

from copolpredictor.inference import CopolymerPredictor  # noqa: E402

BUNDLE = os.path.join(SCRIPT_DIR, "artifacts", "model_bundle")
SPLIT_DIR = os.path.join(SCRIPT_DIR, "artifacts", "data_splits")
SMILES_COLS = ["monomer1_smiles", "monomer2_smiles", "solvent_smiles"]
CLASSES = [0, 1, 2]
CLASS_NAMES = ["Alternating", "Random", "Gradient"]

# tab:train_test_voting_performance in sections/appendix.tex — (acc, prec, f1).
PAPER = {
    "Train": {
        "Alternating": (1.000, 0.957, 0.978),
        "Random": (0.984, 0.982, 0.983),
        "Gradient": (0.983, 0.989, 0.986),
        "Macro": (0.989, 0.976, 0.982),
        "coverage": 0.920,
    },
    "Test": {
        "Alternating": (0.788, 0.667, 0.722),
        "Random": (0.781, 0.812, 0.796),
        "Gradient": (0.845, 0.827, 0.836),
        "Macro": (0.805, 0.768, 0.785),
        "coverage": 0.770,
    },
}
TOL = 0.005


def xgb_predict_local(df_eval, features):
    """XGBoost predictions from the released model bundle."""
    predictor = CopolymerPredictor(BUNDLE)
    return np.asarray(predictor.predict(df_eval[features])).astype(int)


def xgb_predict_api(df_eval, features, api_url, chunk=500):
    """XGBoost predictions from a deployed API's /predict/batch endpoint.

    Sends the precomputed feature rows straight to the model — this verifies
    the deployment serves the same model the bundle does."""
    preds: list[int] = []
    rows = df_eval[features].to_dict("records")
    for start in range(0, len(rows), chunk):
        batch = rows[start : start + chunk]
        resp = requests.post(
            f"{api_url.rstrip('/')}/predict/batch", json={"samples": batch}, timeout=300
        )
        resp.raise_for_status()
        preds.extend(p["predicted_class"] for p in resp.json()["predictions"])
    if len(preds) != len(rows):
        raise RuntimeError(f"API returned {len(preds)} predictions for {len(rows)} rows")
    return np.asarray(preds).astype(int)


def voting_predictions(xgb_predict, df_eval, df_pool, features):
    """XGBoost + nearest-neighbour lookup; return (y_true, retained mask,
    y_true_retained, y_pred_retained) for the voting-agreement subset.

    `xgb_predict(df, features) -> np.ndarray` supplies the XGBoost half
    (local bundle or deployed API)."""
    df_eval = df_eval.dropna(subset=features + ["r_product_class"]).reset_index(drop=True)
    df_pool = df_pool.dropna(subset=features + ["r_product_class"]).reset_index(drop=True)

    y = df_eval["r_product_class"].astype(int).values
    xgb_pred = xgb_predict(df_eval, features)

    all_smiles = set()
    for frame in (df_pool, df_eval):
        for col in SMILES_COLS:
            all_smiles.update(frame[col].dropna().unique())
    fp_dict = compute_fingerprints_for_smiles(list(all_smiles))

    y_pool = df_pool["r_product_class"].astype(int).values
    lookup_pred, _ = compute_naive_baseline_predictions_with_similarity(
        df_eval, df_pool, y_pool, features, fp_dict=fp_dict
    )
    lookup_pred = np.asarray(lookup_pred).astype(int)

    agree = xgb_pred == lookup_pred
    return y, agree, y[agree], xgb_pred[agree]


def per_class_table(y_true, y_pred):
    """Per-class (recall, precision, f1) + macro, matching the paper table."""
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASSES, zero_division=0
    )
    rows = {name: (rec[i], prec[i], f1[i]) for i, name in enumerate(CLASS_NAMES)}
    rows["Macro"] = (float(rec.mean()), float(prec.mean()), float(f1.mean()))
    return rows


def main() -> int:
    import functools
    import json

    parser = argparse.ArgumentParser(
        description="Reproduce the paper's train/test voting-performance table."
    )
    parser.add_argument(
        "--api",
        metavar="URL",
        default=None,
        help="Deployed API base URL. XGBoost predictions are then taken from "
        "<URL>/predict/batch instead of the local model bundle.",
    )
    args = parser.parse_args()

    # Feature list from the artifact (avoids loading the model in --api mode).
    with open(os.path.join(BUNDLE, "meta.json")) as fh:
        features = json.load(fh)["feature_columns"]

    if args.api:
        print(f"XGBoost predictions: deployed API — {args.api}/predict/batch")
        xgb_predict = functools.partial(xgb_predict_api, api_url=args.api)
    else:
        print(f"XGBoost predictions: local model bundle — {BUNDLE}")
        xgb_predict = xgb_predict_local

    train = pd.read_csv(os.path.join(SPLIT_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(SPLIT_DIR, "test.csv"))

    # Lookup pool is the training set for both splits (paper Methods:
    # "the most similar training sample").
    splits = {"Train": (train, train), "Test": (test, train)}

    ok = True
    for split, (df_eval, df_pool) in splits.items():
        y, agree, yt, yp = voting_predictions(xgb_predict, df_eval, df_pool, features)
        coverage = float(agree.mean())
        rows = per_class_table(yt, yp)

        print(
            f"\n{'=' * 68}\n{split} — voting model "
            f"(retained {agree.sum()}/{len(agree)}, coverage {coverage:.3f})\n{'=' * 68}"
        )
        print(f"{'Class':<13}{'Acc':>8}{'Prec':>8}{'F1':>8}   " f"{'paper Acc/Prec/F1':>22}")
        for name in CLASS_NAMES + ["Macro"]:
            acc, prec, f1 = rows[name]
            p_acc, p_prec, p_f1 = PAPER[split][name]
            dev = max(abs(acc - p_acc), abs(prec - p_prec), abs(f1 - p_f1))
            flag = "ok" if dev <= TOL else f"OFF by {dev:.3f}"
            ok &= dev <= TOL
            print(
                f"{name:<13}{acc:>8.3f}{prec:>8.3f}{f1:>8.3f}   "
                f"{p_acc:>6.3f}/{p_prec:.3f}/{p_f1:.3f}  {flag}"
            )
        cov_dev = abs(coverage - PAPER[split]["coverage"])
        ok &= cov_dev <= TOL
        print(
            f"{'coverage':<13}{coverage:>8.3f}{'':>16}   "
            f"{PAPER[split]['coverage']:>6.3f}        "
            f"{'ok' if cov_dev <= TOL else f'OFF by {cov_dev:.3f}'}"
        )

    print(f"\n{'=' * 68}")
    if ok:
        print("REPRODUCED: all values within ±%.3f of the paper table." % TOL)
        return 0
    print("MISMATCH: at least one value deviates from the paper table.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
