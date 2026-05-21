#!/usr/bin/env python3
"""Reproduce the PolyCarp train/test performance results from the paper.

Reproduces ``tab:train_test_voting_performance`` — per-class accuracy
(recall), precision, F1 and coverage for the plain XGBoost classifier and
the *voting* model (XGBoost + nearest-neighbour lookup, predictions retained
only where the two agree) on the train and test splits.

It does NOT retrain. The XGBoost predictions come either from the released
``artifacts/model_bundle`` (default) or, with ``--api``, from a deployed
API's ``/predict/batch`` endpoint — both serve the identical model. The
nearest-neighbour lookup is a deterministic RDKit-fingerprint baseline (not
the trained model); it is always computed locally against the train-only
pool the paper uses.

Usage
-----
    cd copol_prediction
    python reproduce_paper_metrics.py                       # local model bundle
    python reproduce_paper_metrics.py --api http://localhost:8000
    python reproduce_paper_metrics.py --json artifacts/paper_metrics.json

``--json`` writes the full results — aggregate metrics, confusion matrices
*and* per-row individual predictions — to a file. That file is what the API
serves at ``GET /paper_metrics`` and the web UI renders in its Results tab.

Without ``--json`` the script exits non-zero if any reproduced number
deviates from the paper by > 0.005, so it doubles as an artifact regression
check.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import warnings
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "src"))
sys.path.insert(0, SCRIPT_DIR)  # so `analysis.analyze_model` is importable
sys.path.insert(0, os.path.join(SCRIPT_DIR, "api"))  # for baseline_lookup

from analysis.analyze_model import (  # noqa: E402
    compute_fingerprints_for_smiles,
    compute_naive_baseline_predictions_with_similarity,
)
from baseline_lookup import doi_from_source_filename, doi_url  # noqa: E402

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
    """(classes, confidences) from the released model bundle."""
    predictor = CopolymerPredictor(BUNDLE)
    out = predictor.predict_with_confidence(df_eval[features])
    return (
        np.asarray(out["predictions"]).astype(int),
        np.asarray(out["confidence"]).astype(float),
    )


def xgb_predict_api(df_eval, features, api_url, chunk=500):
    """(classes, confidences) from a deployed API's /predict/batch endpoint."""
    classes: list[int] = []
    confs: list[float] = []
    rows = df_eval[features].to_dict("records")
    for start in range(0, len(rows), chunk):
        resp = requests.post(
            f"{api_url.rstrip('/')}/predict/batch",
            json={"samples": rows[start : start + chunk]},
            timeout=300,
        )
        resp.raise_for_status()
        for pred in resp.json()["predictions"]:
            classes.append(pred["predicted_class"])
            confs.append(pred["confidence"])
    if len(classes) != len(rows):
        raise RuntimeError(f"API returned {len(classes)} predictions for {len(rows)} rows")
    return np.asarray(classes).astype(int), np.asarray(confs).astype(float)


def evaluate_split(xgb_predict, df_eval, df_pool, features):
    """Run XGBoost + nearest-neighbour lookup on one split.

    Returns a dict with the cleaned eval frame and aligned arrays:
    y_true, xgb_pred, xgb_conf, lookup_pred, lookup_sim, agree.
    """
    df_eval = df_eval.dropna(subset=features + ["r_product_class"]).reset_index(drop=True)
    df_pool = df_pool.dropna(subset=features + ["r_product_class"]).reset_index(drop=True)

    y = df_eval["r_product_class"].astype(int).values
    xgb_pred, xgb_conf = xgb_predict(df_eval, features)

    all_smiles = set()
    for frame in (df_pool, df_eval):
        for col in SMILES_COLS:
            all_smiles.update(frame[col].dropna().unique())
    fp_dict = compute_fingerprints_for_smiles(list(all_smiles))

    y_pool = df_pool["r_product_class"].astype(int).values
    lookup_pred, lookup_sim = compute_naive_baseline_predictions_with_similarity(
        df_eval, df_pool, y_pool, features, fp_dict=fp_dict
    )
    lookup_pred = np.asarray(lookup_pred).astype(int)

    return {
        "df": df_eval,
        "y": y,
        "xgb_pred": xgb_pred,
        "xgb_conf": xgb_conf,
        "lookup_pred": lookup_pred,
        "lookup_sim": np.asarray(lookup_sim).astype(float),
        "agree": xgb_pred == lookup_pred,
    }


def per_class_table(y_true, y_pred):
    """Per-class (recall, precision, f1) + Macro, matching the paper table."""
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASSES, zero_division=0
    )
    rows = {
        name: {
            "acc": round(float(rec[i]), 3),
            "prec": round(float(prec[i]), 3),
            "f1": round(float(f1[i]), 3),
        }
        for i, name in enumerate(CLASS_NAMES)
    }
    rows["Macro"] = {
        "acc": round(float(rec.mean()), 3),
        "prec": round(float(prec.mean()), 3),
        "f1": round(float(f1.mean()), 3),
    }
    return rows


def model_metrics(y_true, y_pred):
    """Aggregate block: confusion matrix + per-class table."""
    return {
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASSES).tolist(),
        "per_class": per_class_table(y_true, y_pred),
    }


def _cell(row, col, default=None):
    val = row.get(col, default)
    if isinstance(val, float) and np.isnan(val):
        return default
    return val


def individual_predictions(ev):
    """Per-row records for the individual-predictions browser."""
    df = ev["df"]
    records = []
    for i, (_, row) in enumerate(df.iterrows()):
        doi = doi_from_source_filename(_cell(row, "source_filename"))
        records.append(
            {
                "monomer1_smiles": _cell(row, "monomer1_smiles", ""),
                "monomer2_smiles": _cell(row, "monomer2_smiles", ""),
                "monomer1_name": _cell(row, "monomer1_name"),
                "monomer2_name": _cell(row, "monomer2_name"),
                "solvent_name": _cell(row, "solvent"),
                "solvent_smiles": _cell(row, "solvent_smiles"),
                "temperature": _cell(row, "temperature"),
                "method": _cell(row, "method"),
                "polytype": _cell(row, "polymerization_type"),
                "true_class": int(ev["y"][i]),
                "true_class_name": CLASS_NAMES[ev["y"][i]],
                "xgb_class": int(ev["xgb_pred"][i]),
                "xgb_class_name": CLASS_NAMES[ev["xgb_pred"][i]],
                "confidence": round(float(ev["xgb_conf"][i]), 4),
                "lookup_class": int(ev["lookup_pred"][i]),
                "lookup_class_name": CLASS_NAMES[ev["lookup_pred"][i]],
                "lookup_similarity": round(float(ev["lookup_sim"][i]), 4),
                "agree": bool(ev["agree"][i]),
                "correct": bool(ev["xgb_pred"][i] == ev["y"][i]),
                "doi": doi,
                "doi_url": doi_url(doi),
            }
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce the paper's train/test performance results."
    )
    parser.add_argument(
        "--api",
        metavar="URL",
        default=None,
        help="Deployed API base URL. XGBoost predictions are then taken from "
        "<URL>/predict/batch instead of the local model bundle.",
    )
    parser.add_argument(
        "--json",
        metavar="PATH",
        default=None,
        help="Write the full results (aggregate metrics + per-row individual "
        "predictions) to this JSON file. This is the artifact served at "
        "GET /paper_metrics.",
    )
    args = parser.parse_args()

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

    payload = {
        "classes": CLASS_NAMES,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "splits": {},
    }
    ok = True
    for split, (df_eval, df_pool) in splits.items():
        ev = evaluate_split(xgb_predict, df_eval, df_pool, features)
        y, agree = ev["y"], ev["agree"]
        yt, yp = y[agree], ev["xgb_pred"][agree]
        coverage = float(agree.mean())

        xgb_block = model_metrics(y, ev["xgb_pred"])
        xgb_block["accuracy"] = round(float(accuracy_score(y, ev["xgb_pred"])), 4)
        voting_block = model_metrics(yt, yp)
        voting_block["coverage"] = round(coverage, 3)
        voting_block["retained"] = int(agree.sum())

        payload["splits"][split.lower()] = {
            "n": int(len(y)),
            "xgboost": xgb_block,
            "voting": voting_block,
            "predictions": individual_predictions(ev),
        }

        # ── Console table vs the paper ──
        rows = voting_block["per_class"]
        print(
            f"\n{'=' * 68}\n{split} — voting model "
            f"(retained {agree.sum()}/{len(agree)}, coverage {coverage:.3f})\n{'=' * 68}"
        )
        print(f"{'Class':<13}{'Acc':>8}{'Prec':>8}{'F1':>8}   {'paper Acc/Prec/F1':>22}")
        for name in CLASS_NAMES + ["Macro"]:
            m = rows[name]
            p_acc, p_prec, p_f1 = PAPER[split][name]
            dev = max(abs(m["acc"] - p_acc), abs(m["prec"] - p_prec), abs(m["f1"] - p_f1))
            ok &= dev <= TOL
            print(
                f"{name:<13}{m['acc']:>8.3f}{m['prec']:>8.3f}{m['f1']:>8.3f}   "
                f"{p_acc:>6.3f}/{p_prec:.3f}/{p_f1:.3f}  "
                f"{'ok' if dev <= TOL else f'OFF by {dev:.3f}'}"
            )
        cov_dev = abs(coverage - PAPER[split]["coverage"])
        ok &= cov_dev <= TOL
        print(
            f"{'coverage':<13}{coverage:>8.3f}{'':>16}   {PAPER[split]['coverage']:>6.3f}"
            f"        {'ok' if cov_dev <= TOL else f'OFF by {cov_dev:.3f}'}"
        )

    if args.json:
        out_path = args.json if os.path.isabs(args.json) else os.path.join(SCRIPT_DIR, args.json)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, separators=(",", ":"))
        n_pred = sum(len(s["predictions"]) for s in payload["splits"].values())
        print(f"\nWrote {out_path} ({n_pred} individual predictions).")
        return 0

    print(f"\n{'=' * 68}")
    if ok:
        print("REPRODUCED: all values within ±%.3f of the paper table." % TOL)
        return 0
    print("MISMATCH: at least one value deviates from the paper table.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
