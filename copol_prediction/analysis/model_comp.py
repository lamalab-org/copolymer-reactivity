#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimal analyzer for hold-out results (MACRO metrics + macro accuracy).

- Expects JSON files created by save_holdout_metrics_json(...)
  named like: holdout_spec1_poly0_neg1_aug0.json
- Produces:
  - artifacts/plots/heatmap_<METRIC>.png
  - artifacts/plots/bar_<METRIC>.png
  - artifacts/plots/cm_best_<METRIC>.png
  - artifacts/plots/holdout_summary_macro.csv
"""

# ---------- SETTINGS ----------
INPUT_DIR  = "artifacts/experiments_holdout"
OUTPUT_DIR = "artifacts/plots"
# choose one of: "accuracy_macro", "f1_macro", "precision_macro", "recall_macro"
METRIC     = "accuracy_macro"
SAVE_CSV   = True
# ------------------------------

import os, re, glob, json
import numpy as np
import pandas as pd

# Force non-interactive backend & save to files only
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FNAME_RE = re.compile(
    r"holdout_(?:.*)?spec(?P<spec>[01])_poly(?P<poly>[01])_neg(?P<neg>[01])_aug(?P<aug>[01])\.json$"
)


def infer_combo_from_filename(path: str):
    m = FNAME_RE.search(os.path.basename(path))
    if not m:
        return None
    return {
        "remove_specialized": int(m.group("spec")),
        "poly_filter": int(m.group("poly")),
        "neg_data": int(m.group("neg")),
        "augmentation": int(m.group("aug")),
    }


def _macro_accuracy_from_cm(cm: np.ndarray) -> float:
    """
    Balanced accuracy = mean of per-class recalls = mean(diag / row_sum).
    Ignores classes with zero support to avoid NaN bias.
    """
    if cm.size == 0:
        return float("nan")
    row_sums = cm.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class_recall = np.where(row_sums > 0, np.diag(cm) / row_sums, np.nan)
    if np.all(np.isnan(per_class_recall)):
        return float("nan")
    return float(np.nanmean(per_class_recall))


def load_holdout_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    report = payload.get("classification_report", {})
    labels = payload.get("labels", None)
    cm = np.array(payload.get("confusion_matrix", []), dtype=float)

    # --- MACRO metrics from report ---
    macro = report.get("macro avg", {}) if isinstance(report, dict) else {}
    f1_macro        = float(macro.get("f1-score", np.nan))
    precision_macro = float(macro.get("precision", np.nan))
    recall_macro    = float(macro.get("recall", np.nan))

    # --- Macro accuracy (balanced accuracy) from the confusion matrix ---
    accuracy_macro = _macro_accuracy_from_cm(cm)
    # Fallback: if CM missing, use macro recall (equivalent in multi-class)
    if np.isnan(accuracy_macro) and isinstance(macro, dict):
        accuracy_macro = recall_macro

    # Optional: per-class F1 if you ever want to plot these later
    class_f1 = {}
    if isinstance(report, dict):
        for k, v in report.items():
            if isinstance(k, str) and k.isdigit() and isinstance(v, dict):
                try:
                    class_f1[int(k)] = float(v.get("f1-score", np.nan))
                except Exception:
                    class_f1[int(k)] = np.nan

    return {
        "labels": labels,
        "cm": cm,
        "accuracy_macro": accuracy_macro,
        "f1_macro": f1_macro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "class_f1": class_f1,
    }


def build_dataframe_from_dir(directory: str) -> pd.DataFrame:
    rows = []
    files = sorted(glob.glob(os.path.join(directory, "holdout_*.json")))
    print(f"Found {len(files)} JSON files in {directory}")
    for path in files:
        cfg = infer_combo_from_filename(path)
        if cfg is None:
            print(f"[skip] {os.path.basename(path)} (filename doesn’t encode switches)")
            continue
        rec = load_holdout_json(path)
        row = {
            "run": os.path.basename(path).replace(".json", ""),
            "path": path,
            **cfg,
            # macro-only columns:
            "accuracy_macro": rec["accuracy_macro"],
            "f1_macro": rec["f1_macro"],
            "precision_macro": rec["precision_macro"],
            "recall_macro": rec["recall_macro"],
            "labels": rec["labels"],
            "cm": rec["cm"].tolist() if rec["cm"] is not None else None,
        }
        for cls, f1 in rec["class_f1"].items():
            row[f"class_f1_{cls}"] = f1
        rows.append(row)

    if not rows:
        raise FileNotFoundError(
            f"No matching holdout_*.json in '{directory}'. "
            "Make sure your sweep wrote files like 'holdout_spec1_poly0_neg1_aug0.json'."
        )

    df = pd.DataFrame(rows)
    # deterministic 4x4 axes: rows = S/P, cols = N/A
    df["row_code"] = df.apply(lambda r: f"S{r['remove_specialized']}-P{r['poly_filter']}", axis=1)
    df["col_code"] = df.apply(lambda r: f"N{r['neg_data']}-A{r['augmentation']}", axis=1)
    return df


def save_heatmap_metric(df: pd.DataFrame, metric: str, out_path: str):
    pivot = df.pivot_table(index="row_code", columns="col_code", values=metric, aggfunc="mean")
    all_rows = [f"S{s}-P{p}" for s in (0,1) for p in (0,1)]
    all_cols = [f"N{n}-A{a}" for n in (0,1) for a in (0,1)]
    pivot = pivot.reindex(index=all_rows, columns=all_cols)

    plt.figure()
    im = plt.imshow(pivot.values, interpolation="nearest", cmap="Blues")
    plt.title(f"Heatmap of {metric} (hold-out)")
    plt.colorbar(im)
    plt.xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    plt.yticks(np.arange(len(pivot.index)), pivot.index)
    plt.xlabel("neg. data / augmentation")
    plt.ylabel("specialized filter/ polym. filter")

    # annotate
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            txt = "–" if pd.isna(val) else f"{val:.3f}"
            plt.text(j, i, txt, ha="center", va="center", fontsize=12, color="black")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[SAVED] {out_path}")


def save_sorted_bar(df: pd.DataFrame, metric: str, out_path: str):
    tmp = df.sort_values(metric, ascending=True)
    plt.figure()
    plt.barh(tmp["run"], tmp[metric])
    plt.xlabel(f"{metric} (hold-out)")
    plt.title(f"Hold-out {metric} across filter combinations")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.xlim(0.65, 0.85)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[SAVED] {out_path}")


def save_confusion_matrix_of_best(df: pd.DataFrame, metric: str, out_path: str):
    if df[metric].isna().all():
        print(f"[WARN] All {metric} values are NaN; skipping CM.")
        return
    best_idx = df[metric].idxmax()
    row = df.loc[best_idx]
    cm = np.array(row["cm"])
    labels = row.get("labels", [0,1,2])

    plt.figure()
    im = plt.imshow(cm, interpolation="nearest", cmap="Blues", vmin=0, vmax=500)
    plt.title(f"Confusion Matrix (best by {metric}: {row['run']})")
    plt.colorbar(im)
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels)
    plt.yticks(ticks, labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = int(cm[i, j])
            color = "white" if cm[i, j] > thresh else "black"
            plt.text(j, i, val, ha="center", va="center", fontsize=12, color=color)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[SAVED] {out_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = build_dataframe_from_dir(INPUT_DIR)

    # Save summary CSV (macro metrics + macro accuracy)
    if SAVE_CSV:
        csv_path = os.path.join(OUTPUT_DIR, "holdout_summary_macro.csv")
        cols = [
            "run", "remove_specialized", "poly_filter", "neg_data", "augmentation",
            "accuracy_macro", "f1_macro", "precision_macro", "recall_macro"
        ]
        df[cols].to_csv(csv_path, index=False)
        print(f"[SAVED] {csv_path}")

    # Plots (metric selected above)
    save_heatmap_metric(df, METRIC, os.path.join(OUTPUT_DIR, f"heatmap_{METRIC}.png"))
    save_sorted_bar(df, METRIC, os.path.join(OUTPUT_DIR, f"bar_{METRIC}.png"))
    save_confusion_matrix_of_best(df, METRIC, os.path.join(OUTPUT_DIR, f"cm_best_{METRIC}.png"))

    # Quick console top-5
    cols_print = ["run", "accuracy_macro", "f1_macro", "precision_macro", "recall_macro"]
    print("\nTop-5 runs by", METRIC)
    print(df.sort_values(METRIC, ascending=False)[cols_print].head(5).to_string(index=False))

if __name__ == "__main__":
    main()
