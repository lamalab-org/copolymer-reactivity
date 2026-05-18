#!/usr/bin/env python3
"""
Compare experiment results.
"""

import json
import os

import pandas as pd


def load_results(experiment_dir):
    """Load results from experiment directory."""
    results_path = os.path.join(experiment_dir, "results.json")
    if not os.path.exists(results_path):
        return None

    with open(results_path, "r") as f:
        return json.load(f)


def compare_experiments():
    """Compare all experiment results."""
    experiments = {"baseline": "baseline/results", "fingerprint": "fingerprint/results"}

    results = []
    for name, path in experiments.items():
        if os.path.exists(path):
            data = load_results(path)
            if data:
                results.append(
                    {
                        "experiment": data.get("experiment", name),
                        "cv_score": data.get("cv_score"),
                        "holdout_acc": data.get("holdout_accuracy"),
                        "holdout_f1_w": data.get("holdout_f1_weighted"),
                        "holdout_f1_m": data.get("holdout_f1_macro"),
                        "num_features": data.get("num_features"),
                    }
                )

    if not results:
        print("No results found. Run experiments first.")
        return

    df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPARISON")
    print("=" * 70)
    print()
    print(df.to_string(index=False))
    print()

    if len(results) == 2:
        print("=" * 70)
        print("DELTA (experiment 2 - experiment 1)")
        print("=" * 70)
        for metric in ["cv_score", "holdout_acc", "holdout_f1_w", "holdout_f1_m"]:
            if df[metric].notna().all():
                diff = df.iloc[1][metric] - df.iloc[0][metric]
                pct = (diff / df.iloc[0][metric]) * 100
                print(f"{metric:20s}: {diff:+.4f} ({pct:+.2f}%)")
        print()

    # Save to CSV
    df.to_csv("comparison.csv", index=False)
    print(f"Saved to: comparison.csv")


if __name__ == "__main__":
    compare_experiments()
