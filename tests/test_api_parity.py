"""
Regression test for the copolpredictor API.

Catches three classes of regressions in one place:

1. Model-artifact corruption / library skew — the loaded bundle no longer
   reproduces the published `holdout_accuracy` from
   `artifacts/model_bundle/meta.json`.
2. Feature-name mismatch between `/preprocess_all` and the trained model —
   the exact bug this test was written for. If `/preprocess_all` emits a
   key the model doesn't expect (e.g. `solvent_logP` instead of
   `solvent_logp`), the predictor silently drops the column, every feature
   falls back to NaN→0, and API↔direct agreement collapses.
3. Any future change that breaks the `/preprocess_all` → `/predict`
   pipeline end-to-end.

Skips automatically if heavy dependencies (`fastapi`, `rdkit`, the model
bundle, or `test.csv`) are unavailable, so the existing CI matrix that
doesn't install API deps stays green.
"""
from __future__ import annotations

import math
import os
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
API_DIR = ROOT / "copol_prediction" / "api"
ARTIFACTS = ROOT / "copol_prediction" / "artifacts"
BUNDLE_DIR = ARTIFACTS / "model_bundle"
TEST_CSV = ARTIFACTS / "data_splits" / "test.csv"
TRAIN_CSV = ARTIFACTS / "data_splits" / "train.csv"
NEGATIVE_CSV = (
    ROOT / "copol_prediction" / "filter" / "artificial_datapoints"
    / "combined_augmented_for_processing.csv"
)
DATASET_CSV = ROOT / "copol_prediction" / "processed_data.csv"

# Point the API at local paths before importing the module — its module-level
# defaults are the container-absolute paths used in the Docker image.
os.environ.setdefault("MODEL_PATH", str(BUNDLE_DIR))
os.environ.setdefault("DATASET_PATH", str(DATASET_CSV))
os.environ.setdefault("TRAIN_DATA_PATH", str(TRAIN_CSV))
os.environ.setdefault("NEGATIVE_DATA_PATH", str(NEGATIVE_CSV))

sys.path.insert(0, str(API_DIR))
sys.path.insert(0, str(ROOT / "src"))

pd = pytest.importorskip("pandas")
pytest.importorskip("rdkit")
pytest.importorskip("xgboost")
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

if not BUNDLE_DIR.exists():
    pytest.skip(f"Model bundle not found at {BUNDLE_DIR}", allow_module_level=True)
if not TEST_CSV.exists():
    pytest.skip(f"Test split not found at {TEST_CSV}", allow_module_level=True)

from copolpredictor.inference import CopolymerPredictor  # noqa: E402
import app as api_app  # noqa: E402

# Observed values at the time of writing (Opus 4.7 fix for feature-name
# mismatch): direct accuracy 0.7401, API↔direct agreement 96/100, p95
# per-feature drift ~5e-15. Thresholds below sit comfortably above the
# observed values so flaky failures are unlikely, but tight enough that
# the previous bug — where every feature fell back to zero — would fail
# both `_class_agreement` and `_feature_drift` immediately.
MIN_DIRECT_ACCURACY = 0.73
MIN_CLASS_AGREEMENT = 0.90
MAX_P95_FEATURE_DRIFT = 1e-10
PARITY_SAMPLE_SIZE = 50
PARITY_SEED = 0


@pytest.fixture(scope="module")
def predictor():
    return CopolymerPredictor(str(BUNDLE_DIR))


@pytest.fixture(scope="module")
def test_df():
    return pd.read_csv(TEST_CSV)


@pytest.fixture(scope="module")
def client():
    with TestClient(api_app.app) as c:
        yield c


def test_holdout_accuracy_matches_published(predictor, test_df):
    y_true = test_df["r_product_class"].astype(int).values
    y_pred = predictor.predict(test_df[predictor.features])
    acc = (y_pred == y_true).mean()
    assert acc >= MIN_DIRECT_ACCURACY, (
        f"Holdout accuracy {acc:.4f} < {MIN_DIRECT_ACCURACY} "
        f"(published baseline in meta.json: 0.7401)"
    )


def test_preprocess_all_to_predict_matches_direct_predictor(predictor, test_df, client):
    rng = random.Random(PARITY_SEED)
    idxs = rng.sample(range(len(test_df)), PARITY_SAMPLE_SIZE)
    y_direct = predictor.predict(test_df[predictor.features])

    agree = 0
    per_feature_diffs: dict[str, list[float]] = {c: [] for c in predictor.features}
    failures: list[tuple] = []

    for i in idxs:
        row = test_df.iloc[i]
        payload = {
            "monomer1_smiles": row["monomer1_smiles"],
            "monomer2_smiles": row["monomer2_smiles"],
            "solvent_smiles": row["solvent_smiles"],
            "method": row["method"],
            "polytype": row["polymerization_type"],
            "temperature": float(row["temperature"]),
        }
        r = client.post("/preprocess_all", json=payload)
        if r.status_code != 200:
            failures.append(("preprocess_all", i, r.status_code, r.text[:200]))
            continue
        api_features = r.json()["features"]
        r2 = client.post("/predict", json={"features": api_features})
        if r2.status_code != 200:
            failures.append(("predict", i, r2.status_code, r2.text[:200]))
            continue
        if int(r2.json()["predicted_class"]) == int(y_direct[i]):
            agree += 1
        for col in predictor.features:
            a = api_features.get(col)
            b = row[col]
            if a is None or b is None:
                continue
            if isinstance(b, float) and math.isnan(b):
                continue
            per_feature_diffs[col].append(abs(a - b) / max(abs(b), 1e-9))

    assert not failures, f"API calls failed: {failures[:3]} (and {len(failures)-3} more)"

    agreement = agree / PARITY_SAMPLE_SIZE
    assert agreement >= MIN_CLASS_AGREEMENT, (
        f"API↔direct class agreement {agreement:.2f} < {MIN_CLASS_AGREEMENT}; "
        f"likely a feature-name regression in /preprocess_all "
        f"(see assemble_model_features in app.py)"
    )

    all_diffs = [d for diffs in per_feature_diffs.values() for d in diffs]
    if all_diffs:
        all_diffs.sort()
        p95 = all_diffs[int(0.95 * len(all_diffs)) - 1]
        assert p95 <= MAX_P95_FEATURE_DRIFT, (
            f"P95 per-feature drift {p95:.2e} > {MAX_P95_FEATURE_DRIFT}; "
            f"the API and the training CSV are computing different values "
            f"for the same feature names — likely a unit / aggregation bug "
            f"in assemble_model_features"
        )
