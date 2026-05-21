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
import re
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
    ROOT
    / "copol_prediction"
    / "filter"
    / "artificial_datapoints"
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

import app as api_app  # noqa: E402

from copolpredictor.inference import CopolymerPredictor  # noqa: E402

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


def test_health_returns_build_and_runtime_info(client):
    """/health must include build provenance and runtime info — used for
    debugging "which build am I hitting" in production. Outside Docker the
    GIT_SHA env var is unset and we expect "unknown"; CI is expected to
    cover the wired-up case end-to-end via docker-smoke-test.yml."""
    body = client.get("/health").json()
    assert body["model_loaded"] is True
    for field in ("git_sha", "git_branch", "build_time"):
        assert isinstance(body["build"][field], str) and body["build"][field]
    runtime = body["runtime"]
    assert runtime["python_version"].count(".") == 2  # e.g. "3.11.7"
    assert runtime["uptime_seconds"] >= 0.0
    assert runtime["hostname"]


def test_predict_response_uses_human_readable_class_names(client, test_df):
    """`/predict` (and by extension `/predict/batch` + `/optimize_reaction`)
    must key `class_probabilities` by the human-readable class name —
    the same string returned in `predicted_class_name` — not by `class_0`,
    `class_1`, `class_2`. Guards against accidentally regressing to the
    indexed-key form which carried no semantic meaning for callers."""
    row = test_df.iloc[0]
    payload = {
        "monomer1_smiles": row["monomer1_smiles"],
        "monomer2_smiles": row["monomer2_smiles"],
        "solvent_smiles": row["solvent_smiles"],
        "method": row["method"],
        "polytype": row["polymerization_type"],
        "temperature": float(row["temperature"]),
    }
    features = client.post("/preprocess_all", json=payload).json()["features"]
    pred = client.post("/predict", json={"features": features}).json()

    keys = set(pred["class_probabilities"])
    assert keys == {"alternating", "random to block like", "gradient"}, keys
    # The named-class key for the predicted class must reproduce confidence.
    assert pred["class_probabilities"][pred["predicted_class_name"]] == pytest.approx(
        pred["confidence"]
    )


def test_doi_from_source_filename():
    """Unit test for the DOI recovery rule (baseline_lookup)."""
    import baseline_lookup as bl

    assert bl.doi_from_source_filename("10.1002_pol.1959.1203512832.json") == (
        "10.1002/pol.1959.1203512832"
    )
    # .json suffix optional; only the FIRST underscore becomes a slash.
    assert bl.doi_from_source_filename("10.1016_0014-3057(84)90075-2") == (
        "10.1016/0014-3057(84)90075-2"
    )
    # Non-DOI filename (a real paper with no DOI) and missing input -> None.
    assert bl.doi_from_source_filename("Polymer_Science_USSR_2.6_(1961)__457.json") is None
    assert bl.doi_from_source_filename(None) is None
    assert bl.doi_from_source_filename("") is None

    assert bl.doi_url("10.1002/pol.1959.1203512832") == (
        "https://doi.org/10.1002/pol.1959.1203512832"
    )
    assert bl.doi_url(None) is None


def test_nearest_neighbors_carry_resolvable_doi(client, test_df):
    """`/preprocess_all` nearest-neighbour entries must carry `doi`/`doi_url`.
    For a neighbour drawn from a real paper, `doi` is a well-formed DOI and
    `doi_url` is the matching https://doi.org/ link; both may be null for
    synthetic/augmented neighbours."""
    row = test_df.iloc[0]
    body = client.post(
        "/preprocess_all",
        json={
            "monomer1_smiles": row["monomer1_smiles"],
            "monomer2_smiles": row["monomer2_smiles"],
            "solvent_smiles": row["solvent_smiles"],
            "method": row["method"],
            "polytype": row["polymerization_type"],
            "temperature": float(row["temperature"]),
        },
    ).json()
    neighbors = body.get("nearest_neighbors") or []
    assert neighbors, "expected nearest_neighbors in /preprocess_all response"
    resolved = 0
    for nn in neighbors:
        assert "doi" in nn and "doi_url" in nn
        if nn["doi"] is not None:
            assert re.fullmatch(r"10\.\d{4,}/\S+", nn["doi"]), nn["doi"]
            assert nn["doi_url"] == f"https://doi.org/{nn['doi']}"
            resolved += 1
        else:
            assert nn["doi_url"] is None
    assert resolved > 0, "expected at least one neighbour with a recoverable DOI"


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


def test_intermediate_results_are_cached(client, test_df):
    """The memoised helpers must actually serve repeat calls from cache, and
    do so without changing the result. Catches an lru_cache being dropped or
    the unique-solvent memo silently never populating."""
    import app as app_mod
    import reaction_optimization as ro

    # Solvent features: same SMILES twice -> identical result, cache hit.
    app_mod.calculate_solvent_features.cache_clear()
    first = app_mod.calculate_solvent_features("Cc1ccccc1")
    second = app_mod.calculate_solvent_features("Cc1ccccc1")
    assert first == second
    assert app_mod.calculate_solvent_features.cache_info().hits >= 1

    # Monomer JSON parse is memoised: a /preprocess_all call followed by an
    # /optimize_reaction call for the same monomers must reuse the parsed JSON.
    app_mod._load_monomer_json.cache_clear()
    row = test_df.iloc[0]
    payload = {
        "monomer1_smiles": row["monomer1_smiles"],
        "monomer2_smiles": row["monomer2_smiles"],
        "solvent_smiles": row["solvent_smiles"],
        "method": row["method"],
        "polytype": row["polymerization_type"],
        "temperature": float(row["temperature"]),
    }
    assert client.post("/preprocess_all", json=payload).status_code == 200
    assert client.post("/optimize_reaction", json=payload).status_code == 200
    assert app_mod._load_monomer_json.cache_info().hits >= 1

    # Unique-solvent table is memoised by dataset identity.
    ro._UNIQUE_SOLVENTS_MEMO.clear()
    once = ro._unique_solvents(app_mod.dataset_df)
    twice = ro._unique_solvents(app_mod.dataset_df)
    assert once is twice  # same object — served from the memo
    assert len(once) > 0


def test_resolve_temperatures():
    """The temperature_mode keys must map to the documented temperature lists."""
    import reaction_optimization as ro

    assert ro.resolve_temperatures("40-80", base_temperature=60.0) == [40.0, 60.0, 80.0]
    assert ro.resolve_temperatures("20-100", base_temperature=60.0) == [20.0, 60.0, 100.0]
    assert ro.resolve_temperatures("fixed60", base_temperature=999.0) == [60.0]
    # step20 still honours the caller's base + step.
    assert ro.resolve_temperatures("step20", base_temperature=70.0, temperature_step=10.0) == [
        60.0,
        70.0,
        80.0,
    ]
    with pytest.raises(ValueError):
        ro.resolve_temperatures("nonsense", base_temperature=60.0)


_RXN_OPT_PAYLOAD = {
    "monomer1_smiles": "C=Cc1ccccc1",  # styrene
    "monomer2_smiles": "C=C(C)C(=O)OC",  # methyl methacrylate
    "solvent_smiles": "Cc1ccccc1",  # toluene
    "method": "solvent",
    "polytype": "free radical",
    "temperature": 60.0,
}


def test_optimize_reaction_named_solvent_set(client):
    """A named solvent set + fixed temperature mode must produce one grid cell
    per (solvent × temperature), with class_probabilities keyed by class name."""
    r = client.post(
        "/optimize_reaction",
        json={**_RXN_OPT_PAYLOAD, "solvent_set": "aromatic", "temperature_mode": "40-80"},
    )
    assert r.status_code == 200, r.text
    preds = r.json()["predictions"]
    # 5 aromatic solvents × 3 temperatures (40/60/80).
    assert len(preds) == 15, len(preds)
    assert {p["temperature"] for p in preds} == {40.0, 60.0, 80.0}
    for p in preds:
        assert set(p["class_probabilities"]) == {
            "alternating",
            "random to block like",
            "gradient",
        }


def test_find_architecture_switch(client):
    """The counterfactual endpoint must return a baseline prediction and a
    list of condition sets that genuinely flip the predicted architecture,
    ranked by smallest |delta_logp|."""
    r = client.post(
        "/find_architecture_switch",
        json={**_RXN_OPT_PAYLOAD, "solvent_set": "common", "temperature_mode": "20-100"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"] is True
    assert body["baseline"] is not None
    assert body["n_evaluated"] > 0

    baseline_class = body["baseline"]["predicted_class"]
    deltas = []
    for cf in body["counterfactuals"]:
        # Every counterfactual must actually be a *different* architecture.
        assert cf["predicted_class"] != baseline_class
        # delta fields must be present and self-consistent.
        assert "delta_logp" in cf and "delta_temperature" in cf
        deltas.append(abs(cf["delta_logp"]))
        # Each counterfactual carries a nearest-neighbour literature
        # reference; when it resolves to a DOI, doi_url is the matching link.
        assert "reference" in cf and "reference_same_monomers" in cf
        ref = cf["reference"]
        if ref is not None:
            # styrene + methyl methacrylate is well represented in the
            # training data, so the reference must be a same-monomer
            # reaction (not just the closest by fingerprint).
            assert cf["reference_same_monomers"] is True
            if ref.get("doi"):
                assert re.fullmatch(r"10\.\d{4,}/\S+", ref["doi"]), ref["doi"]
                assert ref["doi_url"] == f"https://doi.org/{ref['doi']}"
    # Ranked by smallest |delta_logp|.
    assert deltas == sorted(deltas), deltas
