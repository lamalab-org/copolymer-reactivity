# `copol_prediction/` — Pipeline 2: reactions → model + API

This is **Pipeline 2** of the repo (see the [top-level README](../README.md) for the conceptual map). It takes the curated dataset produced by `data_extraction/`, computes per-monomer descriptors, trains the architecture-classification model, and serves predictions via a FastAPI service.

## Data flow

```
                processed_data.csv             ← dataset (4,969 measurement rows,
                  ▲                              3,791 unique reactions, 1,206 papers)
                  │
                  │ ↓ monomer_feature_calculation.py          (XTB descriptors)
                  │ ↓ create_data_split.py                    (stratified by monomer pair)
                  ▼
       artifacts/data_splits/{train,val,test}.csv
                  │
                  │ ↓ train_final_model.py                    (XGBoost + calibration)
                  ▼
       artifacts/model_bundle/                                ← the released model
                  │
                  │ ↓ api/app.py                              (FastAPI service)
                  ▼
       POST /predict, /preprocess_all, /optimize_reaction,  ← what powers
            /find_architecture_switch, /paper_metrics, ...     polycarp.cheminfo.org

       artifacts/model_bundle/  +  artifacts/data_splits/
                  │
                  │ ↓ reproduce_paper_metrics.py              ← regression check
                  ▼
       artifacts/paper_metrics.json + REPRODUCED on stdout
```

## Classes

The classifier predicts one of three architecture classes, derived from the reactivity-ratio product `r₁·r₂`:

| index | name | rule | meaning |
|---:|---|---|---|
| 0 | `alternating` | `r₁·r₂ < 1` | monomers prefer the comonomer over self-addition |
| 1 | `random` | `1 ≤ r₁·r₂ ≤ 25` | catch-all (covers azeotropic, mildly-alternating, and mildly-blocky systems) |
| 2 | `gradient` | `r₁·r₂ > 25` | monomers prefer self-addition; composition drift along the chain |

The canonical mapping lives in [`api/class_labels.py`](api/class_labels.py) and is imported everywhere the names appear — single source of truth.

## Voting model

Two architecture-prediction models cooperate:

- **XGBoost** on the XTB-derived feature vector — the classifier proper.
- **Lookup**: the architecture of the top-1 nearest-neighbour literature reaction (Tanimoto on Morgan fingerprints of the monomer pair + solvent).

The **voting** layer keeps only predictions where the two agree (the "coverage" metric in the paper). Disagreement is exposed on every `/preprocess_all` response so the web UI can flag low-confidence cases.

## Layout

```
copol_prediction/
├── processed_data.csv                   ← live dataset (the API reads this for the
│                                          nearest-neighbour lookup pool)
│
├── create_data_split.py                 ← monomer-pair-stratified 70/10/20 split
├── train_final_model.py                 ← XGBoost training + calibration
├── monomer_feature_calculation.py       ← XTB descriptor pipeline
├── mayo_lewis_classification.py         ← r1·r2 → class assignment
├── reproduce_paper_metrics.py           ← canonical regression check
├── preprocess_splits_full_features.py   ← splits with full feature set (perm. importance)
├── REPRODUCE.md                         ← reproduction recipe details
│
├── analysis/                            ← paper figures (plot_model_figure, plot_class_curves)
├── api/                                 ← FastAPI service deployed at polycarp.cheminfo.org
├── filter/                              ← curation + augmented-negatives pool
├── utils/                               ← load_data_split helpers
│
└── artifacts/
    ├── model_bundle/                    ← XGBoost model + calibration + metadata
    ├── data_splits/                     ← train/val/test (6,774 rows / 3,387 reactions)
    ├── data_splits_full_features/       ← splits with descriptors for permutation analysis
    └── paper_metrics.json               ← cached output of reproduce_paper_metrics.py
                                           served unmodified by GET /paper_metrics
```

## Common tasks

### Reproduce the paper's metrics

```bash
python reproduce_paper_metrics.py
```

Loads `artifacts/model_bundle/` + `artifacts/data_splits/`, evaluates plain XGBoost and the voting model on both splits, asserts every cell of the paper's table reproduces within ±0.005. Exits non-zero on drift. Also the basis of `tests/test_api_parity.py::test_paper_metrics_endpoint`.

### Re-split the dataset

```bash
python create_data_split.py
```

Reads `processed_data.csv`, applies the paper filter (`r₁·r₂ ≥ 0`, `r₁·r₂` not null, drop rows with NaN features), stratifies by **monomer pair** (`frozenset({canon(m1), canon(m2)})`) so all rows for a given pair land in the same split (prevents leakage), writes `artifacts/data_splits/{train,val,test}.csv` + `split_info.json`.

### Retrain the model

```bash
python train_final_model.py
```

Reads the splits, runs `RandomizedSearchCV` over XGBoost hyper-parameters (5-fold GroupKFold by `monomer_pair_key`), fits the final model on train+val, calibrates on a held-out subset, writes the new `artifacts/model_bundle/`. Run `python reproduce_paper_metrics.py` afterwards to confirm the new bundle matches (or to capture the new numbers if you intend to update the paper).

### Compute monomer descriptors

```bash
python monomer_feature_calculation.py
```

Runs XTB for each unique monomer SMILES that doesn't yet have a cached descriptor file under [`api/molecule_properties/`](api/molecule_properties/). First-time computation per monomer takes ~1–5 min; subsequent runs are cached.

### Serve the API locally

See [`api/README.md`](api/README.md). One-liner with Docker:

```bash
cd api && docker compose up
```

## Library entry point

```python
from copolpredictor.inference import CopolymerPredictor

predictor = CopolymerPredictor("artifacts/model_bundle")
result = predictor.predict_with_confidence(features)
```

See [`src/copolpredictor/`](../src/copolpredictor/) for module docs.

## What's where in the model bundle

```
artifacts/model_bundle/
├── meta.json                ← feature list, class labels, hyper-parameters, training metrics
├── all_metrics.txt          ← train/test per-class metrics (paper Table)
├── voting_test_metrics.json ← voting-layer metrics
├── model.joblib             ← XGBoost classifier
├── model.xgb.json           ← XGBoost native format (for cross-framework loading)
└── calibration.joblib       ← isotonic calibrator
```

`meta.json` is the source of truth for feature schema and class definitions used at inference time; the API loads it on startup and exposes it at `GET /model/info`.
