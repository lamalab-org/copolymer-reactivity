# copolymer-reactivity

A condition-aware machine-learning model and FastAPI service that predicts the **microstructure of radical copolymers** — *alternating*, *random*, or *gradient* — from two monomer structures and the reaction conditions (solvent, temperature, polymerisation mechanism), backed by a curated literature dataset of **3,791 copolymerisations from 1,206 publications**.

The model is served live at <https://polycarp.cheminfo.org> (web UI: [cheminfo-py/polycarp.cheminfo.org](https://github.com/cheminfo-py/polycarp.cheminfo.org)).

## The pipelines

This repo contains **two main pipelines** that produce the dataset, the model, and the deployed service. The diagram below is the conceptual map you'll want to keep in mind when navigating the tree.

```
                                    ┌─────────────────────────────────────────┐
                                    │ Pipeline 1 — literature → reactions     │   data_extraction/
                                    │                                         │
                  Crossref          │   1. score & filter (~300k → 1,851)     │
                  abstract          │   2. embedding-based pre-filter         │
                   →                │   3. download PDFs                      │
                                    │   4. vision-language model extraction   │
                                    │   5. curate + assign reaction_id        │
                                    │                                         │
                                    └─────────────────┬───────────────────────┘
                                                      ▼
                                          processed_data.csv
                                          ── 3,791 unique reactions
                                          ── 1,206 source publications
                                          ── 4,969 measurement rows
                                                      │
                                    ┌─────────────────┴───────────────────────┐
                                    │ Pipeline 2 — reactions → model + API    │   copol_prediction/
                                    │                                         │
                                    │   1. XTB descriptors per monomer        │
                                    │   2. monomer-pair-stratified splits     │
                                    │   3. XGBoost training + calibration     │
                                    │   4. voting layer (model + lookup)      │
                                    │   5. FastAPI service                    │
                                    │                                         │
                                    └────────┬────────────────┬───────────────┘
                                             ▼                ▼
                                    artifacts/model_bundle    api/  ──→  https://polycarp.cheminfo.org
```

The **dataset** (`processed_data.csv`) is the hand-off point between the two pipelines. Everything upstream produces it; everything downstream consumes it.

## Where things live

| Path | What it is | README |
|---|---|---|
| [`data_extraction/`](data_extraction/) | Pipeline 1 — Crossref → VLM extraction → curated reactions | [README](data_extraction/README.md) |
| [`copol_prediction/`](copol_prediction/) | Pipeline 2 — descriptors, splits, training, analysis, deployed API | [README](copol_prediction/README.md) |
| [`copol_prediction/api/`](copol_prediction/api/) | The FastAPI service that powers the web app | [README](copol_prediction/api/README.md) |
| [`copol_prediction/artifacts/`](copol_prediction/artifacts/) | The trained `model_bundle/`, `data_splits/`, and `paper_metrics.json` | — |
| [`database/`](database/) | Dataset-summary figures used in the paper | [README](database/README.md) |
| [`experiments/`](experiments/) | Per-study code: permutation importance, baselines, case studies | [README](experiments/README.md) |
| [`src/copolpredictor/`](src/copolpredictor/) | Inference + data-loading library imported by the API and scripts | — |
| [`src/copolextractor/`](src/copolextractor/) | Library for the literature-extraction pipeline | — |

## Quick start

### Use the deployed API (no install)

```bash
curl https://polycarp.cheminfo.org/api/health
```

### Run the API locally (Docker)

```bash
git clone https://github.com/lamalab-org/copolymer-reactivity
cd copolymer-reactivity/copol_prediction/api
docker compose up           # pulls ghcr.io/lamalab-org/copolymer-reactivity:latest
```

Open <http://localhost:8000/docs> for the interactive OpenAPI documentation.

A SMILES → prediction round-trip in one shell snippet:

```bash
curl -sS -X POST http://localhost:8000/preprocess_all \
  -H 'Content-Type: application/json' \
  -d '{"monomer1_smiles":"C=Cc1ccccc1","monomer2_smiles":"C=C(C)C(=O)OC",
       "solvent_smiles":"ClC(Cl)Cl","temperature":60,
       "method":"solvent","polytype":"free radical"}' \
  | jq '.lookup_class_name'      # "random"
```

### Install as a library

```bash
pip install -e .                # core deps
pip install -e ".[extraction]"  # also install the extraction pipeline
pip install -e ".[training]"    # also install training-only deps
pip install -e ".[testing]"     # also install pytest + plugins
```

## Reproducing the paper's numbers

```bash
cd copol_prediction
python reproduce_paper_metrics.py
```

Reads the committed `artifacts/model_bundle/` and the `artifacts/data_splits/`, evaluates plain-XGBoost and the voting model on both splits, and asserts every cell of the paper's `tab:train_test_voting_performance` reproduces within ±0.005. Exits non-zero on any drift — also used as a regression test in CI. Full instructions in [`copol_prediction/REPRODUCE.md`](copol_prediction/REPRODUCE.md).

## Citation

See [`CITATION.cff`](CITATION.cff) for the canonical machine-readable citation (BibTeX is rendered automatically by GitHub's *"Cite this repository"* button on the right side of the repo page). Each release is also archived on Zenodo with a versioned DOI.

## License

MIT — see [LICENSE](LICENSE).
