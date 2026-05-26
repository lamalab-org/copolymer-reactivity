# `paper_dataset/` — the dataset behind the released model and the paper's numbers

This directory pins the **dataset that the released model in `artifacts/model_bundle/` was trained on** and that the paper's headline counts (*"3,791 copolymerisations from 1,206 publications"*) are computed from. Treat it as read-only; updating it is a deliberate paper-release step (see [When to update](#when-to-update)).

## Contents

| File | What it is | Rows |
|---|---|---:|
| `processed_data.csv` | The cleaned reaction dataset that feeds `copol_prediction/create_data_split.py` → `artifacts/data_splits/` → `artifacts/model_bundle/`. | 7,582 measurement rows = 3,791 unique reactions |
| `grouped_by_unique_monomer_pairs.csv` | One row per unique reaction (`drop_duplicates('reaction_id')`), plus an order-insensitive `monomer_pair_key` and a contiguous integer `group_id`. The 3,791-from-1,206 view. | 3,791 |

`grouped_by_unique_monomer_pairs.csv` is fully regenerable from `processed_data.csv` via `copol_prediction/build_grouped_dataset.py`; it's committed as a convenience for citation and as the artefact the Zenodo deposit archives.

## How this differs from `copol_prediction/processed_data.csv`

The repo has **two `processed_data.csv` files**; they serve different consumers:

| Path | Consumer | Rows | Reactions |
|---|---|---:|---:|
| `copol_prediction/processed_data.csv` | The live API's nearest-neighbour lookup pool (`api/baseline_lookup.py`) | 4,969 | 3,791 |
| `copol_prediction/paper_dataset/processed_data.csv` | Input to `create_data_split.py` (this file) | 7,582 | 3,791 |

Both files contain the same 3,791 unique reactions; they differ only in how many measurement rows are kept per reaction. The live file is denormalised differently for the lookup UX; this file is pinned so the model bundle stays bit-stable against the paper's reported metrics.

## Lineage

```
paper_dataset/processed_data.csv       copol_prediction/processed_data.csv
7,582 rows / 3,791 reactions           4,969 rows / 3,791 reactions
       │                                       │
       │ ↓ create_data_split.py                │ ↓ api/baseline_lookup.py
       │   (filter r1·r2 ≥ 0 + drop NaN        │   (nearest-neighbour lookup pool)
       │    features; stratify by monomer       ▼
       │    pair)                          live API: /preprocess_all → nearest_neighbors
       ▼
artifacts/data_splits/{train,val,test}.csv
6,774 rows / 3,387 reactions
       │
       │ ↓ train_final_model.py
       ▼
artifacts/model_bundle/
       │
       │ ↓ reproduce_paper_metrics.py
       ▼
artifacts/paper_metrics.json
(served by GET /paper_metrics; basis of tab:train_test_voting_performance)
```

## When to update

Updating this directory invalidates the model bundle and every paper-cited number. Only do it when **cutting a new paper revision**:

1. Replace `paper_dataset/processed_data.csv` with the current curated dataset.
2. `python copol_prediction/build_grouped_dataset.py --source paper_dataset/processed_data.csv --output paper_dataset/grouped_by_unique_monomer_pairs.csv` — refresh the grouped view; its row count is the new dataset-size number for the paper.
3. `python copol_prediction/create_data_split.py` — recut `artifacts/data_splits/`.
4. `python copol_prediction/train_final_model.py` — retrain into `artifacts/model_bundle/`.
5. `python copol_prediction/reproduce_paper_metrics.py` — capture the new metrics into `artifacts/paper_metrics.json` and update the paper's `tab:train_test_voting_performance`.
6. Cut a new GitHub release; Zenodo will archive it under a new DOI.
