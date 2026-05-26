# `paper_dataset/` — frozen snapshot underlying the paper's numbers

This directory holds the **immutable dataset snapshot** that the trained model in `artifacts/model_bundle/` was fit on and that the paper's headline numbers (*"3,791 copolymerisations from 1,206 publications"*) are computed from. **Do not regenerate it casually** — changing it would invalidate every cited number until the model is re-trained and the paper is re-released.

## Contents

| File | What it is | Rows |
|---|---|---:|
| `processed_data.csv` | The cleaned reaction dataset as of the **November 2025 snapshot**. Input to `copol_prediction/create_data_split.py`, which produces `artifacts/data_splits/`. | 7,582 measurement rows = 3,791 unique reactions |
| `grouped_by_unique_monomer_pairs.csv` | One row per unique reaction (`drop_duplicates('reaction_id')`), with an order-insensitive `monomer_pair_key` and a contiguous `group_id`. The "3,791" the paper cites. | 3,791 |

## Why this is separate from `copol_prediction/processed_data.csv`

There are **two `processed_data.csv` files** in this repo and they answer different questions:

| Path | Snapshot | Rows | Reactions | Used for |
|---|---|---:|---:|---|
| `copol_prediction/processed_data.csv` | **April 2026**, post-curation cleanup | 4,969 | 3,791 | The **live API's nearest-neighbour lookup pool** — gives users multiple measurement rows per reaction (different solvents, temperatures, etc.) |
| `copol_prediction/paper_dataset/processed_data.csv` | **November 2025**, frozen at paper submission | 7,582 | 3,791 | The **input to `create_data_split.py`** — keeps the train/val/test splits bit-stable against the model bundle and the paper's reported metrics |

Both contain the **same 3,791 unique reactions** (verifiable via `reaction_id`). The April version dropped 2,613 measurement rows judged redundant after curation, but no reactions were removed. The November file is preserved so re-running the splits reproduces the splits the model in `artifacts/model_bundle/` was trained on.

## Reproducibility

`grouped_by_unique_monomer_pairs.csv` is fully regenerable from any `processed_data.csv` snapshot:

```bash
python copol_prediction/build_grouped_dataset.py \
    --source copol_prediction/paper_dataset/processed_data.csv \
    --output copol_prediction/paper_dataset/grouped_by_unique_monomer_pairs.csv
```

Running against the live `copol_prediction/processed_data.csv` produces a slightly different but equivalent file: same 3,791 `reaction_id` values, only the captured-per-row metadata differs (because the live snapshot has been further curated). The script is the canonical recipe; the file is committed only because it underpins the paper's citable numbers and the Zenodo archive.

## Lineage diagram

```
[April 2026 curation pass]                [November 2025 snapshot]
copol_prediction/processed_data.csv       paper_dataset/processed_data.csv
4,969 rows / 3,791 reactions               7,582 rows / 3,791 reactions
       │                                              │
       │ ↓ nearest-neighbour lookup pool             │ ↓ create_data_split.py
       │   (api/baseline_lookup.py)                   │   (filter r1·r2 ≥ 0 + drop NaN
       ▼                                              │    features; stratify by monomer pair)
   live API                                           ▼
   /preprocess_all → nearest_neighbors          artifacts/data_splits/{train,val,test}.csv
                                                6,774 rows / 3,387 reactions
                                                       │
                                                       │ ↓ train_final_model.py
                                                       ▼
                                                artifacts/model_bundle/
                                                       │
                                                       │ ↓ reproduce_paper_metrics.py
                                                       ▼
                                                artifacts/paper_metrics.json
                                                (tab:train_test_voting_performance)
```

## When to regenerate

Only when cutting a **new paper revision**. The flow is:

1. Replace `paper_dataset/processed_data.csv` with the current curated dataset.
2. Re-run `python build_grouped_dataset.py` to refresh the grouped CSV (its counts will change → update the paper).
3. Re-run `python create_data_split.py` to recut the splits.
4. Re-run `python train_final_model.py` to retrain.
5. Re-run `python reproduce_paper_metrics.py` and update the paper's `tab:train_test_voting_performance` with the new numbers.
6. Cut a new GitHub release / Zenodo DOI.
