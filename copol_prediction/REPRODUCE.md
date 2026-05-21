# Reproducing the paper's model-performance results

This reproduces **Table `tab:train_test_voting_performance`** from the paper —
the per-class accuracy, precision, F1 and coverage of the PolyCarp voting model
on the train and test splits.

Nothing is retrained. The script evaluates the **released model bundle**
(`artifacts/model_bundle/`) against the **released data splits**
(`artifacts/data_splits/`), both of which are committed to this repo.

## Prerequisites

```bash
# from the repo root
pip install -e .[testing]      # core deps + rdkit + xgboost + pandas
```

The model bundle and data splits are already in the repo — no download needed.

## 1. Quick check: does the model run on the data?

```python
# from the repo root
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
from copolpredictor.inference import CopolymerPredictor
from sklearn.metrics import accuracy_score

P = CopolymerPredictor("copol_prediction/artifacts/model_bundle")
for split in ["train", "test"]:
    df = pd.read_csv(f"copol_prediction/artifacts/data_splits/{split}.csv")
    df = df.dropna(subset=P.features + ["r_product_class"])
    acc = accuracy_score(df["r_product_class"].astype(int), P.predict(df[P.features]))
    print(f"{split}: plain-XGBoost accuracy = {acc:.4f}")
```

Expected — matching `artifacts/model_bundle/all_metrics.txt`:

```
train: plain-XGBoost accuracy = 0.9217
test:  plain-XGBoost accuracy = 0.7401
```

## 2. Reproduce the paper table

```bash
cd copol_prediction
python reproduce_paper_metrics.py
```

This computes the **voting** model — XGBoost combined with a nearest-neighbour
RDKit-fingerprint lookup, keeping only predictions where the two agree
(`coverage` = retained fraction) — and prints per-class accuracy/precision/F1
for the train and test splits next to the paper's published values.

Expected output (every cell within ±0.005 of the paper):

```
Test — voting model (retained 1045/1358, coverage 0.770)
Class             Acc    Prec      F1        paper Acc/Prec/F1
Alternating     0.788   0.667   0.722    0.788/0.667/0.722  ok
Random          0.781   0.812   0.796    0.781/0.812/0.796  ok
Gradient        0.845   0.827   0.836    0.845/0.827/0.836  ok
Macro           0.805   0.768   0.785    0.805/0.768/0.785  ok
coverage        0.770                    0.770        ok
...
REPRODUCED: all values within ±0.005 of the paper table.
```

The script exits non-zero if any value deviates by more than 0.005, so it
doubles as a regression test for the released artifacts.

## 3. Reproduce against the deployed API

The same table can be reproduced with the XGBoost predictions served by a
running API instead of the local bundle — a check that the deployment serves
the identical model:

```bash
# with the API running (locally or via the GHCR image), from copol_prediction/
python reproduce_paper_metrics.py --api http://localhost:8000
```

`--api` routes the XGBoost half through `POST /predict/batch`. The
nearest-neighbour lookup is a deterministic fingerprint baseline (not the
trained model), so it is always computed locally against the train-only pool
the paper uses — keeping the comparison apples-to-apples regardless of how the
deployed API configures its own lookup pool.

## What the numbers mean

| Quantity | Definition |
|---|---|
| `Acc` (per class) | recall — fraction of that class's retained samples predicted correctly |
| `Prec` (per class) | precision — fraction of retained predictions for that class that are correct |
| `coverage` | retained predictions ÷ all samples in the split (XGBoost and lookup agree) |
| `Macro` | unweighted mean across the three classes |

The published metrics also live with the artifact itself:
`artifacts/model_bundle/voting_test_metrics.json` (test voting metrics) and
`artifacts/model_bundle/all_metrics.txt` (plain-XGBoost train/test breakdown).
