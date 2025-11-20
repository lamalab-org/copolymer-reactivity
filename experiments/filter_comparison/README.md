# Filter Comparison Experiments

This directory contains experiments comparing different data filtering strategies for copolymer microstructure prediction.

## 📁 Structure

```
filter_comparison/
├── sweep_filters.py       # Systematic filter comparison
└── results/               # (to be created)
    ├── filter_results.json
    └── plots/
```

## 🚀 Running Experiments

### Filter Sweep
```bash
python sweep_filters.py
```

This script systematically evaluates model performance with different filtering strategies:
- No filtering (baseline)
- Polymer type filtering
- Method filtering  
- Combined filters

## 📊 Metrics

The sweep evaluates:
- Accuracy
- F1 Score (weighted and macro)
- Precision and Recall
- Dataset size after filtering
- Training time

## 📝 Notes

- All experiments use the same base dataset from `../data/`
- Models are trained with consistent hyperparameters for fair comparison
- Results show trade-offs between data quality and quantity

