# Feature Comparison Experiments

This directory contains experiments comparing different molecular feature representations for copolymer microstructure prediction.

## 📁 Structure

```
feature_comparison/
├── baseline/              # Quantum chemical descriptors
│   ├── train.py           # Training script
│   ├── results_final/     # Final model results
│   └── cache/             # RDKit cache
├── fingerprint/           # Morgan fingerprints
│   ├── train.py           # Training script
│   ├── data_processing_morgan.py  # Feature generation
│   ├── results_final/     # Final model results
│   └── cache/             # RDKit cache
└── comparison/            # Comparative analysis
    ├── compare.py         # Comparison script
    └── plots/             # Generated plots
        ├── comparison_baseline_vs_fingerprint.png
        └── comparison_baseline_vs_fingerprint.pdf
```

## 🚀 Running Experiments

### 1. Train Baseline Model (Quantum Features)
```bash
cd baseline
python train.py
```

### 2. Train Fingerprint Model (Morgan)
```bash
cd fingerprint
python train.py
```

### 3. Compare Models
```bash
cd comparison
python compare.py
```

## 📊 Results

### Model Comparison
- **Baseline**: 15 quantum chemical descriptors
  - Macro F1: 0.817
  - Macro Precision: 0.867
  
- **Morgan Fingerprint**: 4105 features (2048-bit fingerprints + other)
  - Macro F1: 0.665
  - Macro Precision: 0.732

### Key Findings
- Quantum descriptors outperform Morgan fingerprints despite using 273x fewer features
- This suggests domain-specific features are more informative than generic molecular fingerprints
- The comparison plots show performance differences across all three classes

## 📝 Notes

- All experiments use the same train/test splits from `../data/`
- Models are trained with XGBoost using 5-fold cross-validation
- Hyperparameter tuning is performed with Optuna (50 trials)

