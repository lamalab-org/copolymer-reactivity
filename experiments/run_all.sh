#!/bin/bash
# Run all experiments

echo "=========================================="
echo "Running all experiments"
echo "=========================================="
echo ""

# Check if central split exists
if [ ! -f "../copol_prediction/artifacts/data_splits/train.csv" ]; then
    echo "ERROR: Central split not found!"
    echo "Please run first:"
    echo "  cd ../copol_prediction"
    echo "  python create_data_split.py"
    exit 1
fi

# Check if Morgan fingerprint data exists (only derived data needed)
if [ ! -f "feature_comparison/data/train_morgan.csv" ]; then
    echo ">> Creating Morgan fingerprint data (first time setup)"
    python archive/create_train_test_split.py --fingerprints
    echo ""
fi

# Feature Comparison Experiments
echo "=========================================="
echo "FEATURE COMPARISON EXPERIMENTS"
echo "=========================================="
echo ""

# Baseline (Quantum Features)
echo ">> Training Baseline Model (Quantum Features)"
cd feature_comparison/baseline && python train.py && cd ../..
echo ""

# Morgan Fingerprint
echo ">> Training Fingerprint Model (Morgan)"
cd feature_comparison/fingerprint && python train.py && cd ../..
echo ""

# Compare Feature Models
echo ">> Comparing Feature Models"
cd feature_comparison/comparison && python compare.py && cd ../..
echo ""

# Filter Comparison Experiments
echo "=========================================="
echo "FILTER COMPARISON EXPERIMENTS"
echo "=========================================="
echo ""

echo ">> Running Filter Sweep"
cd filter_comparison && python sweep_filters.py && cd ..
echo ""

# Summary
echo "=========================================="
echo "COMPLETE!"
echo "=========================================="
echo ""
echo "Results:"
echo "  - Feature comparison plots: feature_comparison/comparison/plots/"
echo "  - Filter comparison results: filter_comparison/results/"
echo ""

