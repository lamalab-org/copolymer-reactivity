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

echo ">> Comparing quantum-chemical descriptors vs Morgan fingerprints"
cd feature_comparison && python run_comparison.py && cd ..
echo ""

# Reaction Conditions Comparison Experiments
echo "=========================================="
echo "REACTION CONDITIONS COMPARISON EXPERIMENTS"
echo "=========================================="
echo ""

echo ">> Comparing model with vs without reaction condition features"
cd reaction_conditions_comparison && python run_comparison.py && cd ..
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
echo "  - Feature comparison results: feature_comparison/results/"
echo "  - Reaction conditions comparison results: reaction_conditions_comparison/results/"
echo "  - Filter comparison results: filter_comparison/output/"
echo ""

