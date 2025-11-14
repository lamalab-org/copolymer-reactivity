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

# Check if experiment data exists
if [ ! -f "data/train.csv" ] || [ ! -f "data/train_morgan.csv" ]; then
    echo ">> Creating experiment data (first time setup)"
    python create_train_test_split.py --fingerprints
    echo ""
fi

# Baseline
echo ">> Baseline experiment"
cd baseline && python train.py && cd ..
echo ""

# Fingerprint
echo ">> Fingerprint experiment"
cd fingerprint && python train.py && cd ..
echo ""

# Compare
echo ">> Comparing results"
python compare_results.py
echo ""

echo "=========================================="
echo "Complete!"
echo "=========================================="

