#!/bin/bash
# Helper script to run analysis with correct PYTHONPATH

# Set PYTHONPATH to include src directory
export PYTHONPATH="/Users/maraw/PycharmProjects/test/src:$PYTHONPATH"

# Run analysis script with all arguments passed through
python analysis/analyze_model.py "$@"

