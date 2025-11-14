#!/bin/bash
#
# Quick start script for Copolymerization Prediction API
#

set -e

echo "=========================================="
echo "Copolymerization Prediction API Starter"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "Error: app.py not found. Please run this script from the copol_prediction/api directory."
    exit 1
fi

# Check if model exists
if [ ! -d "../artifacts/model_bundle" ]; then
    echo "Error: Model bundle not found at ../artifacts/model_bundle"
    echo "Please train the model first:"
    echo "  cd .. && python train_final_model.py"
    exit 1
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Warning: No virtual environment detected"
    echo "It's recommended to use a virtual environment"
    echo ""
fi

# Install dependencies if needed
echo "Checking dependencies..."
if ! python -c "import fastapi" 2>/dev/null; then
    echo "Installing API dependencies..."
    pip install -r requirements.txt
else
    echo "✓ Dependencies already installed"
fi

echo ""
echo "Starting API server..."
echo "----------------------------------------"
echo "API will be available at:"
echo "  - Main: http://localhost:8000"
echo "  - Docs: http://localhost:8000/docs"
echo "  - ReDoc: http://localhost:8000/redoc"
echo "----------------------------------------"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the API
python app.py

