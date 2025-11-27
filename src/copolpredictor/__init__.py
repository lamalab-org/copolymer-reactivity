"""
Copolymerization prediction package.

This package provides tools for predicting copolymerization reactivity
using machine learning models trained on molecular descriptors.
"""

__version__ = "0.1.0"

# Import main modules
from . import data_processing
from . import data_augmentation
from . import prediction_utils
from . import model_training
from . import evaluation
from . import calibration
from . import holdout_utils

# Import commonly used functions
from .model_training import (
    train_xgboost_with_cv,
    train_final_model,
    save_model_bundle,
    load_model_bundle
)

from .evaluation import (
    evaluate_model,
    print_evaluation_results,
    save_holdout_metrics_json
)

from .calibration import (
    calibrate_model_with_weights,
    calculate_prediction_confidence
)

__all__ = [
    'data_processing',
    'data_augmentation',
    'prediction_utils',
    'model_training',
    'evaluation',
    'calibration',
    'holdout_utils',
    'train_xgboost_with_cv',
    'train_final_model',
    'save_model_bundle',
    'load_model_bundle',
    'evaluate_model',
    'print_evaluation_results',
    'save_holdout_metrics_json',
    'calibrate_model_with_weights',
    'calculate_prediction_confidence',
]


