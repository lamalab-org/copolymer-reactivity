"""
Copolymerization prediction package.

Submodules and re-exported helpers are loaded lazily (PEP 562) so importing
just `copolpredictor.inference` for inference-only deployments doesn't
drag in heavy training/visualization stacks (matplotlib, openai, ...).
"""

import importlib
from typing import Any

__version__ = "0.1.0"

_SUBMODULES = (
    "data_processing",
    "data_augmentation",
    "prediction_utils",
    "model_training",
    "evaluation",
    "calibration",
    "holdout_utils",
    "inference",
)

# name -> (submodule, attribute) for top-level re-exports.
_LAZY_ATTRS = {
    "train_xgboost_with_cv": ("model_training", "train_xgboost_with_cv"),
    "train_final_model": ("model_training", "train_final_model"),
    "save_model_bundle": ("model_training", "save_model_bundle"),
    "load_model_bundle": ("model_training", "load_model_bundle"),
    "evaluate_model": ("evaluation", "evaluate_model"),
    "print_evaluation_results": ("evaluation", "print_evaluation_results"),
    "save_holdout_metrics_json": ("evaluation", "save_holdout_metrics_json"),
    "calibrate_model_with_weights": ("calibration", "calibrate_model_with_weights"),
}


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    if name in _LAZY_ATTRS:
        submod_name, attr = _LAZY_ATTRS[name]
        submod = importlib.import_module(f"{__name__}.{submod_name}")
        return getattr(submod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_SUBMODULES) + list(_LAZY_ATTRS)
