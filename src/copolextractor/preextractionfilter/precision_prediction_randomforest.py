import json
import os
import time
import warnings
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

import copolextractor.utils as utils

# Suppress XGBoost warnings
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")


def preprocess_data(
    data: List[dict], features: Sequence[str], target: str, threshold: float
) -> pd.DataFrame:
    """
    Prepare data for training by converting it into a DataFrame.

    Args:
        data: List of scored entry dicts.
        features: Feature column names (currently unused, kept for call-site
            documentation/compatibility).
        target: Key holding the numeric precision value used for thresholding.
        threshold: Minimum `target` value classified as "precise" (class 1).

    Returns:
        A DataFrame of entries with a non-null `target` value, with an added
        "precision_class" column (1 if `target` > `threshold`, else 0).
    """
    filtered_data = [entry for entry in data if target in entry and entry[target] is not None]
    df = pd.DataFrame(filtered_data)
    df["precision_class"] = (df[target] > threshold).astype(int)
    return df


def build_pipeline(
    numeric_features: Sequence[str], categorical_features: Sequence[str], seed_rf: int
) -> Pipeline:
    """
    Build a preprocessing and modeling pipeline with XGBoost.

    Args:
        numeric_features: Numeric column names, passed through unchanged.
        categorical_features: Categorical column names, one-hot encoded.
        seed_rf: Random seed for the XGBoost classifier.

    Returns:
        An unfitted scikit-learn `Pipeline` with a "preprocessor" and "model" step.
    """
    numeric_transformer = "passthrough"  # No transformation needed for numerical features
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Remove deprecated parameter use_label_encoder
    model = XGBClassifier(random_state=seed_rf, eval_metric="logloss", missing=np.nan)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    return pipeline


def train_model(
    training_data: pd.DataFrame, features: Sequence[str], target: str, seed_rf: int
) -> Pipeline:
    """
    Train the XGBoost model.

    Runs a grid search over XGBoost hyperparameters with 5-fold stratified CV and
    saves the resulting statistics to "./model_training_stats.json".

    Args:
        training_data: DataFrame containing `features` and `target` columns.
        features: Feature column names to train on (mixture of numeric/categorical).
        target: Name of the binary target column to predict.
        seed_rf: Random seed for the XGBoost classifier.

    Returns:
        The best-performing fitted pipeline found by grid search.
    """
    numeric_features = [
        "table_quality",
        "quality_of_number",
        "year",
        "pdf_quality",
        "rxn_number",
    ]
    categorical_features = ["language"]

    pipeline = build_pipeline(numeric_features, categorical_features, seed_rf)

    param_grid = {
        "model__n_estimators": [100, 500, 1000],
        "model__max_depth": [3, 5, 7],
        "model__learning_rate": [0.01, 0.1, 0.2],
        "model__colsample_bytree": [0.6, 0.8, 1.0],
        "model__subsample": [0.6, 0.8, 1.0],
    }

    X = training_data[features]
    y = training_data[target]

    # Ensure all categorical features are treated as strings
    for cat_feature in categorical_features:
        if cat_feature in X.columns:
            X[cat_feature] = X[cat_feature].astype(str)

    cv = StratifiedKFold(n_splits=5)
    scorer = make_scorer(accuracy_score)

    # Add log file for token tracking
    stats_file_path = "./model_training_stats.json"
    stats = {
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": "XGBoost",
        "features": features,
        "hyperparameters": {},  # Will be filled with best parameters later
        "grid_search_params": param_grid,
        "cv_folds": 5,
        "metrics": {},  # Will be filled with results later
    }

    print("Starting GridSearchCV...")
    start_time = time.time()
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring=scorer, n_jobs=-1, verbose=1)
    grid_search.fit(X, y)
    end_time = time.time()
    training_time = end_time - start_time

    print("GridSearchCV completed.")
    print("Best parameters found:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)

    # Update statistics
    stats["hyperparameters"] = grid_search.best_params_
    stats["metrics"] = {"best_cv_score": grid_search.best_score_, "training_time": training_time}

    # Save statistics
    save_training_stats(stats_file_path, stats)

    return grid_search.best_estimator_


def save_training_stats(stats_file_path: Union[str, os.PathLike], stats: Dict[str, Any]) -> None:
    """
    Save or update training statistics in a JSON file.

    Args:
        stats_file_path: Path to the training-statistics JSON file; appended to if
            it already exists and contains a "runs" list.
        stats: Statistics dict for the current run to append/persist.
    """
    if os.path.exists(stats_file_path):
        try:
            with open(stats_file_path, "r", encoding="utf-8") as file:
                existing_stats = json.load(file)
                if "runs" in existing_stats:
                    existing_stats["runs"].append(stats)
                else:
                    existing_stats = {"runs": [stats]}
        except (json.JSONDecodeError, FileNotFoundError):
            existing_stats = {"runs": [stats]}
    else:
        existing_stats = {"runs": [stats]}

    with open(stats_file_path, "w", encoding="utf-8") as file:
        json.dump(existing_stats, file, indent=4)

    print(f"Training statistics saved to {stats_file_path}")


def prepare_entries_for_scoring(
    data: List[dict], features: Sequence[str]
) -> Tuple[List[dict], Dict[str, Any]]:
    """
    Prepare entries for scoring by standardizing field names and logging statistics.

    Args:
        data: List of entry dicts to prepare in place (adds/normalizes
            "rxn_number" from "number_of_reactions"/"rxn_count" where available).
        features: Feature names required to be present (and non-null) for an entry
            to be considered complete.

    Returns:
        A (data, stats) tuple. `data` is the same list, updated in place. `stats`
        summarizes completeness and the count of missing values per feature.
    """
    # Statistics counters
    stats = {
        "total_entries": len(data),
        "complete_entries": 0,
        "entries_with_missing_features": 0,
        "missing_feature_counts": Counter(),
        "converted_rxn_count": 0,
    }

    # Process entries
    for entry in data:
        # Convert number_of_reactions to rxn_number
        if "number_of_reactions" in entry and entry["number_of_reactions"] is not None:
            entry["rxn_number"] = entry["number_of_reactions"]
            stats["converted_rxn_count"] += 1
        elif "rxn_count" in entry and entry["rxn_count"] is not None:
            entry["rxn_number"] = entry["rxn_count"]
            stats["converted_rxn_count"] += 1

        # Check for missing features
        missing_features = [
            feature
            for feature in features
            if feature not in entry or entry[feature] is None or pd.isna(entry[feature])
        ]

        if missing_features:
            stats["entries_with_missing_features"] += 1
            for feature in missing_features:
                stats["missing_feature_counts"][feature] += 1
        else:
            stats["complete_entries"] += 1

    # Log statistics
    print(f"Total entries: {stats['total_entries']}")
    print(
        f"Complete entries (all features present): {stats['complete_entries']} ({stats['complete_entries'] / stats['total_entries'] * 100:.2f}%)"
    )
    print(
        f"Entries with missing features: {stats['entries_with_missing_features']} ({stats['entries_with_missing_features'] / stats['total_entries'] * 100:.2f}%)"
    )
    print(
        f"Converted 'number_of_reactions' or 'rxn_count' to 'rxn_number': {stats['converted_rxn_count']}"
    )

    print("\nMissing feature statistics:")
    for feature, count in stats["missing_feature_counts"].most_common():
        print(f"  - {feature}: {count} entries ({count / stats['total_entries'] * 100:.2f}%)")

    return data, stats


def update_scores(
    data: List[dict], model: Pipeline, features: Sequence[str]
) -> Tuple[List[dict], Dict[str, Any]]:
    """
    Update entries with predictions.

    Args:
        data: List of entry dicts to annotate in place with a "precision_score" key.
        model: Fitted pipeline (as returned by `train_model`) used for prediction.
        features: Feature names the model expects; entries missing any of them are
            skipped (their "precision_score" is set to None).

    Returns:
        A (data, prediction_stats) tuple. `data` is the same list, updated in
        place. `prediction_stats` summarizes prediction successes/failures.
    """
    prediction_stats = {
        "total_entries": len(data),
        "successful_predictions": 0,
        "failed_predictions": 0,
        "prediction_errors": Counter(),
    }

    for entry in data:
        # Check if all required features are present and valid
        missing_features = [
            feature for feature in features if feature not in entry or pd.isna(entry[feature])
        ]
        if missing_features:
            print(
                f"Skipping {entry.get('filename', 'unknown')} due to missing features: {missing_features}"
            )
            entry["precision_score"] = None
            prediction_stats["failed_predictions"] += 1
            prediction_stats["prediction_errors"]["missing_features"] += 1
            continue

        # Create a DataFrame for prediction
        feature_values = pd.DataFrame(
            [{feature: entry.get(feature, pd.NA) for feature in features}]
        )

        # Ensure categorical features are treated as strings
        if "language" in feature_values.columns:
            feature_values["language"] = feature_values["language"].astype(str)

        try:
            # Perform prediction
            prediction = model.predict(feature_values)[0]
            entry["precision_score"] = int(prediction)
            prediction_stats["successful_predictions"] += 1
        except Exception as e:
            # Handle any errors during prediction
            error_type = type(e).__name__
            print(
                f"Error during prediction for {entry.get('filename', 'unknown')}: {error_type}: {e}"
            )
            entry["precision_score"] = None
            prediction_stats["failed_predictions"] += 1
            prediction_stats["prediction_errors"][error_type] += 1

    # Log prediction statistics
    print("\nPrediction statistics:")
    print(f"Total entries processed: {prediction_stats['total_entries']}")
    print(
        f"Successful predictions: {prediction_stats['successful_predictions']} ({prediction_stats['successful_predictions'] / prediction_stats['total_entries'] * 100:.2f}%)"
    )
    print(
        f"Failed predictions: {prediction_stats['failed_predictions']} ({prediction_stats['failed_predictions'] / prediction_stats['total_entries'] * 100:.2f}%)"
    )

    if prediction_stats["prediction_errors"]:
        print("\nPrediction error types:")
        for error_type, count in prediction_stats["prediction_errors"].most_common():
            print(f"  - {error_type}: {count} occurrences")

    return data, prediction_stats


def check_json_files(
    scoring_file: Union[str, os.PathLike],
    output_file: Union[str, os.PathLike],
    pdf_folder: Optional[Union[str, os.PathLike]] = None,
    scoring_data: Optional[List[dict]] = None,
) -> Dict[str, Any]:
    """
    Check if JSON files exist and are valid.
    If pdf_folder and scoring_data are provided, also check for missing JSON files.

    Args:
        scoring_file: Path to the scoring-data JSON file to validate.
        output_file: Path to the output JSON file to validate (may not exist yet).
        pdf_folder: Optional directory of source PDFs; if given together with
            `scoring_data`, also checks that a JSON result exists for each entry.
        scoring_data: Optional list of scored entry dicts (with "filename" keys)
            used for the missing-JSON check.

    Returns:
        A dict summarizing file existence/validity and, if applicable, the count
        and names of missing per-paper JSON result files.
    """
    file_stats = {
        "scoring_file_exists": os.path.exists(scoring_file),
        "output_file_exists": os.path.exists(output_file),
        "scoring_file_valid": False,
        "output_file_valid": False,
        "missing_jsons": 0,
        "missing_json_filenames": [],
    }

    # Check scoring file
    if file_stats["scoring_file_exists"]:
        try:
            with open(scoring_file, "r", encoding="utf-8") as f:
                json.load(f)
            file_stats["scoring_file_valid"] = True
        except json.JSONDecodeError:
            print(f"Warning: {scoring_file} exists but contains invalid JSON")
        except Exception as e:
            print(f"Error checking {scoring_file}: {e}")
    else:
        print(f"Warning: Scoring file {scoring_file} does not exist")

    # Check output file
    if file_stats["output_file_exists"]:
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                json.load(f)
            file_stats["output_file_valid"] = True
        except json.JSONDecodeError:
            print(f"Warning: {output_file} exists but contains invalid JSON")
        except Exception as e:
            print(f"Error checking {output_file}: {e}")

    # Check for missing JSON files if pdf_folder and scoring_data are provided
    if pdf_folder and scoring_data:
        # Get list of PDFs in the folder
        pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]

        # Get list of expected JSON filenames from scoring_data
        expected_jsons = []
        for entry in scoring_data:
            if "filename" in entry:
                # Convert PDF filename to expected JSON filename
                json_filename = entry["filename"].replace(".pdf", ".json")
                expected_jsons.append(json_filename)

        # Check each expected JSON file
        for json_filename in expected_jsons:
            json_path = os.path.join(os.path.dirname(output_file), json_filename)
            if not os.path.exists(json_path):
                file_stats["missing_jsons"] += 1
                file_stats["missing_json_filenames"].append(json_filename)

    # Log file stats
    print("\nFile status:")
    print(
        f"Scoring file ({scoring_file}): {'Exists and valid' if file_stats['scoring_file_valid'] else 'Invalid or missing'}"
    )
    print(
        f"Output file ({output_file}): {'Exists and valid' if file_stats['output_file_valid'] else 'Will be created or overwritten'}"
    )

    if "missing_jsons" in file_stats:
        print(f"Missing JSON files: {file_stats['missing_jsons']}")
        if file_stats["missing_jsons"] > 0 and file_stats["missing_jsons"] <= 10:
            print(f"First missing JSON files: {file_stats['missing_json_filenames'][:10]}")

    return file_stats


def check_missing_pdfs(
    pdf_folder: Union[str, os.PathLike], scoring_data: List[dict]
) -> Dict[str, Any]:
    """
    Check which scoring entries don't have a matching PDF file in `pdf_folder`.

    Papers that the open-access download step (see `PDF_download.py`) couldn't
    resolve are expected to be downloaded manually and dropped into `pdf_folder`
    before this pipeline continues; this reports which ones are still missing.

    Args:
        pdf_folder: Directory expected to contain one PDF per scoring entry
            (filename taken from each entry's "filename" key).
        scoring_data: List of entry dicts with "filename" keys (PDF filenames).

    Returns:
        A dict with "pdfs_expected", "pdfs_found", "pdfs_missing" counts and a
        "missing_pdf_filenames" list of the filenames not found in `pdf_folder`.
    """
    existing_pdfs = {f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")}

    expected_filenames = [entry["filename"] for entry in scoring_data if entry.get("filename")]
    missing_filenames = [f for f in expected_filenames if f not in existing_pdfs]

    stats = {
        "pdfs_expected": len(expected_filenames),
        "pdfs_found": len(expected_filenames) - len(missing_filenames),
        "pdfs_missing": len(missing_filenames),
        "missing_pdf_filenames": missing_filenames,
    }

    print("\nPDF availability:")
    print(f"Expected PDFs: {stats['pdfs_expected']}")
    print(f"Found PDFs: {stats['pdfs_found']}")
    print(f"Missing PDFs: {stats['pdfs_missing']}")
    if missing_filenames:
        print(f"Download these manually and place them in {pdf_folder}, then re-run:")
        for filename in missing_filenames[:10]:
            print(f"  - {filename}")
        if len(missing_filenames) > 10:
            print(f"  ... and {len(missing_filenames) - 10} more")

    return stats


def main(
    training_file: Union[str, os.PathLike],
    scoring_file: Union[str, os.PathLike],
    output_file: Union[str, os.PathLike],
    seed_rf: int,
    threshold: float,
    pdf_folder: Optional[Union[str, os.PathLike]] = None,
) -> None:
    """
    Main function to run the filtering pipeline.

    Trains an XGBoost precision classifier on `training_file`, applies it to
    `scoring_file`, and writes the annotated entries to `output_file`. Also
    persists per-run statistics to "./execution_summary.json".

    Args:
        training_file: Path to the JSON file with labelled training examples.
        scoring_file: Path to the JSON file with entries to score.
        output_file: Destination path for the scored entries JSON file.
        seed_rf: Random seed for the XGBoost classifier.
        threshold: Minimum precision value classified as "precise" (class 1).
        pdf_folder: Optional directory of source PDFs, used to additionally check
            for missing PDFs/JSON results.
    """
    start_time = time.time()
    print(f"Starting pipeline at {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load training and scoring data
    print("\nLoading data...")
    training_data = utils.load_json(training_file)
    print(f"Loaded {len(training_data)} training entries")

    scoring_data = utils.load_json(scoring_file)
    print(f"Loaded {len(scoring_data)} scoring entries")

    # Check JSON files (now with pdf_folder and scoring_data for missing JSON check)
    file_stats = check_json_files(scoring_file, output_file, pdf_folder, scoring_data)

    # Check for missing PDFs if pdf_folder is provided
    missing_pdf_stats = None
    if pdf_folder:
        missing_pdf_stats = check_missing_pdfs(pdf_folder, scoring_data)

    features = [
        "language",
        "table_quality",
        "quality_of_number",
        "year",
        "pdf_quality",
        "rxn_number",
    ]
    target = "precision"

    # Prepare entries for scoring by standardizing field names
    print("\nPreparing entries for scoring...")
    scoring_data, preparation_stats = prepare_entries_for_scoring(scoring_data, features)

    # Prepare training data
    print("\nPreprocessing training data...")
    training_df = preprocess_data(training_data, features, target, threshold)
    print(f"Processed {len(training_df)} training entries with class distribution:")
    print(training_df["precision_class"].value_counts())

    # Train the model
    print("\nTraining the model...")
    model = train_model(training_df, features, "precision_class", seed_rf)

    # Update scoring data
    print("\nUpdating scores for scoring data...")
    updated_data, prediction_stats = update_scores(scoring_data, model, features)

    # Save updated scoring data
    utils.save_json(updated_data, output_file)

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"\nUpdated scoring data saved to {output_file}")
    print(f"Total execution time: {execution_time:.2f} seconds")

    # Save overall run statistics
    stats_summary = {
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "training_file": training_file,
        "scoring_file": scoring_file,
        "output_file": output_file,
        "threshold": threshold,
        "seed": seed_rf,
        "total_execution_time": execution_time,
        "file_stats": file_stats,
        "data_preparation": {
            "total_entries": preparation_stats["total_entries"],
            "complete_entries": preparation_stats["complete_entries"],
            "entries_with_missing_features": preparation_stats["entries_with_missing_features"],
            "converted_rxn_count": preparation_stats["converted_rxn_count"],
            "missing_feature_counts": dict(preparation_stats["missing_feature_counts"]),
        },
        "prediction_results": {
            "total_processed": prediction_stats["total_entries"],
            "successful_predictions": prediction_stats["successful_predictions"],
            "failed_predictions": prediction_stats["failed_predictions"],
            "prediction_errors": dict(prediction_stats["prediction_errors"]),
        },
    }

    # Add missing PDF stats if available
    if missing_pdf_stats:
        stats_summary["missing_pdf_stats"] = {
            "pdfs_expected": missing_pdf_stats["pdfs_expected"],
            "pdfs_found": missing_pdf_stats["pdfs_found"],
            "pdfs_missing": missing_pdf_stats["pdfs_missing"],
        }

    # Save overall statistics
    summary_file = "./execution_summary.json"
    if os.path.exists(summary_file):
        try:
            with open(summary_file, "r", encoding="utf-8") as file:
                summary_data = json.load(file)
                if "runs" in summary_data:
                    summary_data["runs"].append(stats_summary)
                else:
                    summary_data = {"runs": [stats_summary]}
        except (json.JSONDecodeError, FileNotFoundError):
            summary_data = {"runs": [stats_summary]}
    else:
        summary_data = {"runs": [stats_summary]}

    with open(summary_file, "w", encoding="utf-8") as file:
        json.dump(summary_data, file, indent=4)

    print(f"Execution summary saved to {summary_file}")
    print(f"Pipeline completed at {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a model to filter papers")
    parser.add_argument("--training-file", required=True, help="Path to training data JSON file")
    parser.add_argument("--scoring-file", required=True, help="Path to scoring data JSON file")
    parser.add_argument("--output-file", required=True, help="Path to save updated scoring data")
    parser.add_argument("--pdf-folder", help="Path to folder containing PDF files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Threshold for precision classification"
    )

    args = parser.parse_args()

    main(
        args.training_file,
        args.scoring_file,
        args.output_file,
        args.seed,
        args.threshold,
        args.pdf_folder,
    )
