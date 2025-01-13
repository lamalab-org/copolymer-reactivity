import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, accuracy_score
from xgboost import XGBClassifier
import copolextractor.utils as utils
import numpy as np


def preprocess_data(data, features, target, threshold):
    """
    Prepare data for training by converting it into a DataFrame.
    """
    filtered_data = [
        entry for entry in data if target in entry and entry[target] is not None
    ]
    df = pd.DataFrame(filtered_data)
    df["precision_class"] = (df[target] > threshold).astype(int)
    return df


def build_pipeline(numeric_features, categorical_features, seed_rf):
    """
    Build a preprocessing and modeling pipeline with XGBoost.
    """
    numeric_transformer = "passthrough"  # No transformation needed for numerical features
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = XGBClassifier(
        random_state=seed_rf, use_label_encoder=False, eval_metric="logloss", missing=np.nan
    )
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    return pipeline


def train_model(training_data, features, target, seed_rf):
    """
    Train the XGBoost model.
    """

    print("Datentypes:", training_data[features].dtypes)
    print("Fehlende Werte:", training_data[features].isnull().sum())

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

    X['language'] = X['language'].astype(str)

    print(training_data[features].head())

    cv = StratifiedKFold(n_splits=5)
    scorer = make_scorer(accuracy_score)

    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring=scorer, n_jobs=-1)
    print("Starting GridSearchCV...")
    grid_search.fit(X, y)
    print("GridSearchCV completed.")
    print("Best parameters found:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)

    return grid_search.best_estimator_


def update_scores(data, model, features):
    """
    Update entries with predictions.
    """
    for entry in data:
        # Check if all required features are present and valid
        missing_features = [
            feature
            for feature in features
            if feature not in entry or pd.isna(entry[feature])
        ]
        if missing_features:
            print(f"Skipping entry due to missing features: {missing_features}")
            entry["precision_score"] = None
            continue

        # Create a DataFrame for prediction
        feature_values = pd.DataFrame(
            [{feature: entry.get(feature, pd.NA) for feature in features}]
        )

        print(f"Features for prediction: {feature_values}")  # Debug: Show feature data
        try:
            # Perform prediction
            prediction = model.predict(feature_values)[0]
            print(f"Prediction result: {prediction}")  # Debug: Show prediction result
            entry["precision_score"] = int(prediction)
        except Exception as e:
            # Handle any errors during prediction
            print(f"Error during prediction: {e}")
            entry["precision_score"] = None

    return data


def main(training_file, scoring_file, output_file, seed_rf, threshold):
    """
    Main function to run the filtering pipeline.
    """
    # Load training and scoring data
    training_data = utils.load_json(training_file)
    scoring_data = utils.load_json(scoring_file)

    features = [
        "language",
        "table_quality",
        "quality_of_number",
        "year",
        "pdf_quality",
        "rxn_number",
    ]
    target = "precision"

    # Prepare training data
    training_df = preprocess_data(training_data, features, target, threshold)

    # Train the model
    print("Training the model...")
    model = train_model(training_df, features, "precision_class", seed_rf)

    # Update scoring data
    print("Updating scores for scoring data...")
    updated_data = update_scores(scoring_data, model, features)

    # Save updated scoring data
    utils.save_json(updated_data, output_file)
    print(f"Updated scoring data saved to {output_file}")

