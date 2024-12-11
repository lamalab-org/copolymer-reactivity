import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor


# Function to process molecular data
def process_molecular_data(d):
    """
    Processes molecular data by calculating min, max, and mean for specified keys
    and extracting dipole components.
    """
    keys_to_process = [
        "charges",
        "fukui_electrophilicity",
        "fukui_nucleophilicity",
        "fukui_radical",
    ]
    for key in keys_to_process:
        if key in d and isinstance(d[key], dict):
            values = d[key].values()
            d[f"{key}_min"] = min(values)
            d[f"{key}_max"] = max(values)
            d[f"{key}_mean"] = sum(values) / len(values)
    if "dipole" in d and isinstance(d["dipole"], list) and len(d["dipole"]) == 3:
        d["dipole_x"], d["dipole_y"], d["dipole_z"] = d["dipole"]
    return d


# Function to extract features from a DataFrame
def extract_features(df):
    """
    Extract features from 'monomer1_data' and 'monomer2_data' columns
    and add them as new columns to the DataFrame.
    """
    for monomer_key in ["monomer1_data", "monomer2_data"]:
        df[f"{monomer_key}_processed"] = df[monomer_key].apply(
            lambda x: process_molecular_data(x) if isinstance(x, dict) else {}
        )
        keys = [
            "charges_min",
            "charges_max",
            "charges_mean",
            "fukui_electrophilicity_min",
            "fukui_electrophilicity_max",
            "fukui_electrophilicity_mean",
            "fukui_nucleophilicity_min",
            "fukui_nucleophilicity_max",
            "fukui_nucleophilicity_mean",
            "fukui_radical_min",
            "fukui_radical_max",
            "fukui_radical_mean",
            "dipole_x",
            "dipole_y",
            "dipole_z",
        ]
        for key in keys:
            df[f"{monomer_key}_{key}"] = df[f"{monomer_key}_processed"].apply(
                lambda x: x.get(key)
            )
    return df


# Method mappings
method_mapping = {
    "bulk": 0,
    "Bulk": 0,
    "solvent": 1,
    "Solvent": 1,
    "emulsion": 2,
    "suspension": 3,
    "photopolymerization": 10,
    "None": 20,
}

calculation_method_mapping = {
    "Kelen-Tudor": 0,
    "Fineman-Ross": 1,
    "fundamental equation of copolymerisation": 2,
    "Mayo and Lewis": 3,
    "Joshi-Joshi": 4,
    "Schwan and Price": 5,
    "Alfrey-Price": 6,
    "regression analysis": 7,
    "graphical": 8,
    "logarithm plot": 9,
    "Nuclear magnetic resonance spectroscopy": 11,
    "carbon elemental analyses": 12,
    "not provided": 30,
    "None": 30,
}

# Define transformer for preprocessing
numerical_features = [
    "temperature",
    "monomer1_data_charges_min",
    "monomer1_data_charges_max",
    "monomer1_data_charges_mean",
    "monomer1_data_fukui_electrophilicity_min",
    "monomer1_data_fukui_electrophilicity_max",
    "monomer1_data_fukui_electrophilicity_mean",
    "monomer1_data_fukui_nucleophilicity_min",
    "monomer1_data_fukui_nucleophilicity_max",
    "monomer1_data_fukui_nucleophilicity_mean",
    "monomer1_data_fukui_radical_min",
    "monomer1_data_fukui_radical_max",
    "monomer1_data_fukui_radical_mean",
    "monomer1_data_dipole_x",
    "monomer1_data_dipole_y",
    "monomer1_data_dipole_z",
    "monomer2_data_charges_min",
    "monomer2_data_charges_max",
    "monomer2_data_charges_mean",
    "monomer2_data_fukui_electrophilicity_min",
    "monomer2_data_fukui_electrophilicity_max",
    "monomer2_data_fukui_electrophilicity_mean",
    "monomer2_data_fukui_nucleophilicity_min",
    "monomer2_data_fukui_nucleophilicity_max",
    "monomer2_data_fukui_nucleophilicity_mean",
    "monomer2_data_fukui_radical_min",
    "monomer2_data_fukui_radical_max",
    "monomer2_data_fukui_radical_mean",
    "monomer2_data_dipole_x",
    "monomer2_data_dipole_y",
    "monomer2_data_dipole_z",
]


categorical_features = ["polymerization_type"]

transformer = ColumnTransformer(
    [
        ("numerical", StandardScaler(), numerical_features),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)


# Perform model training and evaluation
def train_and_evaluate(df_filtered, df_filtered_flipped, transformer, param_grid):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_scores = []

    all_y_true, all_y_pred = [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(df_filtered), 1):
        print(f"Fold {fold}")

        train = pd.concat(
            [df_filtered.iloc[train_idx], df_filtered_flipped.iloc[train_idx]]
        )
        test = pd.concat(
            [df_filtered.iloc[test_idx], df_filtered_flipped.iloc[test_idx]]
        )

        train["r1r2"] = train["r1"] * train["r2"]
        test["r1r2"] = test["r1"] * test["r2"]

        y_train = np.sign(train["r1r2"].values) * np.sqrt(np.abs(train["r1r2"].values))
        y_test = np.sign(test["r1r2"].values) * np.sqrt(np.abs(test["r1r2"].values))

        X_train = transformer.fit_transform(train)
        X_test = transformer.transform(test)

        model = XGBRegressor(random_state=42)

        random_search = RandomizedSearchCV(
            model,
            param_distributions=param_grid,
            n_iter=100,
            cv=3,
            verbose=1,
            random_state=42,
            n_jobs=-1,
        )
        random_search.fit(X_train, y_train)

        best_model = random_search.best_estimator_
        y_pred_test = best_model.predict(X_test)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred_test)

        mse_test = mean_squared_error(y_test, y_pred_test)
        r2_test = r2_score(y_test, y_pred_test)

        print(f"Test MSE: {mse_test:.4f}, Test R2: {r2_test:.4f}")
        print(f"Best Hyperparameters for Fold {fold}: {random_search.best_params_}")

        fold_scores.append(r2_test)

    avg_r2 = np.mean(fold_scores)
    print(f"\nAverage Test R2: {avg_r2:.4f}")


# Define the parameter grid for RandomizedSearchCV
param_grid = {
    "n_estimators": [100, 500, 1000],
    "max_depth": [3, 5, 7],
    "learning_rate": np.logspace(-3, 0, 10),
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],
    "gamma": [0, 0.1, 0.2],
    "reg_alpha": np.linspace(0, 1, 5),
    "reg_lambda": np.linspace(0, 1, 5),
}

# Main execution
if __name__ == "__main__":
    # Load data from CSV files
    df_filtered = pd.read_csv("../../copol_prediction/output/processed_data.csv")
    df_filtered_flipped = pd.read_csv(
        "../../copol_prediction/output/processed_data_flipped.csv"
    )

    # Preprocess the data
    df_filtered = extract_features(df_filtered)
    df_filtered_flipped = extract_features(df_filtered_flipped)

    # Train and evaluate the model
    train_and_evaluate(df_filtered, df_filtered_flipped, transformer, param_grid)
