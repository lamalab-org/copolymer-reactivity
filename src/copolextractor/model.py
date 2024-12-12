import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
import os

# Function to map categorical features to numerical values
def map_categorical_features(X, column, mapping):
    """
    Map categorical values to numeric values using a predefined mapping.
    Unknown values are mapped to -1. Ensures the output is 2D.
    """
    return X[column].map(mapping).values.reshape(-1, 1)

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
    "temperature", "monomer1_data_charges_min", "monomer1_data_charges_max",
    "monomer1_data_charges_mean", "monomer1_data_fukui_electrophilicity_min",
    "monomer1_data_fukui_electrophilicity_max", "monomer1_data_fukui_electrophilicity_mean",
    "monomer1_data_fukui_nucleophilicity_min", "monomer1_data_fukui_nucleophilicity_max",
    "monomer1_data_fukui_nucleophilicity_mean", "monomer1_data_fukui_radical_min",
    "monomer1_data_fukui_radical_max", "monomer1_data_fukui_radical_mean",
    "monomer1_data_dipole_x", "monomer1_data_dipole_y", "monomer1_data_dipole_z",
    "monomer2_data_charges_min", "monomer2_data_charges_max", "monomer2_data_charges_mean",
    "monomer2_data_fukui_electrophilicity_min", "monomer2_data_fukui_electrophilicity_max",
    "monomer2_data_fukui_electrophilicity_mean", "monomer2_data_fukui_nucleophilicity_min",
    "monomer2_data_fukui_nucleophilicity_max", "monomer2_data_fukui_nucleophilicity_mean",
    "monomer2_data_fukui_radical_min", "monomer2_data_fukui_radical_max",
    "monomer2_data_fukui_radical_mean", "monomer2_data_dipole_x",
    "monomer2_data_dipole_y", "monomer2_data_dipole_z"
]

transformer = ColumnTransformer(
    [
        ("numerical", Pipeline([("scaler", StandardScaler())]), numerical_features),
        ("categorical_method", FunctionTransformer(
            lambda X: map_categorical_features(X, "polymerization_type", method_mapping), validate=False
        ), ["polymerization_type"]),
        ("categorical_calculation", FunctionTransformer(
            lambda X: map_categorical_features(X, "determination_method", calculation_method_mapping), validate=False
        ), ["determination_method"]),
    ]
)


def train_and_evaluate(df_filtered, transformer, param_grid):
    """
    Train and evaluate the model using KFold cross-validation.

    Parameters:
    - df_filtered: Combined and deduplicated dataset.
    - transformer: ColumnTransformer for preprocessing.
    - param_grid: Hyperparameter grid for RandomizedSearchCV.

    Returns:
    - all_y_train_true, all_y_train_pred, all_y_true, all_y_pred: Training and test predictions and ground truths.
    """
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    all_y_train_true, all_y_train_pred, all_y_true, all_y_pred = [], [], [], []
    mse_scores, r2_scores = [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(df_filtered), 1):
        print(f"\nFold {fold}")

        # Train and test datasets
        train = df_filtered.iloc[train_idx]
        test = df_filtered.iloc[test_idx]

        # Print dataset sizes
        print(f"Train set size: {len(train)}")
        print(f"Test set size: {len(test)}")

        # Prepare target variable
        train["r1r2"] = train["r1"] * train["r2"]
        test["r1r2"] = test["r1"] * test["r2"]

        y_train = np.sign(train["r1r2"]) * np.sqrt(np.abs(train["r1r2"]))
        y_test = np.sign(test["r1r2"]) * np.sqrt(np.abs(test["r1r2"]))
        X_train = transformer.fit_transform(train)
        X_test = transformer.transform(test)

        # Train the model
        model = XGBRegressor(random_state=42)
        random_search = RandomizedSearchCV(
            model, param_distributions=param_grid, n_iter=100, cv=3, verbose=1,
            random_state=42, n_jobs=-1)
        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_

        # Make predictions
        y_pred_test = best_model.predict(X_test)
        y_pred_train = best_model.predict(X_train)

        # Compute metrics
        mse_test = mean_squared_error(y_test, y_pred_test)
        r2_test = r2_score(y_test, y_pred_test)

        print(f"Test MSE: {mse_test:.4f}, Test R2: {r2_test:.4f}")
        print(f"Best Hyperparameters for Fold {fold}: {random_search.best_params_}")

        # Store results
        mse_scores.append(mse_test)
        r2_scores.append(r2_test)
        all_y_train_true.extend(y_train)
        all_y_train_pred.extend(y_pred_train)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred_test)

    avg_mse = np.mean(mse_scores)
    avg_r2 = np.mean(r2_scores)
    print(f"\nAverage Test MSE: {avg_mse:.4f}")
    print(f"Average Test R2: {avg_r2:.4f}")

    return all_y_train_true, all_y_train_pred, all_y_true, all_y_pred


# Plotting function
def plot_predictions(all_y_train_true, all_y_train_pred, all_y_true, all_y_pred):
    train_indices = np.random.choice(len(all_y_train_true), 200, replace=False)
    selected_y_train_true = np.array(all_y_train_true)[train_indices]
    selected_y_train_pred = np.array(all_y_train_pred)[train_indices]

    test_indices = np.random.choice(len(all_y_true), 100, replace=False)
    selected_y_true = np.array(all_y_true)[test_indices]
    selected_y_pred = np.array(all_y_pred)[test_indices]

    plt.figure(figsize=(8, 8))
    plt.scatter(selected_y_train_true, selected_y_train_pred, alpha=0.7, label="Train Data", color="#661124")
    plt.plot([min(selected_y_train_true), max(selected_y_train_true)],
             [min(selected_y_train_true), max(selected_y_train_true)],
             color="grey", linestyle="--")
    plt.scatter(selected_y_true, selected_y_pred, alpha=0.7, label="Test Data", color="#194A81")
    plt.xlabel(r"True $r_{\text{product}}$", fontsize=16)
    plt.ylabel(r"Predicted $r_{\text{product}}$", fontsize=16)
    plt.legend(fontsize=12)
    plt.savefig("pred_200_XGBoost_just_monomer.png", dpi=300, bbox_inches="tight")
    plt.show()


def preprocess_and_combine(df_filtered, df_filtered_flipped):
    """
    Combine original and flipped datasets, remove duplicates, and return the deduplicated dataset.

    Parameters:
    - df_filtered: Original dataset.
    - df_filtered_flipped: Flipped dataset.

    Returns:
    - Combined and deduplicated dataset.
    """
    # Combine datasets
    combined_df = pd.concat([df_filtered, df_filtered_flipped], ignore_index=True)

    # Remove duplicates
    combined_df = combined_df.drop_duplicates()

    print(f"Combined dataset size after deduplication: {len(combined_df)}")
    return combined_df


def main(data, data_flipped):
    # Print the current working directory
    print("Current Working Directory:", os.getcwd())

    # Combine and deduplicate datasets
    combined_df = preprocess_and_combine(data, data_flipped)

    # Ensure no missing values in numerical features
    combined_df.dropna(subset=numerical_features, inplace=True)

    # Train and evaluate
    param_grid = {
        "n_estimators": [100, 500, 1000],
        "max_depth": [3, 5, 7],
        "learning_rate": np.logspace(-4, -1, 10),
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.1, 0.2],
        "reg_alpha": np.linspace(0, 1, 5),
        "reg_lambda": np.linspace(0, 1, 5),
    }

    # Call train_and_evaluate with the combined dataset
    all_y_train_true, all_y_train_pred, all_y_true, all_y_pred = train_and_evaluate(
        combined_df, transformer, param_grid
    )

    plot_predictions(all_y_train_true, all_y_train_pred, all_y_true, all_y_pred)

# Main execution
if __name__ == "__main__":

    # Load datasets
    df_filtered = pd.read_csv("../../copol_prediction/output/processed_data_copol.csv")
    df_filtered_flipped = pd.read_csv("../../copol_prediction/output/processed_data_copol_flipped.csv")

    main(df_filtered, df_filtered_flipped)


