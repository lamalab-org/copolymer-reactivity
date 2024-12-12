import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer, PowerTransformer
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
import os
from sklearn.model_selection import GroupKFold
import copolextractor.utils as utils


# Function to map categorical features to numerical values
def map_categorical_features(X, column, mapping):
    """
    Map categorical values to numeric values using a predefined mapping.
    Unknown values are mapped to -1. Ensures the output is 2D.
    """
    return X[column].map(mapping).values.reshape(-1, 1)

# Method mappings
method_mapping = {
    # free radical
    "free radical": 0,
    "Free Radical": 0,
    "radical": 0,
    "Radical": 0,
    "Homogeneous Radical": 0,
    "Radiation-induced": 0,
    # controlled radical
    "atom transfer radical": 2,
    "atom transfer radical polymerization":2,
    "nickel-mediated radical": 3,
    "conventional radical polymerization": 0,

    # bulk
    "bulk": 10,
    "Bulk": 10,

    # emulsion
    "Emulsion": 8,
    "emulsion": 8,

    # anionic
    "Anionic": 4,
    "sodium-catalysed": 4,

    # cationic
    "cationic": 5,

    "thermal": 0,

    "na": 15,
}


calculation_method_mapping = {
    # Kelen-Tudor and variants
    "Kelen-Tudor": 0,
    "Kelen-Tudos": 0,
    "Extended K-T": 0,
    "Extended Kelen-Tudor": 0,
    "extended Kelen-Tudor": 0,
    "extended Kelen-Tudos": 0,
    "K-T": 0,
    "Kelen-Tüdos": 0,

    # Fineman-Ross and variants
    "Fineman-Ross": 1,
    "Fineman and Ross": 1,
    "Fineman-Ross plot": 1,
    "Finnan-Ross": 1,
    "Fineman Ross method": 1,
    "curve-fitting and Fineman-Ross": 1,
    "Finemann and Ross": 1,
    "Finemann-Ross": 1,

    # Mayo-Lewis and related methods
    "Mayo and Lewis": 2,
    "Mayo-Lewis differential method": 2,
    "intersection method of Mayo and Lewis": 2,
    "Mayo-Lewis": 2,
    "Mayo-Lewis integrated equation": 2,

    # Fundamental equations
    "fundamental equation of copolymerisation": 3,
    "integrated equation": 3,
    "differential equation": 3,
    "integrated copolymer equation": 3,

    # Graphical methods and variants
    "graphical": 4,
    "graphical solution of the copolymerization equation": 4,
    "graphical method of intersections": 4,
    "graphical evaluation": 4,
    "graphic solution": 4,
    "intersection method": 4,
    "Intersection method": 4,
    "intersection line method": 4,

    # Regression and least-squares methods
    "regression analysis": 5,
    "nonlinear least squares technique": 5,
    "nonlinear least-square analysis": 5,
    "least-squares fit": 5,
    "least-squares method": 5,
    "nonlinear least-squares method": 5,
    "nonlinear least squares procedure": 5,
    "nonlinear least squares error-in-variables": 5,
    "curve fitting": 5,
    "Curve fitting": 5,
    "curve-fitting": 5,
    "curve fitting method": 5,
    "nonlinear least-squares": 5,
    "non-linear least squares": 5,
    "nonlinear least-squares procedure": 5,

    # NMR and spectroscopy
    "NMR analysis": 6,
    "NMR spectroscopy": 6,
    "13C NMR": 6,
    "NMR": 6,

    # Alfrey-Price and variants
    "Alfrey-Price": 7,
    "Alfrey and Price": 7,
    "Alfrey-Price equation": 7,
    "Alfrey and Goldfinger": 7,

    # Tidwell-Mortimer and related
    "Tidwell-Mortimer": 8,
    "Tidwell and Mortimer": 8,

    # Penultimate models
    "penultimate model": 9,
    "penultimate mu model": 9,
    "penultimate": 9,
    "Mayo–Lewis terminal model": 9,

    # Other named methods
    "Joshi-Joshi": 10,
    "Shtraikhman approach": 10,
    "Yserielve-Brokhina-Roskin": 10,
    "Yezrielev-Brokhina-Roskin": 10,
    "YBR": 10,
    "Jaacks method": 10,
    "Jaacks": 10,
    "RREVM": 10,
    "EVM": 10,
    "EVM Program": 10,
    "Rosenbrock optimization": 10,
    "Nelder and Mead simplex method": 10,

    # Optimizer and average
    "Optimizer": 11,
    "average": 11,

    # Not specified
    "na": 12,
    "not provided": 12,
    "not specified": 12,
    "previous paper": 12,

    # Others
    "elemental analysis": 13,
    "Kjeldahl analysis": 13,
    "Barb": 13
}

# Define transformer for preprocessing
numerical_features = [
    "temperature", "LogP", "monomer1_data_charges_min", "monomer1_data_charges_max",
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
    Train and evaluate the model using GroupKFold cross-validation.

    Parameters:
    - df_filtered: Combined and deduplicated dataset.
    - transformer: ColumnTransformer for preprocessing.
    - param_grid: Hyperparameter grid for RandomizedSearchCV.

    Returns:
    - all_y_train_true, all_y_train_pred, all_y_true, all_y_pred: Training and test predictions and ground truths.
    """

    # Create group IDs for GroupKFold
    df_filtered['group_id'] = df_filtered.apply(
        lambda row: tuple(sorted([row['monomer1_s'], row['monomer2_s']])),
        axis=1
    )

    # Use GroupKFold to ensure the same group does not appear in both train and test
    group_kf = GroupKFold(n_splits=10)
    all_y_train_true, all_y_train_pred, all_y_true, all_y_pred = [], [], [], []
    mse_train_scores, r2_train_scores = [], []
    mse_test_scores, r2_test_scores = [], []

    for fold, (train_idx, test_idx) in enumerate(group_kf.split(df_filtered, groups=df_filtered['group_id']), 1):
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

        pt = PowerTransformer(method="yeo-johnson")

        y_train = pt.fit_transform(train["r1r2"].values.reshape(-1, 1)).ravel()
        y_test = pt.transform(test["r1r2"].values.reshape(-1, 1)).ravel()
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
        y_pred_train = best_model.predict(X_train)
        y_pred_test = best_model.predict(X_test)

        # Compute metrics
        mse_train = mean_squared_error(y_train, y_pred_train)
        r2_train = r2_score(y_train, y_pred_train)
        mse_test = mean_squared_error(y_test, y_pred_test)
        r2_test = r2_score(y_test, y_pred_test)

        print(f"Train MSE: {mse_train:.4f}, Train R2: {r2_train:.4f}")
        print(f"Test MSE: {mse_test:.4f}, Test R2: {r2_test:.4f}")
        print(f"Best Hyperparameters for Fold {fold}: {random_search.best_params_}")

        # Store results
        mse_train_scores.append(mse_train)
        r2_train_scores.append(r2_train)
        mse_test_scores.append(mse_test)
        r2_test_scores.append(r2_test)
        all_y_train_true.extend(y_train)
        all_y_train_pred.extend(y_pred_train)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred_test)

    # Calculate average metrics
    avg_mse_train = np.mean(mse_train_scores)
    avg_r2_train = np.mean(r2_train_scores)
    avg_mse_test = np.mean(mse_test_scores)
    avg_r2_test = np.mean(r2_test_scores)

    print(f"\nAverage Train MSE: {avg_mse_train:.4f}, Average Train R2: {avg_r2_train:.4f}")
    print(f"Average Test MSE: {avg_mse_test:.4f}, Average Test R2: {avg_r2_test:.4f}")

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
    df_filtered = pd.read_csv(df_filtered)
    df_filtered_flipped = pd.read_csv(df_filtered_flipped)

    # Combine datasets
    combined_df = pd.concat([df_filtered, df_filtered_flipped], ignore_index=True)

    # Remove duplicates
    combined_df = combined_df.drop_duplicates()

    print(f"Combined dataset size after deduplication: {len(combined_df)}")
    return combined_df


def main(data, data_flipped, combined):
    # Print the current working directory
    print("Current Working Directory:", os.getcwd())

    if combined:
        # Combine and deduplicate datasets
        combined_df = preprocess_and_combine(data, data_flipped)
    else:
        combined_df = pd.read_csv(data)

    if "LogP" not in combined_df.columns:
        print("LogP column not found. Calculating LogP values...")
        combined_df["LogP"] = combined_df["solvent"].apply(
            lambda solvent: utils.calculate_logP(utils.name_to_smiles(solvent))
            if utils.name_to_smiles(solvent) else None)

    # Ensure no missing values in numerical features
    combined_df.dropna(subset=numerical_features, inplace=True)

    # Train and evaluate
    param_grid = {
        "n_estimators": [100, 500, 1000, 5000],
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
    combined = False

    main(df_filtered, df_filtered_flipped, combined)


