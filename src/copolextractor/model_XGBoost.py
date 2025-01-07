from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import copolextractor.utils as utils
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PowerTransformer
from xgboost import XGBRegressor
import numpy as np
import pandas as pd
import plotly.express as px


# Function to map categorical features to numerical values
def map_categorical_features(X, column, mapping):
    """
    Map categorical values to numeric values using a predefined mapping.
    Unknown values are mapped to -1. Ensures the output is 2D.
    """
    return X[column].map(mapping).values.reshape(-1, 1)


# Define transformer for preprocessing
numerical_features = [
    "temperature", "LogP", "method_1", "method_2", "polymerization_type_1",
    "polymerization_type_2", "monomer1_data_charges_min", "monomer1_data_charges_max",
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
    ]
)


def train_and_evaluate(df_filtered, transformer, param_grid):
    """
    Train and evaluate the model using KFold cross-validation.
    Create and save prediction plots for each fold.

    Returns:
    - all_y_true: List of all true r_product values across folds.
    - all_y_pred: List of all predicted r_product values across folds.
    - mse_test_list: List of Mean Squared Errors for each fold.
    - r2_test_list: List of R2 scores for each fold.
    """

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    pt = PowerTransformer(method="yeo-johnson")
    all_y_true, all_y_pred = [], []
    mse_test_list, r2_test_list = [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(df_filtered), 1):
        print(f"\nFold {fold}")

        train = df_filtered.iloc[train_idx].copy()
        test = df_filtered.iloc[test_idx].copy()

        # Create target variable (r_product)
        train["r1r2"] = train["r1"].values * train["r2"].values
        test["r1r2"] = test["r1"].values * test["r2"].values

        y_train = pt.fit_transform(train["r1r2"].values.reshape(-1, 1)).ravel()
        y_test = pt.transform(test["r1r2"].values.reshape(-1, 1)).ravel()

        X_train = transformer.fit_transform(train)
        X_test = transformer.transform(test)

        model = XGBRegressor(random_state=42)
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=100, cv=3, verbose=1, random_state=42, n_jobs=-1
        )
        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_

        y_pred_test = best_model.predict(X_test)

        mse_test = mean_squared_error(y_test, y_pred_test)
        r2_test = r2_score(y_test, y_pred_test)

        mse_test_list.append(mse_test)
        r2_test_list.append(r2_test)

        print(f"Test MSE: {mse_test:.4f}")
        print(f"Test R2: {r2_test:.4f}")
        print(f"Best Hyperparameters for Fold {fold}: {random_search.best_params_}")

        y_test_true_inv = pt.inverse_transform(y_test.reshape(-1, 1)).ravel()
        y_pred_test_inv = pt.inverse_transform(y_pred_test.reshape(-1, 1)).ravel()

        all_y_true.extend(y_test_true_inv)
        all_y_pred.extend(y_pred_test_inv)

        # Randomly select a subset for plotting
        test_indices = np.random.choice(len(y_test_true_inv), 50, replace=False)
        y_test_true_selected = y_test_true_inv[test_indices]
        y_pred_test_selected = y_pred_test_inv[test_indices]

        # Prepare DataFrame for plotting (including optional columns)
        test_plot_df = pd.DataFrame({
            "True": y_test_true_selected,
            "Predicted": y_pred_test_selected,
            "polymerization_type": test.iloc[test_indices]["polymerization_type"].values,
            "monomer1_name": test.iloc[test_indices]["monomer1_s"].values,
            "monomer2_name": test.iloc[test_indices]["monomer2_s"].values,
            "method": test.iloc[test_indices]["method"].values if "method" in test.columns else None,
            "solvent": test.iloc[test_indices]["solvent"].values if "solvent" in test.columns else None
        })

        # Interactive scatter plot using Plotly
        fig = px.scatter(
            test_plot_df,
            x="True",
            y="Predicted",
            color="polymerization_type",
            hover_data=["monomer1_name", "monomer2_name", "method", "solvent"],
            title=f"Interactive Predicted vs True Values - Fold {fold}",
            labels={"True": "True r_product", "Predicted": "Predicted r_product"}
        )

        fig.add_shape(
            type="line",
            x0=test_plot_df["True"].min(),
            y0=test_plot_df["True"].min(),
            x1=test_plot_df["True"].max(),
            y1=test_plot_df["True"].max(),
            line=dict(color="Gray", dash="dash")
        )

        fig.update_xaxes(range=[-0.1, 2], title_text="True r_product")
        fig.update_yaxes(range=[-0.1, 2], title_text="Predicted r_product")

        fig.show()
        fig.write_html(f"interactive_predictions_fold_{fold}.html")
        print(f"Interactive plot saved as: interactive_predictions_fold_{fold}.html")

    return all_y_true, all_y_pred, mse_test_list, r2_test_list


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
    if combined:
        combined_df = preprocess_and_combine(data, data_flipped)
    else:
        combined_df = pd.read_csv(data)

    print(combined_df["polymerization_type"].unique())

    # Ensure group_id column exists
    if "group_id" not in combined_df.columns:
        print("Creating group_id column...")
        combined_df["group_id"] = combined_df.apply(
            lambda row: tuple(sorted([row["monomer1_s"], row["monomer2_s"]])), axis=1
        )

    if "LogP" not in combined_df.columns:
        print("LogP column not found. Calculating LogP values...")
        combined_df["LogP"] = combined_df["solvent"].apply(
            lambda solvent: utils.calculate_logP(utils.name_to_smiles(solvent)) if utils.name_to_smiles(solvent) else None
        )

    combined_df.dropna(subset=numerical_features, inplace=True)

    param_grid = {
        'n_estimators': [100, 500, 1000, 5000],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': np.logspace(-3, 0, 10),
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2],
        'reg_alpha': np.linspace(0, 1, 10),
        'reg_lambda': np.linspace(0, 1, 10)
    }

    all_y_true, all_y_pred, mse_list, r2_list = train_and_evaluate(combined_df, transformer, param_grid)

    mse_avg = np.mean(mse_list)
    r2_avg = np.mean(r2_list)

    print(f"Average Test MSE: {mse_avg:.4f}")
    print(f"Average Test R2: {r2_avg:.4f}")




