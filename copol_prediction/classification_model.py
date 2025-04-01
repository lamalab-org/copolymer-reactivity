from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

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


def get_r_product_class(r_value):
    """
    Convert r_product value to class labels:
    0: r_product ≤ 0.1
    1: 0.1 < r_product < 0.9
    2: 0.9 ≤ r_product ≤ 1.1
    3: r_product > 1.1
    """
    if r_value < 0.1:
        return 0
    elif 0.1 < r_value < 0.9:
        return 1
    elif 0.9 < r_value < 1.1:
        return 2
    else:  # r_value > 1.1
        return 3


def train_and_evaluate(df_filtered, transformer, param_grid):
    """
    Train and evaluate the classification model using StratifiedKFold cross-validation.
    Creates and saves prediction plots for each fold.
    """
    df_filtered.dropna(subset=['r1', 'r2'], inplace=True)

    # Create target variable for stratification
    df_filtered["r_product"] = df_filtered["r1"].values * df_filtered["r2"].values
    y_all = df_filtered["r_product"].apply(get_r_product_class)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    all_y_true, all_y_pred = [], []
    accuracy_list = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(df_filtered, y_all), 1):
        print(f"\nFold {fold}")

        # Create train and test sets using integer indexing
        train = df_filtered.iloc[train_idx].copy()
        test = df_filtered.iloc[test_idx].copy()

        # Get corresponding y values using integer indexing
        y_train = train["r_product"].apply(get_r_product_class)
        y_test = test["r_product"].apply(get_r_product_class)

        # Print class distribution
        print("\nClass distribution in this fold:")
        print("Training set:")
        train_dist = pd.Series(y_train).value_counts().sort_index()
        print("Class 0 (r≤0.1):", train_dist.get(0, 0))
        print("Class 1 (0.1<r<0.9):", train_dist.get(1, 0))
        print("Class 2 (0.9≤r≤1.1):", train_dist.get(2, 0))
        print("Class 3 (r>1.1):", train_dist.get(3, 0))

        print("\nTest set:")
        test_dist = pd.Series(y_test).value_counts().sort_index()
        print("Class 0 (r≤0.1):", test_dist.get(0, 0))
        print("Class 1 (0.1<r<0.9):", test_dist.get(1, 0))
        print("Class 2 (0.9≤r≤1.1):", test_dist.get(2, 0))
        print("Class 3 (r>1.1):", test_dist.get(3, 0))

        X_train = transformer.fit_transform(train)
        X_test = transformer.transform(test)

        model = XGBClassifier(random_state=42)
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=100, cv=3, verbose=1, random_state=42, n_jobs=-1
        )
        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_

        y_pred_test = best_model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred_test)
        accuracy_list.append(accuracy)

        print(f"Test Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_test))
        print("\nConfusion Matrix:")
        conf_matrix = confusion_matrix(y_test, y_pred_test)
        print(conf_matrix)
        print(f"Best Hyperparameters for Fold {fold}: {random_search.best_params_}")

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred_test)

        # Create confusion matrix heatmap
        labels = ['r≤0.1', '0.1<r<0.9', '0.9≤r≤1.1', 'r>1.1']

        # Ensure conf_matrix matches the labels
        if conf_matrix.shape != (4, 4):
            print(f"Warning: Unexpected confusion matrix shape: {conf_matrix.shape}")
            continue

        fig = ff.create_annotated_heatmap(
            z=conf_matrix,
            x=labels,
            y=labels,
            colorscale='Blues'
        )
        fig.update_layout(
            title=f'Confusion Matrix - Fold {fold}',
            xaxis_title='Predicted',
            yaxis_title='True'
        )

        # Save confusion matrix plot
        fig.write_html(f"confusion_matrix_fold_{fold}.html")
        print(f"Confusion matrix plot saved as: confusion_matrix_fold_{fold}.html")

    return all_y_true, all_y_pred, accuracy_list


def main(data):
    print("Starting training of XGBoost classification model")

    # Initialize counters for total class distribution
    total_class_dist = {0: 0, 1: 0, 2: 0, 3: 0}

    df = pd.read_csv(data)
    print("Original df: ", len(df))
    df = df.replace(['na', 'NA', 'null', 'NULL', '', 'None', 'none', 'nan'], np.nan)

    print("\nInitial number of samples:", len(df))
    print("Initial polymerization types:", df["polymerization_type"].unique())
    print("Initial polymerization methods:", df["method"].unique())

    # Define radical polymerization types
    radical_types = [
        'atom transfer radical polymerization',
        'free radical',
        'photo-induced polymerization',
        'nickel-mediated radical',
        'controlled radical',
        'controlled/living radical',
        'atom-transfer radical polymerization',
        'reversible addition-fragmentation chain transfer polymerization',
        'radical',
        'controlled/living polymerization'
    ]

    # Filter for radical polymerization types and solvent method
    df = df[df['polymerization_type'].isin(radical_types)]
    df = df[df['method'] == 'solvent']

    # Filter for 'source' column with value 'copol'
    df = df[df['original_source'] == 'copol database']
    print(f"Number of samples after filtering by 'source=copol': {len(df)}")

    # Reset index after filtering
    df = df.reset_index(drop=True)

    print("\nSelected radical polymerization types:", radical_types)
    print("\nRemaining polymerization types after filtering:", df["polymerization_type"].unique())
    print("Remaining polymerization methods after filtering:", df["method"].unique())
    print(f"Number of samples after filtering: {len(df)}")

    # Ensure group_id column exists
    if "group_id" not in df.columns:
        print("Creating group_id column...")
        df["group_id"] = df.apply(
            lambda row: tuple(sorted([row["monomer1_s"], row["monomer2_s"]])), axis=1
        )

    # Drop rows with missing values in features
    df.dropna(subset=numerical_features, inplace=True)

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

    all_y_true, all_y_pred, accuracy_list = train_and_evaluate(df, transformer, param_grid)

    # Calculate and print overall metrics
    print("\nOverall Results:")
    print("Average Accuracy:", np.mean(accuracy_list))
    print("\nFinal Classification Report:")
    print(classification_report(all_y_true, all_y_pred))
    print("\nFinal Confusion Matrix:")
    print(confusion_matrix(all_y_true, all_y_pred))

    # Calculate overall class distribution
    df["r_product"] = df["r1"].values * df["r2"].values
    y_all = df["r_product"].apply(get_r_product_class)
    total_dist = pd.Series(y_all).value_counts().sort_index()

    print("\nTotal class distribution in filtered dataset:")
    print("Class 0 (r≤0.1):", total_dist.get(0, 0))
    print("Class 1 (0.1<r<0.9):", total_dist.get(1, 0))
    print("Class 2 (0.9≤r≤1.1):", total_dist.get(2, 0))
    print("Class 3 (r>1.1):", total_dist.get(3, 0))


if __name__ == "__main__":
    data = "./output/processed_data_copol.csv"
    main(data)