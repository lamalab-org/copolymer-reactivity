from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PowerTransformer
from xgboost import XGBRegressor
import numpy as np
import pandas as pd
import plotly.express as px
from collections import Counter


def get_most_common_monomer(df):
    """Find the most frequently occurring monomer in the dataset"""
    monomer_count = Counter()

    # Count occurrences in both monomer1_s and monomer2_s columns
    monomer_count.update(df['monomer1_s'].values)
    monomer_count.update(df['monomer2_s'].values)

    most_common = monomer_count.most_common(1)[0]
    return most_common[0]


def get_monomer_dataset(df, target_monomer):
    """Create dataset for specific monomer"""
    df = df.copy()
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

    print("After filtering method ans technique:", len(df))

    # Filter for 'source' column with value 'copol'
    df = df[df['original_source'] == 'copol database']
    print(f"\nNumber of samples after filtering by 'source=copol': {len(df)}")

    # Reset index after filtering
    df = df.reset_index(drop=True)

    # Filter for entries containing the target monomer
    #monomer_df = df[
        #(df['monomer1_s'] == target_monomer) |
        #(df['monomer2_s'] == target_monomer)
    #].copy()

    monomer_df = df

    # Calculate r_product
    monomer_df['r_product'] = monomer_df['r1'] * monomer_df['r2']

    # Define column groups
    mol1_cols = [col for col in df.columns if col.startswith('monomer1_data_')]
    mol2_cols = [col for col in df.columns if col.startswith('monomer2_data_')]
    condition_cols = ['LogP', 'temperature', 'method_1', 'method_2',
                     'polymerization_type_1', 'polymerization_type_2']

    # Print info about numeric columns
    print("\nColumn info before filtering:")
    for col in mol1_cols + mol2_cols + condition_cols + ['r_product']:
        non_null = monomer_df[col].notna().sum()
        total = len(monomer_df)
        print(f"{col}: {non_null}/{total} non-null values")

    # Drop rows with missing values
    all_feature_cols = mol1_cols + mol2_cols + condition_cols
    monomer_df = monomer_df.dropna(subset=all_feature_cols + ['r_product'])

    print(f"\nRows after filtering: {len(monomer_df)}")
    print("\nRemaining polymerization types:", df["polymerization_type"].unique())
    print("Remaining polymerization methods:", df["method"].unique())

    return monomer_df


def train_and_evaluate_r_product(df):
    mol1_cols = [col for col in df.columns if col.startswith('monomer1_data_')]
    mol2_cols = [col for col in df.columns if col.startswith('monomer2_data_')]
    condition_cols = ['LogP', 'temperature', 'method_1', 'method_2',
                      'polymerization_type_1', 'polymerization_type_2']

    feature_cols = mol1_cols + mol2_cols + condition_cols

    transformer = ColumnTransformer(
        [("all_features", StandardScaler(), feature_cols)],
        remainder='drop'
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    pt = PowerTransformer(method="yeo-johnson")
    all_y_true, all_y_pred = [], []
    mse_test_list, r2_test_list = [], []

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

    for fold, (train_idx, test_idx) in enumerate(kf.split(df), 1):
        print(f"\nFold {fold} - Predicting r_product")

        train = df.iloc[train_idx].copy()
        test = df.iloc[test_idx].copy()

        y_train = pt.fit_transform(train['r_product'].values.reshape(-1, 1)).ravel()
        y_test = pt.transform(test['r_product'].values.reshape(-1, 1)).ravel()

        X_train = transformer.fit_transform(train[feature_cols])
        X_test = transformer.transform(test[feature_cols])

        model = XGBRegressor(random_state=42)
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            scoring='r2',
            n_iter=50, cv=3, verbose=1, random_state=42, n_jobs=-1
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
        print(f"Best parameters: {random_search.best_params_}")

        y_test_true_inv = pt.inverse_transform(y_test.reshape(-1, 1)).ravel()
        y_pred_test_inv = pt.inverse_transform(y_pred_test.reshape(-1, 1)).ravel()

        all_y_true.extend(y_test_true_inv)
        all_y_pred.extend(y_pred_test_inv)

        # Create DataFrame with all information for plotting
        plot_df = pd.DataFrame({
            'True_r_product': y_test_true_inv,
            'Predicted_r_product': y_pred_test_inv,
            'Monomer1': test['monomer1_s'].values,
            'Monomer2': test['monomer2_s'].values,
            'Temperature': test['temperature'].values,
            'LogP': test['LogP'].values
        })

        fig = px.scatter(
            plot_df,
            x='True_r_product',
            y='Predicted_r_product',
            hover_data=['Monomer1', 'Monomer2', 'Temperature', 'LogP'],
            title=f"Predicted vs True r_product - Fold {fold}",
            labels={"True_r_product": "True r_product",
                   "Predicted_r_product": "Predicted r_product"}
        )

        fig.add_shape(
            type="line",
            x0=min(y_test_true_inv),
            y0=min(y_test_true_inv),
            x1=max(y_test_true_inv),
            y1=max(y_test_true_inv),
            line=dict(color="Gray", dash="dash")
        )

        fig.show()
        fig.write_html(f"predictions_r_product_fold_{fold}.html")

    return np.mean(mse_test_list), np.mean(r2_test_list)


def print_monomer_entries(monomer_df):
    """Print detailed information for each entry with the target monomer"""
    print("\nDetailed entries for selected monomer:")
    print("=" * 100)

    for idx, row in monomer_df.iterrows():
        print(f"\nEntry {idx + 1}:")
        print("-" * 50)

        # Parse monomers list from string
        try:
            monomers = eval(row['monomers'])
            monomer1_name = monomers[0]
            monomer2_name = monomers[1]
        except:
            monomer1_name = "Error parsing name"
            monomer2_name = "Error parsing name"

        # Get r_product
        r_product = row['r_product']

        print(f"Monomer 1: {monomer1_name}")
        print(f"Monomer 2: {monomer2_name}")
        print("\nConditions:")
        print(f"Temperature: {row['temperature']}")
        print(f"Method: {'method_1' if row['method_1'] == 1 else 'method_2'}")
        print(f"Polymerization Type: {'type_1' if row['polymerization_type_1'] == 1 else 'type_2'}")
        print(f"LogP: {row['LogP']:.2f}" if pd.notnull(row['LogP']) else "LogP: N/A")
        print(f"r_product: {r_product:.3f}")
        print("-" * 50)


def main():
    # Load the dataset
    print("Loading dataset...")
    df = pd.read_csv("./output/processed_data_copol.csv", na_values=['na', 'NA', ''])

    # Get most common monomer
    target_monomer = get_most_common_monomer(df)
    print(f"\nMost common monomer SMILES: {target_monomer}")

    # Get dataset for this monomer
    monomer_df = get_monomer_dataset(df, target_monomer)
    print(f"\nDataset size: {len(monomer_df)} entries")

    # Print detailed information for each entry
    print_monomer_entries(monomer_df)

    print("\nFeature columns:")
    print("Monomer 1 features:", [col for col in monomer_df.columns if col.startswith('monomer1_data_')])
    print("Monomer 2 features:", [col for col in monomer_df.columns if col.startswith('monomer2_data_')])
    print("Condition features:", ['LogP', 'temperature', 'method_1', 'method_2',
                                  'polymerization_type_1', 'polymerization_type_2'])

    # Train and evaluate model for r_product
    print("\nTraining model for r_product...")
    mse_r_product, r2_r_product = train_and_evaluate_r_product(monomer_df)

    # Print final results
    print("\nFinal Results:")
    print(f"r_product - Average MSE: {mse_r_product:.4f}, Average R2: {r2_r_product:.4f}")


if __name__ == "__main__":
    main()