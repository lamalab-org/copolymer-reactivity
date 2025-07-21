from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from copolextractor import utils
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression


class DistributionBucketRegressor:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.bucket_edges = None
        self.bucket_centers = None
        self.transformer = None
        self.model = None
        self.feature_names = None

    def _create_predefined_buckets(self, r1r2_values):
        predefined_edges = [0.01, 0.1, 0.25, 0.5, 0.75, 0.95, 0.99, 1.05, 1.5, 5.00]
        edges = np.array([0.0] + predefined_edges)
        max_val = r1r2_values.max()
        if max_val > edges[-1]:
            print(f"WARNING: Found values above 2.0 ({max_val:.2f}). Capping them to 2.0.")
            r1r2_values = np.clip(r1r2_values, None, 2.0)

        self.bucket_edges = edges
        self.bucket_centers = (edges[:-1] + edges[1:]) / 2

    def _assign_buckets(self, r1r2_values):
        indices = np.digitize(r1r2_values, self.bucket_edges) - 1
        indices = np.clip(indices, 0, len(self.bucket_centers) - 1)
        return indices

    def prepare_data(self, df):
        """
        Prepare data for model training
        Filters data, creates target variable, and extracts features
        Args:
            df: The DataFrame to process
        Returns:
            tuple: (X, y, actual_features, df) - Transformed features, target variable, feature names, and filtered DataFrame
        """
        print("\nPreparing data for classification modeling...")
        # Check if df is actually a DataFrame
        if not isinstance(df, pd.DataFrame):
            print(f"Error: Expected a DataFrame, got {type(df)} instead")
            return None, None, None, None

        # Create a copy of the DataFrame to avoid modifying the original
        df = df.copy()

        # Filter by radical polymerization types
        original_count = len(df)
        df = df[df['polymerization_type'].isin(utils.RADICAL_TYPES)]
        filtered_count = len(df)
        print(f"Filtered datapoints (radical polymerization): {filtered_count}")
        print(f"Removed datapoints (non-radical polymerization): {original_count - filtered_count}")

        # Create target variable
        if 'r1r2' not in df.columns and 'constant_1' in df.columns and 'constant_2' in df.columns:
            df['r1r2'] = df['constant_1'] * df['constant_2']
            print("Created 'r1r2' column from 'constant_1' and 'constant_2'")

        # Remove rows where r1r2 is less than 0
        original_count = len(df)
        df = df[df['r1r2'] >= 0]
        print(f"Dropped {original_count - len(df)} rows where r1r2 < 0")

        # Filter r1r2 > 100
        # r1r2_high_count = len(df[df['r1r2'] > 100])
        # df = df[df['r1r2'] <= 100]
        # print(f"Dropped {r1r2_high_count} rows where r1r2 > 100")

        # Filter temperature > 150
        # temp_out_of_range_count = len(df[(df['temperature'] < 0) | (df['temperature'] > 150)])
        # df = df[(df['temperature'] >= 0) & (df['temperature'] <= 150)]
        # print(f"Dropped {temp_out_of_range_count} rows where temperature > 150")

        # Create bucket edges based on the range of r1r2 values
        self._create_predefined_buckets(df['r1r2'].values)

        # Assign each r1r2 value to a bucket to create the target variable
        y_buckets = self._assign_buckets(df['r1r2'].values)
        df['bucket'] = y_buckets

        print("Bucket edges:", self.bucket_edges)
        print("Bucket centers:", self.bucket_centers)
        print("Shape of centers:", self.bucket_centers.shape)

        # Print bucket distribution
        bucket_counts = df['bucket'].value_counts().sort_index()
        print(f"\nBucket distribution (showing up to 10 buckets):")
        for i, count in bucket_counts.head(10).items():
            bucket_range = f"{self.bucket_edges[i]:.4f} - {self.bucket_edges[i + 1]:.4f}"
            print(f"  Bucket {i} ({bucket_range}): {count} samples")

        print(f"  ...")
        for i, count in bucket_counts.tail(5).items():
            bucket_range = f"{self.bucket_edges[i]:.4f} - {self.bucket_edges[i + 1]:.4f}"
            print(f"  Bucket {i} ({bucket_range}): {count} samples")

        # Define features
        embedding_features = []
        if 'polytype_emb_1' in df.columns:
            embedding_features.extend(['polytype_emb_1', 'polytype_emb_2'])
        if 'method_emb_1' in df.columns:
            embedding_features.extend(['method_emb_1', 'method_emb_2'])

        numerical_features = ['temperature', 'ip_corrected_1', 'ea_1', 'homo_1', 'lumo_1',
                              'global_electrophilicity_1', 'global_nucleophilicity_1',
                              'best_conformer_energy_1', 'charges_min_1', 'charges_max_1',
                              'charges_mean_1', 'fukui_electrophilicity_min_1',
                              'fukui_electrophilicity_max_1', 'fukui_electrophilicity_mean_1',
                              'fukui_nucleophilicity_min_1', 'fukui_nucleophilicity_max_1',
                              'fukui_nucleophilicity_mean_1', 'fukui_radical_min_1',
                              'fukui_radical_max_1', 'fukui_radical_mean_1', 'dipole_x_1',
                              'dipole_y_1', 'dipole_z_1', 'ip_corrected_2', 'ea_2', 'homo_2',
                              'lumo_2', 'global_electrophilicity_2', 'global_nucleophilicity_2',
                              'charges_min_2', 'charges_max_2', 'charges_mean_2',
                              'fukui_electrophilicity_min_2', 'fukui_electrophilicity_max_2',
                              'fukui_electrophilicity_mean_2', 'fukui_nucleophilicity_min_2',
                              'fukui_nucleophilicity_max_2', 'fukui_nucleophilicity_mean_2',
                              'fukui_radical_min_2', 'fukui_radical_max_2', 'fukui_radical_mean_2',
                              'dipole_x_2', 'dipole_y_2', 'dipole_z_2', 'solvent_logp',"delta_HOMO_LUMO_AA",
                              "delta_HOMO_LUMO_AB", "delta_HOMO_LUMO_BB", "delta_HOMO_LUMO_BA"] + embedding_features

        # Only use categorical features if no embeddings are available
        categorical_features = []
        if not embedding_features:
            categorical_features = ['polymerization_type', 'method']

        # Keep only columns that exist in the DataFrame
        existing_numerical = [col for col in numerical_features if col in df.columns]
        existing_categorical = [col for col in categorical_features if col in df.columns]

        # Handle missing values
        model_features = existing_numerical + existing_categorical
        df = df.dropna(subset=model_features)
        print(f"Remaining rows for modeling: {len(df)}")

        # List for actual feature names
        actual_features = []

        # Process numerical features
        print("\nUsing these numerical features:")
        for feature in existing_numerical:
            actual_features.append(feature)
            print(f"- {feature}")

        # Process categorical features with one-hot encoding
        if existing_categorical:
            print("\nOne-hot encoding these categorical features:")
            for feature in existing_categorical:
                dummies = pd.get_dummies(df[feature], prefix=feature, drop_first=False)
                df = pd.concat([df, dummies], axis=1)
                actual_features.extend(dummies.columns.tolist())
                print(f"- {feature} (creates {len(dummies.columns)} dummy features)")

        if len(actual_features) < 5:
            print(f"Error: Only {len(actual_features)} features found. Not enough for a good model.")
            return None, None, None, None

        print(f"\nFinal feature count for modeling: {len(actual_features)}")

        # Create feature transformer
        self.transformer = ColumnTransformer([
            ('numerical', StandardScaler(), actual_features)
        ], remainder='drop')

        # Transform features
        X = self.transformer.fit_transform(df[actual_features])

        # Target variable
        y = df['bucket'].values

        # Store feature names for later use
        self.feature_names = actual_features

        return X, y, df

    def fit(self, df, best_params=None):
        X, y, df_prepared = self.prepare_data(df)
        kf_splits = utils.create_grouped_kfold_splits(df_prepared, n_splits=5, random_state=self.random_state)

        # Arrays for test results
        test_preds = []
        test_truths = []
        test_uncerts = []

        # Arrays for training results
        train_preds = []
        train_truths = []
        train_uncerts = []

        for train_idx, test_idx in kf_splits:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train = y[train_idx]

            # True values for test and training
            test_true_r1r2 = df_prepared.iloc[test_idx]['r1r2'].values
            train_true_r1r2 = df_prepared.iloc[train_idx]['r1r2'].values

            # Use default or optimized hyperparameters
            if best_params:
                model = XGBClassifier(
                    objective='multi:softprob',
                    eval_metric='mae',
                    random_state=self.random_state,
                    **best_params
                )
            else:
                model = XGBClassifier(
                    objective='multi:softprob',
                    eval_metric='mae',
                    random_state=self.random_state,
                    n_estimators=500,
                    max_depth=6,
                    learning_rate=0.01,
                    subsample=0.8,
                    colsample_bytree=0.8
                )

            model.fit(X_train, y_train)

            # Predictions for test data
            test_proba = model.predict_proba(X_test)

            # Predictions for training data
            train_proba = model.predict_proba(X_train)

            k = 4  # Top-k buckets

            # Calculate test predictions
            test_expected_r1r2 = []
            for i in range(test_proba.shape[0]):
                top_k = np.argsort(test_proba[i])[-k:]
                weights = test_proba[i][top_k]
                centers = self.bucket_centers[top_k]
                test_expected_r1r2.append(np.sum(weights * centers) / np.sum(weights))
            test_expected_r1r2 = np.array(test_expected_r1r2)

            # Calculate training predictions
            train_expected_r1r2 = []
            for i in range(train_proba.shape[0]):
                top_k = np.argsort(train_proba[i])[-k:]
                weights = train_proba[i][top_k]
                centers = self.bucket_centers[top_k]
                train_expected_r1r2.append(np.sum(weights * centers) / np.sum(weights))
            train_expected_r1r2 = np.array(train_expected_r1r2)

            # Uncertainties for test data
            test_uncertainties = np.array([
                np.sum(p * (self.bucket_centers - np.sum(p * self.bucket_centers)) ** 2)
                for p in test_proba
            ])

            # Uncertainties for training data
            train_uncertainties = np.array([
                np.sum(p * (self.bucket_centers - np.sum(p * self.bucket_centers)) ** 2)
                for p in train_proba
            ])

            # Collect results
            test_preds.extend(test_expected_r1r2)
            test_truths.extend(test_true_r1r2)
            test_uncerts.extend(np.sqrt(test_uncertainties))

            train_preds.extend(train_expected_r1r2)
            train_truths.extend(train_true_r1r2)
            train_uncerts.extend(np.sqrt(train_uncertainties))

        self.model = model  # last fold model for future use

        # Return test and training data
        return (
            np.array(test_truths), np.array(test_preds), np.array(test_uncerts),
            np.array(train_truths), np.array(train_preds), np.array(train_uncerts)
        )

    def evaluate(self, test_truths, test_preds, test_uncerts, train_truths, train_preds, train_uncerts):
        # Calculate metrics for test data
        test_rmse = np.sqrt(mean_squared_error(test_truths, test_preds))
        test_r2 = r2_score(test_truths, test_preds)
        test_avg_uncertainty = np.mean(test_uncerts)

        # Calculate metrics for training data
        train_rmse = np.sqrt(mean_squared_error(train_truths, train_preds))
        train_r2 = r2_score(train_truths, train_preds)
        train_avg_uncertainty = np.mean(train_uncerts)

        # Calculate absolute errors
        test_abs_errors = np.abs(test_truths - test_preds)

        # Calculate Pearson correlation for test data
        test_corr, _ = pearsonr(test_uncerts, test_abs_errors)

        # Fit linear regression for comparison (model calibration line)
        test_uncerts_reshaped = np.array(test_uncerts).reshape(-1, 1)
        test_abs_errors_array = np.array(test_abs_errors)
        reg = LinearRegression().fit(test_uncerts_reshaped, test_abs_errors_array)
        slope = reg.coef_[0]
        intercept = reg.intercept_

        # Prepare x values for reference line
        x_vals = np.linspace(0, 2, 100)
        y_ideal = x_vals  # Perfect calibration: y = x
        y_model = slope * x_vals + intercept  # Model linear calibration

        # Plot uncertainty vs. absolute error with quantile bands
        df_test = pd.DataFrame({"uncert": test_uncerts, "abs_err": test_abs_errors})
        df_test["bin"] = pd.qcut(df_test["uncert"], q=5, duplicates='drop')

        bin_means = df_test.groupby("bin", observed=True).mean()
        bin_q10 = df_test.groupby("bin", observed=True).quantile(0.1)
        bin_q90 = df_test.groupby("bin", observed=True).quantile(0.9)

        x_bin_means = bin_means["uncert"]
        y_bin_means = bin_means["abs_err"]
        y_q10 = bin_q10["abs_err"]
        y_q90 = bin_q90["abs_err"]

        plt.figure(figsize=(8, 6))
        plt.scatter(test_uncerts, test_abs_errors, alpha=0.5, label="Samples")
        plt.plot(x_vals, y_ideal, 'r--', label="Ideal Calibration (y = x)")
        plt.plot(x_vals, y_model, 'b-', label=f"Linear Fit (y = {slope:.2f}x + {intercept:.2f})")
        plt.plot(x_bin_means, y_bin_means, 'k-', label="Mean Abs Error (per bin)")
        plt.fill_between(x_bin_means, y_q10, y_q90, color='gray', alpha=0.3, label="10–90% Error Range")

        plt.xlabel("Uncertainty (Std Dev or Variance)")
        plt.ylabel("Absolute Error |True - Predicted|")
        plt.title(f"Uncertainty vs Absolute Error\nPearson r = {test_corr:.2f}")
        plt.grid(True)
        plt.xlim(0, 2)
        plt.ylim(0, 2)
        plt.legend()
        plt.tight_layout()
        plt.savefig("output/classification/uncertainty_vs_error.png")
        plt.close()

        # Parity plot: predicted vs. true r1r2 with distinction between test and train
        plt.figure(figsize=(8, 6))

        # Plot training data in blue with lower alpha for clarity
        plt.scatter(train_truths, train_preds, alpha=0.3, color='blue', label='Training data')

        # Plot test data in red
        plt.scatter(test_truths, test_preds, alpha=0.5, color='red', label='Test data')

        # Add perfect prediction line
        plt.plot([0, 2], [0, 2], 'k--', alpha=0.7, label='Perfect prediction')

        plt.xlabel("True r1r2")
        plt.ylabel("Predicted r1r2 (Expected from Distribution)")
        plt.title("Expected Value from Distribution vs Ground Truth")
        plt.grid(True)
        plt.xlim(0, 2)
        plt.ylim(0, 2)
        plt.legend()
        plt.tight_layout()
        plt.savefig("output/classification/distribution_vs_truth_train_test.png")
        plt.close()

        # Create a second parity plot showing only test data (for clarity)
        plt.figure(figsize=(8, 6))
        plt.scatter(test_truths, test_preds, alpha=0.5, color='red')
        plt.plot([0, 2], [0, 2], 'k--', alpha=0.7, label='Perfect prediction')
        plt.xlabel("True r1r2")
        plt.ylabel("Predicted r1r2 (Expected from Distribution)")
        plt.title("Expected Value from Distribution vs Ground Truth (Test Data Only)")
        plt.grid(True)
        plt.xlim(0, 2)
        plt.ylim(0, 2)
        plt.legend()
        plt.tight_layout()
        plt.savefig("output/classification/distribution_vs_truth_test_only.png")
        plt.close()

        # Add confusion matrix for bucket classification - simplified approach
        from sklearn.metrics import confusion_matrix
        import seaborn as sns

        # Convert continuous r1r2 values to bucket indices
        true_buckets = self._assign_buckets(test_truths)
        pred_buckets = self._assign_buckets(test_preds)

        # Get unique bucket values that are actually used
        unique_buckets = np.unique(np.concatenate([true_buckets, pred_buckets]))
        n_buckets = len(unique_buckets)

        # Create confusion matrix
        cm = confusion_matrix(true_buckets, pred_buckets, labels=np.sort(unique_buckets))

        # Normalize confusion matrix by row (true labels)
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_normalized[np.isnan(cm_normalized)] = 0  # Replace NaN with 0

        # Create labels for the confusion matrix
        bucket_labels = [f"{self.bucket_edges[i]:.2f}-{self.bucket_edges[i + 1]:.2f}" for i in np.sort(unique_buckets)]

        # Plot confusion matrix (normalized)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=bucket_labels, yticklabels=bucket_labels)
        plt.xlabel('Predicted Bucket')
        plt.ylabel('True Bucket')
        plt.title('Confusion Matrix (Normalized)')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig("output/classification/confusion_matrix.png")
        plt.close()

        # Plot raw counts as well
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=bucket_labels, yticklabels=bucket_labels)
        plt.xlabel('Predicted Bucket')
        plt.ylabel('True Bucket')
        plt.title('Confusion Matrix (Counts)')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig("output/classification/confusion_matrix_counts.png")
        plt.close()

        # Baseline RMSE using mean bucket center as constant prediction
        baseline_rmse = np.sqrt(np.mean((test_truths - np.mean(self.bucket_centers)) ** 2))
        print("Baseline RMSE (mean bucket center):", baseline_rmse)

        print(f"Test RMSE: {test_rmse:.4f}, Test R²: {test_r2:.4f}, Test Mean Uncertainty: {test_avg_uncertainty:.4f}")
        print(
            f"Train RMSE: {train_rmse:.4f}, Train R²: {train_r2:.4f}, Train Mean Uncertainty: {train_avg_uncertainty:.4f}")

        return {
            "test": {"rmse": test_rmse, "r2": test_r2, "uncertainty": test_avg_uncertainty},
            "train": {"rmse": train_rmse, "r2": train_r2, "uncertainty": train_avg_uncertainty}
        }

    def predict(self, df_new):
        X_new = self.transformer.transform(df_new[self.feature_names])
        proba = self.model.predict_proba(X_new)
        expected_r1r2 = np.sum(proba * self.bucket_centers, axis=1)

        print("proba shape:", proba.shape)
        print("bucket_centers shape:", self.bucket_centers.shape)

        import matplotlib.pyplot as plt

        for i in range(5):
            print(proba[i])
            plt.figure()

            plt.bar(range(len(self.bucket_centers)), proba[i])
            plt.title(f"Sample {i} - Bucket Probabilities")
            plt.xlabel("Bucket Index")
            plt.ylabel("Probability")
            plt.bar(range(len(self.bucket_centers)), proba[i])
            plt.title(f"Sample {i} - Bucket Probabilities")
            plt.xlabel("Bucket Index")
            plt.ylabel("Probability")
            plt.savefig(f"output/classification/prob/prob_dist_{i}.png")
            plt.savefig("output/classification/prob_dist.png")

        return expected_r1r2
