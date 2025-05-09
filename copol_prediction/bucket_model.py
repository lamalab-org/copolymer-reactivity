"""
Bucket-based classification model for the copolymerization prediction
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from copolextractor import utils as utils


class BucketClassifier:
    """Class for training and evaluating bucket classification models"""

    def __init__(self, n_buckets=100, model_type="xgboost", random_state=42,
                 bucket_transform='log', quantile_based=False):
        """
        Initialize the BucketClassifier
        Args:
            n_buckets: Number of buckets to divide the range of r1r2 values
            model_type: Type of model to train ('xgboost', 'random_forest')
            random_state: Seed for reproducibility
            bucket_transform: Type of transform for buckets ('log', 'sqrt', 'linear')
            quantile_based: Whether to use quantile-based bucketing instead of range-based
        """
        self.n_buckets = n_buckets
        self.model_type = model_type
        self.random_state = random_state
        self.bucket_transform = bucket_transform
        self.quantile_based = quantile_based
        self.best_params = None
        self.feature_names = None
        self.transformer = None
        self.final_model = None
        self.bucket_edges = None
        self.bucket_centers = None
        self.bucket_transform_func = None
        self.bucket_inverse_func = None

    def _create_bucket_transform_funcs(self, max_value):
        """Create transform and inverse transform functions for the buckets"""
        if self.bucket_transform == 'log':
            # Log transform: more granularity for smaller values
            self.bucket_transform_func = lambda x: np.log1p(x)
            self.bucket_inverse_func = lambda x: np.expm1(x)
        elif self.bucket_transform == 'sqrt':
            # Square root transform: moderate increase in granularity for smaller values
            self.bucket_transform_func = lambda x: np.sqrt(x)
            self.bucket_inverse_func = lambda x: np.square(x)
        else:
            # Linear transform: uniform buckets
            self.bucket_transform_func = lambda x: x
            self.bucket_inverse_func = lambda x: x

    def _create_bucket_edges(self, df):
        """
        Create bucket edges based on predefined values or the range of r1r2 values

        Args:
            df: DataFrame with 'r1r2' column

        Returns:
            array: The bucket edges
        """
        # Ensure r1r2 exists in the DataFrame
        if 'r1r2' not in df.columns:
            print("Error: 'r1r2' column not found in DataFrame")
            return None

        # Get the range of r1r2 values in the data
        min_value = df['r1r2'].min()
        max_value = df['r1r2'].max()

        print(f"Range of r1r2 values in data: {min_value} to {max_value}")

        # Use predefined bucket edges
        # These are the predefined category boundaries
        predefined_edges = [
            0.005,
            0.01, 0.02, 0.05, 0.1,
            0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.90, 0.95, 0.98, 0.99, 1.00, 1.05, 1.1, 1.5,
            2.00, 5.00, 15.00
        ]

        # Add 0 as the first edge and ensure the last edge covers all data
        self.bucket_edges = np.array([0.0] + predefined_edges)
        if max_value > self.bucket_edges[-1]:
            print(f"Data maximum ({max_value}) exceeds the highest predefined bucket edge ({self.bucket_edges[-1]})")
            print(f"Adding an additional bucket edge at {max_value * 1.1} to cover all data")
            self.bucket_edges = np.append(self.bucket_edges, max_value * 1.1)

        # Create bucket centers for visualization
        self.bucket_centers = (self.bucket_edges[:-1] + self.bucket_edges[1:]) / 2

        # Adjust number of buckets to match the predefined edges
        self.n_buckets = len(self.bucket_edges) - 1
        print(f"Using {self.n_buckets} predefined buckets instead of dynamically created ones")

        # Print bucket distribution
        print(f"First few bucket edges: {self.bucket_edges[:5]}")
        print(f"Last few bucket edges: {self.bucket_edges[-5:]}")

        return self.bucket_edges

    def predict_with_distribution(self, X_new):
        """
        Make predictions using probability distribution across all buckets

        Args:
            X_new: New feature data to predict on
        Returns:
            array: Predicted r1r2 values using probability distribution
        """
        if self.transformer is None or self.final_model is None or self.bucket_centers is None:
            print(
                "Error: Model not properly initialized. Run prepare_data, train_and_evaluate, and train_final_model first.")
            return None

        # Transform the input features
        X_new_transformed = self.transformer.transform(X_new)

        # Get probability distribution over all buckets
        proba_distribution = self.final_model.predict_proba(X_new_transformed)

        # Calculate expected value (weighted average)
        # For each sample, multiply each bucket center with its probability and sum
        expected_values = np.sum(proba_distribution * self.bucket_centers, axis=1)

        return expected_values


    def _assign_buckets(self, r1r2_values):
        """
        Assign each r1r2 value to a bucket

        Args:
            r1r2_values: Array of r1r2 values

        Returns:
            array: The bucket indices
        """
        if self.bucket_edges is None:
            print("Error: Bucket edges not created yet")
            return None

        # Use numpy's digitize to assign each value to a bucket
        bucket_indices = np.digitize(r1r2_values, self.bucket_edges) - 1

        # Ensure all values are within valid range
        bucket_indices = np.clip(bucket_indices, 0, self.n_buckets - 1)

        return bucket_indices

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
        #r1r2_high_count = len(df[df['r1r2'] > 100])
        #df = df[df['r1r2'] <= 100]
        #print(f"Dropped {r1r2_high_count} rows where r1r2 > 100")

        # Filter temperature > 150
        # temp_out_of_range_count = len(df[(df['temperature'] < 0) | (df['temperature'] > 150)])
        # df = df[(df['temperature'] >= 0) & (df['temperature'] <= 150)]
        # print(f"Dropped {temp_out_of_range_count} rows where temperature > 150")

        # Create bucket edges based on the range of r1r2 values
        self._create_bucket_edges(df)

        # Assign each r1r2 value to a bucket to create the target variable
        y_buckets = self._assign_buckets(df['r1r2'].values)
        df['bucket'] = y_buckets

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
                              'dipole_x_2', 'dipole_y_2', 'dipole_z_2', 'solvent_logp'] + embedding_features

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

        return X, y, actual_features, df

    def get_model_and_param_grid(self):
        """
        Get the model instance and parameter grid based on model_type
        Returns:
            tuple: (model, param_grid) - The model instance and parameter grid for hyperparameter tuning
        """
        if self.model_type == "xgboost":
            model = XGBClassifier(random_state=self.random_state)
            param_grid = {
                'n_estimators': [100, 300, 500],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'min_child_weight': [1, 3, 5],
                'gamma': [0, 0.1, 0.2],
                'objective': ['multi:softmax', 'multi:softprob'],
                'eval_metric': ['mlogloss', 'merror', 'accuracy']
            }
        elif self.model_type == "random_forest":
            model = RandomForestClassifier(random_state=self.random_state)
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt', 'log2']
            }
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        return model, param_grid

    def train_and_evaluate(self, X, y, df):
        """
        Train and evaluate the model using cross-validation
        Args:
            X: Feature matrix
            y: Target variable
            df: DataFrame with reaction IDs for proper train/test splitting
        Returns:
            tuple: (fold_scores, all_predictions) - Cross-validation scores and predictions
        """
        print(f"\nTraining {self.model_type} classification model with {self.n_buckets} buckets...")

        # Create grouped K-Fold splits
        n_splits = 5
        kf_splits = utils.create_grouped_kfold_splits(df, n_splits=n_splits, random_state=self.random_state)
        print(f"\nUsing {n_splits}-fold cross-validation with grouped splits (keeps flipped monomer pairs together)")

        # Get model and parameter grid
        model, param_grid = self.get_model_and_param_grid()

        # Storage for scores and predictions
        fold_scores = []
        all_y_true = []
        all_y_pred = []
        all_y_train_true = []
        all_y_train_pred = []
        all_test_indices = []
        all_train_indices = []
        all_models = []

        # Cross-validation loop
        for fold, (train_idx, test_idx) in enumerate(kf_splits, 1):
            print(f"Fold {fold}")

            # Store indices for later visualization
            all_train_indices.extend(train_idx)
            all_test_indices.extend(test_idx)

            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

            # Hyperparameter optimization
            random_search = RandomizedSearchCV(
                model, param_distributions=param_grid,
                n_iter=20, cv=3, verbose=0, random_state=self.random_state, n_jobs=-1
            )

            # For XGBoost, configure it to not print validation messages
            if self.model_type == "xgboost":
                random_search.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            else:
                random_search.fit(X_train, y_train)

            # Get best model
            best_model = random_search.best_estimator_
            all_models.append(best_model)

            # Store best parameters from the last fold
            self.best_params = random_search.best_params_

            # Print the best hyperparameters
            print("\nBest Hyperparameters:")
            for param, value in random_search.best_params_.items():
                print(f"  {param}: {value}")
            print(f"Best CV score: {random_search.best_score_:.4f}")

            # Make predictions
            y_pred_train = best_model.predict(X_train)
            y_pred_test = best_model.predict(X_test)

            # Store values for later analysis
            all_y_train_true.extend(y_train)
            all_y_train_pred.extend(y_pred_train)
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred_test)

            # Calculate metrics
            accuracy_train = accuracy_score(y_train, y_pred_train)
            accuracy_test = accuracy_score(y_test, y_pred_test)
            f1_test = f1_score(y_test, y_pred_test, average='weighted')
            print(f"Train Accuracy: {accuracy_train:.4f}")
            print(f"Test Accuracy: {accuracy_test:.4f}")
            print(f"Test F1 Score (weighted): {f1_test:.4f}")
            fold_scores.append((accuracy_train, accuracy_test, f1_test))

        # Calculate average scores
        avg_train_accuracy = np.mean([score[0] for score in fold_scores])
        avg_test_accuracy = np.mean([score[1] for score in fold_scores])
        avg_test_f1 = np.mean([score[2] for score in fold_scores])
        print(f"\nAverage Train Accuracy: {avg_train_accuracy:.4f}")
        print(f"\nAverage Test Accuracy: {avg_test_accuracy:.4f}")
        print(f"\nAverage Test F1 Score: {avg_test_f1:.4f}")



        # Convert bucket indices back to r1r2 values for evaluation
        def bucket_to_r1r2(bucket_indices):
            """Convert bucket indices to r1r2 values using bucket centers"""
            return np.array([self.bucket_centers[idx] for idx in bucket_indices])

        # Convert predicted and true bucket indices to r1r2 values
        r1r2_pred = bucket_to_r1r2(all_y_pred)
        r1r2_true = df.iloc[all_test_indices]['r1r2'].values

        # Calculate RMSE between predicted and true r1r2 values
        rmse = np.sqrt(np.mean((r1r2_pred - r1r2_true) ** 2))
        print(f"\nRMSE on r1r2 values: {rmse:.4f}")

        # Additionally: Evaluate predictions using probability distribution
        r1r2_pred_dist = []
        for fold, model in enumerate(all_models):
            # Get test data for this fold
            fold_test_idx = kf_splits[fold][1]
            X_fold_test = X[fold_test_idx]

            # Calculate probability distribution
            proba_dist = model.predict_proba(X_fold_test)

            # Calculate expected values (weighted average)
            expected_values = np.sum(proba_dist * self.bucket_centers, axis=1)
            r1r2_pred_dist.extend(expected_values)

        # Calculate RMSE for distribution-based predictions
        rmse_dist = np.sqrt(np.mean((np.array(r1r2_pred_dist) - r1r2_true) ** 2))
        print(f"\nRMSE with distribution-based prediction: {rmse_dist:.4f}")

        # Package all predictions and indices for later analysis and visualization
        all_predictions = {
            'test_true_buckets': all_y_true,
            'test_pred_buckets': all_y_pred,
            'train_true_buckets': all_y_train_true,
            'train_pred_buckets': all_y_train_pred,
            'test_indices': all_test_indices,
            'train_indices': all_train_indices,
            'avg_test_accuracy': avg_test_accuracy,
            'avg_train_accuracy': avg_train_accuracy,
            'avg_test_f1': avg_test_f1,
            'fold_scores': fold_scores,
            'r1r2_rmse': rmse,
            'bucket_edges': self.bucket_edges,
            'bucket_centers': self.bucket_centers,
            'r1r2_pred_dist': r1r2_pred_dist,
            'r1r2_rmse_dist': rmse_dist
        }

        return fold_scores, all_predictions

    # 2. Create a new method for plotting distribution-based predictions
    def plot_distribution_predictions(self, predictions, df, title="Distribution-based r1r2 Predictions",
                                      save_path=None):
        """
        Plot distribution-based predicted vs true r1r2 values
        Args:
            predictions: Dictionary with prediction results
            df: DataFrame with original r1r2 values
            title: Plot title
            save_path: Where to save the plot
        """
        # Check if distribution-based predictions are available
        if 'r1r2_pred_dist' not in predictions:
            print("No distribution-based predictions available.")
            return

        # Get test indices
        test_indices = predictions['test_indices']

        # Get true r1r2 values
        true_r1r2 = df.iloc[test_indices]['r1r2'].values

        # Get distribution-based predictions
        pred_r1r2 = predictions['r1r2_pred_dist']

        # Calculate R²
        from sklearn.metrics import r2_score
        r2 = r2_score(true_r1r2, pred_r1r2)

        # Create figure
        plt.figure(figsize=(10, 8))

        # Define plot limits
        max_val = max(max(true_r1r2), max(pred_r1r2)) * 1.1
        min_val = 0

        # Plot
        plt.scatter(true_r1r2, pred_r1r2, alpha=0.6)
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')

        # Add labels, title, and R² text
        plt.xlabel('True r1r2')
        plt.ylabel('Predicted r1r2 (Probability Distribution)')
        plt.title(f"{title} (R² = {r2:.4f})")

        # Set limits
        plt.xlim(min_val, 10)
        plt.ylim(min_val, 10)

        # Add grid
        plt.grid(True, linestyle='--', alpha=0.7)

        # Save if path is provided
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
        else:
            plt.tight_layout()
            plt.show()

    def compare_prediction_methods(self, predictions, df, save_path=None):
        """
        Compare hard classification vs distribution-based prediction
        Args:
            predictions: Dictionary with prediction results
            df: DataFrame with original r1r2 values
            save_path: Where to save the plot
        """
        # Check if both predictions are available
        if 'r1r2_pred_dist' not in predictions:
            print("Distribution-based predictions not available.")
            return

        # Get test indices and true values
        test_indices = predictions['test_indices']
        true_r1r2 = df.iloc[test_indices]['r1r2'].values

        # Get predictions
        pred_buckets = predictions['test_pred_buckets']
        pred_r1r2_hard = np.array([self.bucket_centers[idx] for idx in pred_buckets])
        pred_r1r2_dist = predictions['r1r2_pred_dist']

        # Calculate metrics
        from sklearn.metrics import r2_score, mean_squared_error

        r2_hard = r2_score(true_r1r2, pred_r1r2_hard)
        rmse_hard = np.sqrt(mean_squared_error(true_r1r2, pred_r1r2_hard))

        r2_dist = r2_score(true_r1r2, np.array(pred_r1r2_dist))
        rmse_dist = np.sqrt(mean_squared_error(true_r1r2, np.array(pred_r1r2_dist)))

        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Scatter plot for hard prediction
        ax = axes[0, 0]
        ax.scatter(true_r1r2, pred_r1r2_hard, alpha=0.6)
        max_val = max(true_r1r2.max(), pred_r1r2_hard.max()) * 1.1
        ax.plot([0, max_val], [0, max_val], 'r--')
        ax.set_xlabel('True r1r2')
        ax.set_ylabel('Predicted r1r2')
        ax.set_title(f"Hard Prediction\nR² = {r2_hard:.4f}, RMSE = {rmse_hard:.4f}")
        ax.grid(True, linestyle='--', alpha=0.7)

        # 2. Scatter plot for distribution prediction
        ax = axes[0, 1]
        ax.scatter(true_r1r2, pred_r1r2_dist, alpha=0.6)
        max_val = max(true_r1r2.max(), np.array(pred_r1r2_dist).max()) * 1.1
        ax.plot([0, max_val], [0, max_val], 'r--')
        ax.set_xlabel('True r1r2')
        ax.set_ylabel('Predicted r1r2 (Distribution)')
        ax.set_title(f"Distribution Prediction\nR² = {r2_dist:.4f}, RMSE = {rmse_dist:.4f}")
        ax.grid(True, linestyle='--', alpha=0.7)

        # 3. Error distribution for hard prediction
        ax = axes[1, 0]
        abs_errors_hard = np.abs(true_r1r2 - pred_r1r2_hard)
        ax.hist(abs_errors_hard, bins=30, alpha=0.7)
        ax.set_xlabel('Absolute Error')
        ax.set_ylabel('Frequency')
        ax.set_title(f"Hard Prediction Error Distribution\nMean Error = {abs_errors_hard.mean():.4f}")
        ax.grid(True, linestyle='--', alpha=0.7)

        # 4. Error distribution for distribution prediction
        ax = axes[1, 1]
        abs_errors_dist = np.abs(true_r1r2 - np.array(pred_r1r2_dist))
        ax.hist(abs_errors_dist, bins=30, alpha=0.7)
        ax.set_xlabel('Absolute Error')
        ax.set_ylabel('Frequency')
        ax.set_title(f"Distribution Prediction Error Distribution\nMean Error = {abs_errors_dist.mean():.4f}")
        ax.grid(True, linestyle='--', alpha=0.7)

        # Add overall title
        plt.suptitle('Comparison of Prediction Methods', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        # Save and close
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

        # Print comparison summary
        print("\n=== Prediction Methods Comparison ===")
        print(f"Hard Prediction RMSE: {rmse_hard:.4f}, R²: {r2_hard:.4f}")
        print(f"Distribution Prediction RMSE: {rmse_dist:.4f}, R²: {r2_dist:.4f}")

        improvement = (rmse_hard - rmse_dist) / rmse_hard * 100
        print(f"RMSE Improvement: {improvement:.2f}%")

    def train_final_model(self, X, y):
        """
        Train a final model on all data using the best parameters found during cross-validation
        Args:
            X: Feature matrix
            y: Target variable
        Returns:
            model: The trained model
        """
        if self.best_params is None:
            print("Error: No best parameters found. Run train_and_evaluate first.")
            return None

        print("\nTraining final classification model on all data...")

        # Create and train the final model with the best parameters
        if self.model_type == "xgboost":
            final_model = XGBClassifier(**self.best_params, random_state=self.random_state)
        elif self.model_type == "random_forest":
            final_model = RandomForestClassifier(**self.best_params, random_state=self.random_state)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        final_model.fit(X, y)
        print("Final model training completed.")

        # Store the final model
        self.final_model = final_model

        return final_model

    def get_feature_importances(self, model):
        """
        Get feature importances from the model
        Args:
            model: The trained model
        Returns:
            DataFrame: Feature importances sorted in descending order
        """
        if self.feature_names is None:
            print("Error: No feature names found. Run prepare_data first.")
            return None

        # Get feature importances if the model supports it
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_

            # Create DataFrame for feature importances
            importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importances
            })

            # Sort by importance
            importance_df = importance_df.sort_values('Importance', ascending=False)
            return importance_df
        else:
            print(f"Model type {self.model_type} does not support feature importances.")
            return None

    def predict(self, X_new, use_distribution=False):
        """
        Make predictions on new data
        Args:
            X_new: New feature data to predict on
            use_distribution: Whether to use probability distribution for prediction
        Returns:
            array: Predicted r1r2 values
        """
        if self.transformer is None or self.final_model is None or self.bucket_centers is None:
            print(
                "Error: Model not properly initialized. Run prepare_data, train_and_evaluate, and train_final_model first.")
            return None

        if use_distribution:
            # Use probability distribution for prediction
            return self.predict_with_distribution(X_new)
        else:
            # Traditional single-class prediction
            # Transform the input features
            X_new_transformed = self.transformer.transform(X_new)

            # Predict bucket indices
            bucket_indices = self.final_model.predict(X_new_transformed)

            # Convert bucket indices to r1r2 values using bucket centers
            r1r2_pred = np.array([self.bucket_centers[idx] for idx in bucket_indices])

            return r1r2_pred

    def plot_confusion_matrix(self, predictions, title="Confusion Matrix", save_path=None, max_buckets_to_show=20):
        """
        Plot confusion matrix for classification results
        Args:
            predictions: Dictionary with prediction results
            title: Plot title
            save_path: Where to save the plot
            max_buckets_to_show: Maximum number of buckets to include in the plot
        """
        y_true = predictions['test_true_buckets']
        y_pred = predictions['test_pred_buckets']

        # Find the range of buckets actually used
        min_bucket = min(min(y_true), min(y_pred))
        max_bucket = max(max(y_true), max(y_pred))

        # Limit the number of buckets to show
        if max_bucket - min_bucket >= max_buckets_to_show:
            # Split into evenly spaced buckets across the range
            bucket_step = max(1, (max_bucket - min_bucket) // max_buckets_to_show)
            selected_buckets = list(range(min_bucket, max_bucket + 1, bucket_step))

            # Filter data to only include these buckets
            mask_true = np.isin(y_true, selected_buckets)
            mask_pred = np.isin(y_pred, selected_buckets)
            mask = mask_true & mask_pred

            y_true_filtered = np.array(y_true)[mask]
            y_pred_filtered = np.array(y_pred)[mask]

            # Create labels for the confusion matrix
            bucket_labels = [f"{i}\n({self.bucket_centers[i]:.2f})" for i in selected_buckets]
        else:
            # Use all buckets if there are few enough
            y_true_filtered = y_true
            y_pred_filtered = y_pred
            bucket_labels = [f"{i}\n({self.bucket_centers[i]:.2f})" for i in range(min_bucket, max_bucket + 1)]

        # Compute confusion matrix
        cm = confusion_matrix(y_true_filtered, y_pred_filtered,
                              labels=range(min_bucket, max_bucket + 1,
                                           bucket_step if max_bucket - min_bucket >= max_buckets_to_show else 1))

        # Create figure
        plt.figure(figsize=(12, 10))
        ax = plt.subplot()

        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues', cbar=True, square=True)

        # Set labels
        ax.set_xlabel('Predicted Bucket')
        ax.set_ylabel('True Bucket')
        ax.set_title(title)

        # Set x and y tick labels
        ax.set_xticks(np.arange(len(bucket_labels)) + 0.5)
        ax.set_yticks(np.arange(len(bucket_labels)) + 0.5)
        ax.set_xticklabels(bucket_labels, rotation=90)
        ax.set_yticklabels(bucket_labels, rotation=0)

        # Save if path is provided
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
        else:
            plt.tight_layout()
            plt.show()

    def plot_bucket_distribution(self, df, save_path=None):
        """
        Plot the distribution of samples across buckets
        Args:
            df: DataFrame with 'bucket' column
            save_path: Where to save the plot
        """
        plt.figure(figsize=(15, 6))

        # Count samples in each bucket
        bucket_counts = df['bucket'].value_counts().sort_index()

        # Create bucket labels with r1r2 ranges
        bucket_labels = [f"{i}\n({self.bucket_edges[i]:.2f}-{self.bucket_edges[i + 1]:.2f})"
                         for i in bucket_counts.index]

        # Plot
        ax = sns.barplot(x=bucket_counts.index, y=bucket_counts.values, palette='viridis')

        # Add labels and title
        plt.xlabel('Bucket Index (r1r2 range)')
        plt.ylabel('Number of Samples')
        plt.title('Distribution of Samples Across r1r2 Buckets')

        # Rotate x-axis labels for better readability
        if len(bucket_counts) > 20:
            # If too many buckets, only show a subset of labels
            n_ticks = 20
            step = max(1, len(bucket_counts) // n_ticks)
            plt.xticks(np.arange(0, len(bucket_counts), step),
                       [bucket_labels[i] for i in range(0, len(bucket_counts), step)],
                       rotation=90)
        else:
            plt.xticks(range(len(bucket_counts)), bucket_labels, rotation=90)

        # Add grid for better readability
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Save if path is provided
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
        else:
            plt.tight_layout()
            plt.show()

    def plot_r1r2_predictions(self, predictions, df, title="r1r2 Predictions", save_path=None):
        """
        Plot predicted vs true r1r2 values
        Args:
            predictions: Dictionary with prediction results
            df: DataFrame with original r1r2 values
            title: Plot title
            save_path: Where to save the plot
        """
        # Get test indices and predicted bucket indices
        test_indices = predictions['test_indices']
        pred_buckets = predictions['test_pred_buckets']

        # Get true r1r2 values
        true_r1r2 = df.iloc[test_indices]['r1r2'].values

        # Convert predicted bucket indices to r1r2 values
        pred_r1r2 = np.array([self.bucket_centers[idx] for idx in pred_buckets])

        # Calculate R²
        from sklearn.metrics import r2_score
        r2 = r2_score(true_r1r2, pred_r1r2)

        # Create figure
        plt.figure(figsize=(10, 8))

        # Define plot limits
        max_val = max(max(true_r1r2), max(pred_r1r2)) * 1.1
        min_val = 0

        # Plot
        plt.scatter(true_r1r2, pred_r1r2, alpha=0.6)
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')

        # Add labels, title, and R² text
        plt.xlabel('True r1r2')
        plt.ylabel('Predicted r1r2')
        plt.title(f"{title} (R² = {r2:.4f})")

        # Set limits
        plt.xlim(min_val, 10)
        plt.ylim(min_val, 10)

        # Add grid
        plt.grid(True, linestyle='--', alpha=0.7)

        # Save if path is provided
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
        else:
            plt.tight_layout()
            plt.show()

    def plot_error_analysis(self, predictions, df, save_path=None):
        """
        Plot error analysis of r1r2 predictions
        Args:
            predictions: Dictionary with prediction results
            df: DataFrame with original r1r2 values
            save_path: Where to save the plot
        """
        # Get test indices and predicted bucket indices
        test_indices = predictions['test_indices']
        pred_buckets = predictions['test_pred_buckets']

        # Get true r1r2 values
        true_r1r2 = df.iloc[test_indices]['r1r2'].values

        # Convert predicted bucket indices to r1r2 values
        pred_r1r2 = np.array([self.bucket_centers[idx] for idx in pred_buckets])

        # Calculate errors
        abs_errors = np.abs(true_r1r2 - pred_r1r2)
        rel_errors = abs_errors / np.maximum(true_r1r2, 0.0001) * 100

        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Absolute errors vs true r1r2
        ax1.scatter(true_r1r2, abs_errors, alpha=0.6)
        ax1.set_xlabel('True r1r2')
        ax1.set_ylabel('Absolute Error')
        ax1.set_title('Absolute Error vs True r1r2')
        ax1.grid(True, linestyle='--', alpha=0.7)

        # Plot 2: Relative errors vs true r1r2
        ax2.scatter(true_r1r2, rel_errors, alpha=0.6)
        ax2.set_xlabel('True r1r2')
        ax2.set_ylabel('Relative Error (%)')
        ax2.set_title('Relative Error vs True r1r2')
        ax2.grid(True, linestyle='--', alpha=0.7)

        # Limit y-axis for relative errors to avoid extreme values
        ax2.set_ylim(0, min(500, np.percentile(rel_errors, 99)))

        # Add overall title
        plt.suptitle('Error Analysis of Bucket Classification Predictions', fontsize=16)

        # Save if path is provided
        if save_path:
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(save_path)
            plt.close()
        else:
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()