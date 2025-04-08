"""
Model module for the copolymerization prediction model

Contains functions for modeling, training, and evaluating different models
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from copolextractor import utils as utils


class ModelTrainer:
    """Class for training and evaluating different models"""

    def __init__(self, model_type="xgboost", random_state=42):
        """
        Initialize the ModelTrainer

        Args:
            model_type: Type of model to train ('xgboost', 'random_forest', 'svr')
            random_state: Seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.best_params = None
        self.feature_names = None
        self.transformer = None
        self.target_transformer = None

    def prepare_data(self, df):
        """
        Prepare data for model training

        Filters data, creates target variable, and extracts features

        Args:
            df: The DataFrame to process

        Returns:
            tuple: (X, y, actual_features, df) - Transformed features, target variable, feature names, and filtered DataFrame
        """
        print("\nPreparing data for modeling...")

        # Check if df is actually a DataFrame
        import pandas as pd
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
        df['r1r2'] = df['constant_1'] * df['constant_2']

        # Remove rows where r1r2 is less than 0
        original_count = len(df)
        df = df[df['r1r2'] >= 0]
        print(f"Dropped {original_count - len(df)} rows where r1r2 < 0")

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
        X = self.transformer.fit_transform(df)

        # Transform target variable
        self.target_transformer = PowerTransformer(method='yeo-johnson')
        y = self.target_transformer.fit_transform(df['r1r2'].values.reshape(-1, 1)).ravel()

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
            model = XGBRegressor(random_state=self.random_state)
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
        elif self.model_type == "random_forest":
            model = RandomForestRegressor(random_state=self.random_state)
            param_grid = {
                'n_estimators': [100, 300, 500, 1000],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt', 'log2']
            }
        elif self.model_type == "svr":
            model = SVR()
            param_grid = {
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'C': np.logspace(-3, 3, 10),
                'gamma': ['scale', 'auto'] + list(np.logspace(-3, 3, 10)),
                'epsilon': [0.01, 0.05, 0.1, 0.2]
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
        print(f"\nTraining {self.model_type} model...")

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
                n_iter=10, cv=3, scoring='r2', verbose=0, random_state=self.random_state, n_jobs=-1
            )
            random_search.fit(X_train, y_train, eval_set=[(X_test, y_test)] if self.model_type == "xgboost" else None)

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
            r2_train = r2_score(y_train, y_pred_train)
            r2_test = r2_score(y_test, y_pred_test)

            print(f"Train R2: {r2_train:.4f}")
            print(f"Test R2: {r2_test:.4f}")

            fold_scores.append((r2_train, r2_test))

        # Calculate average scores
        avg_train_r2 = np.mean([score[0] for score in fold_scores])
        avg_test_r2 = np.mean([score[1] for score in fold_scores])

        print(f"\nAverage Train R2: {avg_train_r2:.4f}")
        print(f"\nAverage Test R2: {avg_test_r2:.4f}")

        # Transform values back to original scale for better interpretability
        all_y_true_inv = self.target_transformer.inverse_transform(np.array(all_y_true).reshape(-1, 1)).ravel()
        all_y_pred_inv = self.target_transformer.inverse_transform(np.array(all_y_pred).reshape(-1, 1)).ravel()
        all_y_train_true_inv = self.target_transformer.inverse_transform(
            np.array(all_y_train_true).reshape(-1, 1)).ravel()
        all_y_train_pred_inv = self.target_transformer.inverse_transform(
            np.array(all_y_train_pred).reshape(-1, 1)).ravel()

        # Package all predictions and indices for later analysis and visualization
        all_predictions = {
            'test_true': all_y_true_inv,
            'test_pred': all_y_pred_inv,
            'train_true': all_y_train_true_inv,
            'train_pred': all_y_train_pred_inv,
            'test_indices': all_test_indices,
            'train_indices': all_train_indices,
            'avg_test_r2': avg_test_r2,
            'avg_train_r2': avg_train_r2,
            'fold_scores': fold_scores
        }

        return fold_scores, all_predictions

    def train_final_model(self, X, y):
        """
        Train a final model on all data using the best parameters found during cross-validation

        Args:
            X: Feature matrix
            y: Target variable

        Returns:
            model: The trained model
        """
        if self.best_params is None:  # Changed from best_params_ to best_params
            print("Error: No best parameters found. Run train_and_evaluate first.")
            return None

        print("\nTraining final model on all data...")

        # Create and train the final model with the best parameters
        if self.model_type == "xgboost":
            final_model = XGBRegressor(**self.best_params, random_state=self.random_state, scoring='r2')  # Changed here too
        elif self.model_type == "random_forest":
            final_model = RandomForestRegressor(**self.best_params, random_state=self.random_state)  # And here
        elif self.model_type == "svr":
            final_model = SVR(**self.best_params)  # And here
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        final_model.fit(X, y)
        print("Final model training completed.")

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

    def predict(self, X_new):
        """
        Make predictions on new data

        Args:
            X_new: New feature data to predict on

        Returns:
            array: Predictions in original scale
        """
        if self.transformer is None or self.target_transformer is None:
            print("Error: Model not properly initialized. Run prepare_data and train_and_evaluate first.")
            return None

        # Transform the input features
        X_new_transformed = self.transformer.transform(X_new)

        # Predict using the final model
        y_pred = self.final_model.predict(X_new_transformed)

        # Transform predictions back to original scale
        y_pred_orig = self.target_transformer.inverse_transform(y_pred.reshape(-1, 1)).ravel()

        return y_pred_orig

    def evaluate_learning_curve(self, dataframe, model_type=None, random_state=None):
        """
        Evaluate model performance with different dataset sizes

        Args:
            dataframe: DataFrame with features and target
            model_type: Override the model type (optional)
            random_state: Override the random state (optional)
        """
        if model_type is None:
            model_type = self.model_type

        if random_state is None:
            random_state = self.random_state

        print(f"\n=== Learning Curve Analysis ({model_type}) ===")

        # Prepare the data using the same instance (self)
        X, y, features, filtered_df = self.prepare_data(
            dataframe)  # Use a different parameter name to avoid confusion

        if X is None or y is None:
            print("Error preparing data for learning curve analysis.")
            return None

        # Define the sample sizes to test
        # Ensure the sizes don't exceed the actual data size
        total_size = len(filtered_df)
        sample_sizes = [
            min(100, total_size),
            min(250, total_size),
            min(500, total_size),
            min(1000, total_size),
            min(2000, total_size),
            total_size
        ]

        # Remove duplicates and ensure the list is sorted
        sample_sizes = sorted(list(set(sample_sizes)))
        print(f"Testing model with sample sizes: {sample_sizes}")

        # Results storage
        results = {
            'sample_size': [],
            'train_r2': [],
            'test_r2': [],
            'train_std': [],
            'test_std': []
        }

        # For each sample size
        for size in sample_sizes:
            print(f"\nEvaluating model with {size} samples")

            # Sample by reaction ID to ensure proper grouping
            if size < total_size:
                # Sample unique reaction IDs
                unique_ids = filtered_df['reaction_id'].unique()
                np.random.seed(random_state)  # For reproducibility

                # Estimate how many reaction IDs we need to get approximately 'size' rows
                avg_rows_per_id = len(filtered_df) / len(unique_ids)
                n_ids_needed = min(int(size / avg_rows_per_id * 1.1), len(unique_ids))  # Add 10% buffer

                # Sample the reaction IDs
                sampled_ids = np.random.choice(unique_ids, size=n_ids_needed, replace=False)

                # Get rows for these reaction IDs
                sample_mask = filtered_df['reaction_id'].isin(sampled_ids)

                # Get subset of X and y
                X_sample = X[sample_mask]
                y_sample = y[sample_mask]
                df_sample = filtered_df[sample_mask]

                # If we have too many rows, randomly subsample
                if len(X_sample) > size:
                    indices = np.random.choice(len(X_sample), size=size, replace=False)
                    X_sample = X_sample[indices]
                    y_sample = y_sample[indices]
                    df_sample = df_sample.iloc[indices]
            else:
                X_sample = X
                y_sample = y
                df_sample = filtered_df

            print(f"Actual sample size: {len(X_sample)}")

            # Create grouped k-fold splits for cross-validation
            n_splits = 5
            fold_train_scores = []
            fold_test_scores = []

            # Get model with fixed parameters for speed
            if model_type == "xgboost":
                model = XGBRegressor(
                    n_estimators=200,  # Reduced for speed
                    learning_rate=0.05,
                    max_depth=5,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=random_state
                )
            elif model_type == "random_forest":
                model = RandomForestRegressor(
                    n_estimators=100,  # Reduced for speed
                    max_depth=10,
                    random_state=random_state
                )
            elif model_type == "svr":
                model = SVR(kernel='rbf', C=1.0, gamma='scale')
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            # Create grouped k-fold splits
            grouped_kf = utils.create_grouped_kfold_splits(df_sample, n_splits=n_splits,
                                                           random_state=random_state, id_column='reaction_id')

            for fold, (train_idx, test_idx) in enumerate(grouped_kf, 1):
                # Ensure we have enough data in both train and test sets
                if len(train_idx) < 10 or len(test_idx) < 10:
                    print(f"Warning: Fold {fold} has insufficient data, skipping")
                    continue

                # Split data
                X_train, X_test = X_sample[train_idx], X_sample[test_idx]
                y_train, y_test = y_sample[train_idx], y_sample[test_idx]

                print(f"Fold {fold}: Train size = {len(X_train)}, Test size = {len(X_test)}")

                # Fit model
                model.fit(X_train, y_train)

                # Make predictions
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)

                # Calculate metrics
                r2_train = r2_score(y_train, y_pred_train)
                r2_test = r2_score(y_test, y_pred_test)

                fold_train_scores.append(r2_train)
                fold_test_scores.append(r2_test)

                print(f"  Fold {fold} Train R²: {r2_train:.4f}")
                print(f"  Fold {fold} Test R²: {r2_test:.4f}")

            # Skip this sample size if we couldn't compute any valid scores
            if not fold_train_scores or not fold_test_scores:
                print(f"Skipping sample size {size} - insufficient data for valid cross-validation")
                continue

            # Average and std of scores
            avg_train_r2 = np.mean(fold_train_scores)
            avg_test_r2 = np.mean(fold_test_scores)
            std_train_r2 = np.std(fold_train_scores)
            std_test_r2 = np.std(fold_test_scores)

            print(f"Sample size: {size}")
            print(f"Average Train R²: {avg_train_r2:.4f} ± {std_train_r2:.4f}")
            print(f"Average Test R²: {avg_test_r2:.4f} ± {std_test_r2:.4f}")

            # Store results
            results['sample_size'].append(size)
            results['train_r2'].append(avg_train_r2)
            results['test_r2'].append(avg_test_r2)
            results['train_std'].append(std_train_r2)
            results['test_std'].append(std_test_r2)

        # Create a DataFrame with the results
        results_df = pd.DataFrame(results)

        # Save results to CSV
        results_df.to_csv(f'learning_curve_{model_type}_results.csv', index=False)

        print(f"\nLearning curve analysis completed and saved to learning_curve_{model_type}_results.csv")
        return results_df