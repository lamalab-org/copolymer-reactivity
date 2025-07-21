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
from sklearn.multioutput import MultiOutputRegressor

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
        self.final_model = None

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

        # Filter r1r2 > 100
        r1r2_high_count = len(df[df['r1r2'] > 10])
        df = df[df['r1r2'] <= 5]
        print(f"Dropped {r1r2_high_count} rows where r1r2 > 10")

        # Filter temperature > 150
        #temp_out_of_range_count = len(df[(df['temperature'] < 0) | (df['temperature'] > 150)])
        #df = df[(df['temperature'] >= 0) & (df['temperature'] <= 150)]
        #print(f"Dropped {temp_out_of_range_count} rows where temperature > 150")

        # Filter logp > 2.5
        #logp_high_count = len(df[df['solvent_logp'] > 2.5])
        #df = df[df['solvent_logp'] <= 2.5]
        #print(f"Dropped {logp_high_count} rows where solvent_logp > 2.5")

        #total_removed = r1r2_high_count + temp_high_count + logp_high_count
        #print(f"\nTotal removed by new filters: {total_removed} rows")

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
                              'dipole_x_2', 'dipole_y_2', 'dipole_z_2', 'solvent_logp', "delta_HOMO_LUMO_AA",
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
                'n_estimators': [100, 500,],
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
        if self.best_params is None:
            print("Error: No best parameters found. Run train_and_evaluate first.")
            return None

        print("\nTraining final model on all data...")

        # Create and train the final model with the best parameters
        if self.model_type == "xgboost":
            final_model = XGBRegressor(**self.best_params, random_state=self.random_state)
        elif self.model_type == "random_forest":
            final_model = RandomForestRegressor(**self.best_params, random_state=self.random_state)
        elif self.model_type == "svr":
            final_model = SVR(**self.best_params)
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

    def predict(self, X_new):
        """
        Make predictions on new data
        Args:
            X_new: New feature data to predict on
        Returns:
            array: Predictions in original scale
        """
        if self.transformer is None or self.target_transformer is None or self.final_model is None:
            print(
                "Error: Model not properly initialized. Run prepare_data, train_and_evaluate, and train_final_model first.")
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
        Returns:
            DataFrame: Results of the learning curve analysis
        """
        if model_type is None:
            model_type = self.model_type
        if random_state is None:
            random_state = self.random_state

        print(f"\n=== Learning Curve Analysis ({model_type}) ===")

        # Prepare the data using the same instance (self)
        X, y, features, filtered_df = self.prepare_data(dataframe)
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


class DualDistanceModelTrainer(ModelTrainer):
    """Extended ModelTrainer for dual distance prediction approach

    This class inherits from ModelTrainer and uses its prepare_data method
    to handle feature preparation and transformation.
    """

    def __init__(self, model_type="xgboost", random_state=42):
        # Initialize parent class which handles all base functionality
        super().__init__(model_type, random_state)
        # Additional attributes for dual distance approach
        self.model_distance_to_0 = None
        self.model_distance_to_1 = None
        self.best_params_distance_0 = None
        self.best_params_distance_1 = None

    def prepare_dual_targets(self, y_original):
        """
        Prepare two target variables: distance to 0 and distance to 1
        Args:
            y_original: Original target values (r1r2 product)
        Returns:
            tuple: (y_distance_to_0, y_distance_to_1)
        """
        # Calculate distances
        y_distance_to_0 = np.abs(y_original - 0)
        y_distance_to_1 = np.abs(y_original - 1)

        return y_distance_to_0, y_distance_to_1

    def reconstruct_prediction(self, pred_dist_0, pred_dist_1):
        """
        Reconstruct the original prediction from the two distance predictions
        Args:
            pred_dist_0: Predicted distance to 0
            pred_dist_1: Predicted distance to 1
        Returns:
            tuple: (predicted_value, uncertainty_score)
        """
        # The predicted value should satisfy:
        # |pred - 0| ≈ pred_dist_0
        # |pred - 1| ≈ pred_dist_1

        # If both distances are consistent, we can solve for the prediction
        # Case 1: pred < 0.5 (closer to 0)
        # pred = pred_dist_0
        # 1 - pred = pred_dist_1
        # => pred = pred_dist_0, check: 1 - pred_dist_0 ≈ pred_dist_1

        # Case 2: pred >= 0.5 (closer to 1)
        # pred = -pred_dist_0 (not valid if pred should be positive)
        # pred = 1 - pred_dist_1
        # => pred = 1 - pred_dist_1, check: pred ≈ pred_dist_0

        # More robust approach: weighted average based on which anchor point is closer
        pred_from_0 = pred_dist_0
        pred_from_1 = 1 - pred_dist_1

        # Calculate uncertainty as the disagreement between the two predictions
        uncertainty = np.abs(pred_from_0 - pred_from_1)

        # Use weighted average based on predicted distances
        # If closer to 0, trust pred_from_0 more; if closer to 1, trust pred_from_1 more
        weight_0 = 1 / (pred_dist_0 + 1e-6)
        weight_1 = 1 / (pred_dist_1 + 1e-6)

        predicted_value = (weight_0 * pred_from_0 + weight_1 * pred_from_1) / (weight_0 + weight_1)

        # Ensure prediction is non-negative
        predicted_value = np.maximum(predicted_value, 0)

        return predicted_value, uncertainty

    def train_dual_distance_models(self, X, y, df):
        """
        Train two separate models for distance to 0 and distance to 1
        Args:
            X: Feature matrix
            y: Target variable (transformed)
            df: DataFrame with reaction IDs for proper train/test splitting
        Returns:
            dict: Results containing predictions and metrics
        """
        print("\n=== Training Dual Distance XGBoost Models ===")

        # Transform y back to original scale for distance calculation
        y_original = self.target_transformer.inverse_transform(y.reshape(-1, 1)).ravel()

        # Prepare dual targets
        y_dist_0, y_dist_1 = self.prepare_dual_targets(y_original)

        # Create grouped K-Fold splits
        n_splits = 5
        kf_splits = utils.create_grouped_kfold_splits(df, n_splits=n_splits, random_state=self.random_state)

        # Storage for results
        all_test_true = []
        all_test_pred = []
        all_test_uncertainty = []
        all_train_true = []
        all_train_pred = []
        all_train_uncertainty = []
        all_test_indices = []
        all_train_indices = []

        fold_scores = []

        # Get parameter grid for XGBoost
        _, param_grid = self.get_model_and_param_grid()

        print("\n--- Training Model for Distance to 0 ---")
        for fold, (train_idx, test_idx) in enumerate(kf_splits, 1):
            print(f"\nFold {fold}")

            # Store indices
            if fold == 1:  # Only store once
                all_train_indices.extend(train_idx)
                all_test_indices.extend(test_idx)

            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train_dist_0 = y_dist_0[train_idx]
            y_test_dist_0 = y_dist_0[test_idx]
            y_train_dist_1 = y_dist_1[train_idx]
            y_test_dist_1 = y_dist_1[test_idx]
            y_train_original = y_original[train_idx]
            y_test_original = y_original[test_idx]

            # Train model for distance to 0
            print("Training distance-to-0 model...")
            model_0 = XGBRegressor(random_state=self.random_state)
            random_search_0 = RandomizedSearchCV(
                model_0, param_distributions=param_grid,
                n_iter=10, cv=3, scoring='neg_mean_squared_error',
                verbose=0, random_state=self.random_state, n_jobs=-1
            )
            random_search_0.fit(X_train, y_train_dist_0)
            best_model_0 = random_search_0.best_estimator_

            if fold == n_splits:  # Store params from last fold
                self.best_params_distance_0 = random_search_0.best_params_

            # Train model for distance to 1
            print("Training distance-to-1 model...")
            model_1 = XGBRegressor(random_state=self.random_state)
            random_search_1 = RandomizedSearchCV(
                model_1, param_distributions=param_grid,
                n_iter=10, cv=3, scoring='neg_mean_squared_error',
                verbose=0, random_state=self.random_state, n_jobs=-1
            )
            random_search_1.fit(X_train, y_train_dist_1)
            best_model_1 = random_search_1.best_estimator_

            if fold == n_splits:  # Store params from last fold
                self.best_params_distance_1 = random_search_1.best_params_

            # Make predictions for both models
            # Test set
            pred_test_dist_0 = best_model_0.predict(X_test)
            pred_test_dist_1 = best_model_1.predict(X_test)

            # Train set
            pred_train_dist_0 = best_model_0.predict(X_train)
            pred_train_dist_1 = best_model_1.predict(X_train)

            # Reconstruct predictions and calculate uncertainty
            test_preds = []
            test_uncertainties = []
            for i in range(len(X_test)):
                pred, unc = self.reconstruct_prediction(pred_test_dist_0[i], pred_test_dist_1[i])
                test_preds.append(pred)
                test_uncertainties.append(unc)

            train_preds = []
            train_uncertainties = []
            for i in range(len(X_train)):
                pred, unc = self.reconstruct_prediction(pred_train_dist_0[i], pred_train_dist_1[i])
                train_preds.append(pred)
                train_uncertainties.append(unc)

            # Convert to arrays
            test_preds = np.array(test_preds)
            test_uncertainties = np.array(test_uncertainties)
            train_preds = np.array(train_preds)
            train_uncertainties = np.array(train_uncertainties)

            # Store results
            all_test_true.extend(y_test_original)
            all_test_pred.extend(test_preds)
            all_test_uncertainty.extend(test_uncertainties)
            all_train_true.extend(y_train_original)
            all_train_pred.extend(train_preds)
            all_train_uncertainty.extend(train_uncertainties)

            # Calculate metrics
            r2_train = r2_score(y_train_original, train_preds)
            r2_test = r2_score(y_test_original, test_preds)
            rmse_train = np.sqrt(mean_squared_error(y_train_original, train_preds))
            rmse_test = np.sqrt(mean_squared_error(y_test_original, test_preds))

            print(f"Train R²: {r2_train:.4f}, RMSE: {rmse_train:.4f}")
            print(f"Test R²: {r2_test:.4f}, RMSE: {rmse_test:.4f}")
            print(f"Average test uncertainty: {np.mean(test_uncertainties):.4f}")

            fold_scores.append({
                'train_r2': r2_train,
                'test_r2': r2_test,
                'train_rmse': rmse_train,
                'test_rmse': rmse_test,
                'test_uncertainty': np.mean(test_uncertainties)
            })

        # Calculate average metrics
        avg_test_r2 = np.mean([s['test_r2'] for s in fold_scores])
        avg_train_r2 = np.mean([s['train_r2'] for s in fold_scores])
        avg_test_rmse = np.mean([s['test_rmse'] for s in fold_scores])
        avg_train_rmse = np.mean([s['train_rmse'] for s in fold_scores])
        avg_uncertainty = np.mean([s['test_uncertainty'] for s in fold_scores])

        print(f"\n=== Average Metrics ===")
        print(f"Train R²: {avg_train_r2:.4f}")
        print(f"Test R²: {avg_test_r2:.4f}")
        print(f"Train RMSE: {avg_train_rmse:.4f}")
        print(f"Test RMSE: {avg_test_rmse:.4f}")
        print(f"Average Uncertainty: {avg_uncertainty:.4f}")

        # Analyze correlation between uncertainty and error
        all_test_errors = np.abs(np.array(all_test_true) - np.array(all_test_pred))
        uncertainty_error_corr = np.corrcoef(all_test_uncertainty, all_test_errors)[0, 1]
        print(f"\nCorrelation between uncertainty and absolute error: {uncertainty_error_corr:.4f}")

        # Package results
        results = {
            'test_true': np.array(all_test_true),
            'test_pred': np.array(all_test_pred),
            'test_uncertainty': np.array(all_test_uncertainty),
            'train_true': np.array(all_train_true),
            'train_pred': np.array(all_train_pred),
            'train_uncertainty': np.array(all_train_uncertainty),
            'test_indices': all_test_indices,
            'train_indices': all_train_indices,
            'avg_test_r2': avg_test_r2,
            'avg_train_r2': avg_train_r2,
            'avg_test_rmse': avg_test_rmse,
            'avg_train_rmse': avg_train_rmse,
            'avg_uncertainty': avg_uncertainty,
            'uncertainty_error_correlation': uncertainty_error_corr,
            'fold_scores': fold_scores
        }

        return results

    def train_final_dual_models(self, X, y):
        """
        Train final dual distance models on all data
        Args:
            X: Feature matrix
            y: Target variable (transformed)
        Returns:
            tuple: (model_distance_0, model_distance_1)
        """
        print("\n=== Training Final Dual Distance Models ===")

        # Transform y back to original scale
        y_original = self.target_transformer.inverse_transform(y.reshape(-1, 1)).ravel()

        # Prepare dual targets
        y_dist_0, y_dist_1 = self.prepare_dual_targets(y_original)

        # Train model for distance to 0
        print("Training final distance-to-0 model...")
        if self.best_params_distance_0:
            self.model_distance_to_0 = XGBRegressor(**self.best_params_distance_0, random_state=self.random_state)
        else:
            self.model_distance_to_0 = XGBRegressor(random_state=self.random_state)
        self.model_distance_to_0.fit(X, y_dist_0)

        # Train model for distance to 1
        print("Training final distance-to-1 model...")
        if self.best_params_distance_1:
            self.model_distance_to_1 = XGBRegressor(**self.best_params_distance_1, random_state=self.random_state)
        else:
            self.model_distance_to_1 = XGBRegressor(random_state=self.random_state)
        self.model_distance_to_1.fit(X, y_dist_1)

        print("Final dual distance models trained successfully.")

        return self.model_distance_to_0, self.model_distance_to_1

    def predict_with_uncertainty(self, X_new):
        """
        Make predictions on new data with uncertainty estimates
        Args:
            X_new: New feature data to predict on
        Returns:
            tuple: (predictions, uncertainties)
        """
        if self.model_distance_to_0 is None or self.model_distance_to_1 is None:
            print("Error: Dual distance models not trained. Run train_final_dual_models first.")
            return None, None

        # Transform features
        X_new_transformed = self.transformer.transform(X_new)

        # Predict distances
        pred_dist_0 = self.model_distance_to_0.predict(X_new_transformed)
        pred_dist_1 = self.model_distance_to_1.predict(X_new_transformed)

        # Reconstruct predictions with uncertainty
        predictions = []
        uncertainties = []

        for i in range(len(X_new_transformed)):
            pred, unc = self.reconstruct_prediction(pred_dist_0[i], pred_dist_1[i])
            predictions.append(pred)
            uncertainties.append(unc)

        return np.array(predictions), np.array(uncertainties)


"""
Extension to ModelTrainer class for product vs individual r1/r2 prediction approach

This approach trains two models:
1. Direct product model: predicts r1r2 directly
2. Individual model: predicts r1 and r2 separately, then multiplies them
Uncertainty is calculated from the disagreement between these approaches.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV


# from copolpredictor.models import ModelTrainer
# from copolextractor import utils


class ProductVsIndividualModelTrainer(ModelTrainer):
    """Extended ModelTrainer for product vs individual r1/r2 prediction approach"""

    def __init__(self, model_type="xgboost", random_state=42):
        # Initialize parent class
        super().__init__(model_type, random_state)
        # Additional attributes for this approach
        self.model_product = None  # Model for direct r1r2 prediction
        self.model_individual = None  # Model for r1 and r2 separately
        self.best_params_product = None
        self.best_params_individual = None

    def prepare_individual_targets(self, df):
        """
        Prepare individual r1 and r2 targets from the dataframe
        Args:
            df: DataFrame containing constant_1 (r1) and constant_2 (r2)
        Returns:
            tuple: (r1_values, r2_values, r1r2_product)
        """
        # Extract individual reactivity ratios
        r1_values = df['constant_1'].values
        r2_values = df['constant_2'].values
        r1r2_product = r1_values * r2_values

        print(f"\n=== R1/R2 Data Analysis ===")
        print(f"R1 range: [{r1_values.min():.3f}, {r1_values.max():.3f}]")
        print(f"R2 range: [{r2_values.min():.3f}, {r2_values.max():.3f}]")
        print(f"Product range: [{r1r2_product.min():.3f}, {r1r2_product.max():.3f}]")

        # Analyze the inverse relationship
        high_r1_mask = r1_values > 10
        high_r2_mask = r2_values > 10

        if high_r1_mask.sum() > 0:
            print(f"\nWhen R1 > 10 ({high_r1_mask.sum()} samples):")
            print(f"  Mean R2: {r2_values[high_r1_mask].mean():.6f}")
            print(f"  Max R2: {r2_values[high_r1_mask].max():.6f}")

        if high_r2_mask.sum() > 0:
            print(f"\nWhen R2 > 10 ({high_r2_mask.sum()} samples):")
            print(f"  Mean R1: {r1_values[high_r2_mask].mean():.6f}")
            print(f"  Max R1: {r1_values[high_r2_mask].max():.6f}")

        # Check for the chemical constraint: high r1 usually means low r2
        correlation = np.corrcoef(np.log1p(r1_values), np.log1p(r2_values))[0, 1]
        print(f"\nLog-scale correlation between R1 and R2: {correlation:.4f}")

        # Identify the sample with maximum product
        max_prod_idx = np.argmax(r1r2_product)
        print(f"\nMaximum product case:")
        print(f"  R1 = {r1_values[max_prod_idx]:.3f}")
        print(f"  R2 = {r2_values[max_prod_idx]:.3f}")
        print(f"  Product = {r1r2_product[max_prod_idx]:.3f}")

        return r1_values, r2_values, r1r2_product

    def calculate_uncertainty_from_models(self, pred_product_direct, pred_r1, pred_r2):
        """
        Calculate uncertainty by comparing direct product prediction with r1*r2

        Args:
            pred_product_direct: Direct prediction of r1r2 from product model
            pred_r1: Predicted r1 values from individual model
            pred_r2: Predicted r2 values from individual model

        Returns:
            tuple: (final_prediction, uncertainty, pred_product_indirect)
        """
        # Calculate indirect product prediction
        pred_product_indirect = pred_r1 * pred_r2

        # Handle potential NaN or infinite values
        if np.isnan(pred_product_indirect) or np.isinf(pred_product_indirect):
            # Fall back to direct prediction
            return pred_product_direct, 0.5, pred_product_direct

        # Uncertainty is the absolute difference between the two approaches
        uncertainty = np.abs(pred_product_direct - pred_product_indirect)

        # Cap uncertainty at a reasonable maximum
        uncertainty = np.minimum(uncertainty, 10.0)

        # Final prediction is a weighted average
        # Weight based on the magnitude of predictions (avoid division by zero)
        weight_direct = 1.0 / (np.abs(pred_product_direct) + 0.1)
        weight_indirect = 1.0 / (np.abs(pred_product_indirect) + 0.1)

        # Handle edge cases where weights might be infinite
        if np.isinf(weight_direct):
            weight_direct = 10.0
        if np.isinf(weight_indirect):
            weight_indirect = 10.0

        final_prediction = (weight_direct * pred_product_direct +
                            weight_indirect * pred_product_indirect) / (weight_direct + weight_indirect)

        # Ensure non-negative predictions
        final_prediction = np.maximum(final_prediction, 0)
        pred_r1 = np.maximum(pred_r1, 0)
        pred_r2 = np.maximum(pred_r2, 0)

        return final_prediction, uncertainty, pred_product_indirect

    def train_product_vs_individual_models(self, X, y, df):
        """
        Train two separate approaches: direct product prediction and individual r1/r2 prediction

        Args:
            X: Feature matrix (already transformed)
            y: Target variable (transformed r1r2 product)
            df: DataFrame with reaction IDs and original values

        Returns:
            dict: Results containing predictions, metrics, and uncertainty estimates
        """
        print("\n=== Training Product vs Individual R1/R2 Models ===")

        # Get original r1, r2 values
        r1_original, r2_original, r1r2_original = self.prepare_individual_targets(df)

        # Transform individual targets using the same transformer
        # Note: We need to be careful here - the target transformer was fit on r1r2
        # For r1 and r2, we'll use separate transformers
        from sklearn.preprocessing import PowerTransformer

        r1_transformer = PowerTransformer(method='yeo-johnson')
        r2_transformer = PowerTransformer(method='yeo-johnson')

        # Filter out negative values before transformation
        r1_positive = np.maximum(r1_original, 1e-6)
        r2_positive = np.maximum(r2_original, 1e-6)

        r1_transformed = r1_transformer.fit_transform(r1_positive.reshape(-1, 1)).ravel()
        r2_transformed = r2_transformer.fit_transform(r2_positive.reshape(-1, 1)).ravel()

        print(f"R1 transformed range: [{r1_transformed.min():.3f}, {r1_transformed.max():.3f}]")
        print(f"R2 transformed range: [{r2_transformed.min():.3f}, {r2_transformed.max():.3f}]")

        # Create grouped K-Fold splits
        n_splits = 5
        kf_splits = utils.create_grouped_kfold_splits(df, n_splits=n_splits, random_state=self.random_state)

        # Storage for results
        all_test_true = []
        all_test_pred_final = []
        all_test_pred_direct = []
        all_test_pred_indirect = []
        all_test_pred_r1 = []
        all_test_pred_r2 = []
        all_test_uncertainty = []
        all_train_true = []
        all_train_pred_final = []
        all_train_uncertainty = []
        all_test_indices = []
        all_train_indices = []

        fold_scores = []

        # Get parameter grid
        _, param_grid = self.get_model_and_param_grid()

        # Cross-validation loop
        for fold, (train_idx, test_idx) in enumerate(kf_splits, 1):
            print(f"\n--- Fold {fold} ---")

            # Store indices
            if fold == 1:
                all_train_indices.extend(train_idx)
                all_test_indices.extend(test_idx)

            # Split data
            X_train, X_test = X[train_idx], X[test_idx]

            # For product model (using existing transformer)
            y_train_product = y[train_idx]
            y_test_product = y[test_idx]

            # For individual model
            y_train_r1 = r1_transformed[train_idx]
            y_test_r1 = r1_transformed[test_idx]
            y_train_r2 = r2_transformed[train_idx]
            y_test_r2 = r2_transformed[test_idx]

            # Original values for evaluation
            y_train_original = r1r2_original[train_idx]
            y_test_original = r1r2_original[test_idx]
            r1_test_original = r1_original[test_idx]
            r2_test_original = r2_original[test_idx]

            # --- Train Model 1: Direct Product Prediction ---
            print("Training direct product model...")
            model_product = XGBRegressor(random_state=self.random_state)

            # Hyperparameter optimization for product model
            random_search_product = RandomizedSearchCV(
                model_product, param_distributions=param_grid,
                n_iter=10, cv=3, scoring='neg_mean_squared_error',
                verbose=0, random_state=self.random_state, n_jobs=-1
            )
            random_search_product.fit(X_train, y_train_product)
            best_model_product = random_search_product.best_estimator_

            if fold == n_splits:
                self.best_params_product = random_search_product.best_params_

            # --- Train Model 2: Individual R1 and R2 Prediction ---
            print("Training individual r1/r2 model...")
            # Stack targets for multi-output regression
            y_train_individual = np.column_stack([y_train_r1, y_train_r2])
            y_test_individual = np.column_stack([y_test_r1, y_test_r2])

            # Use MultiOutputRegressor to predict both r1 and r2
            base_model_individual = XGBRegressor(random_state=self.random_state)
            model_individual = MultiOutputRegressor(base_model_individual)

            # For multi-output, we'll use fixed hyperparameters for simplicity
            # You could extend this to optimize each output separately
            model_individual.fit(X_train, y_train_individual)

            # Make predictions
            # Test set
            pred_test_product_transformed = best_model_product.predict(X_test)
            pred_test_individual_transformed = model_individual.predict(X_test)

            # Transform back to original scale
            pred_test_product_direct = self.target_transformer.inverse_transform(
                pred_test_product_transformed.reshape(-1, 1)
            ).ravel()

            pred_test_r1 = r1_transformer.inverse_transform(
                pred_test_individual_transformed[:, 0].reshape(-1, 1)
            ).ravel()
            pred_test_r2 = r2_transformer.inverse_transform(
                pred_test_individual_transformed[:, 1].reshape(-1, 1)
            ).ravel()

            # Ensure non-negative predictions (reactivity ratios cannot be negative)
            pred_test_product_direct = np.maximum(pred_test_product_direct, 0)
            pred_test_r1 = np.maximum(pred_test_r1, 0)
            pred_test_r2 = np.maximum(pred_test_r2, 0)

            # Calculate final predictions and uncertainty
            test_pred_final = []
            test_uncertainties = []
            test_pred_indirect = []

            for i in range(len(X_test)):
                final_pred, unc, indirect = self.calculate_uncertainty_from_models(
                    pred_test_product_direct[i],
                    pred_test_r1[i],
                    pred_test_r2[i]
                )
                test_pred_final.append(final_pred)
                test_uncertainties.append(unc)
                test_pred_indirect.append(indirect)

            test_pred_final = np.array(test_pred_final)
            test_uncertainties = np.array(test_uncertainties)
            test_pred_indirect = np.array(test_pred_indirect)

            # Check for NaN values and handle them
            if np.any(np.isnan(test_pred_indirect)):
                print(f"Warning: {np.sum(np.isnan(test_pred_indirect))} NaN values in indirect predictions")
                # Replace NaN with direct predictions as fallback
                nan_mask = np.isnan(test_pred_indirect)
                test_pred_indirect[nan_mask] = pred_test_product_direct[nan_mask]
                test_uncertainties[nan_mask] = 0.5  # Assign moderate uncertainty

            # Train set predictions (for overfitting analysis)
            pred_train_product_transformed = best_model_product.predict(X_train)
            pred_train_individual_transformed = model_individual.predict(X_train)

            pred_train_product_direct = self.target_transformer.inverse_transform(
                pred_train_product_transformed.reshape(-1, 1)
            ).ravel()

            pred_train_r1 = r1_transformer.inverse_transform(
                pred_train_individual_transformed[:, 0].reshape(-1, 1)
            ).ravel()
            pred_train_r2 = r2_transformer.inverse_transform(
                pred_train_individual_transformed[:, 1].reshape(-1, 1)
            ).ravel()

            # Ensure non-negative predictions for train set too
            pred_train_product_direct = np.maximum(pred_train_product_direct, 0)
            pred_train_r1 = np.maximum(pred_train_r1, 0)
            pred_train_r2 = np.maximum(pred_train_r2, 0)

            train_pred_final = []
            train_uncertainties = []

            for i in range(len(X_train)):
                final_pred, unc, _ = self.calculate_uncertainty_from_models(
                    pred_train_product_direct[i],
                    pred_train_r1[i],
                    pred_train_r2[i]
                )
                train_pred_final.append(final_pred)
                train_uncertainties.append(unc)

            train_pred_final = np.array(train_pred_final)
            train_uncertainties = np.array(train_uncertainties)

            # Store results
            all_test_true.extend(y_test_original)
            all_test_pred_final.extend(test_pred_final)
            all_test_pred_direct.extend(pred_test_product_direct)
            all_test_pred_indirect.extend(test_pred_indirect)
            all_test_pred_r1.extend(pred_test_r1)
            all_test_pred_r2.extend(pred_test_r2)
            all_test_uncertainty.extend(test_uncertainties)
            all_train_true.extend(y_train_original)
            all_train_pred_final.extend(train_pred_final)
            all_train_uncertainty.extend(train_uncertainties)

            # Calculate metrics with NaN handling
            # Direct product model
            valid_mask = ~(np.isnan(y_test_original) | np.isnan(pred_test_product_direct))
            if np.sum(valid_mask) > 0:
                r2_test_direct = r2_score(y_test_original[valid_mask], pred_test_product_direct[valid_mask])
                rmse_test_direct = np.sqrt(
                    mean_squared_error(y_test_original[valid_mask], pred_test_product_direct[valid_mask]))
            else:
                r2_test_direct = 0.0
                rmse_test_direct = np.inf
                print("Warning: No valid samples for direct product evaluation")

            # Individual r1/r2 model (using indirect product)
            valid_mask = ~(np.isnan(y_test_original) | np.isnan(test_pred_indirect))
            if np.sum(valid_mask) > 0:
                r2_test_indirect = r2_score(y_test_original[valid_mask], test_pred_indirect[valid_mask])
                rmse_test_indirect = np.sqrt(
                    mean_squared_error(y_test_original[valid_mask], test_pred_indirect[valid_mask]))
            else:
                r2_test_indirect = 0.0
                rmse_test_indirect = np.inf
                print("Warning: No valid samples for indirect product evaluation")

            # Individual r1 and r2 metrics
            valid_mask_r1 = ~(np.isnan(r1_test_original) | np.isnan(pred_test_r1))
            valid_mask_r2 = ~(np.isnan(r2_test_original) | np.isnan(pred_test_r2))

            r2_test_r1 = r2_score(r1_test_original[valid_mask_r1], pred_test_r1[valid_mask_r1]) if np.sum(
                valid_mask_r1) > 0 else 0.0
            r2_test_r2 = r2_score(r2_test_original[valid_mask_r2], pred_test_r2[valid_mask_r2]) if np.sum(
                valid_mask_r2) > 0 else 0.0

            # Combined final prediction
            valid_mask = ~(np.isnan(y_test_original) | np.isnan(test_pred_final))
            if np.sum(valid_mask) > 0:
                r2_test_final = r2_score(y_test_original[valid_mask], test_pred_final[valid_mask])
                rmse_test_final = np.sqrt(mean_squared_error(y_test_original[valid_mask], test_pred_final[valid_mask]))
            else:
                r2_test_final = 0.0
                rmse_test_final = np.inf

            # Train metrics
            valid_mask = ~(np.isnan(y_train_original) | np.isnan(train_pred_final))
            if np.sum(valid_mask) > 0:
                r2_train_final = r2_score(y_train_original[valid_mask], train_pred_final[valid_mask])
                rmse_train_final = np.sqrt(
                    mean_squared_error(y_train_original[valid_mask], train_pred_final[valid_mask]))
            else:
                r2_train_final = 0.0
                rmse_train_final = np.inf

            print(f"\nTest Metrics:")
            print(f"  Direct Product Model - R²: {r2_test_direct:.4f}, RMSE: {rmse_test_direct:.4f}")
            print(f"  Indirect (R1×R2) Model - R²: {r2_test_indirect:.4f}, RMSE: {rmse_test_indirect:.4f}")
            print(f"  Individual R1 - R²: {r2_test_r1:.4f}")
            print(f"  Individual R2 - R²: {r2_test_r2:.4f}")
            print(f"  Final Combined - R²: {r2_test_final:.4f}, RMSE: {rmse_test_final:.4f}")
            print(f"  Average Uncertainty: {np.mean(test_uncertainties):.4f}")

            fold_scores.append({
                'train_r2': r2_train_final,
                'test_r2': r2_test_final,
                'test_r2_direct': r2_test_direct,
                'test_r2_indirect': r2_test_indirect,
                'test_r2_r1': r2_test_r1,
                'test_r2_r2': r2_test_r2,
                'train_rmse': rmse_train_final,
                'test_rmse': rmse_test_final,
                'test_uncertainty': np.mean(test_uncertainties)
            })

        # Calculate average metrics
        avg_test_r2 = np.mean([s['test_r2'] for s in fold_scores])
        avg_test_r2_direct = np.mean([s['test_r2_direct'] for s in fold_scores])
        avg_test_r2_indirect = np.mean([s['test_r2_indirect'] for s in fold_scores])
        avg_uncertainty = np.mean([s['test_uncertainty'] for s in fold_scores])

        print(f"\n=== Average Cross-Validation Metrics ===")
        print(f"Direct Product Model - Test R²: {avg_test_r2_direct:.4f}")
        print(f"Indirect (R1×R2) Model - Test R²: {avg_test_r2_indirect:.4f}")
        print(f"Final Combined Model - Test R²: {avg_test_r2:.4f}")
        print(f"Average Uncertainty: {avg_uncertainty:.4f}")

        # Analyze correlation between uncertainty and error
        all_test_errors = np.abs(np.array(all_test_true) - np.array(all_test_pred_final))
        uncertainty_error_corr = np.corrcoef(all_test_uncertainty, all_test_errors)[0, 1]
        print(f"\nCorrelation between uncertainty and absolute error: {uncertainty_error_corr:.4f}")

        # Store the transformers for later use
        self.r1_transformer = r1_transformer
        self.r2_transformer = r2_transformer

        # Package results
        results = {
            'test_true': np.array(all_test_true),
            'test_pred': np.array(all_test_pred_final),
            'test_pred_direct': np.array(all_test_pred_direct),
            'test_pred_indirect': np.array(all_test_pred_indirect),
            'test_pred_r1': np.array(all_test_pred_r1),
            'test_pred_r2': np.array(all_test_pred_r2),
            'test_uncertainty': np.array(all_test_uncertainty),
            'train_true': np.array(all_train_true),
            'train_pred': np.array(all_train_pred_final),
            'train_uncertainty': np.array(all_train_uncertainty),
            'test_indices': all_test_indices,
            'train_indices': all_train_indices,
            'avg_test_r2': avg_test_r2,
            'avg_test_r2_direct': avg_test_r2_direct,
            'avg_test_r2_indirect': avg_test_r2_indirect,
            'avg_uncertainty': avg_uncertainty,
            'uncertainty_error_correlation': uncertainty_error_corr,
            'fold_scores': fold_scores
        }

        return results

    def train_final_models(self, X, y, df):
        """
        Train final models on all data

        Args:
            X: Feature matrix
            y: Target variable (transformed)
            df: DataFrame with original values

        Returns:
            tuple: (model_product, model_individual)
        """
        print("\n=== Training Final Product vs Individual Models ===")

        # Get original values
        r1_original, r2_original, _ = self.prepare_individual_targets(df)

        # Transform individual targets
        r1_transformed = self.r1_transformer.transform(r1_original.reshape(-1, 1)).ravel()
        r2_transformed = self.r2_transformer.transform(r2_original.reshape(-1, 1)).ravel()

        # Train final product model
        print("Training final direct product model...")
        if self.best_params_product:
            self.model_product = XGBRegressor(**self.best_params_product, random_state=self.random_state)
        else:
            self.model_product = XGBRegressor(random_state=self.random_state)
        self.model_product.fit(X, y)

        # Train final individual model
        print("Training final individual r1/r2 model...")
        y_individual = np.column_stack([r1_transformed, r2_transformed])
        base_model = XGBRegressor(random_state=self.random_state)
        self.model_individual = MultiOutputRegressor(base_model)
        self.model_individual.fit(X, y_individual)

        print("Final models trained successfully.")

        return self.model_product, self.model_individual

    def predict_with_uncertainty(self, X_new):
        """
        Make predictions on new data with uncertainty estimates

        Args:
            X_new: New feature data to predict on

        Returns:
            tuple: (predictions, uncertainties, r1_predictions, r2_predictions)
        """
        if self.model_product is None or self.model_individual is None:
            print("Error: Models not trained. Run train_final_models first.")
            return None, None, None, None

        # Transform features
        X_new_transformed = self.transformer.transform(X_new)

        # Get predictions from both models
        # Direct product prediction
        pred_product_transformed = self.model_product.predict(X_new_transformed)
        pred_product_direct = self.target_transformer.inverse_transform(
            pred_product_transformed.reshape(-1, 1)
        ).ravel()

        # Individual r1/r2 predictions
        pred_individual_transformed = self.model_individual.predict(X_new_transformed)
        pred_r1 = self.r1_transformer.inverse_transform(
            pred_individual_transformed[:, 0].reshape(-1, 1)
        ).ravel()
        pred_r2 = self.r2_transformer.inverse_transform(
            pred_individual_transformed[:, 1].reshape(-1, 1)
        ).ravel()

        # Calculate final predictions and uncertainties
        predictions = []
        uncertainties = []

        for i in range(len(X_new_transformed)):
            final_pred, unc, _ = self.calculate_uncertainty_from_models(
                pred_product_direct[i],
                pred_r1[i],
                pred_r2[i]
            )
            predictions.append(final_pred)
            uncertainties.append(unc)

        return (np.array(predictions), np.array(uncertainties),
                np.array(pred_r1), np.array(pred_r2))