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