"""
Improved neural network implementation with safe PowerTransformer
that prevents NaN issues in the inverse transformation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.model_selection import train_test_split


class SimpleDataset(Dataset):
    """Simple dataset for copolymerization data"""

    def __init__(self, features, targets):
        """Initialize with feature and target tensors"""
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class MSE_R2Loss(nn.Module):
    def __init__(self, r2_weight=0.5):
        super(MSE_R2Loss, self).__init__()
        self.r2_weight = r2_weight
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):
        # MSE loss
        mse_loss = self.mse(y_pred, y_true)

        # R² loss
        y_true_mean = torch.mean(y_true, dim=0, keepdim=True)
        ss_tot = torch.sum((y_true - y_true_mean) ** 2, dim=0)
        ss_res = torch.sum((y_true - y_pred) ** 2, dim=0)

        valid = ss_tot > 1e-6
        r2 = torch.where(valid, 1 - (ss_res / (ss_tot + 1e-8)), torch.tensor(0.0, device=ss_tot.device))
        r2_loss = 1 - torch.mean(r2)

        # Combine both losses
        combined_loss = (1 - self.r2_weight) * mse_loss + self.r2_weight * r2_loss
        return combined_loss


class CoPolymerNN(nn.Module):
    """Neural network with explicit r1*r2 product computation"""

    def __init__(self, input_size, hidden_layers=[256, 128, 64]):
        super(CoPolymerNN, self).__init__()

        # Shared feature extraction
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, hidden_layers[0]),
            nn.BatchNorm1d(hidden_layers[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.BatchNorm1d(hidden_layers[1]),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Separate paths for r1 and r2
        self.r1_layers = nn.Sequential(
            nn.Linear(hidden_layers[1], hidden_layers[2]),
            nn.BatchNorm1d(hidden_layers[2]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_layers[2], 1)
        )

        self.r2_layers = nn.Sequential(
            nn.Linear(hidden_layers[1], hidden_layers[2]),
            nn.BatchNorm1d(hidden_layers[2]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_layers[2], 1)
        )

    def forward(self, x):
        # Shared feature extraction
        shared_features = self.shared_layers(x)

        # Separate predictions for r1 and r2
        r1 = self.r1_layers(shared_features)
        r2 = self.r2_layers(shared_features)

        # Explicitly compute the product r1*r2
        r_product = r1 * r2

        # Concatenate all outputs: [r1, r2, r1*r2]
        outputs = torch.cat([r1, r2, r_product], dim=1)

        return outputs


class SafeTransformer:
    """Wrapper for transformers that handles extreme values and prevents NaN"""

    def __init__(self, transformer=None, method='yeo-johnson', clip_min=-10, clip_max=10):
        self.transformer = PowerTransformer(method=method) if transformer is None else transformer
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.column_means = None
        self.column_stds = None

    def fit(self, X):
        # Store column means and stds for handling anomalies later
        self.column_means = np.mean(X, axis=0)
        self.column_stds = np.std(X, axis=0)
        self.transformer.fit(X)
        return self

    def transform(self, X):
        # Apply the transformation
        X_transformed = self.transformer.transform(X)

        # Clip extreme values to prevent issues in the model
        X_transformed = np.clip(X_transformed, self.clip_min, self.clip_max)

        # Handle any remaining NaNs by replacing with column means
        mask = np.isnan(X_transformed)
        if np.any(mask):
            print(f"Warning: {np.sum(mask)} NaN values found after transformation. Replacing with means.")
            col_indices = np.where(mask)
            for i, j in zip(col_indices[0], col_indices[1]):
                X_transformed[i, j] = 0  # Replace with 0 in transformed space

        return X_transformed

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        # First, clip extreme values in transformed space
        X_clipped = np.clip(X, self.clip_min, self.clip_max)

        # Apply inverse transformation
        try:
            X_original = self.transformer.inverse_transform(X_clipped)

            # Handle any NaNs in the inverse transformation
            mask = np.isnan(X_original)
            if np.any(mask):
                print(f"Warning: {np.sum(mask)} NaN values found after inverse transformation. Replacing with means.")
                col_indices = np.where(mask)
                for i, j in zip(col_indices[0], col_indices[1]):
                    if self.column_means is not None:
                        X_original[i, j] = self.column_means[j]
                    else:
                        X_original[i, j] = 0

            return X_original

        except Exception as e:
            print(f"Error in inverse transform: {e}")
            # Return a safe fallback - column means or zeros
            if self.column_means is not None:
                print("Using column means as fallback for inverse transform")
                return np.tile(self.column_means, (X.shape[0], 1))
            else:
                print("Using zeros as fallback for inverse transform")
                return np.zeros_like(X)


def prepare_data(df, batch_size=32, random_state=42, transform_targets=True):
    """Improved data preparation with safe target transformation"""
    # Ensure r1r2 is calculated
    if 'r1r2' not in df.columns:
        print("Calculating r1r2 from constant_1 and constant_2")
        df['r1r2'] = df['constant_1'] * df['constant_2']

    # Identify feature columns
    monomer1_cols = [col for col in df.columns if col.endswith('_1')
                     and col not in ['constant_1', 'polytype_emb_1', 'method_emb_1']]

    monomer2_cols = [col for col in df.columns if col.endswith('_2')
                     and col not in ['constant_2', 'polytype_emb_2', 'method_emb_2']]

    # Condition columns
    condition_cols = ['temperature', 'solvent_logp']
    if 'polytype_emb_1' in df.columns:
        condition_cols.extend(['polytype_emb_1', 'polytype_emb_2'])
    if 'method_emb_1' in df.columns:
        condition_cols.extend(['method_emb_1', 'method_emb_2'])

    # Target columns
    target_cols = ['constant_1', 'constant_2', 'r1r2']

    # All columns used
    feature_cols = monomer1_cols + monomer2_cols + condition_cols
    all_cols = feature_cols + target_cols

    print(f"Number of feature columns: {len(feature_cols)}")
    print(f"Number of target columns: {len(target_cols)}")

    # Create a copy of the DataFrame with only the needed columns
    df_subset = df[all_cols].copy()

    print(f"Shape of reduced DataFrame: {df_subset.shape}")

    # Clean data AFTER column selection
    # Remove NaN values in target columns
    df_subset = df_subset.dropna(subset=target_cols)
    print(f"Shape after removing NaNs in target columns: {df_subset.shape}")

    # Handle NaN and Inf values in features
    df_clean = df_subset.copy()

    # Check if there are feature columns with too many NaNs
    nan_percentage = df_clean[feature_cols].isna().mean() * 100
    high_nan_cols = nan_percentage[nan_percentage > 50].index.tolist()

    if high_nan_cols:
        print(f"WARNING: These feature columns have > 50% NaN values: {high_nan_cols}")
        print("These columns will be removed")
        remaining_features = [col for col in feature_cols if col not in high_nan_cols]
        df_clean = df_clean.drop(columns=high_nan_cols)
        feature_cols = remaining_features

    # For remaining feature columns: replace NaN with median
    for col in feature_cols:
        if df_clean[col].isna().any():
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)

    # Replace Inf values with large but finite values
    inf_mask = df_clean.replace([np.inf, -np.inf], np.nan).isna() & ~df_clean.isna()
    if inf_mask.any().any():
        for col in feature_cols:
            pos_inf_count = (df_clean[col] == np.inf).sum()
            neg_inf_count = (df_clean[col] == -np.inf).sum()

            if pos_inf_count > 0 or neg_inf_count > 0:
                max_finite = df_clean[col].replace([np.inf, -np.inf], np.nan).max()
                min_finite = df_clean[col].replace([np.inf, -np.inf], np.nan).min()

                # Use large finite values for +/-inf
                pos_inf_replacement = max_finite * 1.5 if not np.isnan(max_finite) else 1e6
                neg_inf_replacement = min_finite * 1.5 if not np.isnan(min_finite) else -1e6

                df_clean[col] = df_clean[col].replace(np.inf, pos_inf_replacement)
                df_clean[col] = df_clean[col].replace(-np.inf, neg_inf_replacement)

    # Extract feature and target values
    X = df_clean[feature_cols].values
    y = df_clean[target_cols].values

    # Feature normalization with RobustScaler
    feature_scaler = RobustScaler()
    X_scaled = feature_scaler.fit_transform(X)

    # Target transformation with SafeTransformer
    if transform_targets:
        # Use SafeTransformer to prevent NaN issues
        target_transformer = SafeTransformer(method='yeo-johnson')
        y_transformed = target_transformer.fit_transform(y)
        y = y_transformed
    else:
        target_transformer = None

    # Train-Test-Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=random_state
    )

    # Create PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Create datasets and dataloaders
    train_dataset = SimpleDataset(X_train_tensor, y_train_tensor)
    test_dataset = SimpleDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, len(feature_cols), feature_scaler, target_transformer


def train_simple_nn(df, epochs=300, lr=5e-4, batch_size=32, random_state=42, transform_targets=True):
    """Training with safe target transformation"""
    try:
        # Set PyTorch random seed for reproducibility
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        # Prepare data
        train_loader, test_loader, input_size, feature_scaler, target_transformer = prepare_data(
            df, batch_size, random_state, transform_targets=transform_targets
        )

        # Create model
        model = CoPolymerNN(input_size=input_size)

        # Combined loss function with lower R2 weight initially
        criterion = MSE_R2Loss(r2_weight=0.1)  # Start with focus on MSE

        # Optimizer with weight decay
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

        # Learning Rate Scheduler with warmup phase
        def lr_lambda(epoch):
            warmup_epochs = 30
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs  # Linear increase to full learning rate
            else:
                return max(0.1, 1.0 - (epoch - warmup_epochs) / (epochs - warmup_epochs))  # Linear decay

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        # Early Stopping parameters
        best_val_loss = float('inf')
        patience = 50
        patience_counter = 0
        best_model_state = None

        # Training statistics
        train_losses = []
        val_losses = []

        # Gradually increase R2 weight after some epochs
        r2_weight_schedule = {
            int(epochs * 0.2): 0.2,  # After 20% of epochs
            int(epochs * 0.4): 0.3,  # After 40% of epochs
            int(epochs * 0.6): 0.4,  # After 60% of epochs
        }

        # Training loop
        for epoch in range(epochs):
            model.train()
            train_loss = 0

            # Update R2 weight according to schedule
            if epoch in r2_weight_schedule:
                criterion.r2_weight = r2_weight_schedule[epoch]
                print(f"Epoch {epoch}: R2 weight increased to {criterion.r2_weight}")

            for features, targets in train_loader:
                # Forward pass
                outputs = model(features)
                loss = criterion(outputs, targets)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()

            # Adjust learning rate
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            # Calculate average training loss
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for features, targets in test_loader:
                    outputs = model(features)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()

            # Calculate average validation loss
            avg_val_loss = val_loss / len(test_loader)
            val_losses.append(avg_val_loss)

            # Check for early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break

            # Print progress
            if (epoch + 1) % 10 == 0:
                print(
                    f'Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.6f}'
                )

        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Evaluate the model
        model.eval()
        r12_true, r12_pred = [], []
        r21_true, r21_pred = [], []
        prod_true, prod_pred = [], []

        with torch.no_grad():
            for features, targets in test_loader:
                outputs = model(features)

                # Store predictions and actual values
                r12_true.extend(targets[:, 0].numpy())
                r12_pred.extend(outputs[:, 0].numpy())

                r21_true.extend(targets[:, 1].numpy())
                r21_pred.extend(outputs[:, 1].numpy())

                prod_true.extend(targets[:, 2].numpy())
                prod_pred.extend(outputs[:, 2].numpy())

        # Calculate R² scores in transformed space
        from sklearn.metrics import r2_score, mean_squared_error

        # Get metrics in transformed space
        metrics_transformed = {
            'r12_r2': r2_score(r12_true, r12_pred),
            'r21_r2': r2_score(r21_true, r21_pred),
            'product_r2': r2_score(prod_true, prod_pred),
            'r12_mse': mean_squared_error(r12_true, r12_pred),
            'r21_mse': mean_squared_error(r21_true, r21_pred),
            'product_mse': mean_squared_error(prod_true, prod_pred)
        }

        # If we transformed the targets, also calculate metrics in original space
        if transform_targets and target_transformer is not None:
            try:
                # Convert to 2D arrays for inverse transform
                r12_true_2d = np.array(r12_true).reshape(-1, 1)
                r12_pred_2d = np.array(r12_pred).reshape(-1, 1)
                r21_true_2d = np.array(r21_true).reshape(-1, 1)
                r21_pred_2d = np.array(r21_pred).reshape(-1, 1)
                prod_true_2d = np.array(prod_true).reshape(-1, 1)
                prod_pred_2d = np.array(prod_pred).reshape(-1, 1)

                # Stack for inverse transform
                all_true_2d = np.column_stack((r12_true_2d, r21_true_2d, prod_true_2d))
                all_pred_2d = np.column_stack((r12_pred_2d, r21_pred_2d, prod_pred_2d))

                # Safe inverse transform
                all_true_orig = target_transformer.inverse_transform(all_true_2d)
                all_pred_orig = target_transformer.inverse_transform(all_pred_2d)

                # Extract back to arrays
                r12_true_orig = all_true_orig[:, 0]
                r12_pred_orig = all_pred_orig[:, 0]
                r21_true_orig = all_true_orig[:, 1]
                r21_pred_orig = all_pred_orig[:, 1]
                prod_true_orig = all_true_orig[:, 2]
                prod_pred_orig = all_pred_orig[:, 2]

                # Calculate metrics in original space - with additional safeguards
                def safe_r2_score(y_true, y_pred):
                    # Replace any remaining NaNs with 0
                    y_true = np.nan_to_num(y_true)
                    y_pred = np.nan_to_num(y_pred)
                    try:
                        return r2_score(y_true, y_pred)
                    except Exception as e:
                        print(f"Error calculating R2 score: {e}")
                        return 0.0

                def safe_mse(y_true, y_pred):
                    # Replace any remaining NaNs with 0
                    y_true = np.nan_to_num(y_true)
                    y_pred = np.nan_to_num(y_pred)
                    try:
                        return mean_squared_error(y_true, y_pred)
                    except Exception as e:
                        print(f"Error calculating MSE: {e}")
                        return float('inf')

                metrics_original = {
                    'r12_r2': safe_r2_score(r12_true_orig, r12_pred_orig),
                    'r21_r2': safe_r2_score(r21_true_orig, r21_pred_orig),
                    'product_r2': safe_r2_score(prod_true_orig, prod_pred_orig),
                    'r12_mse': safe_mse(r12_true_orig, r12_pred_orig),
                    'r21_mse': safe_mse(r21_true_orig, r21_pred_orig),
                    'product_mse': safe_mse(prod_true_orig, prod_pred_orig)
                }

                # Use original space metrics for reporting
                metrics = metrics_original

                # Check consistency in original space - with safeguards
                computed_prod_orig = np.array(r12_pred_orig) * np.array(r21_pred_orig)
                consistency_r2_orig = safe_r2_score(prod_pred_orig, computed_prod_orig)

                print("\nFinal Metrics (Original Space):")
                print(f"R12 R²: {metrics['r12_r2']:.4f}")
                print(f"R21 R²: {metrics['r21_r2']:.4f}")
                print(f"Product R²: {metrics['product_r2']:.4f}")
                print(f"R1*R2 vs predicted product consistency R²: {consistency_r2_orig:.4f}")

            except Exception as e:
                print(f"Error calculating metrics in original space: {e}")
                # Fall back to transformed metrics
                metrics = metrics_transformed
                print("Using metrics in transformed space instead")

            print("\nFinal Metrics (Transformed Space):")
            print(f"R12 R²: {metrics_transformed['r12_r2']:.4f}")
            print(f"R21 R²: {metrics_transformed['r21_r2']:.4f}")
            print(f"Product R²: {metrics_transformed['product_r2']:.4f}")
        else:
            metrics = metrics_transformed
            print("\nFinal Metrics:")
            print(f"R12 R²: {metrics['r12_r2']:.4f}")
            print(f"R21 R²: {metrics['r21_r2']:.4f}")
            print(f"Product R²: {metrics['product_r2']:.4f}")

        # Check consistency between predicted product and calculated r1*r2
        computed_prod = np.array(r12_pred) * np.array(r21_pred)
        consistency_r2 = r2_score(prod_pred, computed_prod)
        print(f"R1*R2 vs predicted product consistency R² (transformed): {consistency_r2:.4f}")

        return model, metrics, (feature_scaler, target_transformer), (train_losses, val_losses)

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        raise


def get_predictions_from_model(model, data_loader, target_transformer=None):
    """Get predictions from a trained model"""
    model.eval()
    all_true_values = []
    all_predicted_values = []

    with torch.no_grad():
        for features, targets in data_loader:
            outputs = model(features)

            # Convert to numpy
            true_batch = targets.numpy()
            pred_batch = outputs.numpy()

            all_true_values.append(true_batch)
            all_predicted_values.append(pred_batch)

    # Combine batches
    true_values = np.vstack(all_true_values)
    predicted_values = np.vstack(all_predicted_values)

    # If target transformer exists, inverse transform the predictions
    if target_transformer is not None:
        try:
            true_values = target_transformer.inverse_transform(true_values)
            predicted_values = target_transformer.inverse_transform(predicted_values)
        except Exception as e:
            print(f"Error in inverse transform during prediction: {e}")
            print("Returning predictions in transformed space")

    return true_values, predicted_values