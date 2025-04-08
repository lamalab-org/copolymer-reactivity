"""
Neural network models for copolymerization prediction
Contains models and functions to train and evaluate neural networks
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchmetrics import R2Score

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WandB not available. Logging will be disabled.")


class MoleculeEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(128, 64)):
        super(MoleculeEncoder, self).__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.SiLU(),
            nn.Dropout(0.1)
        )

        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.LayerNorm(hidden_dims[i + 1]),
                nn.SiLU(),
                nn.Dropout(0.1)
            ))

        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, a=0.01, nonlinearity='linear')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        x = torch.clamp(x, -1e6, 1e6)
        x = self.input_norm(x)
        x = self.input_layer(x)

        for layer in self.hidden_layers:
            residual = x
            x = layer(x)
            if x.shape == residual.shape:
                x = x + residual

        x = self.output_layer(x)
        return x


class ConditionEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(128, 64)):
        super(ConditionEncoder, self).__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dims[1], output_dim)
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, a=0.01, nonlinearity='linear')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        x = torch.clamp(x, -1e6, 1e6)
        x = self.input_norm(x)
        return self.network(x)


class SharedBilinear(nn.Module):
    def __init__(self, mol_embedding_dim, hidden_dim):
        super().__init__()
        self.bilinear = nn.Bilinear(mol_embedding_dim, mol_embedding_dim, hidden_dim)

    def forward(self, x1, x2):
        return self.bilinear(x1, x2)


class ProductPredictor(nn.Module):
    def __init__(self, shared_bilinear, condition_dim, hidden_dim):
        super().__init__()
        self.bilinear = shared_bilinear
        self.condition_proj = nn.Linear(condition_dim, hidden_dim)
        self.output_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, mol_i, mol_j, conditions):
        # Symmetric combination using shared bilinear
        interaction = 0.5 * (self.bilinear(mol_i, mol_j) + self.bilinear(mol_j, mol_i))
        cond_effect = self.condition_proj(conditions)
        combined = torch.cat([interaction, cond_effect], dim=1)
        return self.output_network(combined)


class EdgePredictor(nn.Module):
    def __init__(self, shared_bilinear, condition_dim, hidden_dim):
        super().__init__()
        self.bilinear = shared_bilinear
        self.condition_proj = nn.Linear(condition_dim, hidden_dim)
        self.output_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, mol_i, mol_j, conditions):
        interaction = self.bilinear(mol_i, mol_j)
        cond_effect = self.condition_proj(conditions)
        combined = torch.cat([interaction, cond_effect], dim=1)
        return self.output_network(combined)


class MolecularReactivityPredictor(nn.Module):
    def __init__(self, mol_input_dim, condition_input_dim, embedding_dim=16, hidden_dim=32):
        super().__init__()
        self.mol_encoder = MoleculeEncoder(
            input_dim=mol_input_dim,
            output_dim=embedding_dim,
            hidden_dims=(hidden_dim, hidden_dim // 2)
        )

        self.condition_encoder = ConditionEncoder(
            input_dim=condition_input_dim,
            output_dim=embedding_dim,
            hidden_dims=(hidden_dim, hidden_dim // 2)
        )

        self.shared_bilinear = SharedBilinear(
            mol_embedding_dim=embedding_dim,
            hidden_dim=hidden_dim
        )

        self.edge_predictor = EdgePredictor(
            shared_bilinear=self.shared_bilinear,
            condition_dim=embedding_dim,
            hidden_dim=hidden_dim
        )

        self.product_predictor = ProductPredictor(
            shared_bilinear=self.shared_bilinear,
            condition_dim=embedding_dim,
            hidden_dim=hidden_dim
        )

    def forward(self, molecules, conditions):
        # Add explicit type and value checking
        for i, mol in enumerate(molecules):
            if torch.isnan(mol).any() or torch.isinf(mol).any():
                print(f"Warning: NaN or Inf found in molecule {i} input. Replacing with zeros.")
                molecules[i] = torch.where(torch.isnan(mol) | torch.isinf(mol), torch.zeros_like(mol), mol)

        if torch.isnan(conditions).any() or torch.isinf(conditions).any():
            print("Warning: NaN or Inf found in conditions input. Replacing with zeros.")
            conditions = torch.where(torch.isnan(conditions) | torch.isinf(conditions), torch.zeros_like(conditions),
                                     conditions)

        # Apply strict value clamping
        molecules = [torch.clamp(mol, -10, 10) for mol in molecules]
        conditions = torch.clamp(conditions, -10, 10)

        # Continue with the original forward pass
        mol_embeddings = [self.mol_encoder(mol) for mol in molecules]
        condition_embedding = self.condition_encoder(conditions)
        r_ij = self.edge_predictor(mol_embeddings[0], mol_embeddings[1], condition_embedding)
        r_ji = self.edge_predictor(mol_embeddings[1], mol_embeddings[0], condition_embedding)
        r_product = self.product_predictor(mol_embeddings[0], mol_embeddings[1], condition_embedding)
        return r_ij, r_ji, r_product


class CopolymerDataset(Dataset):
    """Dataset for copolymerization data from processed DataFrame"""

    def __init__(self, df, monomer1_cols, monomer2_cols, condition_cols):
        """
        Initialize dataset from a DataFrame with monomer and condition features

        Args:
            df: DataFrame containing all data
            monomer1_cols: Column names for monomer1 features
            monomer2_cols: Column names for monomer2 features
            condition_cols: Column names for condition features
        """
        self.df = df
        self.monomer1_cols = monomer1_cols
        self.monomer2_cols = monomer2_cols
        self.condition_cols = condition_cols

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Convert data to float explicitly to handle mixed types
        mol_1_features = torch.FloatTensor([float(val) for val in row[self.monomer1_cols].values])
        mol_2_features = torch.FloatTensor([float(val) for val in row[self.monomer2_cols].values])
        condition_features = torch.FloatTensor([float(val) for val in row[self.condition_cols].values])

        # Get r values and product
        r_12 = torch.FloatTensor([float(row['constant_1'])])
        r_21 = torch.FloatTensor([float(row['constant_2'])])
        product = torch.FloatTensor([float(row['r1r2'])])

        return {
            'mol_1': mol_1_features,
            'mol_2': mol_2_features,
            'conditions': condition_features,
            'r_12': r_12,
            'r_21': r_21,
            'product': product
        }


def prepare_data_from_df(df, batch_size=32, train_split=0.8, random_state=42):
    """
    Prepare data for neural network training from a DataFrame created by the XGBoost pipeline

    Args:
        df: DataFrame with preprocessed data
        batch_size: Batch size for DataLoader
        train_split: Proportion of data to use for training
        random_state: Random seed for reproducibility

    Returns:
        train_loader, test_loader: DataLoader objects for training and testing
    """
    # First, identify feature column groups
    mol1_cols = [col for col in df.columns if col.endswith('_1')
                 and col not in ['constant_1', 'constant_conf_1', 'e_value_1', 'q_value_1', 'polytype_emb_1',
                                 'method_emb_1']]

    mol2_cols = [col for col in df.columns if col.endswith('_2')
                 and col not in ['constant_2', 'constant_conf_2', 'e_value_2', 'q_value_2', 'polytype_emb_2',
                                 'method_emb_2']]

    condition_cols = ['temperature', 'solvent_logp']

    # Add embedding columns if they exist
    if 'polytype_emb_1' in df.columns and 'polytype_emb_2' in df.columns:
        condition_cols.extend(['polytype_emb_1', 'polytype_emb_2'])

    if 'method_emb_1' in df.columns and 'method_emb_2' in df.columns:
        condition_cols.extend(['method_emb_1', 'method_emb_2'])

    # Make sure r1r2 column exists
    if 'r1r2' not in df.columns:
        df['r1r2'] = df['constant_1'] * df['constant_2']

    # All columns needed for the model
    all_cols = mol1_cols + mol2_cols + condition_cols + ['constant_1', 'constant_2', 'r1r2']

    # Drop rows with NaN values in any of the columns we need
    print(f"Original dataset size: {len(df)}")
    df_clean = df.dropna(subset=all_cols)
    print(f"Dataset size after dropping NaN values: {len(df_clean)}")

    # Check for infinite values and drop those rows too
    mask_inf = np.isinf(df_clean[all_cols]).any(axis=1)
    if mask_inf.any():
        df_clean = df_clean[~mask_inf]
        print(f"Dataset size after dropping Inf values: {len(df_clean)}")

    # Create dataset instance with the cleaned DataFrame
    dataset = CopolymerDataset(
        df=df_clean,
        monomer1_cols=mol1_cols,
        monomer2_cols=mol2_cols,
        condition_cols=condition_cols
    )

    # Create train/test split
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size

    # Set random seed for reproducibility
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True
    )

    print(f"Created dataloaders with {len(train_dataset)} training samples and {len(test_dataset)} test samples")
    print(f"Monomer 1 features: {len(mol1_cols)}")
    print(f"Monomer 2 features: {len(mol2_cols)}")
    print(f"Condition features: {len(condition_cols)}")

    return train_loader, test_loader, mol1_cols, mol2_cols, condition_cols


def create_nn_model(mol_input_dim, condition_input_dim):
    """
    Create a neural network model with the given input dimensions

    Args:
        mol_input_dim: Number of features for each monomer
        condition_input_dim: Number of features for reaction conditions

    Returns:
        model: The initialized MolecularReactivityPredictor model
    """
    model = MolecularReactivityPredictor(
        mol_input_dim=mol_input_dim,
        condition_input_dim=condition_input_dim,
        embedding_dim=64,
        hidden_dim=128
    )

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Created model with {total_params:,} trainable parameters")

    return model


class R2Loss(nn.Module):
    def __init__(self, reduction='mean'):
        super(R2Loss, self).__init__()
        self.r2_metric = R2Score()
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        """Calculate R² loss using torchmetrics implementation"""
        r2 = self.r2_metric(y_pred, y_true)
        loss = 1 - r2

        if self.reduction == 'mean':
            return loss
        elif self.reduction == 'sum':
            return loss * len(y_true)
        else:
            return loss


def train_step(model, batch, optimizer, criterion, grad_clip=0.1):
    optimizer.zero_grad()

    # Get input tensors
    mol_1 = batch['mol_1']
    mol_2 = batch['mol_2']
    conditions = batch['conditions']

    # Get target tensors
    r_12 = batch['r_12'].reshape(-1, 1)
    r_21 = batch['r_21'].reshape(-1, 1)
    true_product = batch['product'].reshape(-1, 1)

    try:
        # Forward pass with the simplified model
        r_12_pred, r_21_pred, product_pred = model([mol_1, mol_2], conditions)

        # Calculate losses
        loss_12 = criterion(r_12_pred, r_12)
        loss_21 = criterion(r_21_pred, r_21)
        loss_product = criterion(product_pred, true_product)

        # Combine losses
        loss = 0.4 * loss_12 + 0.4 * loss_21 + 0.2 * loss_product

        # Backpropagate and optimize
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        return loss.item()

    except Exception as e:
        print(f"Error in train_step: {str(e)}")
        import traceback
        traceback.print_exc()
        return float('nan')


def evaluate_model(model, dataloader):
    """
    Evaluate model performance on a dataset

    Args:
        model: The neural network model
        dataloader: DataLoader with evaluation data

    Returns:
        metrics: Dictionary with evaluation metrics
    """
    model.eval()
    all_r12_true, all_r12_pred = [], []
    all_r21_true, all_r21_pred = [], []
    all_product_true, all_product_pred = [], []

    with torch.no_grad():
        for batch in dataloader:
            mol_1 = torch.clamp(batch['mol_1'], -10, 10)
            mol_2 = torch.clamp(batch['mol_2'], -10, 10)
            conditions = torch.clamp(batch['conditions'], -10, 10)

            try:
                r_12_pred, r_21_pred, product_pred = model([mol_1, mol_2], conditions)

                # Clip predictions
                r_12_pred = torch.clamp(r_12_pred, -10, 10)
                r_21_pred = torch.clamp(r_21_pred, -10, 10)
                product_pred = torch.clamp(product_pred, -10, 10)

                # Convert tensors to numpy and collect
                all_r12_true.extend(batch['r_12'].numpy().flatten())
                all_r12_pred.extend(r_12_pred.numpy().flatten())
                all_r21_true.extend(batch['r_21'].numpy().flatten())
                all_r21_pred.extend(r_21_pred.numpy().flatten())
                all_product_true.extend(batch['product'].numpy().flatten())
                all_product_pred.extend(product_pred.numpy().flatten())

            except Exception as e:
                print(f"Error in evaluation: {str(e)}")
                continue

    # Convert to numpy arrays
    all_r12_true = np.array(all_r12_true)
    all_r12_pred = np.array(all_r12_pred)
    all_r21_true = np.array(all_r21_true)
    all_r21_pred = np.array(all_r21_pred)
    all_product_true = np.array(all_product_true)
    all_product_pred = np.array(all_product_pred)

    # Calculate metrics
    r12_mse = np.mean((all_r12_true - all_r12_pred) ** 2)
    r12_r2 = 1 - (np.sum((all_r12_true - all_r12_pred) ** 2) /
                  (np.sum((all_r12_true - np.mean(all_r12_true)) ** 2) + 1e-10))

    r21_mse = np.mean((all_r21_true - all_r21_pred) ** 2)
    r21_r2 = 1 - (np.sum((all_r21_true - all_r21_pred) ** 2) /
                  (np.sum((all_r21_true - np.mean(all_r21_true)) ** 2) + 1e-10))

    product_mse = np.mean((all_product_true - all_product_pred) ** 2)
    product_r2 = 1 - (np.sum((all_product_true - all_product_pred) ** 2) /
                      (np.sum((all_product_true - np.mean(all_product_true)) ** 2) + 1e-10))

    return {
        'r12_mse': r12_mse,
        'r12_r2': r12_r2,
        'r21_mse': r21_mse,
        'r21_r2': r21_r2,
        'product_mse': product_mse,
        'product_r2': product_r2
    }


def train_nn_model(train_loader, test_loader, model=None, num_epochs=100, learning_rate=1e-4):
    """
    Train the neural network model

    Args:
        train_loader: DataLoader with training data
        test_loader: DataLoader with test data
        model: The neural network model (if None, one will be created)
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer

    Returns:
        model: The trained model
        metrics: Dictionary with final evaluation metrics
    """
    # Create model if not provided
    if model is None:
        # Get dimensions from the first batch
        for batch in train_loader:
            mol_input_dim = batch['mol_1'].shape[1]
            condition_input_dim = batch['conditions'].shape[1]
            model = create_nn_model(mol_input_dim, condition_input_dim)
            break

    # Initialize WandB logging if available
    if WANDB_AVAILABLE:
        wandb.init(
            project="Copol_prediction",
            config={
                "learning_rate": learning_rate,
                "architecture": type(model).__name__,
                "epochs": num_epochs,
                "optimizer": "AdamW",
                "loss_metric": "R²",
            }
        )

    # Set up optimizer, loss function, and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-6,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    criterion = R2Loss(reduction='mean')

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=20,
        verbose=True,
        min_lr=1e-7
    )

    print("Starting training loop")
    # Training loop
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch}")
        model.train()
        train_loss = 0
        valid_batches = 0

        print("Starting batch processing")
        for batch_idx, batch in enumerate(train_loader):
            print(f"Processing batch {batch_idx}")
            # Print shapes to find issues
            print(f"Mol 1 shape: {batch['mol_1'].shape}, dtype: {batch['mol_1'].dtype}")
            print(f"Mol 2 shape: {batch['mol_2'].shape}, dtype: {batch['mol_2'].dtype}")
            print(f"Conditions shape: {batch['conditions'].shape}, dtype: {batch['conditions'].dtype}")

            try:
                print("Starting train_step")
                loss = train_step(model, batch, optimizer, criterion)
                print(f"Completed train_step, loss: {loss}")
                if not np.isnan(loss):
                    train_loss += loss
                    valid_batches += 1
            except Exception as e:
                print(f"Error in train_step: {e}")
                import traceback
                traceback.print_exc()
                continue

        avg_train_loss = train_loss / max(1, valid_batches)

        # Calculate metrics
        train_metrics = evaluate_model(model, train_loader)
        val_metrics = evaluate_model(model, test_loader)

        # Log to WandB if available
        if WANDB_AVAILABLE:
            wandb.log({
                "epoch": epoch,
                "avg_train_loss": avg_train_loss,
                "train/r12_r2": train_metrics['r12_r2'],
                "train/r21_r2": train_metrics['r21_r2'],
                "train/product_r2": train_metrics['product_r2'],
                "val/r12_r2": val_metrics['r12_r2'],
                "val/r21_r2": val_metrics['r21_r2'],
                "val/product_r2": val_metrics['product_r2'],
            })

        # Print metrics
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"\nEpoch {epoch}:")
            print(f"Training Metrics:")
            print(f"  R12    - R²: {train_metrics['r12_r2']:.6f}")
            print(f"  R21    - R²: {train_metrics['r21_r2']:.6f}")
            print(f"  Product- R²: {train_metrics['product_r2']:.6f}")
            print(f"Validation Metrics:")
            print(f"  R12    - R²: {val_metrics['r12_r2']:.6f}")
            print(f"  R21    - R²: {val_metrics['r21_r2']:.6f}")
            print(f"  Product- R²: {val_metrics['product_r2']:.6f}")

        # Update learning rate
        total_val_r2 = (val_metrics['r12_r2'] + val_metrics['r21_r2'] + val_metrics['product_r2']) / 3
        scheduler.step(1 - total_val_r2)

        # Check for NaN loss
        if np.isnan(avg_train_loss):
            print("Training diverged. Stopping...")
            break

    # Close WandB run if active
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()

    # Final evaluation
    final_metrics = evaluate_model(model, test_loader)

    print("\n=== Neural Network Training Completed ===")
    print(f"Final R² scores:")
    print(f"  R12: {final_metrics['r12_r2']:.6f}")
    print(f"  R21: {final_metrics['r21_r2']:.6f}")
    print(f"  Product: {final_metrics['product_r2']:.6f}")

    return model, final_metrics


class SimpleReactivityNN(nn.Module):
    def __init__(self, mol_input_dim, condition_input_dim):
        super(SimpleReactivityNN, self).__init__()

        # Combined input dimension
        total_input_dim = mol_input_dim * 2 + condition_input_dim

        # Simple feedforward network
        self.backbone = nn.Sequential(
            nn.Linear(total_input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Separate output heads
        self.r12_head = nn.Linear(32, 1)
        self.r21_head = nn.Linear(32, 1)
        self.product_head = nn.Linear(32, 1)

    def forward(self, mol1, mol2, conditions):
        # Simple concatenation of all inputs
        x = torch.cat([mol1, mol2, conditions], dim=1)

        # Apply backbone network
        x = self.backbone(x)

        # Get outputs from different heads
        r12 = self.r12_head(x)
        r21 = self.r21_head(x)
        product = self.product_head(x)

        return r12, r21, product