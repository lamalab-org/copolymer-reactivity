"""
Simplified neural network module for copolymerization prediction
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class SimpleDataset(Dataset):
    """Simple dataset for copolymerization data"""

    def __init__(self, df, feature_cols, target_cols):
        """Initialize with DataFrame and column names"""
        self.features = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.targets = torch.tensor(df[target_cols].values, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class SimpleNN(nn.Module):
    """Simple feedforward neural network"""

    def __init__(self, input_size, hidden_size=64, output_size=3):
        super(SimpleNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):
        return self.model(x)


def prepare_data(df, batch_size=16):
    """Prepare data for neural network training"""
    # Create feature and target columns
    monomer1_cols = [col for col in df.columns if col.endswith('_1')
                     and col not in ['constant_1', 'polytype_emb_1', 'method_emb_1']]

    monomer2_cols = [col for col in df.columns if col.endswith('_2')
                     and col not in ['constant_2', 'polytype_emb_2', 'method_emb_2']]

    # Add condition columns
    condition_cols = ['temperature', 'solvent_logp']
    if 'polytype_emb_1' in df.columns:
        condition_cols.extend(['polytype_emb_1', 'polytype_emb_2'])
    if 'method_emb_1' in df.columns:
        condition_cols.extend(['method_emb_1', 'method_emb_2'])

    # Ensure target columns
    if 'r1r2' not in df.columns:
        df['r1r2'] = df['constant_1'] * df['constant_2']

    target_cols = ['constant_1', 'constant_2', 'r1r2']

    # Clean the data
    feature_cols = monomer1_cols + monomer2_cols + condition_cols
    all_cols = feature_cols + target_cols

    print(f"Original data shape: {df.shape}")

    # Drop rows with NaN or inf
    df_clean = df.dropna(subset=all_cols)
    print(f"Shape after dropping NaNs: {df_clean.shape}")

    # Remove infinities
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna(subset=all_cols)
    print(f"Shape after removing infinities: {df_clean.shape}")

    # Create datasets
    dataset = SimpleDataset(df_clean, feature_cols, target_cols)

    # Split into train/test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader, len(feature_cols)


def train_simple_nn(df, epochs=50, lr=0.001, batch_size=16):
    """Train a simple neural network on the data"""
    try:
        # Prepare data
        train_loader, test_loader, input_size = prepare_data(df, batch_size)

        # Create model
        model = SimpleNN(input_size=input_size)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Training loop
        for epoch in range(epochs):
            model.train()
            train_loss = 0

            for features, targets in train_loader:
                # Forward pass
                outputs = model(features)
                loss = criterion(outputs, targets)

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for features, targets in test_loader:
                    outputs = model(features)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()

            # Print progress
            if (epoch + 1) % 5 == 0:
                print(
                    f'Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(test_loader):.4f}')

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

        # Calculate R² scores
        from sklearn.metrics import r2_score
        metrics = {
            'r12_r2': r2_score(r12_true, r12_pred),
            'r21_r2': r2_score(r21_true, r21_pred),
            'product_r2': r2_score(prod_true, prod_pred)
        }

        print("\nFinal Metrics:")
        print(f"R12 R²: {metrics['r12_r2']:.4f}")
        print(f"R21 R²: {metrics['r21_r2']:.4f}")
        print(f"Product R²: {metrics['product_r2']:.4f}")

        return model, metrics

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        raise
