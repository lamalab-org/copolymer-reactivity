from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import math


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
            nn.Linear(hidden_dim, 1)
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
        # Bilinear naturally provides asymmetry
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
            hidden_dims=(hidden_dim, hidden_dim // 2)  # 64, 32
        )

        self.condition_encoder = ConditionEncoder(
            input_dim=condition_input_dim,
            output_dim=embedding_dim,
            hidden_dims=(hidden_dim, hidden_dim // 2)  # 64, 32
        )

        self.shared_bilinear = SharedBilinear(
            mol_embedding_dim=embedding_dim,  # 32
            hidden_dim=hidden_dim  # 64
        )

        self.edge_predictor = EdgePredictor(
            shared_bilinear=self.shared_bilinear,
            condition_dim=embedding_dim,  # 32
            hidden_dim=hidden_dim  # 64
        )

        self.product_predictor = ProductPredictor(
            shared_bilinear=self.shared_bilinear,
            condition_dim=embedding_dim,  # 32
            hidden_dim=hidden_dim  # 64
        )

    def forward(self, molecules, conditions):
        batch_size = molecules[0].size(0)
        mol_embeddings = [self.mol_encoder(mol) for mol in molecules]
        condition_embedding = self.condition_encoder(conditions)

        r_ij = self.edge_predictor(mol_embeddings[0], mol_embeddings[1], condition_embedding)
        r_ji = self.edge_predictor(mol_embeddings[1], mol_embeddings[0], condition_embedding)
        r_product = self.product_predictor(mol_embeddings[0], mol_embeddings[1], condition_embedding)

        return r_ij, r_ji, r_product


class MolecularReactivityDataset(Dataset):
    def __init__(self, molecule_features_df, condition_features_df, reactivity_df):
        self.mol_features = molecule_features_df
        self.condition_features = condition_features_df
        self.reactions = reactivity_df

    def __len__(self):
        return len(self.reactions)

    def __getitem__(self, idx):
        reaction = self.reactions.iloc[idx]
        mol_1_features = torch.FloatTensor(
            self.mol_features.loc[reaction['mol_1']].values
        )
        mol_2_features = torch.FloatTensor(
            self.mol_features.loc[reaction['mol_2']].values
        )
        condition_features = torch.FloatTensor(
            self.condition_features.loc[reaction['condition_id']].values
        )

        # Get r_12 and r_21 values
        r_12 = torch.FloatTensor([reaction['r_12']])
        r_21 = torch.FloatTensor([reaction['r_21']])

        # Calculate product as r_12 * r_21
        product = r_12 * r_21

        return {
            'mol_1': mol_1_features,
            'mol_2': mol_2_features,
            'conditions': condition_features,
            'r_12': r_12,
            'r_21': r_21,
            'product': product
        }


def create_dataloaders(mol_features_df, condition_features_df, reactivity_df,
                       batch_size=32, train_split=0.8):
    # Create dataset instance
    dataset = MolecularReactivityDataset(
        molecule_features_df=mol_features_df,
        condition_features_df=condition_features_df,
        reactivity_df=reactivity_df
    )

    # Create train/test split
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True  # Added to prevent batch size issues
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True  # Added to prevent batch size issues
    )

    return train_loader, test_loader


def train_step(model, batch, optimizer, criterion, grad_clip=0.1):
    optimizer.zero_grad()

    # Get data
    mol_1 = torch.clamp(batch['mol_1'], -10, 10)
    mol_2 = torch.clamp(batch['mol_2'], -10, 10)
    conditions = torch.clamp(batch['conditions'], -10, 10)
    r_12 = batch['r_12']
    r_21 = batch['r_21']
    true_product = r_12 * r_21

    try:
        # Forward pass
        r_12_pred, r_21_pred, product_pred = model([mol_1, mol_2], conditions)

        # Clip predictions
        r_12_pred = torch.clamp(r_12_pred, -10, 10)
        r_21_pred = torch.clamp(r_21_pred, -10, 10)
        product_pred = torch.clamp(product_pred, -10, 10)

        # Calculate losses
        loss_12 = criterion(r_12_pred, r_12)
        loss_21 = criterion(r_21_pred, r_21)
        loss_product = criterion(product_pred, true_product)

        # Combine losses with weights
        loss = 0.4 * loss_12 + 0.4 * loss_21 + 0.2 * loss_product

        # Check for invalid loss and backward pass
        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            return loss.item()
        return float('nan')

    except Exception as e:
        print(f"Error in train step: {str(e)}")
        return float('nan')


def train_model(model, train_loader, test_loader, num_epochs=100):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-5,
        weight_decay=1e-6,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    criterion = nn.MSELoss(reduction='mean')

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        verbose=True,
        min_lr=1e-7
    )

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        valid_batches = 0

        for batch in train_loader:
            loss = train_step(model, batch, optimizer, criterion)
            if not math.isnan(loss):
                train_loss += loss
                valid_batches += 1

        avg_train_loss = train_loss / max(1, valid_batches)

        # Calculate metrics on training data
        train_metrics = evaluate_model(model, train_loader)

        # Calculate metrics on validation data
        val_metrics = evaluate_model(model, test_loader)

        # Print metrics
        print(f"\nEpoch {epoch}:")
        print(f"Training Metrics:")
        print(f"  R12    - MSE: {train_metrics['r12_mse']:.6f}, R²: {train_metrics['r12_r2']:.6f}")
        print(f"  R21    - MSE: {train_metrics['r21_mse']:.6f}, R²: {train_metrics['r21_r2']:.6f}")
        print(f"  Product- MSE: {train_metrics['product_mse']:.6f}, R²: {train_metrics['product_r2']:.6f}")
        print(f"Validation Metrics:")
        print(f"  R12    - MSE: {val_metrics['r12_mse']:.6f}, R²: {val_metrics['r12_r2']:.6f}")
        print(f"  R21    - MSE: {val_metrics['r21_mse']:.6f}, R²: {val_metrics['r21_r2']:.6f}")
        print(f"  Product- MSE: {val_metrics['product_mse']:.6f}, R²: {val_metrics['product_r2']:.6f}")

        # Use total validation MSE for scheduler
        total_val_mse = (val_metrics['r12_mse'] + val_metrics['r21_mse'] + val_metrics['product_mse']) / 3
        scheduler.step(total_val_mse)

        if math.isnan(avg_train_loss):
            print("Training diverged. Stopping...")
            break


def calculate_metrics(y_true, y_pred):
    """Calculate MSE and R² for given true and predicted values."""
    # Input is already numpy array, no need for detach().numpy()

    # Calculate MSE
    mse = np.mean((y_true - y_pred) ** 2)

    # Calculate R²
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-10))  # Add small epsilon to prevent division by zero

    return mse, r2


def evaluate_model(model, dataloader):
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
                all_product_true.extend((batch['r_12'] * batch['r_21']).numpy().flatten())
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
    r12_mse, r12_r2 = calculate_metrics(all_r12_true, all_r12_pred)
    r21_mse, r21_r2 = calculate_metrics(all_r21_true, all_r21_pred)
    product_mse, product_r2 = calculate_metrics(all_product_true, all_product_pred)

    return {
        'r12_mse': r12_mse,
        'r12_r2': r12_r2,
        'r21_mse': r21_mse,
        'r21_r2': r21_r2,
        'product_mse': product_mse,
        'product_r2': product_r2
    }


def prepare_data_from_csv(csv_path):
    print("Loading data from CSV...")
    df = pd.read_csv(csv_path)
    print(f"Loaded data shape: {df.shape}")

    # Define column groups
    mol1_cols = [col for col in df.columns if col.startswith('monomer1_data_')]
    mol2_cols = [col for col in df.columns if col.startswith('monomer2_data_')]
    condition_cols = ['LogP', 'temperature', 'method_1', 'method_2',
                      'polymerization_type_1', 'polymerization_type_2']
    target_cols = ['r1', 'r2', 'r-product']

    print("\nFound column groups:")
    print(f"Molecule 1 features: {mol1_cols}")
    print(f"Molecule 2 features: {mol2_cols}")
    print(f"Condition features: {condition_cols}")
    print(f"Target variables: {target_cols}")

    # Fill NaN values
    df[mol1_cols] = df[mol1_cols].fillna(0)
    df[mol2_cols] = df[mol2_cols].fillna(0)
    df[['LogP', 'temperature', 'method_1', 'method_2', 'polymerization_type_1', 'polymerization_type_2']] = df[['LogP', 'temperature', 'method_1', 'method_2', 'polymerization_type_1', 'polymerization_type_2']].fillna(0)

    # Create molecule features dataframe
    mol1_features = df[mol1_cols].drop_duplicates()
    mol2_features = df[mol2_cols].drop_duplicates()

    # Create mappings with index as the key
    mol1_features.index = [f'mol1_{i}' for i in range(len(mol1_features))]
    mol2_features.index = [f'mol2_{i}' for i in range(len(mol2_features))]

    # Create molecule mappings using the index
    mol1_mapping = {
        tuple(row.values): idx
        for idx, row in mol1_features.iterrows()
    }
    mol2_mapping = {
        tuple(row.values): idx
        for idx, row in mol2_features.iterrows()
    }

    # Combine molecule features
    mol1_features_renamed = mol1_features.rename(
        columns={col: col.replace('monomer1_data_', '') for col in mol1_cols}
    )
    mol2_features_renamed = mol2_features.rename(
        columns={col: col.replace('monomer2_data_', '') for col in mol2_cols}
    )
    mol_features = pd.concat([mol1_features_renamed, mol2_features_renamed])

    # Create condition features
    unique_conditions = df[condition_cols].drop_duplicates()
    unique_conditions.index = [f'cond_{i}' for i in range(len(unique_conditions))]
    condition_features = unique_conditions

    # Create condition mapping
    condition_mapping = {}
    for idx, row in unique_conditions.iterrows():
        key = tuple(
            str(val) if isinstance(val, str) else round(val, 6) if isinstance(val, float) else val
            for val in row.values
        )
        condition_mapping[key] = idx

    # Create reactions dataframe
    reactions = []
    for _, row in df.iterrows():
        mol1_key = tuple(row[mol1_cols].values)
        mol2_key = tuple(row[mol2_cols].values)
        cond_key = tuple(
            str(val) if isinstance(val, str) else round(val, 6) if isinstance(val, float) else val
            for val in row[condition_cols].values
        )

        reaction = {
            'mol_1': mol1_mapping[mol1_key],
            'mol_2': mol2_mapping[mol2_key],
            'condition_id': condition_mapping[cond_key],
            'r_12': float(row['r1']),
            'r_21': float(row['r2']),
            'product': float(row['r-product'])
        }
        reactions.append(reaction)

    reactivity_df = pd.DataFrame(reactions)

    print(f"\nCreated DataFrames:")
    print(f"Molecule features shape: {mol_features.shape}")
    print(f"Condition features shape: {condition_features.shape}")
    print(f"Reactivity data shape: {reactivity_df.shape}")

    return mol_features, condition_features, reactivity_df


def init_model(mol_input_dim, condition_input_dim):
    # Create model with proper initialization
    model = MolecularReactivityPredictor(
        mol_input_dim=mol_input_dim,
        condition_input_dim=condition_input_dim,
        embedding_dim=64,  # Reduced from original
        hidden_dim=128  # Kept the same
    )

    # Initialize weights with smaller values
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=0.1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    model.apply(init_weights)

    # Print model parameter statistics
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_params += param.numel()
            print(f"{name}: {param.shape}, {torch.min(param).item():.4f} to {torch.max(param).item():.4f}")

    print(f"\nTotal trainable parameters: {total_params:,}")
    return model


def main():
    print("\n=== Starting Molecular Reactivity Prediction ===\n")

    # Load and prepare data
    mol_features, condition_features, reactivity_df = prepare_data_from_csv('output/processed_data_copol.csv')

    # Normalize the features
    mol_scaler = StandardScaler()
    condition_scaler = StandardScaler()

    mol_features_scaled = pd.DataFrame(
        mol_scaler.fit_transform(mol_features),
        index=mol_features.index,
        columns=mol_features.columns
    )

    condition_features_scaled = pd.DataFrame(
        condition_scaler.fit_transform(condition_features),
        index=condition_features.index,
        columns=condition_features.columns
    )

    # Create dataloaders
    train_loader, test_loader = create_dataloaders(
        mol_features_scaled,
        condition_features_scaled,
        reactivity_df,
        batch_size=32
    )

    # Initialize model
    model = init_model(
        mol_input_dim=len(mol_features.columns),
        condition_input_dim=len(condition_features.columns)
    )

    # Train model
    train_model(model, train_loader, test_loader, num_epochs=100)


if __name__ == "__main__":
    main()