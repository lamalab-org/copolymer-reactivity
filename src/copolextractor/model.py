from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import math
import wandb
from sklearn.preprocessing import PowerTransformer
from torchmetrics import R2Score


wandb.init(
    project="Copol_prediction",
    config={
    "architecture": "graph-like",
    "dataset": "copol dataset"
    }
)


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


class R2Loss(nn.Module):
    def __init__(self, reduction='mean'):
        super(R2Loss, self).__init__()
        self.r2_metric = R2Score()
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        """
        Calculate R² loss using torchmetrics implementation
        """
        r2 = self.r2_metric(y_pred, y_true)
        loss = 1 - r2

        if self.reduction == 'mean':
            return loss
        elif self.reduction == 'sum':
            return loss * len(y_true)
        else:
            return loss


def train_model(model, train_loader, test_loader, num_epochs=100, overfit_mode=True):

    criterion = R2Loss(reduction='mean')
    transformers = None
    scheduler = None

    if overfit_mode:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=5e-4,
            weight_decay=0.0
        )
        train_loader = create_overfit_loader(train_loader, num_samples=16)
        test_loader = train_loader

        print(f"Overfitting on {len(train_loader.dataset)} samples")

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            valid_batches = 0

            for batch in train_loader:
                loss = train_step(model, batch, optimizer, criterion, transformers)
                if not math.isnan(loss):
                    train_loss += loss
                    valid_batches += 1

            avg_train_loss = train_loss / max(1, valid_batches)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss (1 - R²): {avg_train_loss:.8f}")

            if avg_train_loss < 0.01:
                print(f"Successfully overfit at epoch {epoch} with R² > {1 - avg_train_loss:.8f}")
                break
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=1e-6,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=20,
            verbose=True,
            min_lr=1e-7
        )

        # Initialize and fit PowerTransformer on training data
        pt_r12 = PowerTransformer(method="yeo-johnson")
        pt_r21 = PowerTransformer(method="yeo-johnson")
        pt_product = PowerTransformer(method="yeo-johnson")

        all_r12 = []
        all_r21 = []
        all_products = []

        for batch in train_loader:
            r_12 = batch['r_12'].numpy().reshape(-1, 1)
            r_21 = batch['r_21'].numpy().reshape(-1, 1)
            product = (r_12.ravel() * r_21.ravel()).reshape(-1, 1)

            all_r12.append(r_12)
            all_r21.append(r_21)
            all_products.append(product)

        # Fit separate transformers
        all_r12 = np.concatenate(all_r12, axis=0)
        all_r21 = np.concatenate(all_r21, axis=0)
        all_products = np.concatenate(all_products, axis=0)

        pt_r12.fit(all_r12)
        pt_r21.fit(all_r21)
        pt_product.fit(all_products)

        transformers = {
            'r12': pt_r12,
            'r21': pt_r21,
            'product': pt_product
        }

        config = {
            "learning_rate": optimizer.param_groups[0]['lr'],
            "architecture": type(model).__name__,
            "epochs": num_epochs,
            "optimizer": type(optimizer).__name__,
            "weight_decay": optimizer.param_groups[0]['weight_decay'],
            "scheduler": type(scheduler).__name__,
            "scheduler_patience": scheduler.patience,
            "scheduler_factor": scheduler.factor,
            "min_lr": scheduler.min_lrs[0],
            "loss_weights": {
                "r12": 0.4,
                "r21": 0.4,
                "product": 0.2
            },
            "batch_size": train_loader.batch_size if hasattr(train_loader,
                                                             'batch_size') else train_loader.dataset.batch_size,
            # "power_transformer": transformers.method
        }

        # Initialize wandb with dynamic config
        wandb.init(
            project="Copol_prediction",
            config={
                "learning_rate": optimizer.param_groups[0]['lr'],
                "architecture": type(model).__name__,
                "epochs": num_epochs,
                "optimizer": type(optimizer).__name__,
                "loss_metric": "R²",  # Updated to show we're using R²
                # ... rest of your config ...
            }
        )

        wandb.define_metric("train/r12_r2", summary="max")  # Now we want to maximize R²
        wandb.define_metric("val/r12_r2", summary="max")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        valid_batches = 0

        for batch in train_loader:
            loss = train_step(model, batch, optimizer, criterion, transformers)
            if not math.isnan(loss):
                train_loss += loss
                valid_batches += 1

        avg_train_loss = train_loss / max(1, valid_batches)

        # Calculate metrics
        train_metrics = evaluate_model(model, train_loader)
        val_metrics = evaluate_model(model, test_loader)

        wandb.log({
            "epoch": epoch,
            "avg_train_loss": avg_train_loss,

            "train/r12_r2": train_metrics['r12_r2'],
            "train/r21_r2": train_metrics['r21_r2'],
            "train/product_r2": train_metrics['product_r2'],

            # Validation metrics
            "val/r12_r2": val_metrics['r12_r2'],
            "val/r21_r2": val_metrics['r21_r2'],
            "val/product_r2": val_metrics['product_r2'],

            "r2_comparison": wandb.plot.line_series(
                xs=[epoch, epoch],
                ys=[[train_metrics['r12_r2']], [val_metrics['r12_r2']]],
                keys=["Train R²", "Val R²"],
                title="R² Comparison",
                xname="epoch"
            )
        })

        metrics_table = wandb.Table(columns=["Metric", "Train", "Validation"])
        metrics_table.add_data("R12 R²", train_metrics['r12_r2'], val_metrics['r12_r2'])
        metrics_table.add_data("R21 R²", train_metrics['r21_r2'], val_metrics['r21_r2'])
        metrics_table.add_data("Product R²", train_metrics['product_r2'], val_metrics['product_r2'])
        wandb.log({"metrics_summary": metrics_table})

        print(f"\nEpoch {epoch}:")
        print(f"Training Metrics:")
        print(f"  R12    - R²: {train_metrics['r12_r2']:.6f}")
        print(f"  R21    - R²: {train_metrics['r21_r2']:.6f}")
        print(f"  Product- R²: {train_metrics['product_r2']:.6f}")
        print(f"Validation Metrics:")
        print(f"  R12    - R²: {val_metrics['r12_r2']:.6f}")
        print(f"  R21    - R²: {val_metrics['r21_r2']:.6f}")
        print(f"  Product- R²: {val_metrics['product_r2']:.6f}")

        total_val_r2 = (val_metrics['r12_r2'] + val_metrics['r21_r2'] + val_metrics['product_r2']) / 3
        scheduler.step(1 - total_val_r2)

        if math.isnan(avg_train_loss):
            print("Training diverged. Stopping...")
            break


def visualize_predictions(model, dataloader, num_samples=200):
    model.eval()
    r12_true, r12_pred = [], []
    r21_true, r21_pred = [], []
    product_true, product_pred = [], []

    with torch.no_grad():
        for batch in dataloader:
            # Get predictions
            r_12_p, r_21_p, product_p = model([batch['mol_1'], batch['mol_2']], batch['conditions'])

            # Store true and predicted values
            r12_true.extend(batch['r_12'].numpy().flatten())
            r12_pred.extend(r_12_p.numpy().flatten())
            r21_true.extend(batch['r_21'].numpy().flatten())
            r21_pred.extend(r_21_p.numpy().flatten())
            true_product = (batch['r_12'] * batch['r_21']).numpy().flatten()
            product_true.extend(true_product)
            product_pred.extend(product_p.numpy().flatten())

            if len(r12_true) >= num_samples:
                break

    # Truncate to desired number of samples
    r12_true = r12_true[:num_samples]
    r12_pred = r12_pred[:num_samples]
    r21_true = r21_true[:num_samples]
    r21_pred = r21_pred[:num_samples]
    product_true = product_true[:num_samples]
    product_pred = product_pred[:num_samples]

    # Create artifact for visualization
    wandb.init(project="Copol_prediction", name="prediction_analysis")

    # Log scatter plots to wandb
    wandb.log({
        "r12_scatter": wandb.plot.scatter(
            wandb.Table(data=[[x, y] for x, y in zip(r12_true, r12_pred)],
                        columns=["True r12", "Predicted r12"]),
            "True r12",
            "Predicted r12",
            title="r12: True vs Predicted"
        ),
        "r21_scatter": wandb.plot.scatter(
            wandb.Table(data=[[x, y] for x, y in zip(r21_true, r21_pred)],
                        columns=["True r21", "Predicted r21"]),
            "True r21",
            "Predicted r21",
            title="r21: True vs Predicted"
        ),
        "product_scatter": wandb.plot.scatter(
            wandb.Table(data=[[x, y] for x, y in zip(product_true, product_pred)],
                        columns=["True product", "Predicted product"]),
            "True product",
            "Predicted product",
            title="Product: True vs Predicted"
        )
    })

    # Calculate and log error metrics
    r12_rmse = np.sqrt(np.mean((np.array(r12_true) - np.array(r12_pred)) ** 2))
    r21_rmse = np.sqrt(np.mean((np.array(r21_true) - np.array(r21_pred)) ** 2))
    product_rmse = np.sqrt(np.mean((np.array(product_true) - np.array(product_pred)) ** 2))

    wandb.log({
        "r12_rmse": r12_rmse,
        "r21_rmse": r21_rmse,
        "product_rmse": product_rmse
    })

    # Print summary statistics
    print("\nPrediction Summary:")
    print(f"R12 RMSE: {r12_rmse:.4f}")
    print(f"R21 RMSE: {r21_rmse:.4f}")
    print(f"Product RMSE: {product_rmse:.4f}")

    wandb.finish()


def train_step(model, batch, optimizer, criterion, transformers=None, grad_clip=0.1):
    optimizer.zero_grad()

    mol_1 = batch['mol_1']
    mol_2 = batch['mol_2']
    conditions = batch['conditions']

    # Ensure targets have shape [batch_size, 1]
    r_12 = batch['r_12'].reshape(-1, 1)
    r_21 = batch['r_21'].reshape(-1, 1)
    true_product = (r_12 * r_21).reshape(-1, 1)

    if transformers is not None:
        # Transform if transformer is available
        r_12 = torch.tensor(transformers['r12'].transform(r_12), dtype=torch.float32)
        r_21 = torch.tensor(transformers['r21'].transform(r_21), dtype=torch.float32)
        true_product = torch.tensor(transformers['product'].transform(true_product), dtype=torch.float32)

    try:
        # Forward pass
        r_12_pred, r_21_pred, product_pred = model([mol_1, mol_2], conditions)

        # Ensure predictions have shape [batch_size, 1]
        r_12_pred = r_12_pred.reshape(-1, 1)
        r_21_pred = r_21_pred.reshape(-1, 1)
        product_pred = product_pred.reshape(-1, 1)

        # Calculate losses
        loss_12 = criterion(r_12_pred, r_12)
        loss_21 = criterion(r_21_pred, r_21)
        loss_product = criterion(product_pred, true_product)

        # Combine losses with weights
        loss = 0.4 * loss_12 + 0.4 * loss_21 + 0.2 * loss_product
        print(f"Loss r1: {loss_12}, loss r2: {loss_21}, loss product: {loss_product}")
        print(f"Overall loss: {loss}")

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


def calculate_metrics(y_true, y_pred):
    """Calculate MSE and R² for given true and predicted values."""

    # Calculate MSE
    mse = np.mean((y_true - y_pred) ** 2)

    # Calculate R²
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-10))

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
            'product': float(row['r1']) * float(row['r2'])
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
        embedding_dim=64,
        hidden_dim=128
    )

    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    model.apply(init_weights)

    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_params += param.numel()
            print(f"{name}: {param.shape}, {torch.min(param).item():.4f} to {torch.max(param).item():.4f}")

    print(f"\nTotal trainable parameters: {total_params:,}")
    return model


def create_overfit_loader(train_loader, num_samples=16):
    # Get one batch of data
    for batch in train_loader:
        overfit_data = {
            'mol_1': batch['mol_1'],
            'mol_2': batch['mol_2'],
            'conditions': batch['conditions'],
            'r_12': batch['r_12'],
            'r_21': batch['r_21'],
            'product': batch['r_12'] * batch['r_21']
        }
        break

    class OverfitDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return num_samples

        def __getitem__(self, idx):
            return {k: v[idx % len(v)] if isinstance(v, torch.Tensor) else v
                    for k, v in self.data.items()}

    # Create dataset and loader
    dataset = OverfitDataset(overfit_data)
    overfit_loader = DataLoader(
        dataset,
        batch_size=min(32, num_samples),
        shuffle=True
    )

    return overfit_loader


def train_overfit(model, train_loader, num_epochs=100):
    # Get small dataset
    overfit_loader = create_overfit_loader(train_loader)

    # Configure for overfitting
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Remove regularization
    for param in model.parameters():
        if hasattr(param, 'requires_grad'):
            param.requires_grad = True

    # Train with very small dataset
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for batch in overfit_loader:
            optimizer.zero_grad()

            r_12_pred, r_21_pred, product_pred = model([batch['mol_1'], batch['mol_2']], batch['conditions'])

            loss_12 = criterion(r_12_pred, batch['r_12'])
            loss_21 = criterion(r_21_pred, batch['r_21'])
            loss_product = criterion(product_pred, batch['r_12'] * batch['r_21'])

            loss = loss_12 + loss_21 + loss_product
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {train_loss:.6f}")
            # Visualize predictions on overfit data
            visualize_predictions(model, overfit_loader, num_samples=32)


def main(training_data):
    print("\n=== Starting Molecular Reactivity Prediction ===\n")

    # Load and prepare data
    mol_features, condition_features, reactivity_df = prepare_data_from_csv(
        training_data)

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
    train_model(model, train_loader, test_loader, num_epochs=500, overfit_mode=True)

    # Visualize predictions
    visualize_predictions(model, test_loader, num_samples=1000)
