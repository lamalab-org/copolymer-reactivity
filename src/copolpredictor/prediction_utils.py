import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold

feature_columns = [
    # Molecular descriptors for Monomer 1
    "charges_min_1",
    "fukui_electrophilicity_min_1",
    "fukui_electrophilicity_max_1",
    "fukui_nucleophilicity_min_1",
    "fukui_radical_min_1",
    "homo_1",
    # Molecular descriptors for Monomer 2
    "charges_min_2",
    "fukui_electrophilicity_min_2",
    "fukui_electrophilicity_max_2",
    "fukui_nucleophilicity_min_2",
    "fukui_radical_min_2",
    "homo_2",
    # HOMO-LUMO differences
    "delta_HOMO_LUMO_AB",
    "delta_HOMO_LUMO_BA",
    # Reaction conditions
    "temperature",
    "polytype_emb_1",
    "polytype_emb_2",
    "solvent_logp",
    "solvent_FractionCSP3",
]


feature_columns_all = [
    # Molecular descriptors for Monomer 1
    "best_conformer_energy_1",
    "ip_1",
    "ip_corrected_1",
    "ea_1",
    "homo_1",
    "lumo_1",
    "global_electrophilicity_1",
    "global_nucleophilicity_1",
    "charges_min_1",
    "charges_max_1",
    "charges_mean_1",
    "fukui_electrophilicity_min_1",
    "fukui_electrophilicity_max_1",
    "fukui_electrophilicity_mean_1",
    "fukui_nucleophilicity_min_1",
    "fukui_nucleophilicity_max_1",
    "fukui_nucleophilicity_mean_1",
    "fukui_radical_min_1",
    "fukui_radical_max_1",
    "fukui_radical_mean_1",
    "dipole_x_1",
    "dipole_y_1",
    "dipole_z_1",
    # Molecular descriptors for Monomer 2
    "best_conformer_energy_2",
    "ip_2",
    "ip_corrected_2",
    "ea_2",
    "homo_2",
    "lumo_2",
    "global_electrophilicity_2",
    "global_nucleophilicity_2",
    "charges_min_2",
    "charges_max_2",
    "charges_mean_2",
    "fukui_electrophilicity_min_2",
    "fukui_electrophilicity_max_2",
    "fukui_electrophilicity_mean_2",
    "fukui_nucleophilicity_min_2",
    "fukui_nucleophilicity_max_2",
    "fukui_nucleophilicity_mean_2",
    "fukui_radical_min_2",
    "fukui_radical_max_2",
    "fukui_radical_mean_2",
    "dipole_x_2",
    "dipole_y_2",
    "dipole_z_2",
    # HOMO-LUMO differences
    "delta_HOMO_LUMO_AA",
    "delta_HOMO_LUMO_AB",
    "delta_HOMO_LUMO_BB",
    "delta_HOMO_LUMO_BA",
    # Other features
    "temperature",
    "solvent_logp",
    "polytype_emb_1",
    "polytype_emb_2",
    "method_emb_1",
    "method_emb_2",
    "solvent_TPSA",
    "solvent_HBD",
    "solvent_FractionCSP3",
]


def compute_quality_weighted_accuracy(y_true, y_pred, num_classes=4):
    """
    Compute the quality-weighted accuracy:
    For each class: (class proportion) * (class accuracy), then average over all classes.

    Args:
        y_true (array-like): True class labels
        y_pred (array-like): Predicted class labels
        num_classes (int): Total number of classes

    Returns:
        float: Quality-weighted accuracy
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    total_samples = len(y_true)
    weighted_components = []

    for cls in range(num_classes):
        cls_mask = y_true == cls
        cls_total = cls_mask.sum()
        if cls_total > 0:
            cls_accuracy = accuracy_score(y_true[cls_mask], y_pred[cls_mask])
            component = (cls_total / total_samples) * cls_accuracy
        else:
            component = 0.0
        weighted_components.append(component)

    quality_weighted_accuracy = sum(weighted_components) / num_classes
    return quality_weighted_accuracy


def create_grouped_kfold_splits(df, n_splits=5, id_column="reaction_id"):
    """
    Create K-Fold splits that ensure no data leakage by keeping all rows
    with the same group ID (e.g., reaction_id) in the same fold.

    Parameters:
        df (pd.DataFrame): Full dataset.
        n_splits (int): Number of folds.
        random_state (int): Random seed for reproducibility.
        id_column (str): Column name used for grouping (e.g., 'reaction_id').

    Returns:
        List of (train_indices, test_indices) tuples.
    """
    if id_column not in df.columns:
        raise ValueError(f"Grouping column '{id_column}' not found in the dataframe.")

    # Use GroupKFold to avoid data leakage
    gkf = GroupKFold(n_splits=n_splits)

    # Create consistent group array
    group_ids = df[id_column].values

    # Dummy y (not used by GroupKFold)
    dummy_y = np.zeros(len(df))

    # Collect train/test splits
    splits = []
    for train_idx, test_idx in gkf.split(df, dummy_y, groups=group_ids):
        splits.append((train_idx, test_idx))

    return splits


def plot_weighted_accuracy_learning_curve(
    X_train,
    y_train,
    X_val,
    y_val,
    class_weights,
    best_params,
    output_path="output/learning_curve_accuracy.png",
    train_sizes=None,
    random_state=42,
):
    """
    Train XGBoost models on increasing amounts of training data and plot weighted class accuracy.

    Parameters:
        X_train, y_train: full training set
        X_val, y_val: fixed validation set
        class_weights: dict mapping class labels to weights
        best_params: best XGBoost parameters (from CV)
        output_path: where to save the output plot
        train_sizes: list of training sizes to test (default = auto)
        random_state: reproducibility seed
    """

    if train_sizes is None:
        train_sizes = [10, 100, 250, 500, 1000, 2000, 3000, len(X_train)]

    accuracies = []

    # Shuffle training indices for sampling
    np.random.seed(random_state)
    permuted_idx = np.random.permutation(len(X_train))

    for size in train_sizes:
        idx_subset = permuted_idx[:size]
        X_subset = X_train.iloc[idx_subset]
        y_subset = y_train.iloc[idx_subset]

        # Sample weights for the subset
        sample_weights_subset = np.array([class_weights[label] for label in y_subset])

        # Train new XGBoost model
        model = xgb.XGBClassifier(
            **best_params, random_state=random_state, use_label_encoder=False, eval_metric="logloss"
        )
        model.fit(X_subset, y_subset, sample_weight=sample_weights_subset)

        # Clean validation set (remove rows with missing values if needed)
        X_val_clean = X_val.dropna()
        y_val_clean = y_val.loc[X_val_clean.index]
        y_pred = model.predict(X_val_clean)

        # Compute per-class accuracy
        cm = confusion_matrix(y_val_clean, y_pred, labels=[0, 1, 2])
        acc_per_class = cm.diagonal() / cm.sum(axis=1)

        # Weighted class accuracy (equal weights)
        weighted_acc = (acc_per_class * np.array([1 / 3, 1 / 3, 1 / 3])).sum()
        accuracies.append(weighted_acc)

        print(f"✔️ Trained with {size} samples → Weighted Accuracy: {weighted_acc:.4f}")

    # Plot learning curve
    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, accuracies, marker="o")
    plt.xlabel("Training Set Size")
    plt.ylabel("Weighted Accuracy (macro-average)")
    plt.title("Learning Curve (fixed validation set)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()

    return train_sizes, accuracies
