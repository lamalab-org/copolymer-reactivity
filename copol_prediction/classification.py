import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from sklearn.metrics import (
    classification_report, confusion_matrix
)
import xgboost as xgb
import matplotlib
import warnings
import datetime

from copolpredictor import data_processing
from error_analysis import perform_error_analysis
from copolpredictor.prediction_utils import feature_columns, compute_quality_weighted_accuracy, create_grouped_kfold_splits
from copolpredictor.data_augmentation import augment_with_gaussian_samples
from permutation_analysis import run_permutation_analysis, create_feature_importance_comparison


warnings.filterwarnings('ignore')
matplotlib.use('Agg')  # Set non-interactive backend


TEST_GROUPS_PATH = "artifacts/test_ids.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42
HOLDOUT_RESULTS_DIR = "artifacts/experiments_holdout"


from pathlib import Path
import shutil, logging
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.metrics import f1_score, make_scorer, accuracy_score, precision_score, recall_score
import joblib


def save_model_bundle(final_model, feature_list, class_labels=(0,1,2), out_dir="artifacts/model_bundle"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, f"{out_dir}/model.joblib")
    # optional native booster
    try:
        final_model.get_booster().save_model(f"{out_dir}/model.xgb.json")
    except Exception as e:
        print(f"[WARN] booster save failed: {e}")
    meta = {
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "feature_columns": list(feature_list),
        "class_labels": list(class_labels),
        "task": "multiclass_r_product_class",
    }
    with open(f"{out_dir}/meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def finalize_selected_filters_and_save(
    df: pd.DataFrame,
    *,
    remove_specialized: bool,
    apply_polymerization_filter: bool,
    add_negative_data: bool,
    use_augmentation: bool,
    random_state: int = 42,
    augmented_points_per_row: int = 5,
    bundle_dir: str = "artifacts/model_bundle",
    holdout_filename: str = "holdout_selected_filters.json",
    search_space: dict | None = None,
    n_search_iter: int = 25,
):
    """
    Use *your chosen filter switches* only.
    Then do a focused hyperparameter search (grouped CV), retrain, evaluate on persistent hold-out,
    and save a deployment bundle for the Streamlit app.
    """
    log = logging.getLogger(__name__)
    log.info("=== Finalizing selected filters (your choice) ===")

    # ----- Build filtered dataset (same logic as in run_classification) -----
    base_df = make_base_dataset_for_holdout(df)

    w = df.copy()
    w = w[w['r1r2'].notna()]
    w = w[w['r1r2'] >= 0]

    if remove_specialized and "llm_specialized_filter" in w.columns:
        w = w[w["llm_specialized_filter"] != "specialized"]

    # target
    bins = [-np.inf, 1, 25, np.inf]; labels = [0,1,2]
    w['r_product_class'] = pd.cut(w['r1r2'], bins=bins, labels=labels, right=False).astype(int)

    if {'constant_1','constant_2'}.issubset(w.columns):
        extreme_mask = (
            ((w['constant_1'] <= 0.1) & (w['constant_2'] > 25)) |
            ((w['constant_2'] <= 0.1) & (w['constant_1'] > 25))
        )
        w.loc[extreme_mask, 'r_product_class'] = 2

    if 'reaction_id' not in w.columns:
        raise ValueError("reaction_id required for grouped split")

    if add_negative_data:
        add_path = "artificial_datapoints/processed_combined_augmented.csv"
        add_df = pd.read_csv(add_path)
        if 'Class' not in add_df.columns:
            raise ValueError("additional CSV must contain 'Class'")
        add_df = add_df.rename(columns={'Class': 'r_product_class'})
        add_df['r_product_class'] = add_df['r_product_class'].astype(int)
        w = pd.concat([w, add_df], ignore_index=True)

    available_features = [c for c in feature_columns if c in w.columns]

    X_all = w[available_features]
    y_all = w['r_product_class'].astype(int)
    mask = ~(pd.isna(X_all).any(axis=1) | pd.isna(y_all))
    w = w[mask].reset_index(drop=True)

    holdout_groups = get_or_create_holdout_groups(base_df, group_col="reaction_id")
    df_hold = w[w['reaction_id'].astype(str).isin(holdout_groups)].reset_index(drop=True)
    df_train = w[~w['reaction_id'].astype(str).isin(holdout_groups)].reset_index(drop=True)

    # class weights
    counts = df_train['r_product_class'].value_counts().sort_index()
    total = len(df_train); k = counts.shape[0]
    cls_w = {int(c): round(total/(k*cnt), 4) for c, cnt in counts.items()}

    # augmentation (train only)
    if use_augmentation:
        df_train_aug = augment_with_gaussian_samples(
            df_train, num_samples=augmented_points_per_row, std_factor=0.3, random_state=random_state
        )
    else:
        df_train_aug = df_train

    X_train = df_train_aug[available_features]
    y_train = df_train_aug['r_product_class'].astype(int).values
    w_train = np.array([cls_w[int(lbl)] for lbl in y_train])

    # ----- Hyperparameter search (focused; grouped CV to avoid leakage) -----
    base_model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        random_state=random_state,
    )

    if search_space is None:
        # sensible, compact space
        search_space = {
            'n_estimators': [300, 500, 700],
            'max_depth': [4, 5, 6],
            'learning_rate': [0.035, 0.05, 0.07],
            'subsample': [0.85, 0.9, 0.95],
            'colsample_bytree': [0.85, 0.9, 1.0],
            'reg_alpha': [0.0, 0.1, 0.3],
            'reg_lambda': [1.0, 1.5, 2.0],
            'min_child_weight': [2, 3, 5],
            'gamma': [0.3, 0.5, 0.7],
        }

    gkf = GroupKFold(n_splits=5)
    groups = df_train_aug['reaction_id'].astype(str).values
    scorer = make_scorer(f1_score, average='weighted')

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=search_space,
        n_iter=min(n_search_iter, sum(len(v) for v in search_space.values())),
        scoring=scorer,
        cv=gkf.split(X_train, y_train, groups),
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train, sample_weight=w_train)
    tuned = search.best_params_
    print("[FINALIZE] best params after focused search:", tuned)

    # ----- Train final model on full train pool -----
    final_model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        random_state=random_state,
        **tuned
    )
    final_model.fit(X_train, y_train, sample_weight=w_train)

    # ----- Evaluate on hold-out and persist -----
    if len(df_hold) > 0:
        X_hold = df_hold[available_features]
        y_hold = df_hold['r_product_class'].astype(int).values
        y_pred = final_model.predict(X_hold)
        acc = accuracy_score(y_hold, y_pred)
        prec = precision_score(y_hold, y_pred, average='weighted')
        rec = recall_score(y_hold, y_pred, average='weighted')
        f1w = f1_score(y_hold, y_pred, average='weighted')
        print(f"[HOLDOUT] acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1_w={f1w:.4f}")

        save_holdout_metrics_json(
            y_true=y_hold, y_pred=y_pred, labels=(0,1,2),
            out_dir="artifacts/experiments_holdout",
            filename=holdout_filename
        )

    # save deployment bundle for Streamlit
    if os.path.exists(bundle_dir):
        shutil.rmtree(bundle_dir)
    save_model_bundle(final_model, available_features, class_labels=(0,1,2), out_dir=bundle_dir)

    with open(os.path.join(bundle_dir, "SELECTED_FILTERS.json"), "w") as f:
        json.dump({
            "filters": {
                "remove_specialized": remove_specialized,
                "apply_polymerization_filter": apply_polymerization_filter,
                "add_negative_data": add_negative_data,
                "use_augmentation": use_augmentation,
            },
            "tuned_params": tuned,
        }, f, indent=2)

    print(f"[OK] Saved bundle to {bundle_dir}")
    return tuned, bundle_dir



def save_holdout_metrics_json(
    y_true,
    y_pred,
    labels=None,                      # pass (0,1,2) if you want fixed axes; otherwise inferred from data
    out_dir: str = HOLDOUT_RESULTS_DIR,
    filename: str | None = None
) -> str:
    """
    Save ONLY hold-out results: classification_report (as dict) + confusion_matrix.
    Returns the output file path.
    """
    os.makedirs(out_dir, exist_ok=True)

    # If labels are not provided, infer them from the data (sorted unique)
    if labels is None:
        labels = sorted(set(map(int, set(y_true))))  # ensure ints for JSON neatness

    report = classification_report(y_true, y_pred, labels=list(labels), output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))

    payload = {
        "timestamp": datetime.datetime.now().isoformat(),
        "labels": list(labels),
        "classification_report": report,   # already JSON-serializable
        "confusion_matrix": cm.tolist(),   # convert to list for JSON
    }

    if filename is None:
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"holdout_{ts}.json"

    fpath = os.path.join(out_dir, filename)
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return fpath


# ---------- Step 1: Build the base-clean dataset (no augmentation, no run-specific filters) ----------
def get_or_create_holdout_groups(base_df: pd.DataFrame, group_col: str = "reaction_id") -> pd.Series:
    """
    Create or load a persistent global hold-out at the GROUP level (reaction_id).
    Uses GroupKFold to pick one fold as the hold-out, ensuring no leakage.
    Returns a Series of unique group IDs for the hold-out.
    """
    if group_col not in base_df.columns:
        raise ValueError(f"'{group_col}' not found in base_df; cannot build grouped hold-out.")

    os.makedirs(os.path.dirname(TEST_GROUPS_PATH), exist_ok=True)

    # If a persistent file exists, load and intersect with current base_df (in case data changed)
    if os.path.exists(TEST_GROUPS_PATH):
        test_groups = pd.read_csv(TEST_GROUPS_PATH)[group_col].astype(str)
        current_groups = base_df[group_col].astype(str).unique()
        test_groups = test_groups[test_groups.isin(current_groups)]
        return test_groups.reset_index(drop=True)

    # Derive n_splits from TEST_SIZE (e.g., 0.2 -> 5)
    n_splits = max(2, int(round(1.0 / TEST_SIZE)))

    # Safety: need at least n_splits unique groups
    n_unique_groups = base_df[group_col].nunique()
    if n_unique_groups < n_splits:
        # fallback to the maximum possible splits (at least 2)
        n_splits = max(2, n_unique_groups)

    # use right group splitting

    gkf = create_grouped_kfold_splits(n_splits=n_splits, df=base_df)
    y = base_df.get("r_product_class", pd.Series([0]*len(base_df)))  # not used but required by API
    groups = base_df[group_col]

    # Deterministic: take the first split as hold-out
    _, test_idx = next(gkf.split(base_df, y, groups=groups))
    holdout_groups = base_df.iloc[test_idx][group_col].astype(str).unique()

    pd.DataFrame({group_col: holdout_groups}).to_csv(TEST_GROUPS_PATH, index=False)
    return pd.Series(holdout_groups)


# ---------- Step 2: Load or create persistent hold-out IDs ----------
def make_base_dataset_for_holdout(df: pd.DataFrame) -> pd.DataFrame:
    base = df.copy()
    # Core cleaning only — keep consistent with your earlier logic:
    base = base[base['r1r2'].notna()]
    base = base[base['r1r2'] >= 0]
    if "llm_specialized_filter" in base.columns:
        base = base[base["llm_specialized_filter"] != "specialized"]

    # 3-class target
    bins = [-np.inf, 1, 25, np.inf]
    labels = [0, 1, 2]
    base['r_product_class'] = pd.cut(base['r1r2'], bins=bins, labels=labels, right=False).astype(int)

    # override for extremes
    if {'constant_1','constant_2'}.issubset(base.columns):
        extreme_mask = (
            ((base['constant_1'] <= 0.1) & (base['constant_2'] > 25)) |
            ((base['constant_2'] <= 0.1) & (base['constant_1'] > 25))
        )
        base.loc[extreme_mask, 'r_product_class'] = 2

    # must have grouping key
    if 'reaction_id' not in base.columns:
        raise ValueError("reaction_id is required for grouped hold-out and CV but is missing in base_df.")

    return base


import os, json, itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from typing import Dict, List, Tuple


# NEW: generic combo generator
def _generate_boolean_combos(search_space: Dict[str, List[bool]]):
    """
    Generate all combinations from a boolean search space.

    Parameters
    ----------
    search_space : dict
        Example:
        {
            "remove_specialized": [False, True],
            "apply_polymerization_filter": [False, True],
            "add_negative_data": [True],            # FIXED -> will not be swept
            "use_augmentation": [False, True],
        }

    Yields
    ------
    combo : dict
        A single combination mapping flag -> bool.
    keys : list[str]
        The stable key order used to build the combo.
    """
    keys = list(search_space.keys())
    for values in itertools.product(*(search_space[k] for k in keys)):
        yield dict(zip(keys, values)), keys


def sweep_filters_and_plot(
    df,
    out_dir="artifacts/experiments_holdout",
    random_state=42,
    augmented_points_per_row=5,
    holdout_remove_specialized_in_base=True,
    # NEW: user-defined search space instead of hard-coded 16-combo grid
    search_space: Dict[str, List[bool]] | None = None,
):
    """
    Sweep over all combinations in 'search_space' and evaluate on a persistent group hold-out.

    If 'search_space' is None, defaults to the original 4-boolean sweep (16 runs).
    To FIX a switch (e.g., always keep negative data), pass a single-value list like
    "add_negative_data": [True].

    Returns
    -------
    results_df : pandas.DataFrame
        Aggregated hold-out metrics for each combination.
    """

    # Default: original 4D sweep
    if search_space is None:
        search_space = {
            "remove_specialized":         [False, True],
            "apply_polymerization_filter":[False, True],
            "add_negative_data":          [False, True],
            "use_augmentation":           [False, True],
        }

    rows = []

    for combo, keys in _generate_boolean_combos(search_space):
        spec = combo.get("remove_specialized", False)
        poly = combo.get("apply_polymerization_filter", False)
        neg  = combo.get("add_negative_data", False)
        aug  = combo.get("use_augmentation", True)

        run_name = f"spec{int(spec)}_poly{int(poly)}_neg{int(neg)}_aug{int(aug)}"
        holdout_filename = f"holdout_{run_name}.json"

        print(f"\n=== Running {run_name} ===")
        res = run_classification(
            df,
            random_state=random_state,
            remove_specialized=spec,
            apply_polymerization_filter=poly,
            add_negative_data=neg,
            use_augmentation=aug,
            use_global_holdout=True,
            holdout_remove_specialized_in_base=holdout_remove_specialized_in_base,
            augmented_points_per_row=augmented_points_per_row,
            # ensure we can load the correct JSON after each run
            holdout_filename=holdout_filename,
        )

        holdout_json_path = os.path.join(out_dir, holdout_filename)
        if not os.path.exists(holdout_json_path):
            print(f"[WARN] Hold-out JSON not found for {run_name}: {holdout_json_path}. Skipping.")
            continue

        with open(holdout_json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        report = payload["classification_report"]
        labels = payload.get("labels", [0, 1, 2])
        cm = np.array(payload["confusion_matrix"], dtype=float)

        # Extract key metrics
        acc      = float(report.get("accuracy", np.nan))
        f1_w     = float(report.get("weighted avg", {}).get("f1-score", np.nan))
        f1_macro = float(report.get("macro avg", {}).get("f1-score", np.nan))
        prec_w   = float(report.get("weighted avg", {}).get("precision", np.nan))
        rec_w    = float(report.get("weighted avg", {}).get("recall", np.nan))

        rows.append({
            "run": run_name,
            "remove_specialized": spec,
            "poly_filter": poly,
            "neg_data": neg,
            "augmentation": aug,
            "accuracy": acc,
            "f1_weighted": f1_w,
            "f1_macro": f1_macro,
            "precision_weighted": prec_w,
            "recall_weighted": rec_w,
            "holdout_json": holdout_json_path,
            "best_params": res.get("best_params", None),
            "labels": labels,
            "cm": cm.tolist(),
        })

    results_df = pd.DataFrame(rows)
    if results_df.empty:
        print("[ERROR] No results collected. Check that run_classification writes the hold-out JSON.")
        return results_df

    # --- Plot 1: Weighted F1 across runs ---
    results_df_sorted = results_df.sort_values("f1_weighted", ascending=True)
    plt.figure()
    plt.barh(results_df_sorted["run"], results_df_sorted["f1_weighted"])
    plt.xlabel("Weighted F1 (hold-out)")
    plt.title("Hold-out performance across filter combinations")
    plt.tight_layout()
    plt.savefig('output/model_comp/F1_score.png')
    plt.show()

    # --- Plot 2: Accuracy across runs ---
    plt.figure()
    plt.barh(results_df_sorted["run"], results_df_sorted["accuracy"])
    plt.xlabel("Accuracy (hold-out)")
    plt.title("Hold-out accuracy across filter combinations")
    plt.tight_layout()
    plt.savefig('output/model_comp/Accuracy.png')
    plt.show()

    # --- Plot 3: Confusion matrix for best run (by weighted F1) ---
    best_idx = results_df["f1_weighted"].idxmax()
    best_row = results_df.loc[best_idx]
    best_cm = np.array(best_row["cm"])
    best_labels = best_row["labels"]

    plt.figure()
    plt.imshow(best_cm, interpolation="nearest")
    plt.title(f"Confusion Matrix (best: {best_row['run']})")
    plt.colorbar()
    tick_marks = np.arange(len(best_labels))
    plt.xticks(tick_marks, best_labels)
    plt.yticks(tick_marks, best_labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    # annotate cells
    for i in range(best_cm.shape[0]):
        for j in range(best_cm.shape[1]):
            plt.text(j, i, int(best_cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig('output/model_comp/Conf_matrix.png')
    plt.show()

    return results_df


def run_classification(
    df,
    random_state=42,
    # switches for filters/features:
    remove_specialized: bool = False,
    apply_polymerization_filter: bool = False,
    add_negative_data: bool = False,
    use_augmentation: bool = True,
    # global hold-out controls:
    use_global_holdout: bool = True,
    holdout_remove_specialized_in_base: bool = True,
    augmented_points_per_row: int = 5,
    holdout_filename=None
):
    """
    Multi-class classification (0:[0,1), 1:[1,25), 2:[25,inf)) with grouped CV by reaction_id.
    Switches let you enable/disable:
      - remove_specialized (LLM),
      - apply_polymerization_filter,
      - add_negative_data (external CSV),
      - use_augmentation (Gaussian augmentation for train only),
      - use_global_holdout (persistent grouped hold-out by reaction_id).
    """

    import logging, sys
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='[%(levelname)s] %(message)s')
    log = logging.getLogger(__name__)
    log.info("=== XGBoost Multi-class Classification for r_product_class ===")

    # -------------------------
    # Core preprocessing
    # -------------------------
    log.info(f"Initial dataset size: {len(df)} samples")

    # Base-clean dataset (no augmentation / optional filters)
    base_df = make_base_dataset_for_holdout(df)

    df = df[df['r1r2'].notna()]
    original_count = len(df)
    df = df[df['r1r2'] >= 0]
    log.info(f"After r1r2 >= 0 filter: {len(df)} (removed {original_count - len(df)})")

    # Range and bins
    x_min, x_max = 0, 5
    bins = np.linspace(x_min, x_max, 51)  # 50 bins between 0 and 3

    plt.figure(figsize=(8, 6))

    # Histogram with raw counts
    plt.hist(
        df["r_product"],
        bins=bins,
        edgecolor="black",
        alpha=0.7
    )

    # Axis limits
    plt.xlim(x_min, x_max)

    plt.xlabel("r-product", fontsize=12)
    plt.ylabel("Number of reactions in dataset", fontsize=12)
    plt.savefig("output/data_analysis/r_product_distribution.png")

    # Toggle: specialized removal
    if remove_specialized and "llm_specialized_filter" in df.columns:
        original_count = len(df)
        df = df[df["llm_specialized_filter"] != "specialized"]
        log.info(f"After removing specialized: {len(df)} (removed {original_count - len(df)})")
    elif "llm_specialized_filter" not in df.columns and remove_specialized:
        log.warning("Column 'llm_specialized_filter' not found — skipping specialized filter.")

    # Classes
    bins = [-np.inf, 1, 25, np.inf];
    labels = [0, 1, 2]
    df['r_product_class'] = pd.cut(df['r1r2'], bins=bins, labels=labels, right=False).astype(int)

    # Override extremes
    if {'constant_1', 'constant_2'}.issubset(df.columns):
        extreme_mask = (
                ((df['constant_1'] <= 0.1) & (df['constant_2'] > 25)) |
                ((df['constant_2'] <= 0.1) & (df['constant_1'] > 25))
        )
        df.loc[extreme_mask, 'r_product_class'] = 2

    # Safety: non-negative again
    original_count = len(df)
    df = df[df['r1r2'] >= 0]
    log.info(f"After r1r2 >= 0 (second pass): {len(df)} (removed {original_count - len(df)})")

    # Feature matrix
    available_features = [col for col in feature_columns if col in df.columns]
    X = df[available_features];
    y = df['r_product_class']

    # Drop NaNs
    original_count = len(df)
    mask = ~(pd.isna(X).any(axis=1) | pd.isna(y))
    df_clean = df[mask].reset_index(drop=True)
    log.info(f"After NaN removal: {len(df_clean)} (removed {original_count - len(df_clean)})")

    # Toggle: add negative data
    if add_negative_data:
        log.info("Adding negative data from 'artificial_datapoints/processed_combined_augmented.csv'")
        negative_data_fraction = 1
        additional_csv_path = "artificial_datapoints/processed_combined_augmented.csv"
        df_new = pd.read_csv(additional_csv_path)
        if 'Class' not in df_new.columns:
            raise ValueError("The additional CSV must contain a 'Class' column.")
        df_new = df_new.rename(columns={'Class': 'r_product_class'})
        df_new['r_product_class'] = df_new['r_product_class'].astype(int)
        df_new_sample = df_new.sample(frac=negative_data_fraction, random_state=random_state)
        df_clean = pd.concat([df_clean, df_new_sample], ignore_index=True)
        log.info(f"Combined dataset size after adding negative data: {len(df_clean)}")

    # Toggle: polymerization filter
    if apply_polymerization_filter:
        original_count = len(df_clean)
        # df_clean = df_clean[df_clean['polymerization_type'].isin(utils.RADICAL_TYPES)]
        # ^^^ uncomment and plug in your set of allowed types
        filtered_count = len(df_clean)
        log.info(f"After polymerization filter: {filtered_count} (removed {original_count - filtered_count})")
    else:
        log.info("Polymerization filter disabled (no rows removed).")

    # -----------------------------
    # Build / load global hold-out
    # -----------------------------

    # Get persistent hold-out group IDs (reaction_id)
    holdout_groups = get_or_create_holdout_groups(base_df, group_col="reaction_id")

    # Split current run into train vs global hold-out by reaction_id
    if 'reaction_id' not in df_clean.columns:
        raise ValueError("reaction_id is required in df_clean for split and CV, but is missing.")

    df_test_holdout = df_clean[df_clean['reaction_id'].astype(str).isin(holdout_groups)].reset_index(drop=True)
    df_train_current = df_clean[~df_clean['reaction_id'].astype(str).isin(holdout_groups)].reset_index(drop=True)

    log.info(
        f"Global hold-out size (rows): {len(df_test_holdout)} across {df_test_holdout['reaction_id'].nunique()} groups")
    log.info(
        f"Current training pool size (rows): {len(df_train_current)} across {df_train_current['reaction_id'].nunique()} groups")

    # -----------------------------
    # Class weights (from CURRENT pool)
    # -----------------------------
    class_counts = df_train_current['r_product_class'].astype(int).value_counts().sort_index()
    total_samples = len(df_train_current)
    n_classes = class_counts.shape[0]
    classes_weighted = True

    if classes_weighted:
        class_weights = {
            cls: round(total_samples / (n_classes * count), 2)
            for cls, count in class_counts.items()
        }
    else:
        class_weights = {int(cls): 1.0 for cls in class_counts.keys()}

    log.info("=== CLASS WEIGHTS (train pool) ===")
    for cls in sorted(class_weights.keys()):
        log.info(f"Class {cls}: weight {class_weights[cls]}")

    # -----------------------------
    # Grouped CV on training pool
    # -----------------------------
    n_splits = 5
    kf_splits = create_grouped_kfold_splits(df_train_current, n_splits=n_splits, id_column='reaction_id')

    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        random_state=random_state,
        eval_metric='mlogloss'
    )

    param_grid = {
        'n_estimators': [100, 300, 600, 800],
        'max_depth': [3, 5, 8],
        'learning_rate': [0.04, 0.05, 0.06, 0.07],
        'subsample': [0.8, 0.9, 0.95],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5, 0.6],
        'reg_lambda': [1, 1.5, 2, 3],
        'min_child_weight': [2, 3, 5, 7],
        'gamma': [0.5, 0.6]
    }

    from sklearn.metrics import make_scorer, f1_score
    scorer = make_scorer(f1_score, average='weighted')

    available_features = [c for c in feature_columns if c in df_train_current.columns]

    # Storage for scores and predictions
    fold_scores = []
    all_y_true = []
    all_y_pred = []
    all_y_pred_proba = []
    all_prediction_confidence = []
    all_models = []
    all_test_indices = []

    # Cross-validation loop
    for fold, (train_idx, val_idx) in enumerate(kf_splits, 1):
        print(f"\nFold {fold}")
        df_train_fold = df_train_current.iloc[train_idx].reset_index(drop=True)
        df_val_fold = df_train_current.iloc[val_idx].reset_index(drop=True)

        # Toggle: augmentation on train only
        if use_augmentation:
            df_train_aug = augment_with_gaussian_samples(
                df_train_fold, num_samples=augmented_points_per_row, std_factor=0.3, random_state=random_state
            )
        else:
            df_train_aug = df_train_fold

        X_train = df_train_aug[available_features]
        y_train = df_train_aug['r_product_class'].values
        X_val = df_val_fold[available_features]
        y_val = df_val_fold['r_product_class'].values
        print(f"Training set size: {len(X_train)}, Test set size: {len(X_val)}")

        sample_weights = np.array([class_weights[label] for label in y_train])

        from sklearn.metrics import make_scorer, f1_score

        scorer = make_scorer(f1_score, average='weighted')

        # Hyperparameter optimization
        random_search = RandomizedSearchCV(
            model, param_distributions=param_grid,
            n_iter=10, cv=3, scoring=scorer, verbose=0, random_state=random_state, n_jobs=-1
        )

        # Fit the XGBoost classifier with randomized hyperparameter search
        random_search.fit(X_train, y_train, sample_weight=sample_weights)

        best_model = random_search.best_estimator_
        best_model.fit(X_train, y_train, sample_weight=sample_weights)

        # Predictions with confidence estimation
        y_pred = best_model.predict(X_val)

        # Calculate prediction confidence using multiple methods
        y_pred_proba = best_model.predict_proba(X_val)  # shape: (n_samples, 5)
        confidence_scores = np.max(y_pred_proba, axis=1)

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        # Calculate fold metrics
        fold_accuracy = accuracy_score(y_val, y_pred)
        fold_precision = precision_score(y_val, y_pred, average='weighted')
        fold_recall = recall_score(y_val, y_pred, average='weighted')
        fold_f1 = f1_score(y_val, y_pred, average='weighted')

        print(f"  Accuracy: {fold_accuracy:.4f}")
        print(f"  Precision: {fold_precision:.4f}")
        print(f"  Recall: {fold_recall:.4f}")
        print(f"  F1-Score: {fold_f1:.4f}")

        # Store results
        fold_scores.append({
            'fold': fold,
            'accuracy': fold_accuracy,
            'precision': fold_precision,
            'recall': fold_recall,
            'f1_score': fold_f1,
            'best_params': random_search.best_params_
        })

        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)
        all_y_pred_proba.extend(y_pred_proba)
        all_prediction_confidence.extend(confidence_scores)
        all_models.append(best_model)



    # CV summary
    overall_accuracy = accuracy_score(all_y_true, all_y_pred)
    overall_precision = precision_score(all_y_true, all_y_pred, average='weighted')
    overall_recall = recall_score(all_y_true, all_y_pred, average='weighted')
    overall_f1 = f1_score(all_y_true, all_y_pred, average='weighted')

    print("\n=== Overall CV Results ===")
    print(f"Accuracy:  {overall_accuracy:.4f}")
    print(f"Precision: {overall_precision:.4f}")
    print(f"Recall:    {overall_recall:.4f}")
    print(f"F1-Score:  {overall_f1:.4f}")
    print("\nClassification report (CV oof):")
    print(classification_report(all_y_true, all_y_pred))

    # NOTE: confusion matrix for 3 classes [0,1,2]
    cm_cv = confusion_matrix(all_y_true, all_y_pred, labels=[0, 1, 2])
    print("CV confusion matrix (labels 0,1,2):")
    print(cm_cv)

    # Optional: feature importance from the first best model
    if len(all_models) > 0:
        plot_feature_importance(all_models[0], available_features)

    # ---------------------------------
    # FINAL TRAIN on ALL train data
    # ---------------------------------
    # Pick overall best params (by mean F1)
    fold_df = pd.DataFrame(fold_scores)
    # argmax by f1
    best_params = fold_df.loc[fold_df['f1_score'].idxmax(), 'best_params']
    print("\n=== Best params from CV ===")
    print(best_params)

    # Rebuild full training set (with augmentation)
    df_train_full = augment_with_gaussian_samples(
        df_train_current,
        num_samples=augmented_points_per_row,
        std_factor=0.3,
        random_state=random_state
    )
    X_train_full = df_train_full[available_features]
    y_train_full = df_train_full['r_product_class'].values
    w_train_full = np.array([class_weights[int(lbl)] for lbl in y_train_full])

    final_model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        random_state=random_state,
        eval_metric='mlogloss',
        **best_params
    )
    final_model.fit(X_train_full, y_train_full, sample_weight=w_train_full)

    # ---------------------------------
    # EVAL on GLOBAL HOLD-OUT
    # ---------------------------------
    if len(df_test_holdout) > 0:
        X_test_hold = df_test_holdout[available_features]
        y_test_hold = df_test_holdout['r_product_class'].values

        y_pred_hold = final_model.predict(X_test_hold)
        y_proba_hold = final_model.predict_proba(X_test_hold)

        acc = accuracy_score(y_test_hold, y_pred_hold)
        prec = precision_score(y_test_hold, y_pred_hold, average='weighted')
        rec = recall_score(y_test_hold, y_pred_hold, average='weighted')
        f1 = f1_score(y_test_hold, y_pred_hold, average='weighted')

        print("\n=== Global Hold-out Results ===")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print("\nClassification report (hold-out):")
        print(classification_report(y_test_hold, y_pred_hold))
        cm_hold = confusion_matrix(y_test_hold, y_pred_hold, labels=[0,1,2])
        print("Hold-out confusion matrix (labels 0,1,2):")
        print(cm_hold)
    else:
        print("\n[Warning] Global hold-out set is empty in this run; no final evaluation printed.")

    # After you compute predictions on the global hold-out:
    y_pred_hold = final_model.predict(X_test_hold)

    # Save ONLY hold-out results (report + confusion matrix)
    out_path_holdout = save_holdout_metrics_json(
        y_true=y_test_hold,
        y_pred=y_pred_hold,
        labels=(0, 1, 2),  # optional; use this to keep matrix axes identical across runs
        out_dir="artifacts/experiments_holdout",
        filename=holdout_filename
    )
    print(f"Saved hold-out metrics JSON → {out_path_holdout}")

    print(f"Saved hold-out metrics JSON → {out_path_holdout}")

    # Return objects if you want to persist/log elsewhere
    return {
        "cv_models": all_models,
        "final_model": final_model,
        "best_params": best_params,
        "cv_scores": fold_scores,
        "class_weights": class_weights
    }

    y_pred_proba_array = np.array(all_y_pred_proba)

    # Create prediction dataframe
    predictions_df = pd.DataFrame({
        'true_label': all_y_true,
        'predicted_label': all_y_pred,
        'confidence_score': all_prediction_confidence,
        'correct_prediction': np.array(all_y_pred) == np.array(all_y_true),
        **{f'proba_class_{i}': y_pred_proba_array[:, i] for i in range(n_classes)}
    })
    predictions_df['original_index'] = all_test_indices

    # Merge with original df_clean
    df_clean_reset = df_clean.reset_index()  # adds 'index' column with original row numbers
    df_merged = df_clean_reset.merge(
        predictions_df, left_on='index', right_on='original_index', how='inner'
    )

    save_results_with_confidence(
        fold_scores_df,
        available_features,
        class_counts,
        overall_accuracy,
        overall_f1,
        df_merged['true_label'].tolist(),
        df_merged['predicted_label'].tolist(),
        df_merged[[f'proba_class_{i}' for i in range(y_pred_proba_array.shape[1])]].values,
        df_merged['confidence_score'].tolist(),
        df_merged
    )

    perform_error_analysis(
        df_merged['true_label'].tolist(),
        df_merged['predicted_label'].tolist(),
        df_merged[[f'proba_class_{i}' for i in range(y_pred_proba_array.shape[1])]].values,
        df_merged['confidence_score'].tolist(),
        df_merged,
        kf_splits,
        detailed_error_analyis=True
    )

    print("\n=== Global Calibration on Full Dataset ===")

    # Step 1: Fit final XGBoost model on full data
    # Use average best_params across folds or pick best from first fold
    best_params = fold_scores[0]['best_params']  # or implement a best-of-all strategy

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    df_clean = df_clean.reset_index(drop=True)
    # Determine group column
    if 'reaction_id' in df_clean.columns:
        group_column = 'reaction_id'
    elif 'group_id' in df_clean.columns:
        group_column = 'group_id'
    else:
        raise ValueError("No group column found!")

    # Perform grouped splitting
    X_train, y_train, X_calib, y_calib, X_val, y_val = grouped_train_calib_val_split(
        X, y, groups=df_clean[group_column], test_size=0.2, calib_size=0.2, random_state=42
    )

    final_model = xgb.XGBClassifier(
        **best_params,
        random_state=42, use_label_encoder=False, eval_metric='logloss'
    )

    sample_weights = np.array([class_weights[label] for label in y_train])
    final_model.fit(X_train, y_train, sample_weight=sample_weights)

    # === NEW: PERMUTATION FEATURE IMPORTANCE ANALYSIS ===
    if run_permutation_analysis_flag:
        print("\n" + "=" * 60)
        print("STARTING PERMUTATION FEATURE IMPORTANCE ANALYSIS")
        print("=" * 60)

        # Use the last fold's test set for permutation analysis
        # (alternatively, you could use the validation set from calibration)
        last_fold_train_idx, last_fold_test_idx = kf_splits[-1]
        X_perm_test = X.iloc[last_fold_test_idx]
        y_perm_test = y.iloc[last_fold_test_idx]

        # Run permutation analysis
        permutation_results = run_permutation_analysis(
            models=final_model,
            X_test=X_val,
            y_test=y_val,
            feature_names=available_features,
            output_dir='output',
            scoring='f1_weighted'
        )

        # Compare tree-based vs permutation importance
        tree_importance = all_models[0].feature_importances_
        perm_importance = permutation_results['permutation_results']['importance_mean'].values

        importance_comparison = create_feature_importance_comparison(
            tree_importance=tree_importance,
            perm_importance=perm_importance,
            feature_names=available_features,
            save_path='output/importance_comparison.png'
        )

        # Add permutation results to return dictionary
        permutation_analysis_results = permutation_results
    else:
        permutation_analysis_results = None

    from sklearn.calibration import calibration_curve

    # === Step 1: Predict on validation set (before calibration) ===
    y_val_pred_uncal = final_model.predict(X_val)
    cm_uncal = confusion_matrix(y_val, y_val_pred_uncal)
    print("\n=== CONFUSION MATRIX before calibration ===")
    print(cm_uncal)

    # === Step 2: Calibrate model using weighted logistic regression ===
    # Isotonic without invalid kwargs
    calibrated_model = calibrate_model_with_weights(
        model=final_model,
        X_calib=X_calib,
        y_calib=y_calib,
        class_weight_dict=class_weights,
        method="sigmoid",  # switch to 'sigmoid' if desired
        calibrator_kwargs={}  # leave empty for isotonic
    )

    # === Step 3: Predict calibrated probabilities on validation set ===
    y_val_proba = calibrated_model.predict_proba(X_val)[:, 1]
    y_val_pred = calibrated_model.predict(X_val)

    from sklearn.metrics import accuracy_score

    # === Step 4: Evaluate calibration ===
    cm_cal = confusion_matrix(y_val, y_val_pred)
    print("\n=== CONFUSION MATRIX after Weighted Calibration ===")
    print(cm_cal)

    # === Step 4.1: Class-wise accuracy and equal-weighted accuracy ===
    class_accuracies = {}
    for cls in [0, 1]:
        cls_mask = y_val == cls
        cls_total = cls_mask.sum()
        cls_correct = (y_val_pred[cls_mask] == y_val[cls_mask]).sum()
        cls_accuracy = cls_correct / cls_total if cls_total > 0 else 0.0

        orig_cls_accuracy = accuracy_score(y_val[cls_mask], y_val_pred_uncal[cls_mask]) if cls_total > 0 else 0.0

        class_accuracies[cls] = {
            'orig': orig_cls_accuracy,
            'cal': cls_accuracy,
            'correct': cls_correct,
            'total': cls_total
        }

    # Compute quality-weighted accuracies
    orig_quality_acc = compute_quality_weighted_accuracy(y_val, y_val_pred_uncal, num_classes=2)
    cal_quality_acc = compute_quality_weighted_accuracy(y_val, y_val_pred, num_classes=2)

    # === Print table ===
    GREEN = '\033[92m'
    BRIGHT_GREEN = '\033[92;1m'
    RESET = '\033[0m'

    for cls in [0, 1]:
        data = class_accuracies[cls]
        print(f"{GREEN}Class {cls:<3} {data['orig']:<10.4f} {data['cal']:<10.4f} "
              f"{data['cal'] - data['orig']:<+10.4f} {data['correct']}/{data['total']}{RESET}")

    print(f"{BRIGHT_GREEN}{'Quality-Weighted Acc:':<20} {orig_quality_acc:<10.4f} "
          f"{cal_quality_acc:<10.4f} {cal_quality_acc - orig_quality_acc:<+10.4f}{RESET}")

    # === Step 5: Align df_clean subset with validation set ===
    #val_indices = X_val.index
    #df_clean_val = df_clean.loc[val_indices].reset_index(drop=True)

    # === Step 6: Calculate entropy-based confidence ===
    calibrated_confidence = calculate_prediction_confidence(calibrated_model, X_val)

    # === Step 9: Final error analysis ===
    print("\n=== Error Analysis: Calibrated Global Model ===")

    #perform_error_analysis(
        #all_y_true=y_val,
        #all_y_pred=y_val_pred,
        #all_y_pred_proba=y_val_proba,
        #all_prediction_confidence=calibrated_confidence,
        #df_clean=df_clean_val,
        #kf_splits=None,
        #detailed_error_analyis=False
    #)


    return {
        'fold_scores': fold_scores_df,
        'overall_metrics': {
            'accuracy': overall_accuracy,
            'precision': overall_precision,
            'recall': overall_recall,
            'f1_score': overall_f1,
        },
        'models': all_models,
        'confidence_scores': all_prediction_confidence
    }


from sklearn.model_selection import GroupShuffleSplit


def grouped_train_calib_val_split(X, y, groups, test_size=0.2, calib_size=0.2, random_state=42):
    """
    Performs a leakage-free split of the dataset into train, calibration, and validation sets,
    using groups (e.g., reaction_id) to ensure that related samples stay together.

    Parameters:
        X: Feature matrix (DataFrame or array)
        y: Target array
        groups: Array-like group labels (e.g., reaction_id)
        test_size: Proportion of data to use as validation set (from full data)
        calib_size: Proportion of training data to use as calibration set (from remaining data after val split)
        random_state: Seed for reproducibility

    Returns:
        X_train, y_train, X_calib, y_calib, X_val, y_val
    """

    # Step 1: Split off the validation set
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, val_idx = next(gss.split(X, y, groups=groups))

    X_train_val = X.iloc[train_val_idx].reset_index(drop=True)
    y_train_val = y.iloc[train_val_idx].reset_index(drop=True)
    groups_train_val = np.array(groups)[train_val_idx]

    X_val = X.iloc[val_idx].reset_index(drop=True)
    y_val = y.iloc[val_idx].reset_index(drop=True)

    # Step 2: From the train_val part, split out the calibration set
    gss2 = GroupShuffleSplit(n_splits=1, test_size=calib_size, random_state=random_state)
    train_idx, calib_idx = next(gss2.split(X_train_val, y_train_val, groups=groups_train_val))

    X_train = X_train_val.iloc[train_idx].reset_index(drop=True)
    y_train = y_train_val.iloc[train_idx].reset_index(drop=True)

    X_calib = X_train_val.iloc[calib_idx].reset_index(drop=True)
    y_calib = y_train_val.iloc[calib_idx].reset_index(drop=True)

    return X_train, y_train, X_calib, y_calib, X_val, y_val


def calculate_prediction_confidence(model, X_test):
    """
    Calculate prediction confidence using entropy-based uncertainty.
    Lower entropy = higher confidence
    """

    # Get probabilities for both classes
    y_pred_proba_both = model.predict_proba(X_test)

    # Avoid log(0) by clipping probabilities
    epsilon = 1e-15
    y_pred_proba_both = np.clip(y_pred_proba_both, epsilon, 1 - epsilon)

    # Calculate entropy: H = -sum(p * log(p))
    entropy = -np.sum(y_pred_proba_both * np.log(y_pred_proba_both), axis=1)

    # Flexible calculation of maximum entropy
    max_entropy = np.log(y_pred_proba_both.shape[1])

    # Convert entropy to confidence: confidence = 1 - (entropy / max_entropy)
    # This gives values from 0 (minimum confidence) to 1 (maximum confidence)
    confidence = 1 - (entropy / max_entropy)

    return confidence


def save_results_with_confidence(fold_scores_df, features, class_counts, overall_accuracy, overall_f1,
                                 y_true, y_pred, y_pred_proba, confidence_scores, df_clean):
    """Saves XGBoost binary classification results with confidence scores and all features"""

    # Save fold scores
    fold_scores_df.to_csv('output/xgboost_binary_fold_scores.csv', index=False)

    n_classes = y_pred_proba.shape[1]

    # Start with basic prediction columns
    predictions_dict = {
        'true_label': y_true,
        'predicted_label': y_pred,
        'confidence_score': confidence_scores,
        'correct_prediction': np.array(y_pred) == np.array(y_true)
    }

    # Add per-class probabilities
    for cls_idx in range(n_classes):
        predictions_dict[f'proba_class_{cls_idx}'] = y_pred_proba[:, cls_idx]

    # Optional: if you still want to keep the original "predicted_probability" field (e.g. for class 1)
    # predictions_dict['predicted_probability'] = y_pred_proba[:, 1]

    # Build the DataFrame
    predictions_df = pd.DataFrame(predictions_dict)

    # Add all available features from the cleaned dataframe
    # First, ensure we have the same number of rows
    if len(predictions_df) == len(df_clean):
        # Add all columns from df_clean except those that might conflict
        exclude_columns = ['true_label', 'predicted_label', 'predicted_probability', 'confidence_score',
                           'correct_prediction']

        for col in df_clean.columns:
            if col not in exclude_columns and col not in predictions_df.columns:
                predictions_df[col] = df_clean[col].values
    else:
        print(f"Warning: Predictions length ({len(predictions_df)}) doesn't match clean data length ({len(df_clean)})")
        print("Features will not be added to predictions file")

    # Save predictions with all features
    predictions_df.to_csv('output/xgboost_predictions_with_confidence.csv', index=False)

    print(f"Saved predictions with {len(predictions_df.columns)} columns including all features")

    # Calculate confidence statistics
    correct_predictions = predictions_df['correct_prediction']
    high_confidence_mask = predictions_df['confidence_score'] > 0.8
    medium_confidence_mask = (predictions_df['confidence_score'] > 0.6) & (predictions_df['confidence_score'] <= 0.8)
    low_confidence_mask = predictions_df['confidence_score'] <= 0.6

    confidence_stats = {
        'high_confidence_accuracy': correct_predictions[
            high_confidence_mask].mean() if high_confidence_mask.sum() > 0 else 0,
        'medium_confidence_accuracy': correct_predictions[
            medium_confidence_mask].mean() if medium_confidence_mask.sum() > 0 else 0,
        'low_confidence_accuracy': correct_predictions[
            low_confidence_mask].mean() if low_confidence_mask.sum() > 0 else 0,
        'high_confidence_count': high_confidence_mask.sum(),
        'medium_confidence_count': medium_confidence_mask.sum(),
        'low_confidence_count': low_confidence_mask.sum(),
        'mean_confidence': predictions_df['confidence_score'].mean(),
        'std_confidence': predictions_df['confidence_score'].std()
    }

    # Save summary statistics
    summary_stats = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
        summary_stats[f'{metric}_mean'] = fold_scores_df[metric].mean()
        summary_stats[f'{metric}_std'] = fold_scores_df[metric].std()

    # Add confidence statistics
    summary_stats.update(confidence_stats)

    summary_df = pd.DataFrame([summary_stats])
    summary_df.to_csv('output/xgboost_binary_summary.csv', index=False)


    print(f"\nResults saved to:")
    print(f"- output/xgboost_binary_fold_scores.csv")
    print(f"- output/xgboost_predictions_with_confidence.csv (with all features)")
    print(f"- output/xgboost_binary_summary.csv")
    print(f"- output/xgboost_feature_importance.png")


def plot_feature_importance(model, feature_names, top_n=10):
    """Plots feature importance for XGBoost model"""

    # Check if model is a CalibratedClassifierCV wrapper
    if hasattr(model, "estimator"):
        model = model.estimator  # Compatible access for scikit-learn ≥0.24

    # Now extract feature importances
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    else:
        print("Model has no feature_importances_. Skipping plot.")
        return

    indices = np.argsort(importance)[::-1][:top_n]

    plt.figure(figsize=(12, 8))
    plt.title(f'Top {top_n} Feature Importances - XGBoost')
    plt.barh(range(top_n), importance[indices], color='#661124')
    plt.yticks(range(top_n), [feature_names[i] for i in indices], fontsize=14)
    plt.xlabel('Feature Importance', fontsize=14)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('output/xgboost_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()


class CalibratedModel:
    def __init__(self, base_model, calibrator):
        self.base_model = base_model
        self.calibrator = calibrator

    def predict_proba(self, X):
        """
        Predict calibrated probabilities for class 1.
        """
        raw_proba = self.base_model.predict_proba(X)[:, 1]

        # Handle sigmoid vs isotonic calibrator
        if hasattr(self.calibrator, "predict_proba"):
            calibrated = self.calibrator.predict_proba(raw_proba.reshape(-1, 1))[:, 1]
        else:
            calibrated = self.calibrator.predict(raw_proba)

        # Return as proper two-column probability matrix: [P(class 0), P(class 1)]
        calibrated = np.clip(calibrated, 1e-6, 1 - 1e-6)
        return np.vstack([1 - calibrated, calibrated]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import numpy as np


def calibrate_model_with_weights(
    model,
    X_calib,
    y_calib,
    class_weight_dict=None,
    method="sigmoid",  # 'sigmoid' or 'isotonic'
    calibrator_kwargs=None
):
    """
    Calibrate a classifier using sigmoid (Platt scaling) or isotonic regression.

    Parameters:
        model:               A trained classifier supporting predict_proba().
        X_calib:             Calibration feature matrix.
        y_calib:             Binary labels for calibration set.
        class_weight_dict:   Optional dict, e.g., {0: 3.0, 1: 1.0}.
        method:              'sigmoid' (Platt scaling) or 'isotonic'.
        calibrator_kwargs:   Optional kwargs for the calibrator.

    Returns:
        CalibratedModel: A wrapper with predict_proba and predict.
    """

    # Step 1: Predict raw probabilities (class 1)
    raw_proba = model.predict_proba(X_calib)[:, 1]

    # Step 2: Sample weights based on class weights
    if class_weight_dict is not None:
        sample_weights = np.array([class_weight_dict[y] for y in y_calib])
    else:
        sample_weights = None

    # Step 3: Fit calibrator
    if method == "sigmoid":
        calibrator_args = calibrator_kwargs or {}
        calibrator = LogisticRegression(**calibrator_args)
        calibrator.fit(raw_proba.reshape(-1, 1), y_calib, sample_weight=sample_weights)

    elif method == "isotonic":
        # IsotonicRegression doesn't accept 'solver' or many LogisticRegression kwargs
        # Only safe kwargs like y_min, y_max, increasing etc.
        calibrator_args = calibrator_kwargs or {}
        calibrator = IsotonicRegression(out_of_bounds="clip", **calibrator_args)
        calibrator.fit(raw_proba, y_calib, sample_weight=sample_weights)

    else:
        raise ValueError(f"Unknown calibration method '{method}'. Use 'sigmoid' or 'isotonic'.")

    return CalibratedModel(model, calibrator)


# Integration into main function
def main(process_data=True):
    """Main function to run binary classification model"""
    import os
    # Input data path
    data_path = "../data_extraction/extracted_reactions.csv"

    # Set random seed for reproducibility
    random_state = 42

    # Create output directory
    os.makedirs("output", exist_ok=True)

    print("=== Copolymerization Binary Classification ===")

    if process_data:
        # Step 1: Load and preprocess data
        print("\nStep 1: Loading and preprocessing data...")
        df = data_processing.load_and_preprocess_data(data_path)

        if df is None or len(df) == 0:
            print("Error: No data available for modeling.")
            return

        # Save processed data
        df.to_csv("output/processed_data.csv", index=False)
    else:
        df = pd.read_csv("llm_specialized_filter/classified_output.csv")

    # YOUR chosen filter combo (this is what you meant by "parameters")
    chosen = dict(
        remove_specialized=False,
        apply_polymerization_filter=False,
        add_negative_data=True,
        use_augmentation=False,
    )

    # optional: a compact search space; omit to use the default above
    custom_search = {
        'n_estimators': [500, 600, 700],
        'max_depth': [4, 5, 6],
        'learning_rate': [0.04, 0.05, 0.06],
        'subsample': [0.85, 0.9, 0.95],
        'colsample_bytree': [0.85, 0.9, 1.0],
        'reg_alpha': [0.0, 0.1, 0.3],
        'reg_lambda': [1.0, 1.5, 2.0],
        'min_child_weight': [2, 3, 5],
        'gamma': [0.3, 0.5, 0.7],
    }

    tuned_params, bundle_dir = finalize_selected_filters_and_save(
        df,  # your processed dataframe from main()
        **chosen,
        random_state=42,
        augmented_points_per_row=5,
        bundle_dir="artifacts/model_bundle",
        holdout_filename="holdout_selected_filters.json",
        search_space=custom_search,  # or None
        n_search_iter=25,
    )
    print("Tuned params:", tuned_params)
    print("Bundle saved at:", bundle_dir)

    # Run Binary Classification model
    print("\nRunning XGBoost Binary Classification model...")
    results = sweep_filters_and_plot(
        df,
        search_space={
            "remove_specialized": [False, True],
            "apply_polymerization_filter": [False, True],
            "add_negative_data": [True],  # FIXED ON
            "use_augmentation": [False, True],
        }
    )

    print(results.sort_values("f1_weighted", ascending=False).head(10))

    print("\n=== Modeling Complete ===")
    print("Results saved to output/")

    return results


if __name__ == "__main__":
    main()