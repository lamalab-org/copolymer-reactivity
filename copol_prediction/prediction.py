"""
Extension to main.py to add bucket-based classification model
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Import the BucketClassifier
from bucket_model import BucketClassifier
import data_processing
import result_visualization as visualization
import models


def run_bucket_classifier(df, random_state):
    """
    Run bucket classification model on the data

    Args:
        df: DataFrame with features and target
        random_state: Seed for reproducibility
    """
    print("\n=== Running Bucket Classification Model ===")

    # Create output_2 directory for classification results
    os.makedirs("output/classification", exist_ok=True)

    # Initialize classifier with predefined category boundaries
    classifier = BucketClassifier(
        n_buckets=21,  # This will be overridden by the predefined boundaries
        model_type="xgboost",
        random_state=random_state,
        # The bucket_transform and quantile_based parameters will be ignored
        # when using predefined bucket edges
        bucket_transform='log',
        quantile_based=False
    )

    # Store the original DataFrame before any processing
    original_df = df.copy()

    # Prepare data for modeling
    X, y, features, df_prepared = classifier.prepare_data(df)

    if X is None or y is None:
        print("Error preparing data for classification model.")
        return

    # Plot the distribution of samples across buckets
    classifier.plot_bucket_distribution(
        df_prepared,
        save_path="output/classification/bucket_distribution.png"
    )
    print("Saved bucket distribution plot to output_2/classification/bucket_distribution.png")

    # Train and evaluate model
    fold_scores, predictions = classifier.train_and_evaluate(X, y, df_prepared)

    # Train final model
    final_model = classifier.train_final_model(X, y)

    # Get feature importances
    importance_df = classifier.get_feature_importances(final_model)

    if importance_df is not None:
        importance_df.to_csv("output_2/classification/feature_importances.csv", index=False)

        # Plot feature importances
        visualization.plot_feature_importances(
            importance_df,
            title="Bucket Classification Feature Importances",
            save_path="output/classification/feature_importances.png"
        )

    # Plot confusion matrix
    classifier.plot_confusion_matrix(
        predictions,
        title="Bucket Classification Confusion Matrix",
        save_path="output/classification/confusion_matrix.png",
        max_buckets_to_show=20
    )

    # Plot r1r2 predictions
    classifier.plot_r1r2_predictions(
        predictions,
        df_prepared,
        title="Bucket Classification r1r2 Predictions",
        save_path="output/classification/r1r2_predictions.png"
    )

    # Plot distribution-based predictions
    classifier.plot_distribution_predictions(
        predictions,
        df_prepared,
        title="Distribution-based r1r2 Predictions",
        save_path="output/classification/r1r2_predictions_distribution.png"
    )

    # Compare both prediction methods
    classifier.compare_prediction_methods(
        predictions,
        df_prepared,
        save_path="output/classification/prediction_methods_comparison.png"
    )

    # Update metrics in metrics_df
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'F1 Score (weighted)', 'RMSE (Hard Prediction)', 'RMSE (Distribution Prediction)'],
        'Value': [
            predictions['avg_test_accuracy'],
            predictions['avg_test_f1'],
            predictions['r1r2_rmse'],
            predictions['r1r2_rmse_dist']
        ]
    })

    # Plot error analysis
    classifier.plot_error_analysis(
        predictions,
        df_prepared,
        save_path="output/classification/error_analysis.png"
    )

    # Save prediction metrics
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'F1 Score (weighted)', 'RMSE on r1r2'],
        'Value': [
            predictions['avg_test_accuracy'],
            predictions['avg_test_f1'],
            predictions['r1r2_rmse']
        ]
    })
    metrics_df.to_csv("output_2/classification/metrics.csv", index=False)

    # Save detailed predictions for error analysis
    save_detailed_predictions(predictions, df_prepared, original_df)

    print("\n=== Bucket Classification Results ===")
    print(f"Average Accuracy: {predictions['avg_test_accuracy']:.4f}")
    print(f"Average F1 Score: {predictions['avg_test_f1']:.4f}")
    print(f"RMSE on r1r2 values: {predictions['r1r2_rmse']:.4f}")
    print("Results saved to output_2/classification/")


def save_detailed_predictions(predictions, df_prepared, original_df):
    """
    Save detailed prediction results for error analysis
    Ensures method and polymerization_type are included in the output_2

    Args:
        predictions: Dictionary with prediction results
        df_prepared: DataFrame with prepared data (including buckets)
        original_df: Original DataFrame with all features
    """
    # Get test indices and predicted bucket indices
    test_indices = predictions['test_indices']
    pred_buckets = predictions['test_pred_buckets']
    true_buckets = predictions['test_true_buckets']

    # Create a results DataFrame
    results_df = pd.DataFrame()

    true_buckets_array = np.array(true_buckets)
    pred_buckets_array = np.array(pred_buckets)

    comparison = (true_buckets_array == pred_buckets_array)
    results_df['bucket_correct'] = np.where(comparison, 1, 0)

    # Add r1r2 data from the prepared DataFrame
    if 'r1r2' in df_prepared.columns:
        true_r1r2 = df_prepared.iloc[test_indices]['r1r2'].values
        results_df['true_r1r2'] = true_r1r2

    # Add predicted r1r2 using bucket centers
    bucket_centers = predictions['bucket_centers']
    pred_r1r2 = np.array([bucket_centers[idx] for idx in pred_buckets])
    results_df['pred_r1r2'] = pred_r1r2

    # Calculate errors
    results_df['abs_error'] = np.abs(results_df['true_r1r2'] - results_df['pred_r1r2'])
    results_df['rel_error'] = results_df['abs_error'] / np.maximum(results_df['true_r1r2'], 0.0001) * 100

    # Add important columns from the original DataFrame
    important_columns = [
        'monomer1_name', 'monomer2_name', 'solvent_name',
        'monomer1_smiles', 'monomer2_smiles', 'solvent_smiles',
        'reaction_id', 'reference', 'temperature', 'polymerization_type',
        'method'  # Explicitly include method
    ]

    # Only add columns that exist
    existing_columns = [col for col in important_columns if col in original_df.columns]

    # For each important column, add it to the results DataFrame
    if 'reaction_id' in df_prepared.columns:
        reaction_ids = df_prepared.iloc[test_indices]['reaction_id'].values

        # Add each column if it exists
        for col in existing_columns:
            if col in original_df.columns:
                # Create a mapping of reaction_id to column value
                id_to_value = dict(zip(original_df['reaction_id'], original_df[col]))
                # Map reaction_ids to the corresponding values
                results_df[col] = [id_to_value.get(rid) for rid in reaction_ids]

    # Verify that method and polymerization_type are included if available in the original data
    for col in ['method', 'polymerization_type']:
        if col not in results_df.columns and col in original_df.columns:
            print(f"Warning: Column '{col}' not transferred properly. Adding it manually.")

            # If we have reaction_id, we can still add the missing column
            if 'reaction_id' in results_df.columns:
                id_to_value = dict(zip(original_df['reaction_id'], original_df[col]))
                results_df[col] = [id_to_value.get(rid) for rid in results_df['reaction_id']]

    # Add distribution-based predictions if available
    if 'r1r2_pred_dist' in predictions:
        results_df['pred_r1r2_dist'] = predictions['r1r2_pred_dist']
        # Calculate errors for distribution-based predictions
        results_df['abs_error_dist'] = np.abs(results_df['true_r1r2'] - results_df['pred_r1r2_dist'])
        results_df['rel_error_dist'] = results_df['abs_error_dist'] / np.maximum(results_df['true_r1r2'], 0.0001) * 100

    # Save the results
    results_df.to_csv("output_2/classification/detailed_predictions.csv", index=False)
    print("Saved detailed predictions to output_2/classification/detailed_predictions.csv")
    print(f"Columns included in detailed predictions: {results_df.columns.tolist()}")


def main(process_data = True):
    """Main function to run both XGBoost regression and bucket classification models"""
    # Input data path
    data_path = "../data_extraction/extracted_reactions.csv"

    # Set random seed for reproducibility
    random_state = 42

    # Create output_2 directory
    os.makedirs("output", exist_ok=True)

    print("=== Copolymerization Prediction Model ===")

    if process_data:
        # Step 1: Load and preprocess data
        print("\nStep 1: Loading and preprocessing data...")
        df = data_processing.load_and_preprocess_data(data_path)

        if df is None or len(df) == 0:
            print("Error: No data available for modeling.")
            return

        # Save processed data
        df.to_csv("output_2/processed_data.csv", index=False)
    else:
        df = pd.read_csv("output/processed_data.csv")

    # Step 2: Run XGBoost regression model (as before)
    print("\nStep 2: Running XGBoost regression model...")
    run_xgboost(df, random_state)

    # Step 3: Run Bucket Classification model
    print("\nStep 3: Running Bucket Classification model...")
    run_bucket_classifier(df, random_state)

    # Step 4: Run Neural Network model (optional)
    # print("\nStep 4: Running improved Neural Network model...")
    # run_neural_network(df, random_state)

    print("\n=== Modeling Complete ===")
    print("Results saved to output_2/")


def run_xgboost(df, random_state):
    """Run XGBoost model and save both train and test predictions for error analysis"""
    # Initialize model trainer
    trainer = models.ModelTrainer(model_type="xgboost", random_state=random_state)

    # Store the original DataFrame before any processing
    original_df = df.copy()

    # Prepare data for modeling
    X, y, features, df_prepared = trainer.prepare_data(df)

    if X is None or y is None:
        print("Error preparing data for XGBoost model.")
        return

    # Train and evaluate model
    fold_scores, predictions = trainer.train_and_evaluate(X, y, df_prepared)

    # Train final model
    final_model = trainer.train_final_model(X, y)

    # Get feature importances
    importance_df = trainer.get_feature_importances(final_model)

    if importance_df is not None:
        importance_df.to_csv("output_2/xgboost_feature_importances.csv", index=False)

    # Create visualizations
    visualization.plot_model_performance(
        predictions,
        title="XGBoost Model Performance",
        save_path="output/xgboost_performance.png"
    )

    if importance_df is not None:
        visualization.plot_feature_importances(
            importance_df,
            title="XGBoost Feature Importances",
            save_path="output/xgboost_importances.png"
        )

    # MODIFIED: Save both train and test results for error analysis
    # Check if the predictions dictionary contains the necessary keys
    required_keys = ['test_pred', 'test_true', 'train_pred', 'train_true', 'test_indices', 'train_indices']
    missing_keys = [key for key in required_keys if key not in predictions]

    if missing_keys:
        print(f"Missing required keys for complete analysis: {missing_keys}")
        print(f"Available keys in predictions: {list(predictions.keys())}")
        return

    # Create a combined DataFrame for both train and test data
    results_df = pd.DataFrame()

    # Process test data
    test_indices = predictions['test_indices']
    test_predictions = predictions['test_pred']
    test_true_values = predictions['test_true']

    # Process train data
    train_indices = predictions['train_indices']
    train_predictions = predictions['train_pred']
    train_true_values = predictions['train_true']

    # Function to add data to the results DataFrame
    def add_data_to_results(indices, predictions, true_values, data_type, df_prepared, original_df, features):
        temp_df = pd.DataFrame()

        # Add a column to identify train/test data
        temp_df['data_type'] = [data_type] * len(indices)

        # Add input features
        if isinstance(indices, list) and len(indices) > 0:
            indices = np.array(indices)

        if all(feature in df_prepared.columns for feature in features):
            for feature in features:
                temp_df[feature] = df_prepared.iloc[indices][feature].reset_index(drop=True)

        # Ensure method and polymerization_type are copied from the prepared or original DataFrame
        special_columns = ['method', 'polymerization_type']
        for col in special_columns:
            # Check if column exists in df_prepared first
            if col in df_prepared.columns:
                temp_df[col] = df_prepared.iloc[indices][col].reset_index(drop=True)
            # If not in df_prepared but in original_df, we'll need to map by reaction_id later

        # Get identification columns from original DataFrame if possible
        if 'reaction_id' in df_prepared.columns:
            reaction_ids = df_prepared.iloc[indices]['reaction_id'].values

            # Important identification columns - include method and polymerization_type
            important_columns = [
                'monomer1_name', 'monomer2_name', 'solvent_name',
                'monomer1_smiles', 'monomer2_smiles', 'solvent_smiles',
                'reaction_id', 'reference', 'source_filename',
                'method', 'polymerization_type'  # Explicitly include these
            ]

            # Check which columns exist in the original DataFrame
            existing_columns = [col for col in important_columns if col in original_df.columns]

            if existing_columns and 'reaction_id' in existing_columns:
                # For each sample, find the corresponding row in the original DataFrame
                for i, reaction_id in enumerate(reaction_ids):
                    original_row = original_df[original_df['reaction_id'] == reaction_id]

                    if not original_row.empty:
                        for col in existing_columns:
                            if i >= len(temp_df):
                                # Expand temp_df if needed
                                new_row = pd.DataFrame([{col: original_row[col].values[0]}])
                                temp_df = pd.concat([temp_df, new_row], ignore_index=True)
                            else:
                                # Add or update the column value
                                temp_df.at[i, col] = original_row[col].values[0]

        # Add true values
        temp_df['true_r1r2'] = true_values

        # Try to add r12 and r21 if they're in the prepared dataframe
        if 'r12' in df_prepared.columns and 'r21' in df_prepared.columns:
            temp_df['true_r12'] = df_prepared.iloc[indices]['r12'].values
            temp_df['true_r21'] = df_prepared.iloc[indices]['r21'].values
        elif 'constant_1' in df_prepared.columns and 'constant_2' in df_prepared.columns:
            temp_df['true_r12'] = df_prepared.iloc[indices]['constant_1'].values
            temp_df['true_r21'] = df_prepared.iloc[indices]['constant_2'].values

        # Add predictions
        temp_df['pred_r1r2'] = predictions

        # Calculate errors
        temp_df['error_r1r2'] = temp_df['true_r1r2'] - temp_df['pred_r1r2']
        temp_df['abs_error_r1r2'] = np.abs(temp_df['error_r1r2'])
        temp_df['rel_error_r1r2'] = np.abs(temp_df['error_r1r2'] / np.maximum(temp_df['true_r1r2'], 0.0001)) * 100

        # Check if any of our important columns are still missing
        for col in ['method', 'polymerization_type']:
            if col not in temp_df.columns and col in original_df.columns:
                print(f"Warning: Column '{col}' not transferred properly. Adding it manually.")

                # If we have reaction_id, we can still add the missing column
                if 'reaction_id' in temp_df.columns:
                    id_to_value = dict(zip(original_df['reaction_id'], original_df[col]))
                    temp_df[col] = [id_to_value.get(rid) for rid in temp_df['reaction_id']]

        return temp_df

    # Add test data
    test_df = add_data_to_results(test_indices, test_predictions, test_true_values, 'test', df_prepared, original_df,
                                  features)
    print(f"Processed {len(test_df)} test samples")

    # Add train data
    train_df = add_data_to_results(train_indices, train_predictions, train_true_values, 'train', df_prepared,
                                   original_df, features)
    print(f"Processed {len(train_df)} train samples")

    # Combine both DataFrames
    results_df = pd.concat([test_df, train_df], ignore_index=True)

    # Save the results
    results_df.to_csv("output_2/xgboost_predictions_for_error_analysis.csv", index=False)
    print(
        f"Saved {len(results_df)} samples ({len(test_df)} test, {len(train_df)} train) to output_2/xgboost_predictions_for_error_analysis.csv")

    # Calculate R² separately for train and test
    # Check for NaN values before calculating metrics
    print("Checking for NaN values...")

    # Filter out NaN values from test data
    test_mask = ~(np.isnan(test_true_values) | np.isnan(test_predictions))
    test_true_filtered = test_true_values[test_mask]
    test_pred_filtered = test_predictions[test_mask]
    print(f"Removed {len(test_true_values) - len(test_true_filtered)} NaN values from test data")

    # Filter out NaN values from train data
    train_mask = ~(np.isnan(train_true_values) | np.isnan(train_predictions))
    train_true_filtered = train_true_values[train_mask]
    train_pred_filtered = train_predictions[train_mask]
    print(f"Removed {len(train_true_values) - len(train_true_filtered)} NaN values from train data")

    # Calculate metrics on filtered data
    from sklearn.metrics import r2_score, mean_squared_error

    if len(test_true_filtered) > 0:
        test_r2 = r2_score(test_true_filtered, test_pred_filtered)
        test_rmse = np.sqrt(mean_squared_error(test_true_filtered, test_pred_filtered))
        print(f"XGBoost Test R² score: {test_r2:.4f}")
        print(f"XGBoost Test RMSE: {test_rmse:.4f}")
    else:
        print("Warning: No valid test data for metric calculation")
        test_r2 = None
        test_rmse = None

    if len(train_true_filtered) > 0:
        train_r2 = r2_score(train_true_filtered, train_pred_filtered)
        train_rmse = np.sqrt(mean_squared_error(train_true_filtered, train_pred_filtered))
        print(f"XGBoost Train R² score: {train_r2:.4f}")
        print(f"XGBoost Train RMSE: {train_rmse:.4f}")
    else:
        print("Warning: No valid train data for metric calculation")
        train_r2 = None
        train_rmse = None

    print(f"XGBoost Avg CV R² score: {predictions['avg_test_r2']:.4f}")

    # Save a summary of metrics
    metrics_summary = pd.DataFrame({
        'Metric': ['R² Score', 'RMSE'],
        'Train': [train_r2, train_rmse],
        'Test': [test_r2, test_rmse],
        'CV_Average': [predictions['avg_test_r2'], None]
    })
    metrics_summary.to_csv("output_2/xgboost_metrics_summary.csv", index=False)

    # Print the columns included in the output_2
    print(f"Columns included in the analysis: {results_df.columns.tolist()}")


def plot_predictions(true_values, predicted_values, title, save_path, x_label="True Values",
                     y_label="Predicted Values"):
    """Plot true vs predicted values with a 0-5 range"""
    plt.figure(figsize=(8, 8))

    # Create the scatter plot
    plt.scatter(true_values, predicted_values, alpha=0.6)

    # Add perfect prediction line
    min_val = 0
    max_val = 5
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')

    # Set limits and labels
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add R² value as text
    r2 = r2_score(true_values, predicted_values)
    plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))

    # Save and close
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_training_history(train_losses, val_losses, save_path):
    """Plot training and validation loss history"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def run_neural_network(df, random_state):
    """Run improved Neural Network model"""
    try:
        # Set PyTorch random seed for reproducibility
        torch.manual_seed(random_state)

        # Train the neural network model with improved settings
        print("Training neural network model...")
        model, metrics, scalers, loss_history = train_simple_nn(
            df,
            epochs=500,  # Increased epochs
            lr=3e-4,  # Adjusted learning rate
            batch_size=64,  # Larger batch size
            random_state=random_state,
            transform_targets=True  # Use PowerTransformer with yeo-johnson
        )

        # Unpack scalers
        feature_scaler, target_transformer = scalers

        # Plot training history
        train_losses, val_losses = loss_history
        plot_training_history(train_losses, val_losses, "output/nn_training_history.png")

        # Save metrics
        metrics_df = pd.DataFrame([{
            'Metric': 'R² Score',
            'R1': metrics['r12_r2'],
            'R2': metrics['r21_r2'],
            'Product': metrics['product_r2'],
            'MSE_R1': metrics['r12_mse'],
            'MSE_R2': metrics['r21_mse'],
            'MSE_Product': metrics['product_mse']
        }])

        metrics_df.to_csv("output_2/nn_metrics.csv", index=False)

        # Prepare a fresh test loader for plotting
        train_loader, test_loader, input_size, new_feature_scaler, new_target_transformer = prepare_data(
            df, batch_size=128, random_state=random_state, transform_targets=True
        )

        # Get predictions for plotting - use the target_transformer to get original scale values
        true_values, predicted_values = get_predictions_from_model(model, test_loader, new_target_transformer)

        # Plot predictions for each target
        plot_predictions(
            true_values[:, 0],
            predicted_values[:, 0],
            title="Neural Network: R1 Predictions",
            save_path="output/nn_r1_predictions.png",
            x_label="True R1",
            y_label="Predicted R1"
        )

        plot_predictions(
            true_values[:, 1],
            predicted_values[:, 1],
            title="Neural Network: R2 Predictions",
            save_path="output/nn_r2_predictions.png",
            x_label="True R2",
            y_label="Predicted R2"
        )

        plot_predictions(
            true_values[:, 2],
            predicted_values[:, 2],
            title="Neural Network: R Product Predictions",
            save_path="output/nn_rproduct_predictions.png",
            x_label="True R Product",
            y_label="Predicted R Product"
        )

        # Additional plot: Computed R1*R2 vs True R Product
        computed_product = predicted_values[:, 0] * predicted_values[:, 1]
        plot_predictions(
            true_values[:, 2],
            computed_product,
            title="Neural Network: Computed R1*R2 vs True Product",
            save_path="output/nn_computed_product.png",
            x_label="True R Product",
            y_label="Computed R1*R2"
        )

        # Distribution plots
        plt.figure(figsize=(10, 6))
        plt.hist(true_values[:, 0], bins=30, alpha=0.5, label='True R1')
        plt.hist(predicted_values[:, 0], bins=30, alpha=0.5, label='Predicted R1')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Distribution of True vs Predicted R1 Values')
        plt.legend()
        plt.tight_layout()
        plt.savefig("output_2/nn_r1_distribution.png")
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.hist(true_values[:, 1], bins=30, alpha=0.5, label='True R2')
        plt.hist(predicted_values[:, 1], bins=30, alpha=0.5, label='Predicted R2')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Distribution of True vs Predicted R2 Values')
        plt.legend()
        plt.tight_layout()
        plt.savefig("output_2/nn_r2_distribution.png")
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.hist(true_values[:, 2], bins=30, alpha=0.5, label='True Product')
        plt.hist(predicted_values[:, 2], bins=30, alpha=0.5, label='Predicted Product')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Distribution of True vs Predicted Product Values')
        plt.legend()
        plt.tight_layout()
        plt.savefig("output_2/nn_product_distribution.png")
        plt.close()

        # Save model
        try:
            model_info = {
                'model_state': model.state_dict(),
                'feature_scaler': feature_scaler,
                'target_transformer': target_transformer,
                'input_size': model.shared_layers[0].in_features
            }
            torch.save(model_info, "output_2/nn_model.pt")
            print("Model saved successfully to output_2/nn_model.pt")
        except Exception as e:
            print(f"Error saving model: {e}")

        avg_r2 = (metrics['r12_r2'] + metrics['r21_r2'] + metrics['product_r2']) / 3
        print(f"Neural Network average R² score: {avg_r2:.4f}")
        print("Prediction plots saved to output_2/ directory")

    except Exception as e:
        print(f"Error running neural network model: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()