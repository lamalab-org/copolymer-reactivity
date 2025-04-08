"""
Simple main script for the copolymerization prediction model
"""

import os
import data_processing
import models
import visualization

# Import neural network module if available
try:
    from old_data_test.new_modules.output import NN_model

    NN_AVAILABLE = True
except ImportError:
    NN_AVAILABLE = False
    print("Neural network module not available.")


def main():
    """Simple main function to run both XGBoost and Neural Network models"""
    # Input data path
    data_path = "../../data_extraction/new/extracted_reactions.csv"

    # Set random seed for reproducibility
    random_state = 42

    # Create output directory
    os.makedirs("output", exist_ok=True)

    print("=== Copolymerization Prediction Model ===")

    # Step 1: Load and preprocess data
    print("\nStep 1: Loading and preprocessing data...")
    df = data_processing.load_and_preprocess_data(data_path)

    if df is None or len(df) == 0:
        print("Error: No data available for modeling.")
        return

    # Save processed data
    df.to_csv("output/processed_data.csv", index=False)

    # Step 2: Run XGBoost model
    print("\nStep 2: Running XGBoost model...")
    #run_xgboost(df, random_state)

    # Step 3: Run Neural Network model if available
    if NN_AVAILABLE:
        print("\nStep 3: Running Neural Network model...")
        run_neural_network(df, random_state)
    else:
        print("\nStep 3: Skipping Neural Network (module not available)")

    print("\n=== Modeling Complete ===")
    print("Results saved to output/")


def run_xgboost(df, random_state):
    """Run XGBoost model"""
    # Initialize model trainer
    trainer = models.ModelTrainer(model_type="xgboost", random_state=random_state)

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
        importance_df.to_csv("output/xgboost_feature_importances.csv", index=False)

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

    print(f"XGBoost R² score: {predictions['avg_test_r2']:.4f}")


def run_neural_network(df, random_state):
    """Run Neural Network model"""
    # Prepare data for neural network
    train_loader, test_loader, mol1_cols, mol2_cols, condition_cols = NN_model.prepare_data_from_df(
        df, batch_size=32, train_split=0.8, random_state=random_state
    )

    # Train the model (use fewer epochs for demonstration)
    model, metrics = NN_model.train_nn_model(
        train_loader, test_loader, num_epochs=50, learning_rate=1e-4
    )

    # Save metrics
    import pandas as pd
    metrics_df = pd.DataFrame([{
        'Metric': 'R² Score',
        'R12': metrics['r12_r2'],
        'R21': metrics['r21_r2'],
        'Product': metrics['product_r2']
    }])

    metrics_df.to_csv("output/nn_metrics.csv", index=False)

    # Save model
    try:
        import torch
        torch.save(model.state_dict(), "output/nn_model.pt")
    except Exception as e:
        print(f"Error saving model: {e}")

    avg_r2 = (metrics['r12_r2'] + metrics['r21_r2'] + metrics['product_r2']) / 3
    print(f"Neural Network average R² score: {avg_r2:.4f}")


if __name__ == "__main__":
    main()