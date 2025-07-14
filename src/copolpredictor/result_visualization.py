"""
Visualization module for the copolymerization prediction model

Contains functions for creating static and interactive plots of model results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


plt.style.use('lamalab.mplstyle')

# Check if plotly is available for interactive plots
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not installed. Interactive plots will not be available.")
    print("Install with: pip install plotly")


def plot_model_performance(predictions, title=None, save_path="model_performance.png", interactive=True):
    """
    Create a scatter plot of true vs predicted values

    Args:
        predictions: Dictionary with prediction data
        title: Plot title (optional)
        save_path: Path to save the static plot
        interactive: Whether to create an interactive plot as well (if plotly is available)
    """
    # Extract prediction data
    y_true_inv = predictions['test_true']
    y_pred_inv = predictions['test_pred']
    y_train_true_inv = predictions['train_true']
    y_train_pred_inv = predictions['train_pred']
    avg_test_r2 = predictions['avg_test_r2']

    # Create arrays for the test data points
    test_data = np.column_stack((y_true_inv, y_pred_inv))
    train_data = np.column_stack((y_train_true_inv, y_train_pred_inv))

    # Randomly select points if there are more than requested
    if len(test_data) > 200:
        np.random.seed(42)
        test_indices = np.random.choice(len(test_data), size=200, replace=False)
        test_data = test_data[test_indices]

    if len(train_data) > 500:
        np.random.seed(43)  # Use different seed than test set
        train_indices = np.random.choice(len(train_data), size=500, replace=False)
        train_data = train_data[train_indices]

    # Create static matplotlib plot
    plt.figure(figsize=(10, 8))

    # Plot the training and test points
    plt.scatter(train_data[:, 0], train_data[:, 1], alpha=0.5, color='#661124',
                label='Trainingset prediction ', s=50)
    plt.scatter(test_data[:, 0], test_data[:, 1], alpha=0.8, color='#194A81',
                label='Testset prediction ',s=50)

    # Add the diagonal line (perfect prediction)
    plt.plot([0, 5], [0, 5], color='gray', linestyle='--')

    # Set axis limits from 0 to 5
    plt.xlim(0, 5)
    plt.ylim(0, 5)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.margins(x=0.1, y=0.1)

    # Add labels and title
    plt.xlabel('True r-product', fontsize=24)
    plt.ylabel('Predicted r-product', fontsize=24)

    # Add legend
    plt.legend(loc='upper left', fontsize=24)

    # Save figure with higher resolution and tighter layout
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Also create interactive plot if requested and plotly is available
    if interactive and PLOTLY_AVAILABLE:
        try:
            create_interactive_plot(predictions)
        except Exception as e:
            print(f"Error creating interactive plot: {e}")


def create_interactive_plot(predictions, df=None, save_path='interactive_model_performance.html'):
    """
    Create an interactive scatter plot of true vs predicted values

    Args:
        predictions: Dictionary with prediction data
        df: DataFrame with original data for hover information (optional)
        save_path: Path to save the HTML file
    """
    if not PLOTLY_AVAILABLE:
        print("Error: Plotly is not available. Cannot create interactive plot.")
        return

    # Extract prediction data
    y_true_inv = predictions['test_true']
    y_pred_inv = predictions['test_pred']
    y_train_true_inv = predictions['train_true']
    y_train_pred_inv = predictions['train_pred']
    avg_test_r2 = predictions['avg_test_r2']

    # Get indices if available
    train_indices = predictions.get('train_indices', [])
    test_indices = predictions.get('test_indices', [])

    # Create figure
    fig = make_subplots(specs=[[{"secondary_y": False}]])

    # If DataFrame is provided, use it for hover info
    if df is not None and len(train_indices) > 0 and len(test_indices) > 0:
        # Format hover text for training data
        train_hover_texts = []
        for i, idx in enumerate(train_indices):
            if i >= len(y_train_true_inv) or idx >= len(df):
                continue  # Skip if index is out of bounds

            row = df.iloc[idx]
            hover_text = f"<b>Monomer 1:</b> {row.get('monomer1_name', 'N/A')}<br>" + \
                         f"<b>Monomer 2:</b> {row.get('monomer2_name', 'N/A')}<br>" + \
                         f"<b>Temperature:</b> {row.get('temperature', 'N/A')} K<br>" + \
                         f"<b>Solvent:</b> {row.get('solvent', 'N/A')}<br>" + \
                         f"<b>Method:</b> {row.get('method', 'N/A')}<br>" + \
                         f"<b>Polymerization type:</b> {row.get('polymerization_type', 'N/A')}<br>" + \
                         f"<b>True Value:</b> {y_train_true_inv[i]:.3f}<br>" + \
                         f"<b>Prediction:</b> {y_train_pred_inv[i]:.3f}"
            train_hover_texts.append(hover_text)

        # Format hover text for test data
        test_hover_texts = []
        for i, idx in enumerate(test_indices):
            if i >= len(y_true_inv) or idx >= len(df):
                continue  # Skip if index is out of bounds

            row = df.iloc[idx]
            hover_text = f"<b>Monomer 1:</b> {row.get('monomer1_name', 'N/A')}<br>" + \
                         f"<b>Monomer 2:</b> {row.get('monomer2_name', 'N/A')}<br>" + \
                         f"<b>Temperature:</b> {row.get('temperature', 'N/A')} K<br>" + \
                         f"<b>Solvent:</b> {row.get('solvent', 'N/A')}<br>" + \
                         f"<b>Method:</b> {row.get('method', 'N/A')}<br>" + \
                         f"<b>Polymerization type:</b> {row.get('polymerization_type', 'N/A')}<br>" + \
                         f"<b>True Value:</b> {y_true_inv[i]:.3f}<br>" + \
                         f"<b>Prediction:</b> {y_pred_inv[i]:.3f}"
            test_hover_texts.append(hover_text)
    else:
        # Simple hover text without detailed data
        train_hover_texts = [f"True: {y_true:.3f}<br>Pred: {y_pred:.3f}"
                             for y_true, y_pred in zip(y_train_true_inv, y_train_pred_inv)]
        test_hover_texts = [f"True: {y_true:.3f}<br>Pred: {y_pred:.3f}"
                            for y_true, y_pred in zip(y_true_inv, y_pred_inv)]

    # Add scatter plots for training data
    fig.add_trace(
        go.Scatter(
            x=y_train_true_inv,
            y=y_train_pred_inv,
            mode='markers',
            marker=dict(
                color='blue',
                size=10,
                opacity=0.5
            ),
            text=train_hover_texts,
            hoverinfo='text',
            name='Training Points'
        )
    )

    # Add scatter plots for test data
    fig.add_trace(
        go.Scatter(
            x=y_true_inv,
            y=y_pred_inv,
            mode='markers',
            marker=dict(
                color='red',
                size=10,
                symbol='x',
                opacity=0.8
            ),
            text=test_hover_texts,
            hoverinfo='text',
            name='Test Points'
        )
    )

    # Add the perfect prediction diagonal line
    fig.add_trace(
        go.Scatter(
            x=[0, 5],
            y=[0, 5],
            mode='lines',
            line=dict(color='green', width=2, dash='dash'),
            name='Perfect Prediction'
        )
    )

    # Update layout
    fig.update_layout(
        title=f'Model Performance (Test R² = {avg_test_r2:.4f})',
        xaxis_title='True r_product',
        yaxis_title='Predicted r_product',
        xaxis=dict(range=[0, 5]),
        yaxis=dict(range=[0, 5]),
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor='rgba(255, 255, 255, 0.5)'
        ),
        hovermode='closest',
        plot_bgcolor='white',
        width=900,
        height=700
    )

    # Add grid lines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    # Add annotation with R² value
    fig.add_annotation(
        x=0.15,
        y=0.95,
        xref='paper',
        yref='paper',
        text=f'R² = {avg_test_r2:.4f}',
        showarrow=False,
        font=dict(size=14),
        bgcolor='white',
        bordercolor='black',
        borderwidth=1,
        borderpad=4
    )

    # Save the interactive plot as an HTML file
    fig.write_html(save_path)
    print(f"Interactive plot saved as '{save_path}'")

    # Also save as static image for reports
    try:
        image_path = save_path.replace('.html', '.png')
        fig.write_image(image_path, scale=2)
        print(f"Static image also saved as '{image_path}'")
    except Exception as e:
        print(f"Could not save static image: {e}")
        print("You may need to install kaleido: pip install kaleido")

    return fig


def plot_feature_importances(importance_df, n_features=20,
                             title="Feature Importances",
                             save_path="feature_importances.png",
                             interactive=True):
    """
    Plot feature importances

    Args:
        importance_df: DataFrame with Feature and Importance columns
        n_features: Number of top features to plot
        title: Plot title
        save_path: Path to save the plot
        interactive: Whether to create an interactive plot as well
    """
    # Ensure we don't try to plot more features than we have
    n_features = min(n_features, len(importance_df))

    # Get top features
    top_features = importance_df.head(n_features)

    # Create static plot
    plt.figure(figsize=(12, 8))
    plt.bar(range(n_features), top_features['Importance'])
    plt.xticks(range(n_features), top_features['Feature'], rotation=90)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Create interactive plot if requested
    if interactive and PLOTLY_AVAILABLE:
        try:
            # Create figure
            fig = go.Figure()

            # Add bar chart
            fig.add_trace(go.Bar(
                x=top_features['Feature'],
                y=top_features['Importance'],
                marker_color='darkblue',
                hovertemplate='<b>%{x}</b><br>Importance: %{y:.4f}<extra></extra>'
            ))

            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title='Feature',
                yaxis_title='Importance',
                xaxis_tickangle=-90,
                plot_bgcolor='white',
                width=1000,
                height=600
            )

            # Add grid lines
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

            # Save the interactive plot
            interactive_path = save_path.replace('.png', '_interactive.html')
            fig.write_html(interactive_path)
            print(f"Interactive feature importance plot saved as '{interactive_path}'")
        except Exception as e:
            print(f"Error creating interactive feature importance plot: {e}")


def plot_learning_curve(results_df, title="Learning Curve", save_path="learning_curve.png"):
    """
    Plot the learning curve from learning curve results

    Args:
        results_df: DataFrame with learning curve results
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 8))

    # Plot the scores
    plt.plot(results_df['sample_size'], results_df['train_r2'], 'o-', color='blue',
             label='Training score')
    plt.plot(results_df['sample_size'], results_df['test_r2'], 'o-', color='red',
             label='Cross-validation score')

    # Add error bands
    plt.fill_between(results_df['sample_size'],
                     np.array(results_df['train_r2']) - np.array(results_df['train_std']),
                     np.array(results_df['train_r2']) + np.array(results_df['train_std']),
                     alpha=0.1, color='blue')
    plt.fill_between(results_df['sample_size'],
                     np.array(results_df['test_r2']) - np.array(results_df['test_std']),
                     np.array(results_df['test_r2']) + np.array(results_df['test_std']),
                     alpha=0.1, color='red')

    # Add a table with the exact values
    table_data = []
    for i in range(len(results_df)):
        table_data.append([
            results_df['sample_size'].iloc[i],
            f"{results_df['train_r2'].iloc[i]:.4f} ± {results_df['train_std'].iloc[i]:.4f}",
            f"{results_df['test_r2'].iloc[i]:.4f} ± {results_df['test_std'].iloc[i]:.4f}"
        ])

    plt.table(
        cellText=table_data,
        colLabels=['Sample Size', 'Train R²', 'Test R²'],
        cellLoc='center',
        loc='lower center',
        bbox=[0.2, -0.4, 0.6, 0.2]
    )

    # Add grid lines for better readability
    plt.grid(True, linestyle='--', alpha=0.7)

    # Set axis limits
    plt.ylim(0, 1.0)
    plt.xlim(min(results_df['sample_size']) - 50, max(results_df['sample_size']) + 50)

    # Add labels and title
    plt.xlabel('Training Set Size', fontsize=12)
    plt.ylabel('R² Score', fontsize=12)
    plt.title(title, fontsize=14)

    # Add legend
    plt.legend(loc='lower right')

    # Save with extra space for the table
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.4)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Also create interactive version if plotly is available
    if PLOTLY_AVAILABLE:
        try:
            fig = go.Figure()

            # Add traces for train and test scores
            fig.add_trace(go.Scatter(
                x=results_df['sample_size'],
                y=results_df['train_r2'],
                mode='lines+markers',
                name='Training score',
                line=dict(color='blue', width=2),
                marker=dict(size=8, color='blue'),
                error_y=dict(
                    type='data',
                    array=results_df['train_std'],
                    visible=True,
                    color='blue',
                    thickness=1,
                    width=3
                )
            ))

            fig.add_trace(go.Scatter(
                x=results_df['sample_size'],
                y=results_df['test_r2'],
                mode='lines+markers',
                name='Cross-validation score',
                line=dict(color='red', width=2),
                marker=dict(size=8, color='red'),
                error_y=dict(
                    type='data',
                    array=results_df['test_std'],
                    visible=True,
                    color='red',
                    thickness=1,
                    width=3
                )
            ))

            # Create a table to display the results
            table_data = [
                ['Sample Size', 'Train R²', 'Test R²']
            ]

            for i in range(len(results_df)):
                table_data.append([
                    str(results_df['sample_size'].iloc[i]),
                    f"{results_df['train_r2'].iloc[i]:.4f} ± {results_df['train_std'].iloc[i]:.4f}",
                    f"{results_df['test_r2'].iloc[i]:.4f} ± {results_df['test_std'].iloc[i]:.4f}"
                ])

            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title='Training Set Size',
                yaxis_title='R² Score',
                xaxis=dict(
                    range=[min(results_df['sample_size']) - 50, max(results_df['sample_size']) + 50]
                ),
                yaxis=dict(range=[0, 1.0]),
                legend=dict(
                    x=0.01,
                    y=0.99,
                    bgcolor='rgba(255, 255, 255, 0.5)'
                ),
                plot_bgcolor='white',
                width=1000,
                height=600
            )

            # Add grid lines
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

            # Save the interactive plot
            interactive_path = save_path.replace('.png', '_interactive.html')
            fig.write_html(interactive_path)
            print(f"Interactive learning curve plot saved as '{interactive_path}'")

        except Exception as e:
            print(f"Error creating interactive learning curve plot: {e}")

        def compare_models(model_results, title="Model Comparison", save_path="model_comparison.png"):
            """
            Create a bar chart comparing different models

            Args:
                model_results: Dictionary with model names as keys and R² scores as values
                title: Plot title
                save_path: Path to save the plot
            """
            # Extract model names and scores
            models = list(model_results.keys())
            train_scores = [results.get('avg_train_r2', 0) for results in model_results.values()]
            test_scores = [results.get('avg_test_r2', 0) for results in model_results.values()]

            # Create the plot
            x = np.arange(len(models))
            width = 0.35

            fig, ax = plt.subplots(figsize=(10, 6))
            train_bars = ax.bar(x - width / 2, train_scores, width, label='Train R²', color='blue', alpha=0.7)
            test_bars = ax.bar(x + width / 2, test_scores, width, label='Test R²', color='red', alpha=0.7)

            # Add some text for labels, title and axes ticks
            ax.set_ylabel('R² Score')
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels(models)
            ax.legend()

            # Add value labels on top of bars
            def add_labels(bars):
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f'{height:.3f}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')

            add_labels(train_bars)
            add_labels(test_bars)

            # Add grid lines
            ax.grid(True, linestyle='--', alpha=0.7, axis='y')

            # Save the plot
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            # Also create interactive version if plotly is available
            if PLOTLY_AVAILABLE:
                try:
                    fig = go.Figure()

                    # Add traces for train and test scores
                    fig.add_trace(go.Bar(
                        x=models,
                        y=train_scores,
                        name='Train R²',
                        marker_color='blue',
                        opacity=0.7,
                        text=[f'{score:.3f}' for score in train_scores],
                        textposition='outside'
                    ))

                    fig.add_trace(go.Bar(
                        x=models,
                        y=test_scores,
                        name='Test R²',
                        marker_color='red',
                        opacity=0.7,
                        text=[f'{score:.3f}' for score in test_scores],
                        textposition='outside'
                    ))

                    # Update layout
                    fig.update_layout(
                        title=title,
                        yaxis_title='R² Score',
                        barmode='group',
                        plot_bgcolor='white',
                        width=900,
                        height=600
                    )

                    # Add grid lines
                    fig.update_xaxes(showgrid=False)
                    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

                    # Save the interactive plot
                    interactive_path = save_path.replace('.png', '_interactive.html')
                    fig.write_html(interactive_path)
                    print(f"Interactive model comparison plot saved as '{interactive_path}'")

                except Exception as e:
                    print(f"Error creating interactive model comparison plot: {e}")