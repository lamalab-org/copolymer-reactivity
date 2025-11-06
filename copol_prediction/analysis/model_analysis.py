import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from sklearn.calibration import calibration_curve
from sklearn.metrics import classification_report


def load_predictions(path="output/xgboost_predictions_with_confidence.csv"):
    return pd.read_csv(path)


def add_entropy(predictions_df):
    proba_cols = [f'proba_class_{i}' for i in range(3)]
    probs = predictions_df[proba_cols].values
    entropies = entropy(probs.T)  # shape must be (n_classes, n_samples)
    predictions_df['entropy'] = entropies
    return predictions_df

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_confidence_vs_r1r2(predictions_df, n_bins=30, r1r2_min=1e-3, r1r2_max=100):
    plt.figure(figsize=(10, 6))

    # === 1. Plot scatter as usual ===
    sns.scatterplot(
        data=predictions_df,
        x='r1r2',
        y='confidence_score',
        hue='predicted_label',
        alpha=0.5,
        palette='Set2',
        edgecolor=None,
        s=20
    )

    # === 2. Class boundaries ===
    plt.axvline(1, color='gray', linestyle='--')
    plt.axvline(25, color='gray', linestyle='--')

    # === 3. Create log-spaced bins but keep linear scale ===
    bins = np.logspace(np.log10(r1r2_min), np.log10(r1r2_max), n_bins + 1)
    df = predictions_df.copy()
    df = df[(df['r1r2'] >= r1r2_min) & (df['r1r2'] <= r1r2_max)]
    df['r1r2_bin'] = pd.cut(df['r1r2'], bins=bins)
    df['bin_mid'] = df['r1r2_bin'].apply(lambda x: x.mid if pd.notnull(x) else np.nan)

    # === 4. Aggregate per bin ===
    grouped = df.groupby('bin_mid')['confidence_score'].mean().reset_index()
    grouped.columns = ['r1r2_mid', 'mean_confidence']

    # === 5. Plot trend line on linear x-axis ===
    sns.lineplot(
        data=grouped,
        x='r1r2_mid',
        y='mean_confidence',
        color='black',
        linewidth=2,
        label='Binned Mean Confidence'
    )

    # === 6. Final formatting ===
    plt.xlabel("r1r2")
    plt.ylabel("Max Probability (Confidence)")
    plt.title("Prediction Confidence vs. r1r2 (Log-Binned, Linear Axis)")
    plt.xlim(0, 10)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig("output/confidence_vs_r1r2_logbin_linearx.png", dpi=300)
    print("✅ Saved: output/confidence_vs_r1r2_logbin_linearx.png")

import pandas as pd
import plotly.graph_objects as go
import os
import webbrowser

def plot_top_monomer_pairs_with_class_symbols(predictions_df, top_n=10, outdir="output/monomer_pair_plots", open_browser=True):
    os.makedirs(outdir, exist_ok=True)

    # Create unordered monomer pair key
    predictions_df['monomer_pair_key'] = predictions_df.apply(
        lambda row: tuple(sorted([row['monomer1_smiles'], row['monomer2_smiles']])),
        axis=1
    )

    top_pairs = predictions_df['monomer_pair_key'].value_counts().nlargest(top_n).index.tolist()

    # Define symbol map for true classes
    symbol_map = {
        0: 'circle',
        1: 'square',
        2: 'cross'
    }

    for i, pair in enumerate(top_pairs, 1):
        df_pair = predictions_df[predictions_df['monomer_pair_key'] == pair].copy()

        df_pair['is_correct'] = df_pair['correct_prediction'].map({True: '✔ Correct', False: '✘ Incorrect'})
        df_pair['predicted_confidence'] = df_pair.apply(
            lambda row: row[f'proba_class_{row["predicted_label"]}'], axis=1
        )

        mon1 = df_pair.iloc[0]['monomer1_name']
        mon2 = df_pair.iloc[0]['monomer2_name']
        title = f"Monomer Pair {i}: {mon1} + {mon2}"

        # Create one trace per true class, with different symbols
        traces = []
        buttons = []
        true_classes = sorted(df_pair['true_label'].unique())

        for j, cls in enumerate(true_classes):
            df_cls = df_pair[df_pair['true_label'] == cls]
            trace = go.Scatter(
                x=df_cls['r1r2'],
                y=df_cls['predicted_confidence'],
                mode='markers',
                name=f"True Class {cls}",
                marker=dict(
                    color=['green' if c else 'red' for c in df_cls['correct_prediction']],
                    symbol=symbol_map.get(cls, 'circle'),
                    size=10,
                    opacity=0.75,
                    line=dict(width=1, color='black')
                ),
                text=[
                    f"<b>{m1} + {m2}</b><br>Solvent: {solv}<br>Temp: {temp}°C<br>"
                    f"True: {true} | Pred: {pred}"
                    for m1, m2, solv, temp, type, true, pred in zip(
                        df_cls['monomer1_name'], df_cls['monomer2_name'],
                        df_cls['solvent'], df_cls['temperature'], df_cls['polymerization_type'],
                        df_cls['true_label'], df_cls['predicted_label']
                    )
                ],
                hoverinfo='text',
                visible=True
            )
            traces.append(trace)

        # Optional: dropdown to toggle by true class
        for j, cls in enumerate(true_classes):
            vis = [False] * len(traces)
            vis[j] = True
            buttons.append(dict(label=f"Only Class {cls}", method="update", args=[{"visible": vis}]))

        # Add "All"
        buttons.insert(0, dict(label="All", method="update", args=[{"visible": [True] * len(traces)}]))

        fig = go.Figure(data=traces)
        fig.update_layout(
            updatemenus=[dict(
                type="dropdown",
                x=0.0, y=1.2,
                showactive=True,
                buttons=buttons,
                direction="down"
            )],
            title=title,
            xaxis_title="r1r2 (True Value)",
            yaxis_title="Confidence of Predicted Class",
            height=500,
            width=800,
            legend_title="True Class (Symbol Shape)"
        )

        filename = f"{outdir}/monomer_pair_{i}_{mon1}_{mon2}_symbols.html".replace(" ", "_")
        fig.write_html(filename)
        print(f"✅ Saved: {filename}")

        if open_browser:
            webbrowser.open_new_tab(f"file://{os.path.abspath(filename)}")

            # === Classification Report ===
            y_true = df_pair['true_label']
            y_pred = df_pair['predicted_label']
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

            # Extract per-class and macro metrics
            metrics_summary = "\n".join([
                f"Class {cls}: P={report[str(cls)]['precision']:.2f}, R={report[str(cls)]['recall']:.2f}, F1={report[str(cls)]['f1-score']:.2f}"
                for cls in sorted(df_pair['true_label'].unique())
            ])
            metrics_summary += f"\nMacro Avg: P={report['macro avg']['precision']:.2f}, R={report['macro avg']['recall']:.2f}, F1={report['macro avg']['f1-score']:.2f}"

            # Print to console
            print(f"\n📊 Classification Report for Pair {i}: {mon1} + {mon2}")
            print(metrics_summary)

            # Optionally: include in figure annotation
            fig.add_annotation(
                text=f"<b>Metrics:</b><br>{metrics_summary.replace(chr(10), '<br>')}",
                xref="paper", yref="paper",
                x=1.02, y=1.0,
                showarrow=False,
                align='left',
                font=dict(size=12),
                bordercolor="black",
                borderwidth=1,
                bgcolor="white"
            )


def plot_all_in_subplots(predictions_df, save_path="output/summary_plots.png"):
    sns.set(style="whitegrid")

    fig, axs = plt.subplots(2, 2, figsize=(16, 12))

    # === 1. Calibration Curve (Top-Left) ===
    for cls in range(3):
        y_true_cls = (predictions_df['true_label'] == cls).astype(int)
        prob_cls = predictions_df[f'proba_class_{cls}']
        prob_true, prob_pred = calibration_curve(y_true_cls, prob_cls, n_bins=10, strategy='uniform')
        axs[0, 0].plot(prob_pred, prob_true, marker='o', label=f"Class {cls}")

    axs[0, 0].plot([0, 1], [0, 1], 'k--', label='Ideal')
    axs[0, 0].set_title("Calibration Curves per Class")
    axs[0, 0].set_xlabel("Predicted Probability")
    axs[0, 0].set_ylabel("Empirical Accuracy")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # === 2–4. Stacked Class Probabilities for each Predicted Class ===
    proba_cols = [f'proba_class_{i}' for i in range(3)]

    for cls in range(3):
        row, col = divmod(cls + 1, 2)  # Place class 0 in [0,1], 1 in [1,0], 2 in [1,1]
        ax = axs[row, col]
        subset = predictions_df[predictions_df['predicted_label'] == cls]

        ax.hist(
            [subset[col_name] for col_name in proba_cols],
            bins=20,
            stacked=True,
            label=[f'Class {i}' for i in range(3)],
            color=['#4c72b0', '#55a868', '#c44e52']
        )
        ax.set_title(f"Predicted Class = {cls}")
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Sample Count")
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\n✅ Saved combined plot to: {save_path}")


def main():
    print("🔍 Loading predictions and generating plots...")
    df = load_predictions()
    plot_all_in_subplots(df)
    plot_confidence_vs_r1r2(df)

    # Load predictions with group IDs
    pred_path = "output/xgboost_predictions_with_confidence.csv"
    pred_df = pd.read_csv(pred_path)

    # Ensure monomer pair keys are available
    pred_df['monomer_pair_key'] = pred_df.apply(
        lambda row: tuple(sorted([row['monomer1_smiles'], row['monomer2_smiles']])),
        axis=1
    )
    pair_to_id = {pair: idx for idx, pair in enumerate(pred_df['monomer_pair_key'].unique())}
    pred_df['group_id'] = pred_df['monomer_pair_key'].map(pair_to_id)

    # Group by monomer pair
    grouped = pred_df.groupby('group_id')

    # Analyze prediction consistency
    consistency_stats = []
    for gid, group in grouped:
        unique_true = set(group['true_label'])
        unique_pred = set(group['predicted_label'])
        agreement = (group['true_label'] == group['predicted_label']).sum()
        consistency_stats.append({
            'group_id': gid,
            'n_samples': len(group),
            'n_true_classes': len(unique_true),
            'n_pred_classes': len(unique_pred),
            'accuracy_within_group': agreement / len(group),
            'pred_classes': list(unique_pred),
            'true_classes': list(unique_true)
        })

    consistency_df = pd.DataFrame(consistency_stats)

    # Save overview
    consistency_df.to_csv("output/data_analysis/prediction_consistency_per_monomer_pair.csv", index=False)

    # Show inconsistent examples
    inconsistent = consistency_df[consistency_df['n_pred_classes'] > 1]
    print("\n❗ Inconsistent predictions for same monomer pairs:")
    print(inconsistent.head(10).to_string(index=False))

    plot_top_monomer_pairs_with_class_symbols(df, top_n=10)


if __name__ == "__main__":
    main()
