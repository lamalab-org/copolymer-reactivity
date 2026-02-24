"""
Create a bar plot comparing different models for data extraction.

This script creates a grouped bar plot comparing various LLM models
across different evaluation metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add copol_prediction to path to import plot_config
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'copol_prediction' / 'analysis'))
from plot_config import (
    SEQUENTIAL_COLORS, 
    TWO_COL_WIDTH_INCH
)

# Define the dataset with original values (from table)
# Removed: Number of Calls, Fuzzy Matching Score
# Execution Time divided by 10 (per paper), Rate of Empty Entries as decimal
columns = [
    "Model",
    "Execution\nTime (s)",
    "Cost ($)",
    "Rate of Empty\nEntries",
    "Precision",
]

data_list = [
    ("GPT4 Vision", 537.81 / 10, 1.51, 13.99 / 100, 0.81),
    ("GPT4-o", 302.85 / 10, 0.58, 7.64 / 100, 0.68),
    ("Claude Opus Vision", 622.13 / 10, 0.37, 24.90 / 100, 0.10),
]

# Use colors from plot_config
color_list = SEQUENTIAL_COLORS[:len(data_list)]

# Create a DataFrame
df = pd.DataFrame(data_list, columns=columns)

# Prepare data for bar plot
# Each metric gets its own plot
all_metrics = columns[1:]  # All metrics except Model
models = df["Model"].tolist()
num_models = len(models)

# Set up the plot with TWO_COL_WIDTH and four subplots (1x4, side by side)
# For 4 subplots side by side, reduce individual plot width
height = TWO_COL_WIDTH_INCH * (5/14) * 0.9  # Reduced height
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(TWO_COL_WIDTH_INCH * 1.0, height))
axes = [ax1, ax2, ax3, ax4]

# Helper function to plot single metric bar plot
def plot_single_metric(ax, metric, title_label, ylabel):
    x = np.arange(1)  # Only one metric per plot
    width = 0.12  # Narrower bars
    spacing = 0.02
    
    # Calculate positions for each model
    positions = []
    for i in range(num_models):
        offset = (i - num_models / 2 + 0.5) * (width + spacing)
        positions.append(x[0] + offset)
    
    bars = []
    for i, model in enumerate(models):
        value = df.loc[i, metric]
        bar = ax.bar(positions[i], value, width, label=model, color=color_list[i], alpha=0.8)
        bars.append(bar)
    
    ax.set_title(title_label, fontsize=10, loc='left', fontweight='bold')
    ax.set_xlabel('')  # No x-label needed for single metric
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xticks([0])
    ax.set_xticklabels([metric], rotation=45, ha='right', wrap=True, fontsize=5)
    
    # Return handles and labels for legend (don't show legend in individual plots)
    handles, labels = ax.get_legend_handles_labels()
    
    ax.grid(False)
    ax.tick_params(labelsize=6)
    # Remove top and right spines (keep bottom and left for axes)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(bottom=0)
    
    return handles, labels

# Plot each metric separately and collect legend handles
handles1, labels1 = plot_single_metric(ax1, all_metrics[0], 'a', 'Execution Time (s)')
handles2, labels2 = plot_single_metric(ax2, all_metrics[1], 'b', 'Cost ($)')
handles3, labels3 = plot_single_metric(ax3, all_metrics[2], 'c', 'Rate of Empty Entries')
handles4, labels4 = plot_single_metric(ax4, all_metrics[3], 'd', 'Precision')

# Create common legend below all plots
fig.legend(handles1, labels1, loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False, fontsize=7)

# Set face colors
for ax in axes:
    ax.set_facecolor('white')
fig.patch.set_facecolor('white')

# Adjust layout to prevent label cutoff and make room for legend
plt.tight_layout(rect=[0, 0.02, 1, 1])

# Create output directory if it doesn't exist
output_dir = Path(__file__).parent / 'figures'
output_dir.mkdir(exist_ok=True)

# Save the plot
output_path = output_dir / 'model_comparison_barplot.pdf'
fig.savefig(output_path, bbox_inches='tight', dpi=300)
print(f"Plot saved to: {output_path}")

# Also save as PNG
output_path_png = output_dir / 'model_comparison_barplot.png'
fig.savefig(output_path_png, bbox_inches='tight', dpi=300)
print(f"Plot saved to: {output_path_png}")

# Try to show plot, but don't fail if display is not available
try:
    plt.show()
except Exception as e:
    print(f"Note: Could not display plot interactively: {e}")
    print("Plots have been saved successfully.")

