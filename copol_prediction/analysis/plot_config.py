"""
Plot configuration for copolymerization prediction analysis.

Defines colors, styles, and matplotlib settings for consistent visualizations.

HOW TO CUSTOMIZE COLORS:
1. Edit the hex color codes below (format: '#RRGGBB')
2. CLASS_COLORS: Colors for each prediction class
3. COMPARISON_COLORS: Colors for before/after comparisons
4. SEQUENTIAL_COLORS: Colors for multi-series plots
5. All changes apply automatically to analyze_model.py plots

The matplotlib style is loaded from plots_and_figures/lamalab.mplstyle
"""

from pathlib import Path

import matplotlib.pyplot as plt

# ============================================================================
# COLOR SCHEME
# ============================================================================

# Main colors for classes (CHANGE THESE TO YOUR PREFERRED COLORS)
CLASS_COLORS = {
    0: "#3A3B73",  # Class 0 (< 1) - Blue
    1: "#e27f07",  # Class 1 (1-25) - Orange
    2: "#6a040f",  # Class 2 (> 25) - Red
}

# Comparison colors (Original vs Filtered, Before vs After, etc.)
COMPARISON_COLORS = {
    "original": "#661124",  # Light red/coral
    "filtered": "#143D60",  # Light green
    "correct": "#2266ac",  # Light green
    "incorrect": "#920506",  # Light red
    "train": "#920506",  # Light blue
    "test": "#2266ac",  # Light orange
}

# Sequential colors for multi-element plots
# Custom palette for monomer/network and comparison plots
SEQUENTIAL_COLORS = [
    "#0a0e38",
    "#3e3888",
    "#1e8db9",
    "#9ed5f2",
    "#6a040f",
    "#bb1818",
    "#fe5318",
    "#e27f07",
    "#ffbc57",
]

# Consistent two-line palette for lab time-series panels
LAB_SERIES_COLORS = {
    "series_1": SEQUENTIAL_COLORS[2],  # cool
    "series_2": SEQUENTIAL_COLORS[5],  # warm
}

# Monomer-specific colors used in lab/case-study plots
MONOMER_COLORS = {
    "AN": SEQUENTIAL_COLORS[5],  # acrylonitrile
    "VP": SEQUENTIAL_COLORS[2],  # N-vinyl-5-pyrrolidone
    "VA": SEQUENTIAL_COLORS[7],  # vinyl acetate
    "BA": SEQUENTIAL_COLORS[3],  # butyl acrylate
}

# Categorical colors (for heatmaps, etc.)
HEATMAP_CMAP = "Blues"
DIVERGING_CMAP = "RdYlGn"

# Neutral colors
NEUTRAL_COLORS = {
    "grid": "#e0e0e0",
    "boundary": "#666666",
    "text": "#333333",
    "background": "#ffffff",
}

# Highlight colors
HIGHLIGHT_COLORS = {
    "threshold": "#e27f07",
    "mean": "#b20404",
    "median": "#b20404",
}


# ============================================================================
# FIGURE SIZES (based on golden ratio)
# ============================================================================

# Golden ratio
golden = 1.618

# Column widths
ONE_COL_WIDTH_INCH = 3
TWO_COL_WIDTH_INCH = 7

# Heights based on golden ratio
ONE_COL_GOLDEN_RATIO_HEIGHT_INCH = ONE_COL_WIDTH_INCH / golden
TWO_COL_GOLDEN_RATIO_HEIGHT_INCH = TWO_COL_WIDTH_INCH / golden

# Figure size for the Mayo–Lewis "class curves" 1x4 panel figure
# (3 class panels + 1 explanatory panel).
# Kept slightly shorter so each panel is closer to square.
CLASS_CURVES_FIGSIZE_INCH = (TWO_COL_WIDTH_INCH * 2.1, 4.0)


# ============================================================================
# STYLE SETTINGS
# ============================================================================


def apply_plot_style(style_file="lamalab.mplstyle"):
    """
    Apply matplotlib style from lamalab.mplstyle.

    Args:
        style_file: Name of the style file (in plots_and_figures directory)
    """
    # Try to find style file
    style_paths = [
        Path(__file__).parent.parent / "plots_and_figures" / style_file,
        Path(__file__).parent / style_file,
        Path(style_file),
    ]

    for path in style_paths:
        if path.exists():
            plt.style.use(str(path))
            return

    # Fallback: apply basic settings
    print(f"Warning: Style file '{style_file}' not found, using default settings")
    apply_default_style()


def apply_default_style():
    """Apply default matplotlib style settings."""
    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.color": NEUTRAL_COLORS["grid"],
            "axes.edgecolor": NEUTRAL_COLORS["text"],
            "axes.linewidth": 1.0,
            "figure.facecolor": NEUTRAL_COLORS["background"],
            "axes.facecolor": NEUTRAL_COLORS["background"],
        }
    )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def get_class_color(class_id):
    """Get color for a specific class."""
    return CLASS_COLORS.get(class_id, SEQUENTIAL_COLORS[class_id % len(SEQUENTIAL_COLORS)])


def get_class_colors(class_ids):
    """Get list of colors for multiple classes."""
    return [get_class_color(cid) for cid in class_ids]


def get_comparison_colors():
    """Get colors for comparison plots (original vs filtered)."""
    return COMPARISON_COLORS["original"], COMPARISON_COLORS["filtered"]


def get_monomer_color(monomer_key: str) -> str:
    """
    Get color for a monomer key used in plots.

    Args:
        monomer_key: e.g. 'AN', 'VP'
    """
    key = monomer_key.strip().upper()
    if key in MONOMER_COLORS:
        return MONOMER_COLORS[key]
    return SEQUENTIAL_COLORS[hash(key) % len(SEQUENTIAL_COLORS)]


def setup_plot_style():
    """
    Setup complete plot style for analysis.
    Call this once at the beginning of your script.
    """
    apply_plot_style()

    # Set color cycle for sequential plots
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=SEQUENTIAL_COLORS)


# ============================================================================
# PLOT-SPECIFIC CONFIGURATIONS
# ============================================================================

CONFUSION_MATRIX_CONFIG = {
    "cmap": HEATMAP_CMAP,
    "values_format": "d",
    "colorbar": True,
}

CONFIDENCE_PLOT_CONFIG = {
    "bins": 30,
    "alpha": 0.6,
    "edgecolor": "black",
    "linewidth": 0.5,
}

FEATURE_IMPORTANCE_CONFIG = {
    "color": "#661124",  # Dark red
    "top_n": 20,
}

CALIBRATION_CONFIG = {
    "marker": "o",
    "linewidth": 2,
    "markersize": 6,
    "n_bins": 7,
    "strategy": "quantile",  # 'quantile': same number of samples per bin; 'uniform': equal width
}

ERROR_ANALYSIS_CONFIG = {
    "bins": 20,
    "alpha": 0.6,
    "edgecolor": "black",
}


# ============================================================================
# CLASS LABELS
# ============================================================================

CLASS_LABELS = {
    0: "Class 0:\nAlternating",
    1: "Class 1:\nRandom",
    2: "Class 2:\nGradient",
}

CLASS_LABELS_SHORT = {
    0: "Alternating",
    1: "Random",
    2: "Gradient",
}

CLASS_LABELS_LONG = {
    0: "Class 0: Alternating",
    1: "Class 1: Random",
    2: "Class 2: Gradient",
}


def get_class_label(class_id, style="default"):
    """
    Get label for a class.

    Args:
        class_id: Class ID (0, 1, or 2)
        style: 'default', 'short', or 'long'
    """
    if style == "short":
        return CLASS_LABELS_SHORT.get(class_id, f"Class {class_id}")
    elif style == "long":
        return CLASS_LABELS_LONG.get(class_id, f"Class {class_id}")
    else:
        return CLASS_LABELS.get(class_id, f"Class {class_id}")
