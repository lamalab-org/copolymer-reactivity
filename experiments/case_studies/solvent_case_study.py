import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def confidence_from_distance_to_one(
    values,
    mode: str = "log10",
    eps: float = 1e-12,
    d0: float = 0.5,
    clip: tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """
    Convert a distance-to-1 measure into a confidence score in [0, 1].

    Parameters
    ----------
    values
        Positive numeric values (e.g., r1*r2) for each solvent.
    mode
        "log10" (recommended): distance = |log10(value)| (symmetric for 0.1 and 10)
        "linear": distance = |value - 1|
    eps
        Small constant to avoid log(0).
    d0
        Scale for the saturating mapping: conf = d / (d + d0).
        Smaller d0 -> confidence rises faster with distance.
    clip
        Output range clamp.

    Returns
    -------
    np.ndarray
        Confidence in [0, 1], same length as values.
    """
    v = np.asarray(values, dtype=float)
    v_safe = np.maximum(v, eps)

    if mode == "log10":
        d = np.abs(np.log10(v_safe))
    elif mode == "linear":
        d = np.abs(v - 1.0)
    else:
        raise ValueError("mode must be 'log10' or 'linear'")

    conf = d / (d + d0)
    return np.clip(conf, clip[0], clip[1])


def map_confidence_to_alpha(
    conf,
    alpha_min: float,
    alpha_max: float,
    gamma: float = 0.35,
) -> np.ndarray:
    """
    Map confidence in [0, 1] to alpha in [alpha_min, alpha_max] using a power-law
    transform (gamma < 1 increases visual contrast).
    """
    c = np.clip(np.asarray(conf, dtype=float), 0.0, 1.0)
    return alpha_min + (c ** gamma) * (alpha_max - alpha_min)


def plot_minsk_vs_model_heatstrip(
    solvents,
    minsk_classes,
    model_classes,
    model_confidence,
    class_colors,
    minsk_confidence=None,
    row_labels=("Minsk et al.", "Copolymer Model"),
    figsize=None,
    minsk_alpha_min: float = 0.02,
    minsk_alpha_max: float = 1.0,
    model_alpha_min: float = 0.05,
    model_alpha_max: float = 1.0,
    alpha_gamma: float = 0.35,
    show_cell_text: bool = False,
    text_color: str = "black",
    x_label: str = "Solvent",
    cell_edgecolor: str = "white",
    cell_linewidth: float = 2.0,
    legend_anchor_x: float = 1.18,
    opacity_text_y: float = 0.25,
    right_margin: float = 0.78,
    keep_square_cells: bool = True,
):
    """
    Two-row heatstrip for a single monomer-pair case study across solvents.

    Encoding
    --------
    - Color: class (discrete, via class_colors)
    - Opacity: confidence
        * Minsk et al.: derived confidence (e.g., distance-to-1), optional
        * Copolymer Model: predicted class probability
    """

    solvents = list(solvents)
    n = len(solvents)

    minsk_classes = np.asarray(minsk_classes)
    model_classes = np.asarray(model_classes)
    model_confidence = np.asarray(model_confidence, dtype=float)

    if len(minsk_classes) != n or len(model_classes) != n or len(model_confidence) != n:
        raise ValueError("Input lengths must match the number of solvents.")

    model_alphas = map_confidence_to_alpha(
        model_confidence,
        alpha_min=model_alpha_min,
        alpha_max=model_alpha_max,
        gamma=alpha_gamma,
    )

    if minsk_confidence is None:
        minsk_alphas = np.full(n, minsk_alpha_max)
    else:
        minsk_confidence = np.asarray(minsk_confidence, dtype=float)
        if len(minsk_confidence) != n:
            raise ValueError("minsk_confidence must match the number of solvents.")
        minsk_alphas = map_confidence_to_alpha(
            minsk_confidence,
            alpha_min=minsk_alpha_min,
            alpha_max=minsk_alpha_max,
            gamma=alpha_gamma,
        )

    # Bigger cells / more readable
    if figsize is None:
        figsize = (max(8, 0.6 * n), 3.6)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, n)
    ax.set_ylim(0, 2)

    rows = [
        (row_labels[0], 1, minsk_classes, minsk_alphas),
        (row_labels[1], 0, model_classes, model_alphas),
    ]

    for _, y0, classes, alphas in rows:
        for i in range(n):
            cls = int(classes[i])
            if cls not in class_colors:
                raise ValueError(f"Class {cls} missing in class_colors.")

            ax.add_patch(
                patches.Rectangle(
                    (i, y0),
                    1,
                    1,
                    facecolor=class_colors[cls],
                    edgecolor=cell_edgecolor,
                    linewidth=cell_linewidth,
                    alpha=float(alphas[i]),
                )
            )

            if show_cell_text:
                ax.text(
                    i + 0.5,
                    y0 + 0.5,
                    str(cls),
                    ha="center",
                    va="center",
                    fontsize=11,
                    color=text_color,
                )

    # Axes / ticks
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels([row_labels[1], row_labels[0]], fontsize=12)

    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_xticklabels(solvents, rotation=45, ha="right", fontsize=11)

    ax.set_xlabel(x_label, fontsize=12)

    # Clean frame
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0)

    # Class legend
    handles = [
        patches.Patch(facecolor=class_colors[c], edgecolor="none", label=f"Class {c}", alpha=1.0)
        for c in sorted(class_colors.keys())
    ]
    ax.legend(
        handles=handles,
        title="Reactivity class",
        bbox_to_anchor=(legend_anchor_x, 1.0),
        loc="upper left",
        frameon=False,
        fontsize=11,
        title_fontsize=12,
    )

    # Opacity explanation
    ax.text(
        legend_anchor_x,
        opacity_text_y,
        "Opacity encodes confidence\n"
        "Minsk et al.: distance to 1\n"
        "Copolymer Model: predicted probability",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
    )

    # Reserve space on the right for legend + text
    fig.subplots_adjust(right=right_margin)

    if keep_square_cells:
        ax.set_aspect("equal")

    return fig, ax


# ----------------------------
# Example usage (replace with your values)
# ----------------------------
solvents = [
    "Benzene",
    "o-Dichlorobenzene",
    "Benzonitrile",
    "Dioxane",
    "Bis(2-methoxyethyl) ether",
    "Ethanol",
    "2-(2-Methoxyethoxy)ethanol",
    "DMSO",
    "Methanol",
]

minsk_classes = [1, 1, 1, 1, 1, 0, 0, 0, 0]
minsk_values = [3.13, 3.13, 3.24, 1.75, 1.53, 0.43, 0.45, 0.32, 0.21]

minsk_confidence = confidence_from_distance_to_one(
    minsk_values,
    mode="log10",
    d0=0.4,
)

model_classes = [1, 1, 1, 1, 0, 0, 0, 1, 0]
model_confidence = [0.609, 0.686, 0.395, 0.430, 0.583, 0.493, 0.760, 0.367, 0.569]

CLASS_COLORS = {
    0: "#3A3B73",  # < 1
    1: "#e27f07",  # 1–25
    2: "#6a040f",  # > 25
}

fig, ax = plot_minsk_vs_model_heatstrip(
    solvents=solvents,
    minsk_classes=minsk_classes,
    model_classes=model_classes,
    model_confidence=model_confidence,
    minsk_confidence=minsk_confidence,
    class_colors=CLASS_COLORS,
    row_labels=("Minsk et al.", "Copolymer Model"),
    minsk_alpha_min=0.02,
    model_alpha_min=0.05,
    alpha_gamma=0.35,
    opacity_text_y=0.25,
)

fig.savefig("solvent_case_study.png", dpi=300, bbox_inches="tight")
fig.savefig("solvent_case_study.pdf", bbox_inches="tight")
plt.close(fig)
