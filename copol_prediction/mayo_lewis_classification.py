import numpy as np


# ============================================================
# Integration helper (NumPy 1.x / 2.x compatible)
# ============================================================
def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """
    Trapezoidal integration compatible with NumPy variants:
    - NumPy >= 2.0 may not have np.trapz
    - some older stacks may not have np.trapezoid
    """
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    if hasattr(np, "trapz"):
        return float(np.trapz(y, x))
    # Manual trapezoid rule
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    return float(np.sum((y[1:] + y[:-1]) * (x[1:] - x[:-1]) * 0.5))


# ============================================================
# Mayo–Lewis equation
# ============================================================
def mayo_lewis(f1: np.ndarray, r1: float, r2: float) -> np.ndarray:
    f2 = 1.0 - f1
    denom = r1 * f1**2 + 2.0 * f1 * f2 + r2 * f2**2
    # Avoid warnings for degenerate parameterizations; callers may filter non-finite curves.
    with np.errstate(divide="ignore", invalid="ignore"):
        return (r1 * f1**2 + f1 * f2) / denom


# ============================================================
# Find interior diagonal crossings
# ============================================================
def find_diagonal_crossings(f1: np.ndarray, F1: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    diff = F1 - f1
    crossings = []

    for i in range(len(f1) - 1):
        # exact hit
        if abs(diff[i]) < tol and 1e-6 < f1[i] < 1.0 - 1e-6:
            crossings.append(f1[i])

        # sign change
        if diff[i] * diff[i + 1] < 0:
            x0, x1 = f1[i], f1[i + 1]
            y0, y1 = diff[i], diff[i + 1]
            x_cross = x0 - y0 * (x1 - x0) / (y1 - y0)

            if 1e-6 < x_cross < 1.0 - 1e-6:
                crossings.append(x_cross)

    if len(crossings) == 0:
        return np.array([])

    return np.unique(np.round(crossings, 6))


# ============================================================
# Curve descriptors
# ============================================================
def compute_curve_descriptors(r1: float, r2: float, n_points: int = 5000) -> dict:
    """
    Compute geometric descriptors of the Mayo–Lewis curve for given (r1, r2).
    """
    f1 = np.linspace(1e-4, 1.0 - 1e-4, n_points)
    F1 = mayo_lewis(f1, r1, r2)
    D = F1 - f1

    # total deviation from random
    I_rand = _trapz(np.abs(D), f1)

    # inner diagonal crossings
    crossings = find_diagonal_crossings(f1, F1)
    has_crossing = len(crossings) > 0

    if has_crossing:
        crossing_main = crossings[np.argmin(np.abs(crossings - 0.5))]
        crossing_distance = float(abs(crossing_main - 0.5))
    else:
        crossing_main = None
        crossing_distance = None

    return {
        "f1": f1,
        "F1": F1,
        "D": D,
        "I_rand": float(I_rand),
        "crossings": crossings,
        "has_crossing": has_crossing,
        "crossing_main": crossing_main,
        "crossing_distance": crossing_distance,
    }


# ============================================================
# Classification (random / gradient / alternating)
# ============================================================
def classify_curve(
    I_rand: float,
    has_crossing: bool,
    crossing_distance: float | None,
    rand_threshold: float = 0.02,
    alternating_threshold: float = 0.14,
    alternating_crossing_window: float = 0.06,
    gradient_integral_threshold: float = 0.08,
    gradient_crossing_threshold: float = 0.3,
) -> str:
    """
    3 classes:
      - random
      - gradient
      - alternating

    Logic:
      1) random if total deviation is small
      2) alternating if deviation is very large AND there is a central crossing
      3) gradient if deviation is sufficiently large and
         - no inner crossing exists, OR
         - crossing is far from 0.5
      4) else random
    """

    # 1) nearly random / weak blocky
    if I_rand < rand_threshold:
        return "random"

    # 2) true alternating: strong deviation + central crossing
    if has_crossing and crossing_distance is not None:
        if I_rand >= alternating_threshold and crossing_distance <= alternating_crossing_window:
            return "alternating"

    # 3) gradient:
    #    - non-random with no crossing
    #    - OR non-random with strongly shifted crossing
    if I_rand >= gradient_integral_threshold:
        if not has_crossing:
            return "gradient"
        if crossing_distance is not None and crossing_distance >= gradient_crossing_threshold:
            return "gradient"

    # 4) fallback
    return "random"


def classify_reactivity_curve(
    r1: float,
    r2: float,
    n_points: int = 5000,
    rand_threshold: float = 0.02,
    alternating_threshold: float = 0.14,
    alternating_crossing_window: float = 0.06,
    gradient_integral_threshold: float = 0.08,
    gradient_crossing_threshold: float = 0.3,
) -> dict:
    """
    High-level helper: classify a given (r1, r2) pair.

    Returns a dictionary with:
      - class_id   : int in {0, 1, 2}
      - class_name : str in {"alternating", "gradient", "random"}
      - I_rand, crossings, has_crossing, crossing_main, crossing_distance, f1, F1, D

    Mapping to existing numeric classes:
      0 -> alternating
      1 -> random
      2 -> gradient
    """
    desc = compute_curve_descriptors(r1, r2, n_points=n_points)

    label = classify_curve(
        I_rand=desc["I_rand"],
        has_crossing=desc["has_crossing"],
        crossing_distance=desc["crossing_distance"],
        rand_threshold=rand_threshold,
        alternating_threshold=alternating_threshold,
        alternating_crossing_window=alternating_crossing_window,
        gradient_integral_threshold=gradient_integral_threshold,
        gradient_crossing_threshold=gradient_crossing_threshold,
    )

    label_to_id = {
        "alternating": 0,
        "random": 1,
        "gradient": 2,
    }
    class_id = label_to_id[label]

    desc["class_id"] = class_id
    desc["class_name"] = label
    return desc


# ============================================================
# Optional plotting helper for manual inspection
# ============================================================
def plot_analysis(
    r1: float,
    r2: float,
    n_points: int = 5000,
    rand_threshold: float = 0.02,
    alternating_threshold: float = 0.14,
    alternating_crossing_window: float = 0.06,
    gradient_integral_threshold: float = 0.08,
    gradient_crossing_threshold: float = 0.12,
) -> None:
    """
    Convenience function to visualize the Mayo–Lewis curve and
    print the derived descriptors and class.
    """
    import matplotlib.pyplot as plt

    result = compute_curve_descriptors(r1, r2, n_points=n_points)

    class_label = classify_curve(
        I_rand=result["I_rand"],
        has_crossing=result["has_crossing"],
        crossing_distance=result["crossing_distance"],
        rand_threshold=rand_threshold,
        alternating_threshold=alternating_threshold,
        alternating_crossing_window=alternating_crossing_window,
        gradient_integral_threshold=gradient_integral_threshold,
        gradient_crossing_threshold=gradient_crossing_threshold,
    )

    print("----------------------------------")
    print(f"r1 = {r1}, r2 = {r2}")
    print(f"I_rand              = {result['I_rand']:.6f}")
    print(f"has_crossing        = {result['has_crossing']}")
    print(f"crossings           = {result['crossings']}")
    print(f"main crossing       = {result['crossing_main']}")
    print(f"distance to 0.5     = {result['crossing_distance']}")
    print(f"class               = {class_label}")

    f1 = result["f1"]
    F1 = result["F1"]

    plt.figure(figsize=(6, 5))
    plt.plot(f1, F1, label="Mayo-Lewis curve")
    plt.plot(f1, f1, "--", label="random line")

    if result["has_crossing"]:
        plt.scatter(
            result["crossings"],
            result["crossings"],
            zorder=3,
            label="diagonal crossing(s)",
        )

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.xlabel("f1 (monomer fraction in feed)")
    plt.ylabel("F1 (fraction in polymer)")
    plt.title(f"class = {class_label}")

    plt.legend(frameon=False)
    plt.show()
