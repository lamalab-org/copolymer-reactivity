"""Canonical model class labels.

Single source of truth for the index -> human-readable name mapping of the
copolymer-architecture classes. Imported by the API (`app`), the
nearest-neighbour lookup (`baseline_lookup`) and the reaction-optimization
grid (`reaction_optimization`).

Kept in its own leaf module — it imports nothing from the package — so every
consumer can import it at module level without risking a circular import.
"""

from typing import Dict

# Class index as produced by the multiclass model -> human-readable label.
# Class 1 is the catch-all "random" class — covers everything not unambiguously
# alternating or gradient (see PR #37 for the rationale behind dropping the
# earlier "(to blocky)" qualifier across the analysis pipeline; this is the
# matching rename for the API/UI-facing labels).
CLASS_LABELS: Dict[int, str] = {
    0: "alternating",
    1: "random",
    2: "gradient",
}
