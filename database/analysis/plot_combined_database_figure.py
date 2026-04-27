#!/usr/bin/env python3
"""
Dataset analysis (multi-panel figure + basic statistics).

Layout (2x2):
  A) Monomer chemical space / co-occurrence network (single, no time split)
     - node fill color: monomer class (keeps existing coloring)
     - node outline color: Δr (computed from constant_1/constant_2)
  B) Temporal evolution: monomer category share over time (stacked area)
  C) Publication year distribution
  D) Distribution of reaction temperatures
  E) Distribution of solvent boiling points
"""

from __future__ import annotations

from pathlib import Path
from collections import defaultdict
import json
import re
import sys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


# --- Optional deps for chemically meaningful layout ---
try:
    from rdkit import Chem  # type: ignore
    from rdkit.Chem import AllChem  # type: ignore

    RDKIT_AVAILABLE = True
except Exception:
    RDKIT_AVAILABLE = False

try:
    from sklearn.decomposition import PCA  # type: ignore
    from sklearn.manifold import TSNE  # type: ignore

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "copol_prediction" / "analysis"))

from plot_config import SEQUENTIAL_COLORS, TWO_COL_WIDTH_INCH, setup_plot_style  # noqa: E402


DATA_PATH = PROJECT_ROOT / "copol_prediction" / "processed_data.csv"
SOLVENT_BOILING_POINTS_FILE = Path(__file__).parent / "solvent_boiling_points_c.json"
OUTPUT_DIR = Path(__file__).parent / "figures"

# Shared bar/hist styling for bottom panels (C+D/E)
BOTTOM_PANEL_BAR_COLOR = "#6BAED6"
BOTTOM_PANEL_BAR_ALPHA = 0.65


def _count_raw_llm_extractions() -> dict[str, int]:
    """
    Count publications and extracted reactions from raw LLM JSON outputs.

    Source:
    `data_extraction/artifacts/llm/extractions/model_output_GPT4-o/*.json`
    """
    base = PROJECT_ROOT / "data_extraction" / "artifacts" / "llm" / "extractions" / "model_output_GPT4-o"
    if not base.exists():
        return {"raw_publications": 0, "raw_reactions": 0, "raw_parse_errors": 0}

    files = list(base.glob("*.json"))
    raw_publications = len(files)
    raw_reactions = 0
    raw_parse_errors = 0

    for p in files:
        try:
            payload = json.loads(p.read_text())
        except Exception:
            raw_parse_errors += 1
            continue

        reactions = payload.get("reactions")
        if not isinstance(reactions, list):
            continue

        for r in reactions:
            if not isinstance(r, dict):
                continue
            conds = r.get("reaction_conditions")
            # Count each condition as one reaction datapoint.
            if isinstance(conds, list) and len(conds) > 0:
                raw_reactions += len(conds)
            else:
                raw_reactions += 1

    return {
        "raw_publications": raw_publications,
        "raw_reactions": raw_reactions,
        "raw_parse_errors": raw_parse_errors,
    }


def _count_artifacts_extracted_reactions_csv() -> dict[str, int]:
    """
    Count publications and reactions from the consolidated artifacts CSV.

    Source:
    `data_extraction/artifacts/datasets/extracted_reactions.csv`
    """
    p = PROJECT_ROOT / "data_extraction" / "artifacts" / "datasets" / "extracted_reactions.csv"
    if not p.exists():
        return {"csv_publications": 0, "csv_reactions": 0}

    dfa = pd.read_csv(p)
    csv_reactions = int(len(dfa))

    pub_key_col = None
    for cand in ("PDF_name", "source_filename", "original_source", "source"):
        if cand in dfa.columns:
            pub_key_col = cand
            break
    if pub_key_col is None:
        csv_publications = 0
    else:
        keys = dfa[pub_key_col].dropna().astype(str).str.strip()
        keys = keys[keys.astype(bool)]
        csv_publications = int(keys.nunique())

    return {"csv_publications": csv_publications, "csv_reactions": csv_reactions}


def load_solvent_boiling_points() -> dict[str, float | None]:
    """
    Load curated solvent boiling points (°C) from JSON.
    Keys are solvent names as they appear in the dataset (case-insensitive; stored in lowercase).
    Values are floats (°C) or null (meaning: treat as missing / exclude).
    """
    if not SOLVENT_BOILING_POINTS_FILE.exists():
        raise FileNotFoundError(f"Missing boiling point file: {SOLVENT_BOILING_POINTS_FILE}")
    payload = json.loads(SOLVENT_BOILING_POINTS_FILE.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Boiling point file must be a JSON object: {SOLVENT_BOILING_POINTS_FILE}")
    out: dict[str, float | None] = {}
    for k, v in payload.items():
        if k is None:
            continue
        key = str(k).strip().lower()
        if not key:
            continue
        if v is None:
            out[key] = None
            continue
        try:
            out[key] = float(v)
        except Exception:
            out[key] = None
    return out


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    return df


def _get_publication_year_series(df: pd.DataFrame) -> pd.Series:
    """
    Return a numeric publication year series if available in the dataset.
    (No network calls; relies on a local column such as `publication_year`.)
    """
    if "publication_year" not in df.columns:
        return pd.Series([], dtype="float64")
    return pd.to_numeric(df["publication_year"], errors="coerce")


def _norm_name(x) -> str:
    return "" if pd.isna(x) else str(x).strip().lower()


def _norm_smiles(x) -> str:
    return "" if pd.isna(x) else str(x).strip()


def _has_any(s: str, patterns) -> bool:
    return any(p in s for p in patterns)


def _has_double_bond(smi: str) -> bool:
    if not smi:
        return False
    if "C=C" in smi or "(=C" in smi or "=C(" in smi:
        return True
    return bool(re.search(r"C=.*=?.*C", smi)) or ("=" in smi)


def classify_monomer(monomer_name, monomer_smiles) -> str:
    """
    9-class monomer classification scheme (same logic as existing analysis scripts).
    """
    name = _norm_name(monomer_name)
    smi = _norm_smiles(monomer_smiles)
    if "acrylonitrile" in name or "methacrylonitrile" in name or _has_any(smi, ["C=CC#N", "C=C(C)C#N"]):
        return "(Meth)acrylonitriles"
    if any(k in name for k in ["maleic", "maleate", "fumar", "fumarate", "itaconic", "itaconate", "aconitate"]):
        return "Anhydrides/Diacids"
    if (
        ("methacry" in name)
        or ("acryl" in name and "amide" not in name and "nitrile" not in name)
        or "acrylic acid" in name
        or _has_any(
            smi,
            [
                "C=CC(=O)O",
                "C=CC(=O)OC",
                "C=CC(=O)[O-]",
                "C=C(C)C(=O)O",
                "C=C(C)C(=O)OC",
                "C=C(C)C(=O)[O-]",
            ],
        )
    ):
        return "(Meth)acrylates"
    if "acrylamide" in name or "methacrylamide" in name or "maleimide" in name or _has_any(
        smi, ["C=CC(=O)N", "C=C(C)C(=O)N"]
    ):
        return "(Meth)acrylamides/imides"
    if "styrene" in name or "methylstyrene" in name or "chlorostyrene" in name or "methoxystyrene" in name or "styrene sulfonate" in name or _has_any(
        smi, ["C=Cc1ccccc1", "C=CC1=CC=CC=C1"]
    ):
        return "Styrenics"
    if "butadiene" in name or "isoprene" in name or "chloroprene" in name or "diene" in name or _has_any(
        smi, ["C=CC=C", "C=C-C=C"]
    ):
        return "Conjugated Dienes"
    if "vinyl" in name or re.search(r"\b\d*-?vinyl", name):
        return "Vinyl Derivatives"
    if (
        name in ["ethylene", "propylene", "propene", "isobutylene"]
        or re.search(r"\b\d+-?(hexene|octene)\b", name)
        or "tetrafluoroethylene" in name
        or "chlorotrifluoroethylene" in name
        or ("ethylene" in name and any(k in name for k in ["fluoro", "chloro", "trifluoro", "tetrafluoro"]))
        or (_has_double_bond(smi) and "c1ccccc1" not in smi.lower())
    ):
        return "Olefins"
    return "Other"


PREFERRED_CLASS_ORDER = [
    "(Meth)acrylonitriles",
    "Anhydrides/Diacids",
    "(Meth)acrylates",
    "(Meth)acrylamides/imides",
    "Styrenics",
    "Conjugated Dienes",
    "Vinyl Derivatives",
    "Olefins",
    "Other",
]


def class_color_mapping(classes: set[str]) -> dict[str, str]:
    """
    Deterministic class -> color mapping consistent across panels.
    """
    class_to_color: dict[str, str] = {}
    for i, class_name in enumerate(PREFERRED_CLASS_ORDER):
        if class_name in classes:
            class_to_color[class_name] = SEQUENTIAL_COLORS[i % len(SEQUENTIAL_COLORS)]
    for class_name in sorted(classes - set(PREFERRED_CLASS_ORDER)):
        class_to_color[class_name] = SEQUENTIAL_COLORS[len(class_to_color) % len(SEQUENTIAL_COLORS)]
    return class_to_color


def calculate_morgan_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048):
    if not RDKIT_AVAILABLE:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)
    except Exception:
        return None


def compute_positions(
    nodes: list[str],
    cache_path: Path,
    method: str = "tsne",
    random_state: int = 42,
) -> dict[str, tuple[float, float]]:
    """
    Compute 2D positions for nodes.
    Uses a cache to avoid recomputing expensive t-SNE across runs.
    """
    if cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text())
            if payload.get("method") == method and payload.get("random_state") == random_state:
                pos = {k: tuple(v) for k, v in payload.get("pos", {}).items()}
                if len(pos) >= int(0.9 * len(nodes)):
                    return pos
        except Exception:
            pass

    pos_2d: dict[str, tuple[float, float]] = {}

    fingerprints = []
    valid_nodes = []
    if RDKIT_AVAILABLE:
        for smi in nodes:
            fp = calculate_morgan_fingerprint(smi)
            if fp is not None:
                fingerprints.append(fp)
                valid_nodes.append(smi)

    if fingerprints and SKLEARN_AVAILABLE and method in {"tsne", "pca"}:
        X = np.array(fingerprints)
        try:
            if method == "tsne" and len(valid_nodes) >= 3:
                tsne = TSNE(
                    n_components=2,
                    random_state=random_state,
                    perplexity=min(30, max(2, len(valid_nodes) - 1)),
                    init="random",
                    learning_rate="auto",
                )
                coords = tsne.fit_transform(X)
            else:
                pca = PCA(n_components=2, random_state=random_state)
                coords = pca.fit_transform(X)
            for i, smi in enumerate(valid_nodes):
                pos_2d[smi] = (float(coords[i, 0]), float(coords[i, 1]))
        except Exception:
            pos_2d = {}

    # Fill missing with network layouts (deterministic-ish with seed)
    missing = set(nodes) - set(pos_2d.keys())
    if missing:
        G_tmp = nx.Graph()
        G_tmp.add_nodes_from(missing)
        try:
            pos_missing = nx.kamada_kawai_layout(G_tmp, seed=random_state)
        except Exception:
            pos_missing = nx.spring_layout(G_tmp, seed=random_state)
        pos_2d.update({k: (float(v[0]), float(v[1])) for k, v in pos_missing.items()})

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"method": method, "random_state": random_state, "pos": pos_2d}))
    except Exception:
        pass

    return pos_2d


def build_monomer_network(df: pd.DataFrame, min_degree: int = 5):
    G = nx.Graph()
    monomer_classes: dict[str, str] = {}
    edge_r_product_values: defaultdict[tuple[str, str], list[float]] = defaultdict(list)

    for _, row in df.iterrows():
        m1 = row.get("monomer1_smiles")
        m2 = row.get("monomer2_smiles")
        if pd.isna(m1) or pd.isna(m2):
            continue

        m1_name = row.get("monomer1_name", m1)
        m2_name = row.get("monomer2_name", m2)

        if m1 not in monomer_classes:
            monomer_classes[m1] = classify_monomer(m1_name, m1)
        if m2 not in monomer_classes:
            monomer_classes[m2] = classify_monomer(m2_name, m2)

        if m1 not in G:
            G.add_node(m1, name=m1_name, class_name=monomer_classes[m1])
        if m2 not in G:
            G.add_node(m2, name=m2_name, class_name=monomer_classes[m2])

        if G.has_edge(m1, m2):
            G[m1][m2]["weight"] += 1
        else:
            G.add_edge(m1, m2, weight=1)

        # Collect r-product (r1*r2) for this monomer pair / edge.
        r12 = row.get("r1r2")
        if pd.isna(r12):
            try:
                r1 = float(row.get("constant_1"))
                r2 = float(row.get("constant_2"))
                r12 = r1 * r2 if (np.isfinite(r1) and np.isfinite(r2)) else np.nan
            except Exception:
                r12 = np.nan
        try:
            r12f = float(r12)
        except Exception:
            r12f = float("nan")
        if np.isfinite(r12f) and r12f > 0:
            a, b = (m1, m2) if str(m1) <= str(m2) else (m2, m1)
            edge_r_product_values[(a, b)].append(r12f)

    degrees = dict(G.degree())
    frequent_nodes = {n for n, d in degrees.items() if d >= min_degree}
    Gf = G.subgraph(frequent_nodes).copy()

    # Edge Δ(r-product): spread as max/min ratio -> log10(ratio+1)
    edge_delta = {}
    for u, v in Gf.edges():
        a, b = (u, v) if str(u) <= str(v) else (v, u)
        vals = edge_r_product_values.get((a, b), [])
        if len(vals) < 2:
            edge_delta[(u, v)] = 0.0
            continue
        mn, mx = float(np.min(vals)), float(np.max(vals))
        if mn <= 0 or not np.isfinite(mn) or not np.isfinite(mx):
            edge_delta[(u, v)] = 0.0
            continue
        ratio = mx / mn
        edge_delta[(u, v)] = float(np.log10(ratio + 1.0))

    return Gf, edge_delta


def plot_panel_a_network(ax: plt.Axes, df: pd.DataFrame) -> None:
    Gf, edge_delta = build_monomer_network(df, min_degree=5)

    ax.set_title("A  Monomer chemical space", loc="left", fontsize=14, pad=6)
    ax.axis("off")

    if Gf.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No network nodes after filtering", ha="center", va="center", transform=ax.transAxes)
        return

    all_classes = set(Gf.nodes[n].get("class_name", "Other") for n in Gf.nodes())
    class_to_color = class_color_mapping(all_classes)

    pos_cache = OUTPUT_DIR / "monomer_network_positions_tsne.json"
    pos = compute_positions(list(Gf.nodes()), cache_path=pos_cache, method="tsne")
    pos_period = {n: pos[n] for n in Gf.nodes() if n in pos}

    edges = list(Gf.edges())
    if edges:
        deltas = np.array([edge_delta.get((u, v), edge_delta.get((v, u), 0.0)) for u, v in edges], dtype=float)
        finite = np.isfinite(deltas)
        dmax = float(np.max(deltas[finite])) if finite.any() else 0.0
        if dmax <= 0:
            edge_widths = np.full(len(edges), 0.6)
        else:
            t = np.clip(deltas / dmax, 0, 1) ** 2.2
            edge_widths = 0.30 + t * 9.70
        ec = nx.draw_networkx_edges(
            Gf,
            pos_period,
            ax=ax,
            alpha=0.45,
            width=edge_widths.tolist(),
            edge_color="gray",
        )
        try:
            ec.set_rasterized(True)
        except Exception:
            pass

    deg = dict(Gf.degree())
    node_list = list(Gf.nodes())
    node_sizes = [deg[n] * 28 for n in node_list]
    node_fill = [class_to_color.get(Gf.nodes[n].get("class_name", "Other"), "#CCCCCC") for n in node_list]

    def _darken_if_too_light(hex_color: str, lum_thresh: float = 0.78, amount: float = 0.12) -> str:
        r, g, b = mcolors.to_rgb(hex_color)
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        if lum <= lum_thresh:
            return hex_color
        r2, g2, b2 = (r * (1 - amount), g * (1 - amount), b * (1 - amount))
        return mcolors.to_hex((r2, g2, b2))

    node_fill = [_darken_if_too_light(c) for c in node_fill]
    nc = nx.draw_networkx_nodes(
        Gf,
        pos_period,
        nodelist=node_list,
        node_size=node_sizes,
        node_color=node_fill,
        edgecolors="none",
        linewidths=0.0,
        alpha=0.9,
        ax=ax,
    )
    try:
        nc.set_rasterized(True)
    except Exception:
        pass


def plot_panel_b_temporal(
    ax: plt.Axes,
    df: pd.DataFrame,
    rolling_window_years: int = 5,
) -> tuple[list[str], dict[str, str]]:
    ax.set_title("B  Monomer temporal evolution", loc="left", fontsize=14, pad=6)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Share", fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(False)
    pub_year_series = _get_publication_year_series(df)
    if pub_year_series.empty:
        ax.text(0.5, 0.5, "No temporal data (missing publication_year)", ha="center", va="center", transform=ax.transAxes)
        return ([], {})

    class_counts_by_year: defaultdict[str, defaultdict[int, float]] = defaultdict(lambda: defaultdict(float))

    for i, row in df.iterrows():
        try:
            yv = pub_year_series.iloc[int(i)]
            year = int(yv) if np.isfinite(yv) else None
        except Exception:
            year = None
        if year is None:
            continue
        y = int(year)

        m1_smi = row.get("monomer1_smiles")
        m2_smi = row.get("monomer2_smiles")
        if pd.isna(m1_smi) or pd.isna(m2_smi):
            continue

        m1_name = row.get("monomer1_name", m1_smi)
        m2_name = row.get("monomer2_name", m2_smi)
        c1 = classify_monomer(m1_name, m1_smi)
        c2 = classify_monomer(m2_name, m2_smi)
        class_counts_by_year[c1][y] += 1
        class_counts_by_year[c2][y] += 1

    years = sorted({y for c in class_counts_by_year for y in class_counts_by_year[c].keys()})
    if not years:
        ax.text(0.5, 0.5, "No temporal data (missing years)", ha="center", va="center", transform=ax.transAxes)
        return ([], {})

    classes_in_data = set(class_counts_by_year.keys())
    class_sorted = [c for c in PREFERRED_CLASS_ORDER if c in classes_in_data] + sorted(classes_in_data - set(PREFERRED_CLASS_ORDER))
    class_to_color = class_color_mapping(classes_in_data)

    counts = pd.DataFrame(index=pd.Index(years, name="year"), columns=class_sorted, data=0.0)
    for cls in class_sorted:
        for y, v in class_counts_by_year.get(cls, {}).items():
            if y in counts.index:
                counts.loc[y, cls] = float(v)

    w = int(max(1, rolling_window_years))
    if w > 1:
        counts_sm = counts.rolling(window=w, min_periods=max(1, w // 2), center=True).mean()
    else:
        counts_sm = counts

    totals = counts_sm.sum(axis=1)
    proportions = []
    labels = []
    colors = []
    for i, cat in enumerate(class_sorted):
        prop = (counts_sm[cat] / totals).fillna(0.0).to_numpy()
        proportions.append(prop)
        labels.append(cat)
        colors.append(class_to_color.get(cat, SEQUENTIAL_COLORS[i % len(SEQUENTIAL_COLORS)]))

    ax.stackplot(counts_sm.index.to_numpy(), *proportions, labels=labels, colors=colors, alpha=0.85)
    ax.set_xlim(int(min(years)) - 2, int(max(years)) + 2)
    return (labels, class_to_color)


def plot_panel_publication_years(ax: plt.Axes, df: pd.DataFrame) -> None:
    ax.set_title("C  Publication year distribution", loc="left", fontsize=14, pad=6)
    ax.set_xlabel("Publication year", fontsize=12)
    ax.set_ylabel("Number of papers", fontsize=12)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    pub_year_series = _get_publication_year_series(df)
    if pub_year_series.empty:
        ax.text(0.5, 0.5, "No publication years available", ha="center", va="center", transform=ax.transAxes)
        return
    years = pub_year_series.dropna().astype(int)
    bin_size = 5
    bin_start = (years // bin_size) * bin_size
    counts = bin_start.value_counts().sort_index()
    x = counts.index.values.astype(int)
    y = counts.values

    ax.bar(
        x,
        y,
        width=5.0,
        align="edge",
        color=BOTTOM_PANEL_BAR_COLOR,
        alpha=BOTTOM_PANEL_BAR_ALPHA,
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_xlim(int(x.min()) - 2, int(x.max()) + 7)
    ax.set_xticks(np.arange(int(x.min()), int(x.max()) + 1, 20))
    ax.tick_params(labelsize=12)


def plot_panel_c_distributions(ax_temp: plt.Axes, ax_logp: plt.Axes, df: pd.DataFrame) -> None:
    ax_temp.set_title("D  Distribution of reaction temperatures", loc="left", fontsize=14, pad=6)
    ax_logp.set_title("E  Distribution of solvent boiling points", loc="left", fontsize=14, pad=6)

    t = pd.to_numeric(df.get("temperature"), errors="coerce").dropna()
    ax_temp.hist(t, bins=30, color=BOTTOM_PANEL_BAR_COLOR, alpha=BOTTOM_PANEL_BAR_ALPHA, edgecolor="black", linewidth=0.4)
    ax_temp.set_xlabel("Temperature", fontsize=12)
    ax_temp.set_ylabel("Count", fontsize=12)
    ax_temp.grid(False)
    ax_temp.spines["top"].set_visible(False)
    ax_temp.spines["right"].set_visible(False)

    solvent_name_col = None
    for cand in ("solvent_name", "solvent", "Solvent", "solvent_common_name"):
        if cand in df.columns:
            solvent_name_col = cand
            break
    if solvent_name_col is None:
        ax_logp.text(0.5, 0.5, "No solvent name column (expected solvent_name/solvent)", ha="center", va="center", transform=ax_logp.transAxes)
        ax_logp.set_axis_off()
        return

    bp_by_name = load_solvent_boiling_points()
    alias_to_canonical = {
        "1,4-dioxan": "1,4-dioxane",
        "dioxan": "1,4-dioxane",
        "p-dioxane": "1,4-dioxane",
        "dioxane": "1,4-dioxane",
        "dimethylformamide": "n,n-dimethylformamide",
        "dimethyl formamide": "n,n-dimethylformamide",
        "n,n-dimethyl formamide": "n,n-dimethylformamide",
        "dimethylformamid": "n,n-dimethylformamide",
        "methyl ethyl ketone": "2-butanone",
        "ethyl methyl ketone": "2-butanone",
        "butanone": "2-butanone",
        "isopropanol": "isopropyl alcohol",
        "tert-butanol": "tert-butyl alcohol",
        "tert butyl alcohol": "tert-butyl alcohol",
        "ethyl alcohol": "ethanol",
        "methylene chloride": "dichloromethane",
        "1,2-dichloroethane": "dichloroethane",
        "ethylene dichloride": "dichloroethane",
        "benzol": "benzene",
        "benzointrile": "benzonitrile",
        "acetronitrile": "acetonitrile",
        "n-methyl-2-pyrrolidone": "n-methylpyrrolidone",
        "iso-octane": "isooctane",
        "n-heptane": "heptane",
        "n-hexane": "hexane",
        "n-butanol": "n-butyl alcohol",
        "liquid sulfur dioxide": "sulfur dioxide",
        "dimethylsulfoxide": "dimethyl sulfoxide",
        "dry chlorobenzene": "chlorobenzene",
        "glacial acetic acid": "acetic acid",
        "deuterated dioxane": "1,4-dioxane",
    }
    name_series = df[solvent_name_col].astype(str).str.strip()
    name_lower = name_series.str.lower()
    name_canonical = name_lower.map(lambda n: alias_to_canonical.get(n, n))
    bp_series = name_canonical.map(lambda n: bp_by_name.get(n))
    bp_vals = pd.to_numeric(bp_series, errors="coerce").dropna().to_numpy(dtype=float).tolist()

    if not bp_vals:
        ax_logp.text(0.5, 0.5, f"No boiling point values found in {SOLVENT_BOILING_POINTS_FILE.name}", ha="center", va="center", transform=ax_logp.transAxes)
        ax_logp.set_axis_off()
        return

    bp = pd.Series(bp_vals, dtype="float64")
    ax_logp.hist(bp, bins=30, color=BOTTOM_PANEL_BAR_COLOR, alpha=BOTTOM_PANEL_BAR_ALPHA, edgecolor="black", linewidth=0.4)
    ax_logp.set_xlabel("Boiling point (°C)", fontsize=12)
    ax_logp.set_ylabel("Count", fontsize=12)
    ax_logp.grid(False)
    ax_logp.spines["top"].set_visible(False)
    ax_logp.spines["right"].set_visible(False)


def print_basic_dataset_stats(df: pd.DataFrame) -> None:
    print("\n=== Basic dataset statistics ===")

    raw_counts = _count_raw_llm_extractions()
    print("\n--- Raw extraction (LLM outputs) ---")
    print(f"Publications (files): {raw_counts['raw_publications']}")
    print(f"Number of reactions (expanded by reaction_conditions): {raw_counts['raw_reactions']}")
    if raw_counts["raw_parse_errors"] > 0:
        print(f"Parse errors: {raw_counts['raw_parse_errors']}")

    csv_counts = _count_artifacts_extracted_reactions_csv()
    print("\n--- Extracted dataset (artifacts CSV) ---")
    print(f"Publications (unique): {csv_counts['csv_publications']}")
    print(f"Number of reactions (rows): {csv_counts['csv_reactions']}")

    pub_key_col = None
    for cand in ("PDF_name", "source_filename", "original_source", "source"):
        if cand in df.columns:
            pub_key_col = cand
            break
    if pub_key_col is not None:
        pub_keys = df[pub_key_col].dropna().astype(str).str.strip()
        pub_keys = pub_keys[pub_keys.astype(bool)]
        processed_publications = int(pub_keys.nunique())
    else:
        processed_publications = 0

    if "reaction_id" in df.columns:
        rxn = df["reaction_id"].dropna().astype(str).str.strip()
        rxn = rxn[rxn.astype(bool)]
        processed_reactions = int(rxn.nunique())
    else:
        processed_reactions = int(len(df))

    print("\n--- Processed dataset (processed_data.csv) ---")
    if pub_key_col is not None:
        print(f"Publications (unique by `{pub_key_col}`): {processed_publications}")
    else:
        print("Publications: (no identifier column found)")
    if "reaction_id" in df.columns:
        print(f"Number of reactions (unique by `reaction_id`): {processed_reactions}")
    else:
        print(f"Number of reactions (rows): {processed_reactions}")

    m1 = df.get("monomer1_smiles")
    m2 = df.get("monomer2_smiles")
    if m1 is not None and m2 is not None:
        monomers = pd.concat([m1.dropna().astype(str), m2.dropna().astype(str)], ignore_index=True)
        monomers = monomers[monomers.astype(bool)]
        print(f"Unique monomers: {monomers.nunique()}")

        m1s = m1.dropna().astype(str)
        m2s = m2.dropna().astype(str)
        pairs = pd.DataFrame({"a": m1s, "b": m2s})
        pairs = pairs[pairs["a"].astype(bool) & pairs["b"].astype(bool)]
        pairs["p1"] = np.where(pairs["a"] <= pairs["b"], pairs["a"], pairs["b"])
        pairs["p2"] = np.where(pairs["a"] <= pairs["b"], pairs["b"], pairs["a"])
        print(f"Unique monomer pairs (unordered): {len(pairs.drop_duplicates(subset=['p1','p2']))}")

    solvent_name_col = None
    for cand in ("solvent_name", "solvent", "Solvent", "solvent_common_name"):
        if cand in df.columns:
            solvent_name_col = cand
            break
    if solvent_name_col:
        bp_by_name = load_solvent_boiling_points()
        alias_to_canonical = {
            "1,4-dioxan": "1,4-dioxane",
            "dioxan": "1,4-dioxane",
            "p-dioxane": "1,4-dioxane",
            "dioxane": "1,4-dioxane",
            "dimethylformamide": "n,n-dimethylformamide",
            "dimethyl formamide": "n,n-dimethylformamide",
            "n,n-dimethyl formamide": "n,n-dimethylformamide",
            "dimethylformamid": "n,n-dimethylformamide",
            "methyl ethyl ketone": "2-butanone",
            "ethyl methyl ketone": "2-butanone",
            "butanone": "2-butanone",
            "isopropanol": "isopropyl alcohol",
            "tert-butanol": "tert-butyl alcohol",
            "tert butyl alcohol": "tert-butyl alcohol",
            "ethyl alcohol": "ethanol",
            "methylene chloride": "dichloromethane",
            "1,2-dichloroethane": "dichloroethane",
            "ethylene dichloride": "dichloroethane",
            "benzol": "benzene",
            "benzointrile": "benzonitrile",
            "acetronitrile": "acetonitrile",
            "n-methyl-2-pyrrolidone": "n-methylpyrrolidone",
            "iso-octane": "isooctane",
            "n-heptane": "heptane",
            "n-hexane": "hexane",
            "n-butanol": "n-butyl alcohol",
            "liquid sulfur dioxide": "sulfur dioxide",
            "dimethylsulfoxide": "dimethyl sulfoxide",
            "dry chlorobenzene": "chlorobenzene",
            "glacial acetic acid": "acetic acid",
            "deuterated dioxane": "1,4-dioxane",
        }
        ignore_names = {"na", "nan", "none", "no solvent", "bulk", "monomer"}

        s = df[solvent_name_col].astype(str).str.strip()
        sl = s.str.lower()
        canon = sl.map(lambda n: alias_to_canonical.get(n, n))
        canon = canon[canon.astype(bool) & ~canon.isin(ignore_names)]

        print(f"Unique solvents (canonical, excl placeholders): {canon.nunique()}")

        bp_series = canon.map(lambda n: bp_by_name.get(n))
        bp_vals = pd.to_numeric(bp_series, errors="coerce").dropna()
        if not bp_vals.empty:
            print(f"Solvent boiling points (°C): {bp_vals.min():.2f} .. {bp_vals.max():.2f} (n={len(bp_vals)})")

    if "temperature" in df.columns:
        t = pd.to_numeric(df["temperature"], errors="coerce").dropna()
        if not t.empty:
            print(f"Temperature: {t.min():.2f} .. {t.max():.2f} (n={len(t)})")

    py = _get_publication_year_series(df).dropna()
    if not py.empty:
        py_i = py.astype(int)
        print(f"Publication years: {py_i.min()} .. {py_i.max()} (n={len(py_i)})")
    else:
        print("Publication years: (missing `publication_year` column)")


def main() -> None:
    setup_plot_style()
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = load_data()
    print_basic_dataset_stats(df)

    fig = plt.figure(figsize=(TWO_COL_WIDTH_INCH * 1.65, TWO_COL_WIDTH_INCH * 1.05), layout="constrained")
    gs = fig.add_gridspec(3, 6, height_ratios=[1.15, 0.26, 0.85])

    ax_a = fig.add_subplot(gs[0, 0:3])
    ax_b = fig.add_subplot(gs[0, 3:6])
    ax_pub = fig.add_subplot(gs[2, 0:2])
    ax_temp = fig.add_subplot(gs[2, 2:4])
    ax_logp = fig.add_subplot(gs[2, 4:6])

    plot_panel_a_network(ax_a, df)
    legend_labels, class_to_color = plot_panel_b_temporal(ax_b, df, rolling_window_years=10)
    plot_panel_publication_years(ax_pub, df)
    plot_panel_c_distributions(ax_temp, ax_logp, df)

    if legend_labels and class_to_color:
        handles = [
            plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=class_to_color[l], markersize=8, label=l)
            for l in legend_labels
            if l in class_to_color
        ]
        if handles:
            fig.legend(
                handles=handles,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.515),
                ncol=min(5, max(1, len(handles))),
                frameon=True,
                framealpha=0.9,
                edgecolor="none",
                fontsize=12,
                handletextpad=0.5,
                columnspacing=1.0,
            )

    base = OUTPUT_DIR / "dataset_analysis"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    fig.savefig(base.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"✓ Saved {base}.pdf and {base}.png")


if False and __name__ == "__main__":
    main()

"""
NOTE: The remainder of this file used to contain multiple duplicated copies of the
same script, which caused repeated prints and repeated figure generation.
It has been intentionally removed.
"""
import numpy as np
import pandas as pd


# --- Optional deps for chemically meaningful layout ---
try:
    from rdkit import Chem  # type: ignore
    from rdkit.Chem import AllChem  # type: ignore

    RDKIT_AVAILABLE = True
except Exception:
    RDKIT_AVAILABLE = False

try:
    from sklearn.decomposition import PCA  # type: ignore
    from sklearn.manifold import TSNE  # type: ignore

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "copol_prediction" / "analysis"))

from plot_config import SEQUENTIAL_COLORS, TWO_COL_WIDTH_INCH, setup_plot_style  # noqa: E402


DATA_PATH = PROJECT_ROOT / "copol_prediction" / "processed_data.csv"
SOLVENT_BOILING_POINTS_FILE = Path(__file__).parent / "solvent_boiling_points_c.json"
OUTPUT_DIR = Path(__file__).parent / "figures"

# Shared bar/hist styling for bottom panels (C+D/E)
# Light blue to match the manuscript style
BOTTOM_PANEL_BAR_COLOR = "#6BAED6"
BOTTOM_PANEL_BAR_ALPHA = 0.65


def _count_raw_llm_extractions() -> dict[str, int]:
    """
    Count publications and extracted reactions from raw LLM JSON outputs.

    Source:
    `data_extraction/artifacts/llm/extractions/model_output_GPT4-o/*.json`
    """
    base = PROJECT_ROOT / "data_extraction" / "artifacts" / "llm" / "extractions" / "model_output_GPT4-o"
    if not base.exists():
        return {"raw_publications": 0, "raw_reactions": 0, "raw_parse_errors": 0}

    files = list(base.glob("*.json"))
    raw_publications = len(files)
    raw_reactions = 0
    raw_parse_errors = 0

    for p in files:
        try:
            payload = json.loads(p.read_text())
        except Exception:
            raw_parse_errors += 1
            continue

        reactions = payload.get("reactions")
        if not isinstance(reactions, list):
            continue

        for r in reactions:
            if not isinstance(r, dict):
                continue
            conds = r.get("reaction_conditions")
            # Count each condition as one reaction datapoint (as requested).
            if isinstance(conds, list) and len(conds) > 0:
                raw_reactions += len(conds)
            else:
                raw_reactions += 1

    return {
        "raw_publications": raw_publications,
        "raw_reactions": raw_reactions,
        "raw_parse_errors": raw_parse_errors,
    }


def _count_artifacts_extracted_reactions_csv() -> dict[str, int]:
    """
    Count publications and reactions from the consolidated artifacts CSV.

    Source:
    `data_extraction/artifacts/datasets/extracted_reactions.csv`
    """
    p = PROJECT_ROOT / "data_extraction" / "artifacts" / "datasets" / "extracted_reactions.csv"
    if not p.exists():
        return {"csv_publications": 0, "csv_reactions": 0}

    dfa = pd.read_csv(p)
    csv_reactions = int(len(dfa))

    pub_key_col = None
    for cand in ("PDF_name", "source_filename", "original_source", "source"):
        if cand in dfa.columns:
            pub_key_col = cand
            break
    if pub_key_col is None:
        csv_publications = 0
    else:
        keys = dfa[pub_key_col].dropna().astype(str).str.strip()
        keys = keys[keys.astype(bool)]
        csv_publications = int(keys.nunique())

    return {"csv_publications": csv_publications, "csv_reactions": csv_reactions}


def load_solvent_boiling_points() -> dict[str, float | None]:
    """
    Load curated solvent boiling points (°C) from JSON.
    Keys are solvent names as they appear in the dataset (case-insensitive; stored in lowercase).
    Values are floats (°C) or null (meaning: treat as missing / exclude).
    """
    if not SOLVENT_BOILING_POINTS_FILE.exists():
        raise FileNotFoundError(f"Missing boiling point file: {SOLVENT_BOILING_POINTS_FILE}")
    payload = json.loads(SOLVENT_BOILING_POINTS_FILE.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Boiling point file must be a JSON object: {SOLVENT_BOILING_POINTS_FILE}")
    out: dict[str, float | None] = {}
    for k, v in payload.items():
        if k is None:
            continue
        key = str(k).strip().lower()
        if not key:
            continue
        if v is None:
            out[key] = None
            continue
        try:
            out[key] = float(v)
        except Exception:
            out[key] = None
    return out


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    return df


def _get_publication_year_series(df: pd.DataFrame) -> pd.Series:
    """
    Return a numeric publication year series if available in the dataset.
    (No network calls; relies on a local column such as `publication_year`.)
    """
    if "publication_year" not in df.columns:
        return pd.Series([], dtype="float64")
    return pd.to_numeric(df["publication_year"], errors="coerce")


def _norm_name(x) -> str:
    return "" if pd.isna(x) else str(x).strip().lower()


def _norm_smiles(x) -> str:
    return "" if pd.isna(x) else str(x).strip()


def _has_any(s: str, patterns) -> bool:
    return any(p in s for p in patterns)


def _has_double_bond(smi: str) -> bool:
    if not smi:
        return False
    if "C=C" in smi or "(=C" in smi or "=C(" in smi:
        return True
    return bool(re.search(r"C=.*=?.*C", smi)) or ("=" in smi)


def classify_monomer(monomer_name, monomer_smiles) -> str:
    """
    9-class monomer classification scheme (same logic as existing analysis scripts).
    """
    name = _norm_name(monomer_name)
    smi = _norm_smiles(monomer_smiles)
    if "acrylonitrile" in name or "methacrylonitrile" in name or _has_any(smi, ["C=CC#N", "C=C(C)C#N"]):
        return "(Meth)acrylonitriles"
    if any(k in name for k in ["maleic", "maleate", "fumar", "fumarate", "itaconic", "itaconate", "aconitate"]):
        return "Anhydrides/Diacids"
    if (
        ("methacry" in name)
        or ("acryl" in name and "amide" not in name and "nitrile" not in name)
        or "acrylic acid" in name
        or _has_any(
            smi,
            [
                "C=CC(=O)O",
                "C=CC(=O)OC",
                "C=CC(=O)[O-]",
                "C=C(C)C(=O)O",
                "C=C(C)C(=O)OC",
                "C=C(C)C(=O)[O-]",
            ],
        )
    ):
        return "(Meth)acrylates"
    if (
        "acrylamide" in name
        or "methacrylamide" in name
        or "maleimide" in name
        or _has_any(smi, ["C=CC(=O)N", "C=C(C)C(=O)N"])
    ):
        return "(Meth)acrylamides/imides"
    if (
        "styrene" in name
        or "methylstyrene" in name
        or "chlorostyrene" in name
        or "methoxystyrene" in name
        or "styrene sulfonate" in name
        or _has_any(smi, ["C=Cc1ccccc1", "C=CC1=CC=CC=C1"])
    ):
        return "Styrenics"
    if "butadiene" in name or "isoprene" in name or "chloroprene" in name or "diene" in name or _has_any(
        smi, ["C=CC=C", "C=C-C=C"]
    ):
        return "Conjugated Dienes"
    if "vinyl" in name or re.search(r"\b\d*-?vinyl", name):
        return "Vinyl Derivatives"
    if (
        name in ["ethylene", "propylene", "propene", "isobutylene"]
        or re.search(r"\b\d+-?(hexene|octene)\b", name)
        or "tetrafluoroethylene" in name
        or "chlorotrifluoroethylene" in name
        or ("ethylene" in name and any(k in name for k in ["fluoro", "chloro", "trifluoro", "tetrafluoro"]))
        or (_has_double_bond(smi) and "c1ccccc1" not in smi.lower())
    ):
        return "Olefins"
    return "Other"


PREFERRED_CLASS_ORDER = [
    "(Meth)acrylonitriles",
    "Anhydrides/Diacids",
    "(Meth)acrylates",
    "(Meth)acrylamides/imides",
    "Styrenics",
    "Conjugated Dienes",
    "Vinyl Derivatives",
    "Olefins",
    "Other",
]


def class_color_mapping(classes: set[str]) -> dict[str, str]:
    """
    Deterministic class -> color mapping consistent across panels.
    """
    class_to_color: dict[str, str] = {}
    for i, class_name in enumerate(PREFERRED_CLASS_ORDER):
        if class_name in classes:
            class_to_color[class_name] = SEQUENTIAL_COLORS[i % len(SEQUENTIAL_COLORS)]
    for class_name in sorted(classes - set(PREFERRED_CLASS_ORDER)):
        class_to_color[class_name] = SEQUENTIAL_COLORS[len(class_to_color) % len(SEQUENTIAL_COLORS)]
    return class_to_color


def calculate_morgan_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048):
    if not RDKIT_AVAILABLE:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)
    except Exception:
        return None


def compute_positions(
    nodes: list[str],
    cache_path: Path,
    method: str = "tsne",
    random_state: int = 42,
) -> dict[str, tuple[float, float]]:
    """
    Compute 2D positions for nodes.
    Uses a cache to avoid recomputing expensive t-SNE across runs.
    """
    if cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text())
            if payload.get("method") == method and payload.get("random_state") == random_state:
                pos = {k: tuple(v) for k, v in payload.get("pos", {}).items()}
                if len(pos) >= int(0.9 * len(nodes)):
                    return pos
        except Exception:
            pass

    pos_2d: dict[str, tuple[float, float]] = {}

    fingerprints = []
    valid_nodes = []
    if RDKIT_AVAILABLE:
        for smi in nodes:
            fp = calculate_morgan_fingerprint(smi)
            if fp is not None:
                fingerprints.append(fp)
                valid_nodes.append(smi)

    if fingerprints and SKLEARN_AVAILABLE and method in {"tsne", "pca"}:
        X = np.array(fingerprints)
        try:
            if method == "tsne" and len(valid_nodes) >= 3:
                tsne = TSNE(
                    n_components=2,
                    random_state=random_state,
                    perplexity=min(30, max(2, len(valid_nodes) - 1)),
                    init="random",
                    learning_rate="auto",
                )
                coords = tsne.fit_transform(X)
            else:
                pca = PCA(n_components=2, random_state=random_state)
                coords = pca.fit_transform(X)
            for i, smi in enumerate(valid_nodes):
                pos_2d[smi] = (float(coords[i, 0]), float(coords[i, 1]))
        except Exception:
            pos_2d = {}

    # Fill missing with network layouts (deterministic-ish with seed)
    missing = set(nodes) - set(pos_2d.keys())
    if missing:
        G_tmp = nx.Graph()
        G_tmp.add_nodes_from(missing)
        try:
            pos_missing = nx.kamada_kawai_layout(G_tmp, seed=random_state)
        except Exception:
            pos_missing = nx.spring_layout(G_tmp, seed=random_state)
        pos_2d.update({k: (float(v[0]), float(v[1])) for k, v in pos_missing.items()})

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"method": method, "random_state": random_state, "pos": pos_2d}))
    except Exception:
        pass

    return pos_2d


def build_monomer_network(df: pd.DataFrame, min_degree: int = 5):
    G = nx.Graph()
    monomer_classes: dict[str, str] = {}
    edge_r_product_values: defaultdict[tuple[str, str], list[float]] = defaultdict(list)

    for _, row in df.iterrows():
        m1 = row.get("monomer1_smiles")
        m2 = row.get("monomer2_smiles")
        if pd.isna(m1) or pd.isna(m2):
            continue

        m1_name = row.get("monomer1_name", m1)
        m2_name = row.get("monomer2_name", m2)

        if m1 not in monomer_classes:
            monomer_classes[m1] = classify_monomer(m1_name, m1)
        if m2 not in monomer_classes:
            monomer_classes[m2] = classify_monomer(m2_name, m2)

        if m1 not in G:
            G.add_node(m1, name=m1_name, class_name=monomer_classes[m1])
        if m2 not in G:
            G.add_node(m2, name=m2_name, class_name=monomer_classes[m2])

        if G.has_edge(m1, m2):
            G[m1][m2]["weight"] += 1
        else:
            G.add_edge(m1, m2, weight=1)

        # Collect r-product (r1*r2) for this monomer pair / edge.
        # Prefer explicit r1r2 column, otherwise compute from constant_1/constant_2.
        r12 = row.get("r1r2")
        if pd.isna(r12):
            try:
                r1 = float(row.get("constant_1"))
                r2 = float(row.get("constant_2"))
                r12 = r1 * r2 if (np.isfinite(r1) and np.isfinite(r2)) else np.nan
            except Exception:
                r12 = np.nan
        try:
            r12f = float(r12)
        except Exception:
            r12f = float("nan")
        if np.isfinite(r12f) and r12f > 0:
            a, b = (m1, m2) if str(m1) <= str(m2) else (m2, m1)
            edge_r_product_values[(a, b)].append(r12f)

    degrees = dict(G.degree())
    frequent_nodes = {n for n, d in degrees.items() if d >= min_degree}
    Gf = G.subgraph(frequent_nodes).copy()

    # Edge Δ(r-product): use spread as max/min ratio (scale-invariant),
    # transformed with log10(ratio+1) to compress extreme values.
    edge_delta = {}
    for u, v in Gf.edges():
        a, b = (u, v) if str(u) <= str(v) else (v, u)
        vals = edge_r_product_values.get((a, b), [])
        if len(vals) < 2:
            edge_delta[(u, v)] = 0.0
            continue
        mn, mx = float(np.min(vals)), float(np.max(vals))
        if mn <= 0 or not np.isfinite(mn) or not np.isfinite(mx):
            edge_delta[(u, v)] = 0.0
            continue
        ratio = mx / mn
        edge_delta[(u, v)] = float(np.log10(ratio + 1.0))

    return Gf, edge_delta


def plot_panel_a_network(ax: plt.Axes, df: pd.DataFrame) -> None:
    Gf, edge_delta = build_monomer_network(df, min_degree=5)

    ax.set_title("A  Monomer chemical space", loc="left", fontsize=14, pad=6)
    ax.axis("off")

    if Gf.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No network nodes after filtering", ha="center", va="center", transform=ax.transAxes)
        return

    # Class -> color mapping (shared across panels)
    all_classes = set(Gf.nodes[n].get("class_name", "Other") for n in Gf.nodes())
    class_to_color = class_color_mapping(all_classes)

    # Positions (cached)
    pos_cache = OUTPUT_DIR / "monomer_network_positions_tsne.json"
    pos = compute_positions(list(Gf.nodes()), cache_path=pos_cache, method="tsne")
    pos_period = {n: pos[n] for n in Gf.nodes() if n in pos}

    # Edges: light gray, thickness by Δ(r-product) for that monomer pair
    edges = list(Gf.edges())
    if edges:
        deltas = np.array([edge_delta.get((u, v), edge_delta.get((v, u), 0.0)) for u, v in edges], dtype=float)
        finite = np.isfinite(deltas)
        dmax = float(np.max(deltas[finite])) if finite.any() else 0.0
        if dmax <= 0:
            edge_widths = np.full(len(edges), 0.6)
        else:
            # Stronger scaling so differences are clearly visible
            t = np.clip(deltas / dmax, 0, 1) ** 2.2
            edge_widths = 0.30 + t * 9.70  # ~0.30 .. 10.0
        ec = nx.draw_networkx_edges(
            Gf,
            pos_period,
            ax=ax,
            alpha=0.45,
            width=edge_widths.tolist(),
            edge_color="gray",
        )
        try:
            ec.set_rasterized(True)
        except Exception:
            pass

    # Nodes: size by degree, fill by class
    deg = dict(Gf.degree())
    node_list = list(Gf.nodes())
    node_sizes = [deg[n] * 28 for n in node_list]
    node_fill = [class_to_color.get(Gf.nodes[n].get("class_name", "Other"), "#CCCCCC") for n in node_list]

    # Very light fills can read as "white/transparent" against gray edges; slightly darken locally for panel A.
    def _darken_if_too_light(hex_color: str, lum_thresh: float = 0.78, amount: float = 0.12) -> str:
        r, g, b = mcolors.to_rgb(hex_color)
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        if lum <= lum_thresh:
            return hex_color
        r2, g2, b2 = (r * (1 - amount), g * (1 - amount), b * (1 - amount))
        return mcolors.to_hex((r2, g2, b2))

    node_fill = [_darken_if_too_light(c) for c in node_fill]
    nc = nx.draw_networkx_nodes(
        Gf,
        pos_period,
        nodelist=node_list,
        node_size=node_sizes,
        node_color=node_fill,
        edgecolors="none",
        linewidths=0.0,
        alpha=0.9,
        ax=ax,
    )
    try:
        nc.set_rasterized(True)
    except Exception:
        pass


def plot_panel_b_temporal(
    ax: plt.Axes,
    df: pd.DataFrame,
    rolling_window_years: int = 5,
) -> tuple[list[str], dict[str, str]]:
    ax.set_title("B  Monomer temporal evolution", loc="left", fontsize=14, pad=6)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Share", fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(False)
    pub_year_series = _get_publication_year_series(df)
    if pub_year_series.empty:
        ax.text(0.5, 0.5, "No temporal data (missing publication_year)", ha="center", va="center", transform=ax.transAxes)
        return ([], {})

    # Count monomer class occurrences per year (then smooth with a rolling mean).
    class_counts_by_year: defaultdict[str, defaultdict[int, float]] = defaultdict(lambda: defaultdict(float))

    for i, row in df.iterrows():
        try:
            yv = pub_year_series.iloc[int(i)]
            year = int(yv) if np.isfinite(yv) else None
        except Exception:
            year = None
        if year is None:
            continue
        y = int(year)

        m1_smi = row.get("monomer1_smiles")
        m2_smi = row.get("monomer2_smiles")
        if pd.isna(m1_smi) or pd.isna(m2_smi):
            continue

        m1_name = row.get("monomer1_name", m1_smi)
        m2_name = row.get("monomer2_name", m2_smi)
        c1 = classify_monomer(m1_name, m1_smi)
        c2 = classify_monomer(m2_name, m2_smi)
        class_counts_by_year[c1][y] += 1
        class_counts_by_year[c2][y] += 1

    years = sorted({y for c in class_counts_by_year for y in class_counts_by_year[c].keys()})
    if not years:
        ax.text(0.5, 0.5, "No temporal data (missing years)", ha="center", va="center", transform=ax.transAxes)
        return ([], {})

    classes_in_data = set(class_counts_by_year.keys())
    class_sorted = [c for c in PREFERRED_CLASS_ORDER if c in classes_in_data] + sorted(classes_in_data - set(PREFERRED_CLASS_ORDER))
    class_to_color = class_color_mapping(classes_in_data)

    # Build counts table (rows = years, cols = classes)
    counts = pd.DataFrame(index=pd.Index(years, name="year"), columns=class_sorted, data=0.0)
    for cls in class_sorted:
        for y, v in class_counts_by_year.get(cls, {}).items():
            if y in counts.index:
                counts.loc[y, cls] = float(v)

    # Smooth with rolling mean over years.
    w = int(max(1, rolling_window_years))
    if w > 1:
        counts_sm = counts.rolling(window=w, min_periods=max(1, w // 2), center=True).mean()
    else:
        counts_sm = counts

    totals = counts_sm.sum(axis=1)
    proportions = []
    labels = []
    colors = []
    for i, cat in enumerate(class_sorted):
        prop = (counts_sm[cat] / totals).fillna(0.0).to_numpy()
        proportions.append(prop)
        labels.append(cat)
        colors.append(class_to_color.get(cat, SEQUENTIAL_COLORS[i % len(SEQUENTIAL_COLORS)]))

    ax.stackplot(counts_sm.index.to_numpy(), *proportions, labels=labels, colors=colors, alpha=0.85)
    ax.set_xlim(int(min(years)) - 2, int(max(years)) + 2)
    return (labels, class_to_color)


def plot_panel_publication_years(ax: plt.Axes, df: pd.DataFrame) -> None:
    ax.set_title("C  Publication year distribution", loc="left", fontsize=14, pad=6)
    ax.set_xlabel("Publication year", fontsize=12)
    ax.set_ylabel("Number of papers", fontsize=12)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    pub_year_series = _get_publication_year_series(df)
    if pub_year_series.empty:
        ax.text(0.5, 0.5, "No publication years available", ha="center", va="center", transform=ax.transAxes)
        return
    years = pub_year_series.dropna().astype(int)
    # Bin to 5-year periods so the plot isn't one bar per year
    bin_size = 5
    bin_start = (years // bin_size) * bin_size
    counts = bin_start.value_counts().sort_index()
    x = counts.index.values.astype(int)
    y = counts.values

    # Use full bin width so bars touch (no visual gaps)
    ax.bar(
        x,
        y,
        width=5.0,
        align="edge",
        color=BOTTOM_PANEL_BAR_COLOR,
        alpha=BOTTOM_PANEL_BAR_ALPHA,
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_xlim(int(x.min()) - 2, int(x.max()) + 7)
    ax.set_xticks(np.arange(int(x.min()), int(x.max()) + 1, 20))
    ax.tick_params(labelsize=12)


def plot_panel_c_distributions(ax_temp: plt.Axes, ax_logp: plt.Axes, df: pd.DataFrame) -> None:
    ax_temp.set_title("D  Distribution of reaction temperatures", loc="left", fontsize=14, pad=6)
    ax_logp.set_title("E  Distribution of solvent boiling points", loc="left", fontsize=14, pad=6)

    # Temperature histogram
    t = pd.to_numeric(df.get("temperature"), errors="coerce").dropna()
    ax_temp.hist(t, bins=30, color=BOTTOM_PANEL_BAR_COLOR, alpha=BOTTOM_PANEL_BAR_ALPHA, edgecolor="black", linewidth=0.4)
    ax_temp.set_xlabel("Temperature", fontsize=12)
    ax_temp.set_ylabel("Count", fontsize=12)
    ax_temp.grid(False)
    ax_temp.spines["top"].set_visible(False)
    ax_temp.spines["right"].set_visible(False)

    # Solvent boiling point histogram (from curated JSON)
    solvent_name_col = None
    for cand in ("solvent_name", "solvent", "Solvent", "solvent_common_name"):
        if cand in df.columns:
            solvent_name_col = cand
            break
    if solvent_name_col is None:
        ax_logp.text(0.5, 0.5, "No solvent name column (expected solvent_name/solvent)", ha="center", va="center", transform=ax_logp.transAxes)
        ax_logp.set_axis_off()
        return

    bp_by_name = load_solvent_boiling_points()
    # If the dataset contains synonyms/typos, normalize them here (map -> canonical name key).
    alias_to_canonical = {
        "1,4-dioxan": "1,4-dioxane",
        "dioxan": "1,4-dioxane",
        "p-dioxane": "1,4-dioxane",
        "dioxane": "1,4-dioxane",
        "dimethylformamide": "n,n-dimethylformamide",
        "dimethyl formamide": "n,n-dimethylformamide",
        "n,n-dimethyl formamide": "n,n-dimethylformamide",
        "dimethylformamid": "n,n-dimethylformamide",
        "methyl ethyl ketone": "2-butanone",
        "ethyl methyl ketone": "2-butanone",
        "butanone": "2-butanone",
        "isopropanol": "isopropyl alcohol",
        "tert-butanol": "tert-butyl alcohol",
        "tert butyl alcohol": "tert-butyl alcohol",
        "ethyl alcohol": "ethanol",
        "methylene chloride": "dichloromethane",
        "1,2-dichloroethane": "dichloroethane",
        "ethylene dichloride": "dichloroethane",
        "benzol": "benzene",
        "benzointrile": "benzonitrile",
        "acetronitrile": "acetonitrile",
        "n-methyl-2-pyrrolidone": "n-methylpyrrolidone",
        "iso-octane": "isooctane",
        "n-heptane": "heptane",
        "n-hexane": "hexane",
        "n-butanol": "n-butyl alcohol",
        "liquid sulfur dioxide": "sulfur dioxide",
        "dimethylsulfoxide": "dimethyl sulfoxide",
        "dry chlorobenzene": "chlorobenzene",
        "glacial acetic acid": "acetic acid",
        "deuterated dioxane": "1,4-dioxane",
    }
    name_series = df[solvent_name_col].astype(str).str.strip()
    name_lower = name_series.str.lower()
    name_canonical = name_lower.map(lambda n: alias_to_canonical.get(n, n))
    bp_series = name_canonical.map(lambda n: bp_by_name.get(n))
    bp_vals = pd.to_numeric(bp_series, errors="coerce").dropna().to_numpy(dtype=float).tolist()

    if not bp_vals:
        ax_logp.text(0.5, 0.5, f"No boiling point values found in {SOLVENT_BOILING_POINTS_FILE.name}", ha="center", va="center", transform=ax_logp.transAxes)
        ax_logp.set_axis_off()
        return

    bp = pd.Series(bp_vals, dtype="float64")
    ax_logp.hist(bp, bins=30, color=BOTTOM_PANEL_BAR_COLOR, alpha=BOTTOM_PANEL_BAR_ALPHA, edgecolor="black", linewidth=0.4)
    ax_logp.set_xlabel("Boiling point (°C)", fontsize=12)
    ax_logp.set_ylabel("Count", fontsize=12)
    ax_logp.grid(False)
    ax_logp.spines["top"].set_visible(False)
    ax_logp.spines["right"].set_visible(False)


def print_basic_dataset_stats(df: pd.DataFrame) -> None:
    """
    Print basic dataset statistics for the curated (non-network) analysis.
    """
    print("\n=== Basic dataset statistics ===")

    raw_counts = _count_raw_llm_extractions()
    print("\n--- Raw extraction (LLM outputs) ---")
    print(f"Publications (files): {raw_counts['raw_publications']}")
    print(f"Number of reactions (expanded by reaction_conditions): {raw_counts['raw_reactions']}")
    if raw_counts["raw_parse_errors"] > 0:
        print(f"Parse errors: {raw_counts['raw_parse_errors']}")

    csv_counts = _count_artifacts_extracted_reactions_csv()
    print("\n--- Extracted dataset (artifacts CSV) ---")
    print(f"Publications (unique): {csv_counts['csv_publications']}")
    print(f"Number of reactions (rows): {csv_counts['csv_reactions']}")

    pub_key_col = None
    for cand in ("PDF_name", "source_filename", "original_source", "source"):
        if cand in df.columns:
            pub_key_col = cand
            break
    if pub_key_col is not None:
        pub_keys = df[pub_key_col].dropna().astype(str).str.strip()
        pub_keys = pub_keys[pub_keys.astype(bool)]
        processed_publications = int(pub_keys.nunique())
    else:
        processed_publications = 0

    if "reaction_id" in df.columns:
        rxn = df["reaction_id"].dropna().astype(str).str.strip()
        rxn = rxn[rxn.astype(bool)]
        processed_reactions = int(rxn.nunique())
    else:
        processed_reactions = int(len(df))

    print("\n--- Processed dataset (processed_data.csv) ---")
    if pub_key_col is not None:
        print(f"Publications (unique by `{pub_key_col}`): {processed_publications}")
    else:
        print("Publications: (no identifier column found)")
    if "reaction_id" in df.columns:
        print(f"Number of reactions (unique by `reaction_id`): {processed_reactions}")
    else:
        print(f"Number of reactions (rows): {processed_reactions}")

    # Monomers + pairs
    m1 = df.get("monomer1_smiles")
    m2 = df.get("monomer2_smiles")
    if m1 is not None and m2 is not None:
        monomers = pd.concat([m1.dropna().astype(str), m2.dropna().astype(str)], ignore_index=True)
        monomers = monomers[monomers.astype(bool)]
        print(f"Unique monomers: {monomers.nunique()}")

        m1s = m1.dropna().astype(str)
        m2s = m2.dropna().astype(str)
        pairs = pd.DataFrame({"a": m1s, "b": m2s})
        pairs = pairs[pairs["a"].astype(bool) & pairs["b"].astype(bool)]
        pairs["p1"] = np.where(pairs["a"] <= pairs["b"], pairs["a"], pairs["b"])
        pairs["p2"] = np.where(pairs["a"] <= pairs["b"], pairs["b"], pairs["a"])
        print(f"Unique monomer pairs (unordered): {len(pairs.drop_duplicates(subset=['p1','p2']))}")

    # Solvents + boiling point coverage
    solvent_name_col = None
    for cand in ("solvent_name", "solvent", "Solvent", "solvent_common_name"):
        if cand in df.columns:
            solvent_name_col = cand
            break
    if solvent_name_col:
        bp_by_name = load_solvent_boiling_points()
        alias_to_canonical = {
            "1,4-dioxan": "1,4-dioxane",
            "dioxan": "1,4-dioxane",
            "p-dioxane": "1,4-dioxane",
            "dioxane": "1,4-dioxane",
            "dimethylformamide": "n,n-dimethylformamide",
            "dimethyl formamide": "n,n-dimethylformamide",
            "n,n-dimethyl formamide": "n,n-dimethylformamide",
            "dimethylformamid": "n,n-dimethylformamide",
            "methyl ethyl ketone": "2-butanone",
            "ethyl methyl ketone": "2-butanone",
            "butanone": "2-butanone",
            "isopropanol": "isopropyl alcohol",
            "tert-butanol": "tert-butyl alcohol",
            "tert butyl alcohol": "tert-butyl alcohol",
            "ethyl alcohol": "ethanol",
            "methylene chloride": "dichloromethane",
            "1,2-dichloroethane": "dichloroethane",
            "ethylene dichloride": "dichloroethane",
            "benzol": "benzene",
            "benzointrile": "benzonitrile",
            "acetronitrile": "acetonitrile",
            "n-methyl-2-pyrrolidone": "n-methylpyrrolidone",
            "iso-octane": "isooctane",
            "n-heptane": "heptane",
            "n-hexane": "hexane",
            "n-butanol": "n-butyl alcohol",
            "liquid sulfur dioxide": "sulfur dioxide",
            "dimethylsulfoxide": "dimethyl sulfoxide",
            "dry chlorobenzene": "chlorobenzene",
            "glacial acetic acid": "acetic acid",
            "deuterated dioxane": "1,4-dioxane",
        }
        ignore_names = {"na", "nan", "none", "no solvent", "bulk", "monomer"}

        s = df[solvent_name_col].astype(str).str.strip()
        sl = s.str.lower()
        canon = sl.map(lambda n: alias_to_canonical.get(n, n))
        canon = canon[canon.astype(bool) & ~canon.isin(ignore_names)]

        print(f"Unique solvents (canonical, excl placeholders): {canon.nunique()}")

        bp_series = canon.map(lambda n: bp_by_name.get(n))
        bp_vals = pd.to_numeric(bp_series, errors="coerce").dropna()
        if not bp_vals.empty:
            print(f"Solvent boiling points (°C): {bp_vals.min():.2f} .. {bp_vals.max():.2f} (n={len(bp_vals)})")

    # Temperature
    if "temperature" in df.columns:
        t = pd.to_numeric(df["temperature"], errors="coerce").dropna()
        if not t.empty:
            print(f"Temperature: {t.min():.2f} .. {t.max():.2f} (n={len(t)})")

    # Publication years
    py = _get_publication_year_series(df).dropna()
    if not py.empty:
        py_i = py.astype(int)
        print(f"Publication years: {py_i.min()} .. {py_i.max()} (n={len(py_i)})")
    else:
        print("Publication years: (missing `publication_year` column)")


def main() -> None:
    setup_plot_style()
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = load_data()
    print_basic_dataset_stats(df)

    fig = plt.figure(figsize=(TWO_COL_WIDTH_INCH * 1.65, TWO_COL_WIDTH_INCH * 1.05), layout="constrained")
    # Add a thin spacer row between top and bottom panels for the shared legend
    gs = fig.add_gridspec(3, 6, height_ratios=[1.15, 0.26, 0.85])

    ax_a = fig.add_subplot(gs[0, 0:3])
    ax_b = fig.add_subplot(gs[0, 3:6])
    ax_pub = fig.add_subplot(gs[2, 0:2])
    ax_temp = fig.add_subplot(gs[2, 2:4])
    ax_logp = fig.add_subplot(gs[2, 4:6])

    plot_panel_a_network(ax_a, df)
    legend_labels, class_to_color = plot_panel_b_temporal(ax_b, df, rolling_window_years=10)
    plot_panel_publication_years(ax_pub, df)
    plot_panel_c_distributions(ax_temp, ax_logp, df)

    # Shared legend for monomer classes: centered above panels A+B
    if legend_labels and class_to_color:
        handles = [
            plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=class_to_color[l], markersize=8, label=l)
            for l in legend_labels
            if l in class_to_color
        ]
        if handles:
            fig.legend(
                handles=handles,
                loc="upper center",
                # Place between the two plot rows
                bbox_to_anchor=(0.5, 0.515),
                ncol=min(5, max(1, len(handles))),
                frameon=True,
                framealpha=0.9,
                edgecolor="none",
                fontsize=12,
                handletextpad=0.5,
                columnspacing=1.0,
            )

    base = OUTPUT_DIR / "dataset_analysis"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    fig.savefig(base.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"✓ Saved {base}.pdf and {base}.png")


if False and __name__ == "__main__":
    main()


def _count_artifacts_extracted_reactions_csv() -> dict[str, int]:
    """
    Count publications and reactions from the consolidated artifacts CSV.

    Source:
    `data_extraction/artifacts/datasets/extracted_reactions.csv`
    """
    p = PROJECT_ROOT / "data_extraction" / "artifacts" / "datasets" / "extracted_reactions.csv"
    if not p.exists():
        return {"csv_publications": 0, "csv_reactions": 0}

    dfa = pd.read_csv(p)
    csv_reactions = int(len(dfa))

    pub_key_col = None
    for cand in ("PDF_name", "source_filename", "original_source", "source"):
        if cand in dfa.columns:
            pub_key_col = cand
            break
    if pub_key_col is None:
        csv_publications = 0
    else:
        keys = dfa[pub_key_col].dropna().astype(str).str.strip()
        keys = keys[keys.astype(bool)]
        csv_publications = int(keys.nunique())

    return {"csv_publications": csv_publications, "csv_reactions": csv_reactions}


def load_solvent_boiling_points() -> dict[str, float | None]:
    """
    Load curated solvent boiling points (°C) from JSON.
    Keys are solvent names as they appear in the dataset (case-insensitive; stored in lowercase).
    Values are floats (°C) or null (meaning: treat as missing / exclude).
    """
    if not SOLVENT_BOILING_POINTS_FILE.exists():
        raise FileNotFoundError(f"Missing boiling point file: {SOLVENT_BOILING_POINTS_FILE}")
    payload = json.loads(SOLVENT_BOILING_POINTS_FILE.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Boiling point file must be a JSON object: {SOLVENT_BOILING_POINTS_FILE}")
    out: dict[str, float | None] = {}
    for k, v in payload.items():
        if k is None:
            continue
        key = str(k).strip().lower()
        if not key:
            continue
        if v is None:
            out[key] = None
            continue
        try:
            out[key] = float(v)
        except Exception:
            out[key] = None
    return out


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    return df


def _get_publication_year_series(df: pd.DataFrame) -> pd.Series:
    """
    Return a numeric publication year series if available in the dataset.
    (No network calls; relies on a local column such as `publication_year`.)
    """
    if "publication_year" not in df.columns:
        return pd.Series([], dtype="float64")
    return pd.to_numeric(df["publication_year"], errors="coerce")


def _norm_name(x) -> str:
    return "" if pd.isna(x) else str(x).strip().lower()


def _norm_smiles(x) -> str:
    return "" if pd.isna(x) else str(x).strip()


def _has_any(s: str, patterns) -> bool:
    return any(p in s for p in patterns)


def _has_double_bond(smi: str) -> bool:
    if not smi:
        return False
    if "C=C" in smi or "(=C" in smi or "=C(" in smi:
        return True
    return bool(re.search(r"C=.*=?.*C", smi)) or ("=" in smi)


def classify_monomer(monomer_name, monomer_smiles) -> str:
    """
    9-class monomer classification scheme (same logic as existing analysis scripts).
    """
    name = _norm_name(monomer_name)
    smi = _norm_smiles(monomer_smiles)
    if "acrylonitrile" in name or "methacrylonitrile" in name or _has_any(smi, ["C=CC#N", "C=C(C)C#N"]):
        return "(Meth)acrylonitriles"
    if any(k in name for k in ["maleic", "maleate", "fumar", "fumarate", "itaconic", "itaconate", "aconitate"]):
        return "Anhydrides/Diacids"
    if (
        ("methacry" in name)
        or ("acryl" in name and "amide" not in name and "nitrile" not in name)
        or "acrylic acid" in name
        or _has_any(
            smi,
            [
                "C=CC(=O)O",
                "C=CC(=O)OC",
                "C=CC(=O)[O-]",
                "C=C(C)C(=O)O",
                "C=C(C)C(=O)OC",
                "C=C(C)C(=O)[O-]",
            ],
        )
    ):
        return "(Meth)acrylates"
    if "acrylamide" in name or "methacrylamide" in name or "maleimide" in name or _has_any(
        smi, ["C=CC(=O)N", "C=C(C)C(=O)N"]
    ):
        return "(Meth)acrylamides/imides"
    if (
        "styrene" in name
        or "methylstyrene" in name
        or "chlorostyrene" in name
        or "methoxystyrene" in name
        or "styrene sulfonate" in name
        or _has_any(smi, ["C=Cc1ccccc1", "C=CC1=CC=CC=C1"])
    ):
        return "Styrenics"
    if "butadiene" in name or "isoprene" in name or "chloroprene" in name or "diene" in name or _has_any(
        smi, ["C=CC=C", "C=C-C=C"]
    ):
        return "Conjugated Dienes"
    if "vinyl" in name or re.search(r"\b\d*-?vinyl", name):
        return "Vinyl Derivatives"
    if (
        name in ["ethylene", "propylene", "propene", "isobutylene"]
        or re.search(r"\b\d+-?(hexene|octene)\b", name)
        or "tetrafluoroethylene" in name
        or "chlorotrifluoroethylene" in name
        or ("ethylene" in name and any(k in name for k in ["fluoro", "chloro", "trifluoro", "tetrafluoro"]))
        or (_has_double_bond(smi) and "c1ccccc1" not in smi.lower())
    ):
        return "Olefins"
    return "Other"


PREFERRED_CLASS_ORDER = [
    "(Meth)acrylonitriles",
    "Anhydrides/Diacids",
    "(Meth)acrylates",
    "(Meth)acrylamides/imides",
    "Styrenics",
    "Conjugated Dienes",
    "Vinyl Derivatives",
    "Olefins",
    "Other",
]


def class_color_mapping(classes: set[str]) -> dict[str, str]:
    """
    Deterministic class -> color mapping consistent across panels.
    """
    class_to_color: dict[str, str] = {}
    for i, class_name in enumerate(PREFERRED_CLASS_ORDER):
        if class_name in classes:
            class_to_color[class_name] = SEQUENTIAL_COLORS[i % len(SEQUENTIAL_COLORS)]
    for class_name in sorted(classes - set(PREFERRED_CLASS_ORDER)):
        class_to_color[class_name] = SEQUENTIAL_COLORS[len(class_to_color) % len(SEQUENTIAL_COLORS)]
    return class_to_color


def calculate_morgan_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048):
    if not RDKIT_AVAILABLE:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)
    except Exception:
        return None


def compute_positions(
    nodes: list[str],
    cache_path: Path,
    method: str = "tsne",
    random_state: int = 42,
) -> dict[str, tuple[float, float]]:
    """
    Compute 2D positions for nodes.
    Uses a cache to avoid recomputing expensive t-SNE across runs.
    """
    if cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text())
            if payload.get("method") == method and payload.get("random_state") == random_state:
                pos = {k: tuple(v) for k, v in payload.get("pos", {}).items()}
                if len(pos) >= int(0.9 * len(nodes)):
                    return pos
        except Exception:
            pass

    pos_2d: dict[str, tuple[float, float]] = {}

    fingerprints = []
    valid_nodes = []
    if RDKIT_AVAILABLE:
        for smi in nodes:
            fp = calculate_morgan_fingerprint(smi)
            if fp is not None:
                fingerprints.append(fp)
                valid_nodes.append(smi)

    if fingerprints and SKLEARN_AVAILABLE and method in {"tsne", "pca"}:
        X = np.array(fingerprints)
        try:
            if method == "tsne" and len(valid_nodes) >= 3:
                tsne = TSNE(
                    n_components=2,
                    random_state=random_state,
                    perplexity=min(30, max(2, len(valid_nodes) - 1)),
                    init="random",
                    learning_rate="auto",
                )
                coords = tsne.fit_transform(X)
            else:
                pca = PCA(n_components=2, random_state=random_state)
                coords = pca.fit_transform(X)
            for i, smi in enumerate(valid_nodes):
                pos_2d[smi] = (float(coords[i, 0]), float(coords[i, 1]))
        except Exception:
            pos_2d = {}

    # Fill missing with network layouts (deterministic-ish with seed)
    missing = set(nodes) - set(pos_2d.keys())
    if missing:
        G_tmp = nx.Graph()
        G_tmp.add_nodes_from(missing)
        try:
            pos_missing = nx.kamada_kawai_layout(G_tmp, seed=random_state)
        except Exception:
            pos_missing = nx.spring_layout(G_tmp, seed=random_state)
        pos_2d.update({k: (float(v[0]), float(v[1])) for k, v in pos_missing.items()})

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"method": method, "random_state": random_state, "pos": pos_2d}))
    except Exception:
        pass

    return pos_2d


def compute_delta_r(row: pd.Series) -> float | None:
    """
    Δr for a reaction entry.
    Uses |log10(r1) - log10(r2)|, which is scale-invariant and robust across magnitudes.
    """
    r1 = row.get("constant_1")
    r2 = row.get("constant_2")
    try:
        r1 = float(r1)
        r2 = float(r2)
    except Exception:
        return None
    if not (np.isfinite(r1) and np.isfinite(r2)) or r1 <= 0 or r2 <= 0:
        return None
    return float(abs(np.log10(r1) - np.log10(r2)))


def build_monomer_network(df: pd.DataFrame, min_degree: int = 5):
    G = nx.Graph()
    monomer_classes: dict[str, str] = {}
    edge_r_product_values: defaultdict[tuple[str, str], list[float]] = defaultdict(list)

    for _, row in df.iterrows():
        m1 = row.get("monomer1_smiles")
        m2 = row.get("monomer2_smiles")
        if pd.isna(m1) or pd.isna(m2):
            continue

        m1_name = row.get("monomer1_name", m1)
        m2_name = row.get("monomer2_name", m2)

        if m1 not in monomer_classes:
            monomer_classes[m1] = classify_monomer(m1_name, m1)
        if m2 not in monomer_classes:
            monomer_classes[m2] = classify_monomer(m2_name, m2)

        if m1 not in G:
            G.add_node(m1, name=m1_name, class_name=monomer_classes[m1])
        if m2 not in G:
            G.add_node(m2, name=m2_name, class_name=monomer_classes[m2])

        if G.has_edge(m1, m2):
            G[m1][m2]["weight"] += 1
        else:
            G.add_edge(m1, m2, weight=1)

        # Collect r-product (r1*r2) for this monomer pair / edge.
        # Prefer explicit r1r2 column, otherwise compute from constant_1/constant_2.
        r12 = row.get("r1r2")
        if pd.isna(r12):
            try:
                r1 = float(row.get("constant_1"))
                r2 = float(row.get("constant_2"))
                r12 = r1 * r2 if (np.isfinite(r1) and np.isfinite(r2)) else np.nan
            except Exception:
                r12 = np.nan
        try:
            r12f = float(r12)
        except Exception:
            r12f = float("nan")
        if np.isfinite(r12f) and r12f > 0:
            a, b = (m1, m2) if str(m1) <= str(m2) else (m2, m1)
            edge_r_product_values[(a, b)].append(r12f)

    degrees = dict(G.degree())
    frequent_nodes = {n for n, d in degrees.items() if d >= min_degree}
    Gf = G.subgraph(frequent_nodes).copy()

    # Edge Δ(r-product): use spread as max/min ratio (scale-invariant),
    # transformed with log10(ratio+1) to compress extreme values.
    edge_delta = {}
    for u, v in Gf.edges():
        a, b = (u, v) if str(u) <= str(v) else (v, u)
        vals = edge_r_product_values.get((a, b), [])
        if len(vals) < 2:
            edge_delta[(u, v)] = 0.0
            continue
        mn, mx = float(np.min(vals)), float(np.max(vals))
        if mn <= 0 or not np.isfinite(mn) or not np.isfinite(mx):
            edge_delta[(u, v)] = 0.0
            continue
        ratio = mx / mn
        edge_delta[(u, v)] = float(np.log10(ratio + 1.0))

    return Gf, edge_delta


def plot_panel_a_network(ax: plt.Axes, df: pd.DataFrame) -> None:
    Gf, edge_delta = build_monomer_network(df, min_degree=5)

    ax.set_title("A  Monomer chemical space", loc="left", fontsize=14, pad=6)
    ax.axis("off")

    if Gf.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No network nodes after filtering", ha="center", va="center", transform=ax.transAxes)
        return

    # Class -> color mapping (shared across panels)
    all_classes = set(Gf.nodes[n].get("class_name", "Other") for n in Gf.nodes())
    class_to_color = class_color_mapping(all_classes)

    # Positions (cached)
    pos_cache = OUTPUT_DIR / "monomer_network_positions_tsne.json"
    pos = compute_positions(list(Gf.nodes()), cache_path=pos_cache, method="tsne")
    pos_period = {n: pos[n] for n in Gf.nodes() if n in pos}

    # Edges: light gray, thickness by Δ(r-product) for that monomer pair
    edges = list(Gf.edges())
    if edges:
        deltas = np.array([edge_delta.get((u, v), edge_delta.get((v, u), 0.0)) for u, v in edges], dtype=float)
        finite = np.isfinite(deltas)
        dmax = float(np.max(deltas[finite])) if finite.any() else 0.0
        if dmax <= 0:
            edge_widths = np.full(len(edges), 0.6)
        else:
            # Stronger scaling so differences are clearly visible
            t = np.clip(deltas / dmax, 0, 1) ** 2.2
            edge_widths = 0.30 + t * 9.70  # ~0.30 .. 10.0
        # Rasterize to avoid occasional hollow-marker artifacts in vector backends (e.g. Preview)
        ec = nx.draw_networkx_edges(
            Gf,
            pos_period,
            ax=ax,
            alpha=0.45,
            width=edge_widths.tolist(),
            edge_color="gray",
        )
        try:
            ec.set_rasterized(True)
        except Exception:
            pass

    # Nodes: size by degree, fill by class
    deg = dict(Gf.degree())
    node_list = list(Gf.nodes())
    node_sizes = [deg[n] * 28 for n in node_list]
    node_fill = [class_to_color.get(Gf.nodes[n].get("class_name", "Other"), "#CCCCCC") for n in node_list]

    # Very light fills can read as "white/transparent" against gray edges; slightly darken locally for panel A.
    def _darken_if_too_light(hex_color: str, lum_thresh: float = 0.78, amount: float = 0.12) -> str:
        r, g, b = mcolors.to_rgb(hex_color)
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        if lum <= lum_thresh:
            return hex_color
        r2, g2, b2 = (r * (1 - amount), g * (1 - amount), b * (1 - amount))
        return mcolors.to_hex((r2, g2, b2))

    node_fill = [_darken_if_too_light(c) for c in node_fill]
    nc = nx.draw_networkx_nodes(
        Gf,
        pos_period,
        nodelist=node_list,
        node_size=node_sizes,
        node_color=node_fill,
        edgecolors="none",
        linewidths=0.0,
        alpha=0.9,
        ax=ax,
    )
    try:
        nc.set_rasterized(True)
    except Exception:
        pass


def plot_panel_b_temporal(
    ax: plt.Axes,
    df: pd.DataFrame,
    rolling_window_years: int = 5,
) -> tuple[list[str], dict[str, str]]:
    ax.set_title("B  Monomer temporal evolution", loc="left", fontsize=14, pad=6)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Share", fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(False)
    pub_year_series = _get_publication_year_series(df)
    if pub_year_series.empty:
        ax.text(
            0.5,
            0.5,
            "No temporal data (missing publication_year)",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return ([], {})

    # Count monomer class occurrences per year (then smooth with a rolling mean).
    class_counts_by_year: defaultdict[str, defaultdict[int, float]] = defaultdict(lambda: defaultdict(float))

    for i, row in df.iterrows():
        try:
            yv = pub_year_series.iloc[int(i)]
            year = int(yv) if np.isfinite(yv) else None
        except Exception:
            year = None
        if year is None:
            continue
        y = int(year)

        m1_smi = row.get("monomer1_smiles")
        m2_smi = row.get("monomer2_smiles")
        if pd.isna(m1_smi) or pd.isna(m2_smi):
            continue

        m1_name = row.get("monomer1_name", m1_smi)
        m2_name = row.get("monomer2_name", m2_smi)
        c1 = classify_monomer(m1_name, m1_smi)
        c2 = classify_monomer(m2_name, m2_smi)
        class_counts_by_year[c1][y] += 1
        class_counts_by_year[c2][y] += 1

    years = sorted({y for c in class_counts_by_year for y in class_counts_by_year[c].keys()})
    if not years:
        ax.text(0.5, 0.5, "No temporal data (missing years)", ha="center", va="center", transform=ax.transAxes)
        return ([], {})

    classes_in_data = set(class_counts_by_year.keys())
    class_sorted = [c for c in PREFERRED_CLASS_ORDER if c in classes_in_data] + sorted(
        classes_in_data - set(PREFERRED_CLASS_ORDER)
    )
    class_to_color = class_color_mapping(classes_in_data)

    # Build counts table (rows = years, cols = classes)
    counts = pd.DataFrame(index=pd.Index(years, name="year"), columns=class_sorted, data=0.0)
    for cls in class_sorted:
        for y, v in class_counts_by_year.get(cls, {}).items():
            if y in counts.index:
                counts.loc[y, cls] = float(v)

    # Smooth with rolling mean over years.
    w = int(max(1, rolling_window_years))
    if w > 1:
        counts_sm = counts.rolling(window=w, min_periods=max(1, w // 2), center=True).mean()
    else:
        counts_sm = counts

    totals = counts_sm.sum(axis=1)
    proportions = []
    labels = []
    colors = []
    for i, cat in enumerate(class_sorted):
        prop = (counts_sm[cat] / totals).fillna(0.0).to_numpy()
        proportions.append(prop)
        labels.append(cat)
        colors.append(class_to_color.get(cat, SEQUENTIAL_COLORS[i % len(SEQUENTIAL_COLORS)]))

    ax.stackplot(counts_sm.index.to_numpy(), *proportions, labels=labels, colors=colors, alpha=0.85)
    ax.set_xlim(int(min(years)) - 2, int(max(years)) + 2)
    return (labels, class_to_color)


def plot_panel_publication_years(ax: plt.Axes, df: pd.DataFrame) -> None:
    ax.set_title("C  Publication year distribution", loc="left", fontsize=14, pad=6)
    ax.set_xlabel("Publication year", fontsize=12)
    ax.set_ylabel("Number of papers", fontsize=12)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    pub_year_series = _get_publication_year_series(df)
    if pub_year_series.empty:
        ax.text(0.5, 0.5, "No publication years available", ha="center", va="center", transform=ax.transAxes)
        return
    years = pub_year_series.dropna().astype(int)
    # Bin to 5-year periods so the plot isn't one bar per year
    bin_size = 5
    bin_start = (years // bin_size) * bin_size
    counts = bin_start.value_counts().sort_index()
    x = counts.index.values.astype(int)
    y = counts.values

    # Use full bin width so bars touch (no visual gaps)
    ax.bar(
        x,
        y,
        width=5.0,
        align="edge",
        color=BOTTOM_PANEL_BAR_COLOR,
        alpha=BOTTOM_PANEL_BAR_ALPHA,
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_xlim(int(x.min()) - 2, int(x.max()) + 7)
    ax.set_xticks(np.arange(int(x.min()), int(x.max()) + 1, 20))
    ax.tick_params(labelsize=12)


def plot_panel_c_distributions(ax_temp: plt.Axes, ax_logp: plt.Axes, df: pd.DataFrame) -> None:
    ax_temp.set_title("D  Distribution of reaction temperatures", loc="left", fontsize=14, pad=6)
    ax_logp.set_title("E  Distribution of solvent boiling points", loc="left", fontsize=14, pad=6)

    # Temperature histogram
    t = pd.to_numeric(df.get("temperature"), errors="coerce").dropna()
    ax_temp.hist(
        t, bins=30, color=BOTTOM_PANEL_BAR_COLOR, alpha=BOTTOM_PANEL_BAR_ALPHA, edgecolor="black", linewidth=0.4
    )
    ax_temp.set_xlabel("Temperature", fontsize=12)
    ax_temp.set_ylabel("Count", fontsize=12)
    ax_temp.grid(False)
    ax_temp.spines["top"].set_visible(False)
    ax_temp.spines["right"].set_visible(False)

    # Solvent boiling point histogram (from curated JSON)
    solvent_name_col = None
    for cand in ("solvent_name", "solvent", "Solvent", "solvent_common_name"):
        if cand in df.columns:
            solvent_name_col = cand
            break
    if solvent_name_col is None:
        ax_logp.text(
            0.5,
            0.5,
            "No solvent name column (expected solvent_name/solvent)",
            ha="center",
            va="center",
            transform=ax_logp.transAxes,
        )
        ax_logp.set_axis_off()
        return

    bp_by_name = load_solvent_boiling_points()
    # If the dataset contains synonyms/typos, normalize them here (map -> canonical name key).
    alias_to_canonical = {
        "1,4-dioxan": "1,4-dioxane",
        "dioxan": "1,4-dioxane",
        "p-dioxane": "1,4-dioxane",
        "dioxane": "1,4-dioxane",
        "dimethylformamide": "n,n-dimethylformamide",
        "dimethyl formamide": "n,n-dimethylformamide",
        "n,n-dimethyl formamide": "n,n-dimethylformamide",
        "dimethylformamid": "n,n-dimethylformamide",
        "methyl ethyl ketone": "2-butanone",
        "ethyl methyl ketone": "2-butanone",
        "butanone": "2-butanone",
        "isopropanol": "isopropyl alcohol",
        "tert-butanol": "tert-butyl alcohol",
        "tert butyl alcohol": "tert-butyl alcohol",
        "ethyl alcohol": "ethanol",
        "methylene chloride": "dichloromethane",
        "1,2-dichloroethane": "dichloroethane",
        "ethylene dichloride": "dichloroethane",
        "benzol": "benzene",
        "benzointrile": "benzonitrile",
        "acetronitrile": "acetonitrile",
        "n-methyl-2-pyrrolidone": "n-methylpyrrolidone",
        "iso-octane": "isooctane",
        "n-heptane": "heptane",
        "n-hexane": "hexane",
        "n-butanol": "n-butyl alcohol",
        "liquid sulfur dioxide": "sulfur dioxide",
        "dimethylsulfoxide": "dimethyl sulfoxide",
        "dry chlorobenzene": "chlorobenzene",
        "glacial acetic acid": "acetic acid",
        "deuterated dioxane": "1,4-dioxane",
    }
    name_series = df[solvent_name_col].astype(str).str.strip()
    name_lower = name_series.str.lower()
    name_canonical = name_lower.map(lambda n: alias_to_canonical.get(n, n))
    bp_series = name_canonical.map(lambda n: bp_by_name.get(n))
    bp_vals = pd.to_numeric(bp_series, errors="coerce").dropna().to_numpy(dtype=float).tolist()

    if not bp_vals:
        ax_logp.text(
            0.5,
            0.5,
            f"No boiling point values found in {SOLVENT_BOILING_POINTS_FILE.name}",
            ha="center",
            va="center",
            transform=ax_logp.transAxes,
        )
        ax_logp.set_axis_off()
        return

    bp = pd.Series(bp_vals, dtype="float64")
    ax_logp.hist(
        bp, bins=30, color=BOTTOM_PANEL_BAR_COLOR, alpha=BOTTOM_PANEL_BAR_ALPHA, edgecolor="black", linewidth=0.4
    )
    ax_logp.set_xlabel("Boiling point (°C)", fontsize=12)
    ax_logp.set_ylabel("Count", fontsize=12)
    ax_logp.grid(False)
    ax_logp.spines["top"].set_visible(False)
    ax_logp.spines["right"].set_visible(False)


def print_basic_dataset_stats(df: pd.DataFrame) -> None:
    """
    Print basic dataset statistics for the curated (non-network) analysis.
    """
    print("\n=== Basic dataset statistics ===")

    raw_counts = _count_raw_llm_extractions()
    print("\n--- Raw extraction (LLM outputs) ---")
    print(f"Publications (files): {raw_counts['raw_publications']}")
    print(f"Number of reactions (expanded by reaction_conditions): {raw_counts['raw_reactions']}")
    if raw_counts["raw_parse_errors"] > 0:
        print(f"Parse errors: {raw_counts['raw_parse_errors']}")

    csv_counts = _count_artifacts_extracted_reactions_csv()
    print("\n--- Extracted dataset (artifacts CSV) ---")
    print(f"Publications (unique): {csv_counts['csv_publications']}")
    print(f"Number of reactions (rows): {csv_counts['csv_reactions']}")

    pub_key_col = None
    for cand in ("PDF_name", "source_filename", "original_source", "source"):
        if cand in df.columns:
            pub_key_col = cand
            break
    if pub_key_col is not None:
        pub_keys = df[pub_key_col].dropna().astype(str).str.strip()
        pub_keys = pub_keys[pub_keys.astype(bool)]
        processed_publications = int(pub_keys.nunique())
    else:
        processed_publications = 0

    if "reaction_id" in df.columns:
        rxn = df["reaction_id"].dropna().astype(str).str.strip()
        rxn = rxn[rxn.astype(bool)]
        processed_reactions = int(rxn.nunique())
    else:
        processed_reactions = int(len(df))

    print("\n--- Processed dataset (processed_data.csv) ---")
    if pub_key_col is not None:
        print(f"Publications (unique by `{pub_key_col}`): {processed_publications}")
    else:
        print("Publications: (no identifier column found)")
    if "reaction_id" in df.columns:
        print(f"Number of reactions (unique by `reaction_id`): {processed_reactions}")
    else:
        print(f"Number of reactions (rows): {processed_reactions}")

    # Monomers + pairs
    m1 = df.get("monomer1_smiles")
    m2 = df.get("monomer2_smiles")
    if m1 is not None and m2 is not None:
        monomers = pd.concat([m1.dropna().astype(str), m2.dropna().astype(str)], ignore_index=True)
        monomers = monomers[monomers.astype(bool)]
        print(f"Unique monomers: {monomers.nunique()}")

        m1s = m1.dropna().astype(str)
        m2s = m2.dropna().astype(str)
        pairs = pd.DataFrame({"a": m1s, "b": m2s})
        pairs = pairs[pairs["a"].astype(bool) & pairs["b"].astype(bool)]
        pairs["p1"] = np.where(pairs["a"] <= pairs["b"], pairs["a"], pairs["b"])
        pairs["p2"] = np.where(pairs["a"] <= pairs["b"], pairs["b"], pairs["a"])
        print(f"Unique monomer pairs (unordered): {len(pairs.drop_duplicates(subset=['p1','p2']))}")

    # Solvents + boiling point coverage
    solvent_name_col = None
    for cand in ("solvent_name", "solvent", "Solvent", "solvent_common_name"):
        if cand in df.columns:
            solvent_name_col = cand
            break
    if solvent_name_col:
        bp_by_name = load_solvent_boiling_points()
        alias_to_canonical = {
            "1,4-dioxan": "1,4-dioxane",
            "dioxan": "1,4-dioxane",
            "p-dioxane": "1,4-dioxane",
            "dioxane": "1,4-dioxane",
            "dimethylformamide": "n,n-dimethylformamide",
            "dimethyl formamide": "n,n-dimethylformamide",
            "n,n-dimethyl formamide": "n,n-dimethylformamide",
            "dimethylformamid": "n,n-dimethylformamide",
            "methyl ethyl ketone": "2-butanone",
            "ethyl methyl ketone": "2-butanone",
            "butanone": "2-butanone",
            "isopropanol": "isopropyl alcohol",
            "tert-butanol": "tert-butyl alcohol",
            "tert butyl alcohol": "tert-butyl alcohol",
            "ethyl alcohol": "ethanol",
            "methylene chloride": "dichloromethane",
            "1,2-dichloroethane": "dichloroethane",
            "ethylene dichloride": "dichloroethane",
            "benzol": "benzene",
            "benzointrile": "benzonitrile",
            "acetronitrile": "acetonitrile",
            "n-methyl-2-pyrrolidone": "n-methylpyrrolidone",
            "iso-octane": "isooctane",
            "n-heptane": "heptane",
            "n-hexane": "hexane",
            "n-butanol": "n-butyl alcohol",
            "liquid sulfur dioxide": "sulfur dioxide",
            "dimethylsulfoxide": "dimethyl sulfoxide",
            "dry chlorobenzene": "chlorobenzene",
            "glacial acetic acid": "acetic acid",
            "deuterated dioxane": "1,4-dioxane",
        }
        ignore_names = {"na", "nan", "none", "no solvent", "bulk", "monomer"}

        s = df[solvent_name_col].astype(str).str.strip()
        sl = s.str.lower()
        canon = sl.map(lambda n: alias_to_canonical.get(n, n))
        canon = canon[canon.astype(bool) & ~canon.isin(ignore_names)]

        print(f"Unique solvents (canonical, excl placeholders): {canon.nunique()}")

        bp_series = canon.map(lambda n: bp_by_name.get(n))
        bp_vals = pd.to_numeric(bp_series, errors="coerce").dropna()
        if not bp_vals.empty:
            print(f"Solvent boiling points (°C): {bp_vals.min():.2f} .. {bp_vals.max():.2f} (n={len(bp_vals)})")

    # Temperature
    if "temperature" in df.columns:
        t = pd.to_numeric(df["temperature"], errors="coerce").dropna()
        if not t.empty:
            print(f"Temperature: {t.min():.2f} .. {t.max():.2f} (n={len(t)})")

    # Publication years
    py = _get_publication_year_series(df).dropna()
    if not py.empty:
        py_i = py.astype(int)
        print(f"Publication years: {py_i.min()} .. {py_i.max()} (n={len(py_i)})")
    else:
        print("Publication years: (missing `publication_year` column)")


def main() -> None:
    setup_plot_style()
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = load_data()
    print_basic_dataset_stats(df)

    fig = plt.figure(figsize=(TWO_COL_WIDTH_INCH * 1.65, TWO_COL_WIDTH_INCH * 1.05), layout="constrained")
    # Add a thin spacer row between top and bottom panels for the shared legend
    gs = fig.add_gridspec(3, 6, height_ratios=[1.15, 0.26, 0.85])

    ax_a = fig.add_subplot(gs[0, 0:3])
    ax_b = fig.add_subplot(gs[0, 3:6])
    ax_pub = fig.add_subplot(gs[2, 0:2])
    ax_temp = fig.add_subplot(gs[2, 2:4])
    ax_logp = fig.add_subplot(gs[2, 4:6])

    plot_panel_a_network(ax_a, df)
    legend_labels, class_to_color = plot_panel_b_temporal(ax_b, df, rolling_window_years=10)
    plot_panel_publication_years(ax_pub, df)
    plot_panel_c_distributions(ax_temp, ax_logp, df)

    # Shared legend for monomer classes: centered above panels A+B
    if legend_labels and class_to_color:
        handles = [
            plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=class_to_color[l], markersize=8, label=l)
            for l in legend_labels
            if l in class_to_color
        ]
        if handles:
            fig.legend(
                handles=handles,
                loc="upper center",
                # Place between the two plot rows
                bbox_to_anchor=(0.5, 0.515),
                ncol=min(5, max(1, len(handles))),
                frameon=True,
                framealpha=0.9,
                edgecolor="none",
                fontsize=12,
                handletextpad=0.5,
                columnspacing=1.0,
            )

    base = OUTPUT_DIR / "dataset_analysis"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    fig.savefig(base.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"✓ Saved {base}.pdf and {base}.png")


if False and __name__ == "__main__":
    main()

def load_solvent_boiling_points() -> dict[str, float | None]:
    """
    Load curated solvent boiling points (°C) from JSON.
    Keys are solvent names as they appear in the dataset (case-insensitive; stored in lowercase).
    Values are floats (°C) or null (meaning: treat as missing / exclude).
    """
    if not SOLVENT_BOILING_POINTS_FILE.exists():
        raise FileNotFoundError(f"Missing boiling point file: {SOLVENT_BOILING_POINTS_FILE}")
    payload = json.loads(SOLVENT_BOILING_POINTS_FILE.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Boiling point file must be a JSON object: {SOLVENT_BOILING_POINTS_FILE}")
    out: dict[str, float | None] = {}
    for k, v in payload.items():
        if k is None:
            continue
        key = str(k).strip().lower()
        if not key:
            continue
        if v is None:
            out[key] = None
            continue
        try:
            out[key] = float(v)
        except Exception:
            out[key] = None
    return out


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    return df


def _first_nonempty(*values):
    for v in values:
        if v is None:
            continue
        if isinstance(v, float) and pd.isna(v):
            continue
        s = str(v).strip()
        if s and s.lower() != "nan":
            return s
    return None


def _get_publication_year_series(df: pd.DataFrame) -> pd.Series:
    """
    Return a numeric publication year series if available in the dataset.
    (No network calls; relies on a local column such as `publication_year`.)
    """
    if "publication_year" not in df.columns:
        return pd.Series([], dtype="float64")
    return pd.to_numeric(df["publication_year"], errors="coerce")


def _norm_name(x) -> str:
    return "" if pd.isna(x) else str(x).strip().lower()


def _norm_smiles(x) -> str:
    return "" if pd.isna(x) else str(x).strip()


def _has_any(s: str, patterns) -> bool:
    return any(p in s for p in patterns)


def _has_double_bond(smi: str) -> bool:
    if not smi:
        return False
    if "C=C" in smi or "(=C" in smi or "=C(" in smi:
        return True
    return bool(re.search(r"C=.*=?.*C", smi)) or ("=" in smi)


def classify_monomer(monomer_name, monomer_smiles) -> str:
    """
    9-class monomer classification scheme (same logic as existing analysis scripts).
    """
    name = _norm_name(monomer_name)
    smi = _norm_smiles(monomer_smiles)
    if "acrylonitrile" in name or "methacrylonitrile" in name or _has_any(smi, ["C=CC#N", "C=C(C)C#N"]):
        return "(Meth)acrylonitriles"
    if any(k in name for k in ["maleic", "maleate", "fumar", "fumarate", "itaconic", "itaconate", "aconitate"]):
        return "Anhydrides/Diacids"
    if (
        ("methacry" in name)
        or ("acryl" in name and "amide" not in name and "nitrile" not in name)
        or "acrylic acid" in name
        or _has_any(
            smi,
            [
                "C=CC(=O)O",
                "C=CC(=O)OC",
                "C=CC(=O)[O-]",
                "C=C(C)C(=O)O",
                "C=C(C)C(=O)OC",
                "C=C(C)C(=O)[O-]",
            ],
        )
    ):
        return "(Meth)acrylates"
    if "acrylamide" in name or "methacrylamide" in name or "maleimide" in name or _has_any(smi, ["C=CC(=O)N", "C=C(C)C(=O)N"]):
        return "(Meth)acrylamides/imides"
    if "styrene" in name or "methylstyrene" in name or "chlorostyrene" in name or "methoxystyrene" in name or "styrene sulfonate" in name or _has_any(smi, ["C=Cc1ccccc1", "C=CC1=CC=CC=C1"]):
        return "Styrenics"
    if "butadiene" in name or "isoprene" in name or "chloroprene" in name or "diene" in name or _has_any(smi, ["C=CC=C", "C=C-C=C"]):
        return "Conjugated Dienes"
    if "vinyl" in name or re.search(r"\b\d*-?vinyl", name):
        return "Vinyl Derivatives"
    if (
        name in ["ethylene", "propylene", "propene", "isobutylene"]
        or re.search(r"\b\d+-?(hexene|octene)\b", name)
        or "tetrafluoroethylene" in name
        or "chlorotrifluoroethylene" in name
        or ("ethylene" in name and any(k in name for k in ["fluoro", "chloro", "trifluoro", "tetrafluoro"]))
        or (_has_double_bond(smi) and "c1ccccc1" not in smi.lower())
    ):
        return "Olefins"
    return "Other"


PREFERRED_CLASS_ORDER = [
    "(Meth)acrylonitriles",
    "Anhydrides/Diacids",
    "(Meth)acrylates",
    "(Meth)acrylamides/imides",
    "Styrenics",
    "Conjugated Dienes",
    "Vinyl Derivatives",
    "Olefins",
    "Other",
]


def class_color_mapping(classes: set[str]) -> dict[str, str]:
    """
    Deterministic class -> color mapping consistent across panels.
    """
    class_to_color: dict[str, str] = {}
    for i, class_name in enumerate(PREFERRED_CLASS_ORDER):
        if class_name in classes:
            class_to_color[class_name] = SEQUENTIAL_COLORS[i % len(SEQUENTIAL_COLORS)]
    for class_name in sorted(classes - set(PREFERRED_CLASS_ORDER)):
        class_to_color[class_name] = SEQUENTIAL_COLORS[len(class_to_color) % len(SEQUENTIAL_COLORS)]
    return class_to_color


def calculate_morgan_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048):
    if not RDKIT_AVAILABLE:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)
    except Exception:
        return None


def compute_positions(
    nodes: list[str],
    cache_path: Path,
    method: str = "tsne",
    random_state: int = 42,
) -> dict[str, tuple[float, float]]:
    """
    Compute 2D positions for nodes.
    Uses a cache to avoid recomputing expensive t-SNE across runs.
    """
    if cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text())
            if payload.get("method") == method and payload.get("random_state") == random_state:
                pos = {k: tuple(v) for k, v in payload.get("pos", {}).items()}
                if len(pos) >= int(0.9 * len(nodes)):
                    return pos
        except Exception:
            pass

    pos_2d: dict[str, tuple[float, float]] = {}

    fingerprints = []
    valid_nodes = []
    if RDKIT_AVAILABLE:
        for smi in nodes:
            fp = calculate_morgan_fingerprint(smi)
            if fp is not None:
                fingerprints.append(fp)
                valid_nodes.append(smi)

    if fingerprints and SKLEARN_AVAILABLE and method in {"tsne", "pca"}:
        X = np.array(fingerprints)
        try:
            if method == "tsne" and len(valid_nodes) >= 3:
                tsne = TSNE(
                    n_components=2,
                    random_state=random_state,
                    perplexity=min(30, max(2, len(valid_nodes) - 1)),
                    init="random",
                    learning_rate="auto",
                )
                coords = tsne.fit_transform(X)
            else:
                pca = PCA(n_components=2, random_state=random_state)
                coords = pca.fit_transform(X)
            for i, smi in enumerate(valid_nodes):
                pos_2d[smi] = (float(coords[i, 0]), float(coords[i, 1]))
        except Exception:
            pos_2d = {}

    # Fill missing with network layouts (deterministic-ish with seed)
    missing = set(nodes) - set(pos_2d.keys())
    if missing:
        G_tmp = nx.Graph()
        G_tmp.add_nodes_from(missing)
        try:
            pos_missing = nx.kamada_kawai_layout(G_tmp, seed=random_state)
        except Exception:
            pos_missing = nx.spring_layout(G_tmp, seed=random_state)
        pos_2d.update({k: (float(v[0]), float(v[1])) for k, v in pos_missing.items()})

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"method": method, "random_state": random_state, "pos": pos_2d}))
    except Exception:
        pass

    return pos_2d


def compute_delta_r(row: pd.Series) -> float | None:
    """
    Δr for a reaction entry.
    Uses |log10(r1) - log10(r2)|, which is scale-invariant and robust across magnitudes.
    """
    r1 = row.get("constant_1")
    r2 = row.get("constant_2")
    try:
        r1 = float(r1)
        r2 = float(r2)
    except Exception:
        return None
    if not (np.isfinite(r1) and np.isfinite(r2)) or r1 <= 0 or r2 <= 0:
        return None
    return float(abs(np.log10(r1) - np.log10(r2)))


def build_monomer_network(df: pd.DataFrame, min_degree: int = 5):
    G = nx.Graph()
    monomer_classes: dict[str, str] = {}
    edge_r_product_values: defaultdict[tuple[str, str], list[float]] = defaultdict(list)

    for _, row in df.iterrows():
        m1 = row.get("monomer1_smiles")
        m2 = row.get("monomer2_smiles")
        if pd.isna(m1) or pd.isna(m2):
            continue

        m1_name = row.get("monomer1_name", m1)
        m2_name = row.get("monomer2_name", m2)

        if m1 not in monomer_classes:
            monomer_classes[m1] = classify_monomer(m1_name, m1)
        if m2 not in monomer_classes:
            monomer_classes[m2] = classify_monomer(m2_name, m2)

        if m1 not in G:
            G.add_node(m1, name=m1_name, class_name=monomer_classes[m1])
        if m2 not in G:
            G.add_node(m2, name=m2_name, class_name=monomer_classes[m2])

        if G.has_edge(m1, m2):
            G[m1][m2]["weight"] += 1
        else:
            G.add_edge(m1, m2, weight=1)

        # Collect r-product (r1*r2) for this monomer pair / edge.
        # Prefer explicit r1r2 column, otherwise compute from constant_1/constant_2.
        r12 = row.get("r1r2")
        if pd.isna(r12):
            try:
                r1 = float(row.get("constant_1"))
                r2 = float(row.get("constant_2"))
                r12 = r1 * r2 if (np.isfinite(r1) and np.isfinite(r2)) else np.nan
            except Exception:
                r12 = np.nan
        try:
            r12f = float(r12)
        except Exception:
            r12f = float("nan")
        if np.isfinite(r12f) and r12f > 0:
            a, b = (m1, m2) if str(m1) <= str(m2) else (m2, m1)
            edge_r_product_values[(a, b)].append(r12f)

    degrees = dict(G.degree())
    frequent_nodes = {n for n, d in degrees.items() if d >= min_degree}
    Gf = G.subgraph(frequent_nodes).copy()

    # Edge Δ(r-product): use spread as max/min ratio (scale-invariant),
    # transformed with log10(ratio+1) to compress extreme values.
    edge_delta = {}
    for u, v in Gf.edges():
        a, b = (u, v) if str(u) <= str(v) else (v, u)
        vals = edge_r_product_values.get((a, b), [])
        if len(vals) < 2:
            edge_delta[(u, v)] = 0.0
            continue
        mn, mx = float(np.min(vals)), float(np.max(vals))
        if mn <= 0 or not np.isfinite(mn) or not np.isfinite(mx):
            edge_delta[(u, v)] = 0.0
            continue
        ratio = mx / mn
        edge_delta[(u, v)] = float(np.log10(ratio + 1.0))

    return Gf, edge_delta


def plot_panel_a_network(ax: plt.Axes, df: pd.DataFrame) -> None:
    Gf, edge_delta = build_monomer_network(df, min_degree=5)

    ax.set_title("A  Monomer chemical space", loc="left", fontsize=14, pad=6)
    ax.axis("off")

    if Gf.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No network nodes after filtering", ha="center", va="center", transform=ax.transAxes)
        return

    # Class -> color mapping (shared across panels)
    all_classes = set(Gf.nodes[n].get("class_name", "Other") for n in Gf.nodes())
    class_to_color = class_color_mapping(all_classes)

    # Positions (cached)
    pos_cache = OUTPUT_DIR / "monomer_network_positions_tsne.json"
    pos = compute_positions(list(Gf.nodes()), cache_path=pos_cache, method="tsne")
    pos_period = {n: pos[n] for n in Gf.nodes() if n in pos}

    # Edges: light gray, thickness by Δ(r-product) for that monomer pair
    edges = list(Gf.edges())
    if edges:
        deltas = np.array([edge_delta.get((u, v), edge_delta.get((v, u), 0.0)) for u, v in edges], dtype=float)
        finite = np.isfinite(deltas)
        dmax = float(np.max(deltas[finite])) if finite.any() else 0.0
        if dmax <= 0:
            edge_widths = np.full(len(edges), 0.6)
        else:
            # Stronger scaling so differences are clearly visible
            t = np.clip(deltas / dmax, 0, 1) ** 2.2
            edge_widths = 0.30 + t * 9.70  # ~0.30 .. 10.0
        # Rasterize to avoid occasional hollow-marker artifacts in vector backends (e.g. Preview)
        ec = nx.draw_networkx_edges(
            Gf,
            pos_period,
            ax=ax,
            alpha=0.45,
            width=edge_widths.tolist(),
            edge_color="gray",
        )
        try:
            ec.set_rasterized(True)
        except Exception:
            pass

    # Nodes: size by degree, fill by class
    deg = dict(Gf.degree())
    node_list = list(Gf.nodes())
    node_sizes = [deg[n] * 28 for n in node_list]
    node_fill = [class_to_color.get(Gf.nodes[n].get("class_name", "Other"), "#CCCCCC") for n in node_list]
    # Very light fills can read as "white/transparent" against gray edges; slightly darken locally for panel A.
    def _darken_if_too_light(hex_color: str, lum_thresh: float = 0.78, amount: float = 0.12) -> str:
        r, g, b = mcolors.to_rgb(hex_color)
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        if lum <= lum_thresh:
            return hex_color
        r2, g2, b2 = (r * (1 - amount), g * (1 - amount), b * (1 - amount))
        return mcolors.to_hex((r2, g2, b2))

    node_fill = [_darken_if_too_light(c) for c in node_fill]
    nc = nx.draw_networkx_nodes(
        Gf,
        pos_period,
        nodelist=node_list,
        node_size=node_sizes,
        node_color=node_fill,
        edgecolors="none",
        linewidths=0.0,
        alpha=0.9,
        ax=ax,
    )
    try:
        nc.set_rasterized(True)
    except Exception:
        pass

    # Legend: monomer classes (keep compact)
    # Intentionally no legend here: top row should show only one legend overall (in panel B).


def create_year_bin(year: int | None, bin_size: int = 5) -> str | None:
    if year is None:
        return None
    if bin_size <= 1:
        return str(year)
    bin_start = (year // bin_size) * bin_size
    bin_end = bin_start + bin_size - 1
    return f"{bin_start}-{bin_end}"


def bin_to_numeric(bin_label: str) -> float:
    parts = bin_label.split("-")
    if len(parts) == 2:
        return (int(parts[0]) + int(parts[1])) / 2
    return float(int(parts[0]))


def plot_panel_b_temporal(
    ax: plt.Axes,
    df: pd.DataFrame,
    rolling_window_years: int = 5,
) -> tuple[list[str], dict[str, str]]:
    ax.set_title("B  Monomer temporal evolution", loc="left", fontsize=14, pad=6)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Share", fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(False)
    pub_year_series = _get_publication_year_series(df)
    if pub_year_series.empty:
        ax.text(0.5, 0.5, "No temporal data (missing publication_year)", ha="center", va="center", transform=ax.transAxes)
        return ([], {})

    # Count monomer class occurrences per year (then smooth with a rolling mean).
    class_counts_by_year: defaultdict[str, defaultdict[int, float]] = defaultdict(lambda: defaultdict(float))

    for i, row in df.iterrows():
        try:
            yv = pub_year_series.iloc[int(i)]
            year = int(yv) if np.isfinite(yv) else None
        except Exception:
            year = None
        if year is None:
            continue
        y = int(year)

        m1_smi = row.get("monomer1_smiles")
        m2_smi = row.get("monomer2_smiles")
        if pd.isna(m1_smi) or pd.isna(m2_smi):
            continue

        m1_name = row.get("monomer1_name", m1_smi)
        m2_name = row.get("monomer2_name", m2_smi)
        c1 = classify_monomer(m1_name, m1_smi)
        c2 = classify_monomer(m2_name, m2_smi)
        class_counts_by_year[c1][y] += 1
        class_counts_by_year[c2][y] += 1

    years = sorted({y for c in class_counts_by_year for y in class_counts_by_year[c].keys()})
    if not years:
        ax.text(0.5, 0.5, "No temporal data (missing years)", ha="center", va="center", transform=ax.transAxes)
        return ([], {})

    classes_in_data = set(class_counts_by_year.keys())
    class_sorted = [c for c in PREFERRED_CLASS_ORDER if c in classes_in_data] + sorted(classes_in_data - set(PREFERRED_CLASS_ORDER))
    class_to_color = class_color_mapping(classes_in_data)

    # Build counts table (rows = years, cols = classes)
    counts = pd.DataFrame(index=pd.Index(years, name="year"), columns=class_sorted, data=0.0)
    for cls in class_sorted:
        for y, v in class_counts_by_year.get(cls, {}).items():
            if y in counts.index:
                counts.loc[y, cls] = float(v)

    # Smooth with rolling mean over years.
    w = int(max(1, rolling_window_years))
    if w > 1:
        counts_sm = counts.rolling(window=w, min_periods=max(1, w // 2), center=True).mean()
    else:
        counts_sm = counts

    totals = counts_sm.sum(axis=1)
    proportions = []
    labels = []
    colors = []
    for i, cat in enumerate(class_sorted):
        prop = (counts_sm[cat] / totals).fillna(0.0).to_numpy()
        proportions.append(prop)
        labels.append(cat)
        colors.append(class_to_color.get(cat, SEQUENTIAL_COLORS[i % len(SEQUENTIAL_COLORS)]))

    ax.stackplot(counts_sm.index.to_numpy(), *proportions, labels=labels, colors=colors, alpha=0.85)
    ax.set_xlim(int(min(years)) - 2, int(max(years)) + 2)
    return (labels, class_to_color)


def plot_panel_publication_years(ax: plt.Axes, df: pd.DataFrame) -> None:
    ax.set_title("C  Publication year distribution", loc="left", fontsize=14, pad=6)
    ax.set_xlabel("Publication year", fontsize=12)
    ax.set_ylabel("Number of papers", fontsize=12)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    pub_year_series = _get_publication_year_series(df)
    if pub_year_series.empty:
        ax.text(0.5, 0.5, "No publication years available", ha="center", va="center", transform=ax.transAxes)
        return
    years = pub_year_series.dropna().astype(int)
    # Bin to 5-year periods so the plot isn't one bar per year
    bin_size = 5
    bin_start = (years // bin_size) * bin_size
    counts = bin_start.value_counts().sort_index()
    x = counts.index.values.astype(int)
    y = counts.values

    # Use full bin width so bars touch (no visual gaps)
    ax.bar(
        x,
        y,
        width=5.0,
        align="edge",
        color=BOTTOM_PANEL_BAR_COLOR,
        alpha=BOTTOM_PANEL_BAR_ALPHA,
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_xlim(int(x.min()) - 2, int(x.max()) + 7)
    ax.set_xticks(np.arange(int(x.min()), int(x.max()) + 1, 20))
    ax.tick_params(labelsize=12)


def plot_panel_c_distributions(ax_temp: plt.Axes, ax_logp: plt.Axes, df: pd.DataFrame) -> None:
    ax_temp.set_title("D  Distribution of reaction temperatures", loc="left", fontsize=14, pad=6)
    ax_logp.set_title("E  Distribution of solvent boiling points", loc="left", fontsize=14, pad=6)

    # Temperature histogram
    t = pd.to_numeric(df.get("temperature"), errors="coerce").dropna()
    ax_temp.hist(t, bins=30, color=BOTTOM_PANEL_BAR_COLOR, alpha=BOTTOM_PANEL_BAR_ALPHA, edgecolor="black", linewidth=0.4)
    ax_temp.set_xlabel("Temperature", fontsize=12)
    ax_temp.set_ylabel("Count", fontsize=12)
    ax_temp.grid(False)
    ax_temp.spines["top"].set_visible(False)
    ax_temp.spines["right"].set_visible(False)

    # Solvent boiling point histogram (from curated JSON)
    solvent_name_col = None
    for cand in ("solvent_name", "solvent", "Solvent", "solvent_common_name"):
        if cand in df.columns:
            solvent_name_col = cand
            break
    if solvent_name_col is None:
        ax_logp.text(0.5, 0.5, "No solvent name column (expected solvent_name/solvent)", ha="center", va="center", transform=ax_logp.transAxes)
        ax_logp.set_axis_off()
        return

    bp_by_name = load_solvent_boiling_points()
    # If the dataset contains synonyms/typos, normalize them here (map -> canonical name key).
    alias_to_canonical = {
        "1,4-dioxan": "1,4-dioxane",
        "dioxan": "1,4-dioxane",
        "p-dioxane": "1,4-dioxane",
        "dioxane": "1,4-dioxane",
        "dimethylformamide": "n,n-dimethylformamide",
        "dimethyl formamide": "n,n-dimethylformamide",
        "n,n-dimethyl formamide": "n,n-dimethylformamide",
        "dimethylformamid": "n,n-dimethylformamide",
        "methyl ethyl ketone": "2-butanone",
        "ethyl methyl ketone": "2-butanone",
        "butanone": "2-butanone",
        "isopropanol": "isopropyl alcohol",
        "tert-butanol": "tert-butyl alcohol",
        "tert butyl alcohol": "tert-butyl alcohol",
        "ethyl alcohol": "ethanol",
        "methylene chloride": "dichloromethane",
        "1,2-dichloroethane": "dichloroethane",
        "ethylene dichloride": "dichloroethane",
        "benzol": "benzene",
        "benzointrile": "benzonitrile",
        "acetronitrile": "acetonitrile",
        "n-methyl-2-pyrrolidone": "n-methylpyrrolidone",
        "iso-octane": "isooctane",
        "n-heptane": "heptane",
        "n-hexane": "hexane",
        "n-butanol": "n-butyl alcohol",
        "liquid sulfur dioxide": "sulfur dioxide",
        "dimethylsulfoxide": "dimethyl sulfoxide",
        "dry chlorobenzene": "chlorobenzene",
        "glacial acetic acid": "acetic acid",
        "deuterated dioxane": "1,4-dioxane",
    }
    name_series = df[solvent_name_col].astype(str).str.strip()
    name_lower = name_series.str.lower()
    name_canonical = name_lower.map(lambda n: alias_to_canonical.get(n, n))
    bp_series = name_canonical.map(lambda n: bp_by_name.get(n))
    bp_vals = pd.to_numeric(bp_series, errors="coerce").dropna().to_numpy(dtype=float).tolist()

    if not bp_vals:
        ax_logp.text(0.5, 0.5, f"No boiling point values found in {SOLVENT_BOILING_POINTS_FILE.name}", ha="center", va="center", transform=ax_logp.transAxes)
        ax_logp.set_axis_off()
        return

    # Boiling point stats/coverage are printed once in `print_basic_dataset_stats`.

    bp = pd.Series(bp_vals, dtype="float64")
    ax_logp.hist(bp, bins=30, color=BOTTOM_PANEL_BAR_COLOR, alpha=BOTTOM_PANEL_BAR_ALPHA, edgecolor="black", linewidth=0.4)
    ax_logp.set_xlabel("Boiling point (°C)", fontsize=12)
    ax_logp.set_ylabel("Count", fontsize=12)
    ax_logp.grid(False)
    ax_logp.spines["top"].set_visible(False)
    ax_logp.spines["right"].set_visible(False)


def print_basic_dataset_stats(df: pd.DataFrame) -> None:
    """
    Print basic dataset statistics for the curated (non-network) analysis.
    """
    print("\n=== Basic dataset statistics ===")

    raw_counts = _count_raw_llm_extractions()
    print("\n--- Raw extraction (LLM outputs) ---")
    print(f"Publications (files): {raw_counts['raw_publications']}")
    print(f"Number of reactions (expanded by reaction_conditions): {raw_counts['raw_reactions']}")
    if raw_counts["raw_parse_errors"] > 0:
        print(f"Parse errors: {raw_counts['raw_parse_errors']}")

    csv_counts = _count_artifacts_extracted_reactions_csv()
    print("\n--- Extracted dataset (artifacts CSV) ---")
    print(f"Publications (unique): {csv_counts['csv_publications']}")
    print(f"Number of reactions (rows): {csv_counts['csv_reactions']}")

    pub_key_col = None
    for cand in ("PDF_name", "source_filename", "original_source", "source"):
        if cand in df.columns:
            pub_key_col = cand
            break
    if pub_key_col is not None:
        pub_keys = df[pub_key_col].dropna().astype(str).str.strip()
        pub_keys = pub_keys[pub_keys.astype(bool)]
        processed_publications = int(pub_keys.nunique())
    else:
        processed_publications = 0

    if "reaction_id" in df.columns:
        rxn = df["reaction_id"].dropna().astype(str).str.strip()
        rxn = rxn[rxn.astype(bool)]
        processed_reactions = int(rxn.nunique())
    else:
        processed_reactions = int(len(df))

    print("\n--- Processed dataset (processed_data.csv) ---")
    if pub_key_col is not None:
        print(f"Publications (unique by `{pub_key_col}`): {processed_publications}")
    else:
        print("Publications: (no identifier column found)")
    if "reaction_id" in df.columns:
        print(f"Number of reactions (unique by `reaction_id`): {processed_reactions}")
    else:
        print(f"Number of reactions (rows): {processed_reactions}")

    # Canonicalize mirrored monomer entries (unordered monomer pair + swapped constants)
    m1 = df.get("monomer1_smiles")
    m2 = df.get("monomer2_smiles")
    c1 = df.get("constant_1")
    c2 = df.get("constant_2")
    if m1 is not None and m2 is not None and c1 is not None and c2 is not None:
        m1s = m1.astype(str)
        m2s = m2.astype(str)
        c1n = pd.to_numeric(c1, errors="coerce")
        c2n = pd.to_numeric(c2, errors="coerce")
        swap = m1s > m2s
        cm1 = m1s.where(~swap, m2s)
        cm2 = m2s.where(~swap, m1s)
        cc1 = c1n.where(~swap, c2n)
        cc2 = c2n.where(~swap, c1n)

        subset = ["_m1", "_m2", "_c1", "_c2"]
        tmp = df.copy()
        tmp["_m1"] = cm1
        tmp["_m2"] = cm2
        tmp["_c1"] = cc1
        tmp["_c2"] = cc2

        # Add condition columns when present to avoid collapsing distinct experiments
        for col in ("temperature", "solvent", "solvent_name", "Solvent", "publication_year", "reaction_id"):
            if col in tmp.columns:
                subset.append(col)

        # (Previously printed "Datapoints (unmirrored, best-effort)" here.)

    # Monomers + pairs
    if m1 is not None and m2 is not None:
        monomers = pd.concat([m1.dropna().astype(str), m2.dropna().astype(str)], ignore_index=True)
        monomers = monomers[monomers.astype(bool)]
        print(f"Unique monomers: {monomers.nunique()}")

        m1s = m1.dropna().astype(str)
        m2s = m2.dropna().astype(str)
        pairs = pd.DataFrame({"a": m1s, "b": m2s})
        pairs = pairs[pairs["a"].astype(bool) & pairs["b"].astype(bool)]
        pairs["p1"] = np.where(pairs["a"] <= pairs["b"], pairs["a"], pairs["b"])
        pairs["p2"] = np.where(pairs["a"] <= pairs["b"], pairs["b"], pairs["a"])
        print(f"Unique monomer pairs (unordered): {len(pairs.drop_duplicates(subset=['p1','p2']))}")

    # Solvents + boiling point coverage
    solvent_name_col = None
    for cand in ("solvent_name", "solvent", "Solvent", "solvent_common_name"):
        if cand in df.columns:
            solvent_name_col = cand
            break
    if solvent_name_col:
        bp_by_name = load_solvent_boiling_points()
        alias_to_canonical = {
            "1,4-dioxan": "1,4-dioxane",
            "dioxan": "1,4-dioxane",
            "p-dioxane": "1,4-dioxane",
            "dioxane": "1,4-dioxane",
            "dimethylformamide": "n,n-dimethylformamide",
            "dimethyl formamide": "n,n-dimethylformamide",
            "n,n-dimethyl formamide": "n,n-dimethylformamide",
            "dimethylformamid": "n,n-dimethylformamide",
            "methyl ethyl ketone": "2-butanone",
            "ethyl methyl ketone": "2-butanone",
            "butanone": "2-butanone",
            "isopropanol": "isopropyl alcohol",
            "tert-butanol": "tert-butyl alcohol",
            "tert butyl alcohol": "tert-butyl alcohol",
            "ethyl alcohol": "ethanol",
            "methylene chloride": "dichloromethane",
            "1,2-dichloroethane": "dichloroethane",
            "ethylene dichloride": "dichloroethane",
            "benzol": "benzene",
            "benzointrile": "benzonitrile",
            "acetronitrile": "acetonitrile",
            "n-methyl-2-pyrrolidone": "n-methylpyrrolidone",
            "iso-octane": "isooctane",
            "n-heptane": "heptane",
            "n-hexane": "hexane",
            "n-butanol": "n-butyl alcohol",
            "liquid sulfur dioxide": "sulfur dioxide",
            "dimethylsulfoxide": "dimethyl sulfoxide",
            "dry chlorobenzene": "chlorobenzene",
            "glacial acetic acid": "acetic acid",
            "deuterated dioxane": "1,4-dioxane",
        }
        ignore_names = {"na", "nan", "none", "no solvent", "bulk", "monomer"}

        s = df[solvent_name_col].astype(str).str.strip()
        sl = s.str.lower()
        canon = sl.map(lambda n: alias_to_canonical.get(n, n))
        canon = canon[canon.astype(bool) & ~canon.isin(ignore_names)]

        print(f"Unique solvents (canonical, excl placeholders): {canon.nunique()}")

        bp_series = canon.map(lambda n: bp_by_name.get(n))
        bp_vals = pd.to_numeric(bp_series, errors="coerce").dropna()
        if not bp_vals.empty:
            print(f"Solvent boiling points (°C): {bp_vals.min():.2f} .. {bp_vals.max():.2f} (n={len(bp_vals)})")

    # Temperature
    if "temperature" in df.columns:
        t = pd.to_numeric(df["temperature"], errors="coerce").dropna()
        if not t.empty:
            print(f"Temperature: {t.min():.2f} .. {t.max():.2f} (n={len(t)})")

    # Publication years
    py = _get_publication_year_series(df).dropna()
    if not py.empty:
        py_i = py.astype(int)
        print(f"Publication years: {py_i.min()} .. {py_i.max()} (n={len(py_i)})")
    else:
        print("Publication years: (missing `publication_year` column)")


def main() -> None:
    setup_plot_style()
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = load_data()
    print_basic_dataset_stats(df)

    fig = plt.figure(figsize=(TWO_COL_WIDTH_INCH * 1.65, TWO_COL_WIDTH_INCH * 1.05), layout="constrained")
    # Add a thin spacer row between top and bottom panels for the shared legend
    gs = fig.add_gridspec(3, 6, height_ratios=[1.15, 0.26, 0.85])

    ax_a = fig.add_subplot(gs[0, 0:3])
    ax_b = fig.add_subplot(gs[0, 3:6])
    ax_pub = fig.add_subplot(gs[2, 0:2])
    ax_temp = fig.add_subplot(gs[2, 2:4])
    ax_logp = fig.add_subplot(gs[2, 4:6])

    plot_panel_a_network(ax_a, df)
    legend_labels, class_to_color = plot_panel_b_temporal(ax_b, df, rolling_window_years=10)
    plot_panel_publication_years(ax_pub, df)
    plot_panel_c_distributions(ax_temp, ax_logp, df)

    # Shared legend for monomer classes: centered above panels A+B
    if legend_labels and class_to_color:
        handles = [
            plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=class_to_color[l], markersize=8, label=l)
            for l in legend_labels
            if l in class_to_color
        ]
        if handles:
            fig.legend(
                handles=handles,
                loc="upper center",
                # Place between the two plot rows
                bbox_to_anchor=(0.5, 0.515),
                ncol=min(5, max(1, len(handles))),
                frameon=True,
                framealpha=0.9,
                edgecolor="none",
                fontsize=12,
                handletextpad=0.5,
                columnspacing=1.0,
            )

    base = OUTPUT_DIR / "dataset_analysis"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    fig.savefig(base.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"✓ Saved {base}.pdf and {base}.png")


if __name__ == "__main__":
    main()

