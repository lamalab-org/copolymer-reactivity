import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import streamlit as st

# === your modules ===
from copolpredictor.data_processing import (
    load_molecular_data, molecular_features,
    add_orbital_interaction_features
)
import copolextractor.utils as cx_utils

BUNDLE_DIR = Path("../copol_prediction/artifacts/model_bundle")
MODEL_PATH = BUNDLE_DIR / "model.joblib"
META_PATH  = BUNDLE_DIR / "meta.json"

st.set_page_config(page_title="Copol Predict – Inference", layout="centered")
st.title("Copol Predict – Inference")

@st.cache_resource
def load_bundle():
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    return model, meta["feature_columns"], meta.get("class_labels", [0,1,2])

model, FEATURE_COLUMNS, CLASS_LABELS = load_bundle()
st.caption(f"Loaded {len(FEATURE_COLUMNS)} features | Classes: {CLASS_LABELS}")

# ---- load PCA maps produced during training (by process_embeddings) ----
@st.cache_resource
def load_pca_maps():
    method_map_path = Path("output/method_emb_pca_values.json")
    polytype_map_path = Path("output/polytype_emb_pca_values.json")
    method_pca = json.loads(method_map_path.read_text()) if method_map_path.exists() else {}
    polytype_pca = json.loads(polytype_map_path.read_text()) if polytype_map_path.exists() else {}
    return method_pca, polytype_pca

METHOD_PCA_MAP, POLYTYPE_PCA_MAP = load_pca_maps()

# ---- small solvent featurizer (same columns as your add_solvent_features) ----
def solvent_smiles_from_name(solvent_name: str | None) -> str | None:
    if not solvent_name:
        return None
    try:
        smi = cx_utils.name_to_smiles(solvent_name, force_retry=True)
        return smi
    except Exception:
        return None

def solvent_rdkit_features(solvent_smiles: str | None) -> dict:
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors
    except Exception:
        # RDKit not installed: return empty; alignment will fill with zeros
        return {}
    if not solvent_smiles:
        return {}
    mol = Chem.MolFromSmiles(solvent_smiles)
    if mol is None:
        return {}
    return {
        'solvent_logP': Descriptors.MolLogP(mol),
        'solvent_TPSA': rdMolDescriptors.CalcTPSA(mol),
        'solvent_HBA': rdMolDescriptors.CalcNumHBA(mol),
        'solvent_HBD': rdMolDescriptors.CalcNumHBD(mol),
        'solvent_FractionCSP3': Descriptors.FractionCSP3(mol),
        'solvent_MolMR': Descriptors.MolMR(mol),
        'solvent_LabuteASA': rdMolDescriptors.CalcLabuteASA(mol),
        'solvent_NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
        'solvent_RingCount': Descriptors.RingCount(mol),
        'solvent_HeavyAtomCount': Descriptors.HeavyAtomCount(mol),
    }

# ---- Feature builder using YOUR preprocessing logic ----
class FeatureBuilder:
    def __init__(self, feature_columns: list[str]):
        self.feature_columns = feature_columns

    def _monomer_dict_from_smiles(self, smiles: str, suffix: str) -> dict:
        """
        Load per-monomer JSON feature file and keep only numeric entries,
        then suffix keys with _1/_2 like in training.
        """
        d = load_molecular_data(smiles)  # returns big dict incl. lists we trimmed in training
        if d is None:
            return {}

        # keep numeric only (your molecular_features() does this, too)
        keep = {k: v for k, v in d.items() if isinstance(v, float)}
        # rename to *_1 / *_2
        return {f"{k}_{suffix}": v for k, v in keep.items()}

    def _orbital_deltas(self, row_like: dict) -> dict:
        """
        Your add_orbital_interaction_features expects a DataFrame with columns:
        homo_1, lumo_1, homo_2, lumo_2. Those should be present in the monomer JSONs.
        We’ll compute deltas and return them as dict.
        """
        df = pd.DataFrame([row_like])
        try:
            df = add_orbital_interaction_features(df)
        except Exception:
            # if keys are not available, just return nothing
            return {}
        return {
            "delta_HOMO_LUMO_AA": df.loc[0, "delta_HOMO_LUMO_AA"] if "delta_HOMO_LUMO_AA" in df else 0.0,
            "delta_HOMO_LUMO_AB": df.loc[0, "delta_HOMO_LUMO_AB"] if "delta_HOMO_LUMO_AB" in df else 0.0,
            "delta_HOMO_LUMO_BB": df.loc[0, "delta_HOMO_LUMO_BB"] if "delta_HOMO_LUMO_BB" in df else 0.0,
            "delta_HOMO_LUMO_BA": df.loc[0, "delta_HOMO_LUMO_BA"] if "delta_HOMO_LUMO_BA" in df else 0.0,
        }

    def _embed_with_pca(self, value: str | None, pca_map: dict, prefix: str) -> dict:
        if not value:
            return {f"{prefix}_1": 0.0, f"{prefix}_2": 0.0}
        v = value.strip()
        entry = pca_map.get(v) or pca_map.get(v.lower())
        if not entry:
            # unseen → neutral zeros (alternatively, you could reject)
            return {f"{prefix}_1": 0.0, f"{prefix}_2": 0.0}
        return {f"{prefix}_1": float(entry["pca_1"]), f"{prefix}_2": float(entry["pca_2"])}

    def build(self, monomer_a_name: str, monomer_b_name: str,
              solvent_name: str | None, temperature_c: float | None,
              method: str | None, poly_type: str | None) -> pd.DataFrame:
        if not monomer_a_name or not monomer_b_name:
            raise ValueError("Both monomer names are required.")

        # 1) names → SMILES (your cached resolver)
        smi_a = cx_utils.name_to_smiles(monomer_a_name, force_retry=True)
        smi_b = cx_utils.name_to_smiles(monomer_b_name, force_retry=True)
        if not smi_a or not smi_b:
            raise ValueError("Could not resolve one or both monomer names to SMILES. "
                             "Ensure the names are resolvable or pre-cache them.")

        # 2) per-monomer features from your JSON property files
        a_feats = self._monomer_dict_from_smiles(smi_a, "1")
        b_feats = self._monomer_dict_from_smiles(smi_b, "2")

        # 3) orbital deltas (needs homo_1/lumo_1/homo_2/lumo_2 fields)
        merged_for_orbitals = {}
        merged_for_orbitals.update({k.replace("_1", ""): v for k, v in a_feats.items() if k.endswith("_1")})
        merged_for_orbitals.update({k.replace("_2", ""): v for k, v in b_feats.items() if k.endswith("_2")})
        orbital = self._orbital_deltas(merged_for_orbitals)

        # 4) solvent: resolve to SMILES then RDKit descriptors (same names as in training)
        solv_smi = solvent_smiles_from_name(solvent_name) if solvent_name else None
        solv_feats = solvent_rdkit_features(solv_smi)

        # 5) method/type embeddings via precomputed PCA maps
        method_feats = self._embed_with_pca(method, METHOD_PCA_MAP, "method_emb")
        type_feats   = self._embed_with_pca(poly_type, POLYTYPE_PCA_MAP, "polytype_emb")

        # 6) temperature feature (use same column name you used in training, e.g., 'temperature')
        temp_feats = {"temperature": float(temperature_c)} if temperature_c is not None else {"temperature": 0.0}

        # 7) combine all features we have
        raw = {}
        raw.update(a_feats)
        raw.update(b_feats)
        raw.update(orbital)
        raw.update(solv_feats)
        raw.update(method_feats)
        raw.update(type_feats)
        raw.update(temp_feats)

        # 8) align to model’s expected columns; fill missing with 0.0
        aligned = {col: float(raw.get(col, 0.0)) for col in self.feature_columns}
        return pd.DataFrame([aligned])

FB = FeatureBuilder(FEATURE_COLUMNS)

# ---------------- UI ----------------
st.subheader("Inputs")
with st.form("prediction_form"):
    monomer_a = st.text_input("Monomer A (required)", placeholder="e.g., styrene")
    monomer_b = st.text_input("Monomer B (required)", placeholder="e.g., methyl methacrylate")

    col1, col2 = st.columns(2)
    with col1:
        solvent = st.text_input("Solvent (optional)", placeholder="e.g., toluene / DMF / THF")
        method  = st.text_input("Polymerization method (optional)", placeholder="e.g., RAFT / ATRP / FRP")
    with col2:
        temperature = st.number_input("Temperature (°C, optional)", value=0.0, step=1.0, format="%.1f")
        poly_type = st.text_input("Polymerization type (optional)", placeholder="e.g., radical / anionic")

    submit = st.form_submit_button("Predict", type="primary")

if submit:
    try:
        # Turn empty strings into None for optional fields
        solvent_in = solvent or None
        method_in = method or None
        polytype_in = poly_type or None
        temp_in = None if (temperature is None) else float(temperature)

        feats = FB.build(
            monomer_a_name=monomer_a,
            monomer_b_name=monomer_b,
            solvent_name=solvent_in,
            temperature_c=temp_in,
            method=method_in,
            poly_type=polytype_in
        )

        y = model.predict(feats)[0]
        proba = model.predict_proba(feats)[0] if hasattr(model, "predict_proba") else None
        conf = float(np.max(proba)) if proba is not None else None

        st.success(f"Predicted class: {int(y)}")
        if proba is not None:
            st.write("Class probabilities:", {str(i): float(p) for i, p in enumerate(proba)})
            st.metric("Confidence", f"{conf:.2f}")

        with st.expander("Debug: aligned feature vector"):
            st.dataframe(feats.T.rename(columns={0: "value"}))

    except Exception as e:
        st.error(f"{type(e).__name__}: {e}")
