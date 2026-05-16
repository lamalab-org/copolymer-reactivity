import json
import os
import time

import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from copolextractor.utils import name_to_smiles
from copolpredictor.data_processing import (
    add_molecular_features,
    add_orbital_interaction_features,
    add_solvent_features,
    create_flipped_dataset,
    process_embeddings,
)

# --- Configuration ---
TEMP_ONLY_INPUT = "temperature_only.csv"
SOLV_AND_TEMP_INPUT = "solvent_and_temp.csv"

TEMP_ONLY_OUTPUT = "augmented_temperature_only.csv"
SOLV_AND_TEMP_OUTPUT = "augmented_solvent_and_temperature.csv"

NUM_VARIANTS_PER_ROW = 5
DELAY_BETWEEN_CALLS = 1.5
OPENAI_MODEL = "gpt-4"
client = OpenAI()


# --- LLM Cache Setup ---
CACHE_FILE = "llm_cache.json"
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        llm_cache = json.load(f)
else:
    llm_cache = {}


# --- Key function ---
def make_cache_key(m1, m2, poly, solvent, vary_solvent):
    if vary_solvent:
        return f"{m1}|||{m2}|||{poly}|||VARY"
    else:
        return f"{m1}|||{m2}|||{poly}|||{solvent}"


# --- Prompt Logic ---
def generate_augmented_conditions(
    m1,
    m2,
    poly_type,
    fixed_solvent=None,
    vary_solvent=True,
    num_solvents=3,
    num_temps_per_solvent=2,
    model="gpt-4",
):
    """
    Generates augmented experimental conditions using OpenAI API.

    If vary_solvent is True:
        Returns num_solvents × num_temps_per_solvent combinations
    Else:
        Returns num_temps_per_solvent variants with fixed solvent

    Output format:
    [
        {"Solvent": "X", "Temperatures": [T1, T2]},
        ...
    ]
    """
    key = make_cache_key(m1, m2, poly_type, fixed_solvent, vary_solvent)

    if key in llm_cache:
        print(f"🔁 Loaded cached LLM result for {key}")
        return llm_cache[key]

    system_prompt = (
        "You are a polymer chemistry expert. Based on two monomers and a polymerization type, "
        "propose realistic experimental conditions. Solvents must dissolve both monomers; "
        "temperatures must be realistic for the given polymerization type."
    )

    if vary_solvent:
        user_prompt = (
            f"Monomer 1: {m1}\n"
            f"Monomer 2: {m2}\n"
            f"Polymerization type: {poly_type}\n"
            f"Original solvent: {fixed_solvent}\n"
            f"Suggest {num_solvents} similar solvents that could be used.\n"
            f"For each solvent, propose {num_temps_per_solvent} realistic reaction temperatures in Celsius.\n"
            "Format your response as JSON:\n"
            '[{"Solvent": "...", "Temperatures": [.., ..]}, ...]'
        )
    else:
        user_prompt = (
            f"Monomer 1: {m1}\n"
            f"Monomer 2: {m2}\n"
            f"Solvent: {fixed_solvent}\n"
            f"Polymerization type: {poly_type}\n"
            f"Propose {num_temps_per_solvent} realistic temperatures in Celsius for this reaction.\n"
            'Format your response as JSON list: [{{"Temperature": ...}}, ...]'
        )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
        )
        content = response.choices[0].message.content
        print(f"\n🔍 LLM response for {m1} + {m2} [{poly_type}]:\n{content}\n")

        parsed = json.loads(content)
        llm_cache[key] = parsed  # Save to in-memory cache

        # Write cache to disk
        with open(CACHE_FILE, "w") as f:
            json.dump(llm_cache, f, indent=2)

        # Save updated cache to disk
        with open(CACHE_FILE, "w") as f:
            json.dump(llm_cache, f, indent=2)

        return parsed

    except Exception as e:
        print(f"❌ Error for {m1} + {m2}: {e}")
        return []


# --- Augmentation Loop ---
def augment_dataset(df, vary_solvent=True):
    """Core loop to augment data with LLM calls."""
    augmented = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        m1 = row.get("monomer_1")
        m2 = row.get("monomer_2")
        cls = row.get("Class")
        poly = row.get("polymerization_type")
        original_solvent = row.get("Solvent")

        if pd.isna(m1) or pd.isna(m2) or pd.isna(cls) or pd.isna(poly):
            continue

        variants = generate_augmented_conditions(
            m1,
            m2,
            poly,
            fixed_solvent=original_solvent,
            vary_solvent=vary_solvent,
            num_solvents=5,
            num_temps_per_solvent=5,
        )

        smiles_m1 = name_to_smiles(m1)
        smiles_m2 = name_to_smiles(m2)
        print(f"Converted {m1} and {m2} to smiles: {smiles_m1} and {smiles_m2}")

        if vary_solvent:
            # variants: [{"Solvent": "...", "Temperatures": [.., ..]}, ...]
            for v in variants or []:
                solv = v.get("Solvent")
                temps = v.get("Temperatures", []) or []
                for T in temps:
                    new_row = {
                        "Class": cls,
                        "monomer1_name": m1,
                        "monomer2_name": m2,
                        "monomer1_smiles": smiles_m1,
                        "monomer2_smiles": smiles_m2,
                        "polymerization_type": poly,
                        "solvent": solv,
                        "solvent_smiles": name_to_smiles(solv) if pd.notna(solv) else None,
                        "temperature": T,
                        "source": "LLM_generated",
                        "method": "solvent+temperature",
                    }
                    augmented.append(new_row)
        else:
            # variants: [{"Temperature": ...}, ...]  (fixed Solvent)
            for v in variants or []:
                T = v.get("Temperature")
                new_row = {
                    "Class": cls,
                    "monomer1_name": m1,
                    "monomer2_name": m2,
                    "monomer1_smiles": smiles_m1,
                    "monomer2_smiles": smiles_m2,
                    "polymerization_type": poly,
                    "solvent": original_solvent,
                    "solvent_smiles": (
                        name_to_smiles(original_solvent) if pd.notna(original_solvent) else None
                    ),
                    "temperature": T,
                    "source": "LLM_generated",
                    "method": "temperature_only",
                }
                augmented.append(new_row)

        time.sleep(DELAY_BETWEEN_CALLS)

    df_out = pd.DataFrame(augmented)

    if "temperature" in df_out.columns:
        n_missing = df_out["temperature"].isna().sum()
        if n_missing:
            print(f"⚠️ Warnung: {n_missing} Zeilen ohne Temperatur – prüfe LLM-Ausgabe/Parsing.")
    return df_out


def preprocess_data(df):
    """
    Perform data preprocessing on a given DataFrame.

    Steps:
    - Add solvent-based features
    - Add molecular descriptors (e.g., RDKit, fingerprints)
    - Add orbital interaction features
    - Generate flipped dataset (monomer1 <-> monomer2)
    - Create unique reaction IDs (before flipping)
    """
    print("🔧 Starting preprocessing...")

    # Add solvent features
    print("➕ Adding solvent features...")
    df = add_solvent_features(df)
    print(f"✅ Shape after solvent features: {df.shape}")

    # Add molecular features
    print("➕ Adding molecular features...")
    df = add_molecular_features(df)
    print(f"✅ Shape after molecular features: {df.shape}")

    if len(df) == 0:
        print("❌ No data left after adding molecular features. Please check feature availability.")
        return None

    # Add orbital interaction features
    print("➕ Adding orbital interaction features...")
    df = add_orbital_interaction_features(df)

    # Load embedding dictionaries from JSON
    with open("../output/polytype_emb_pca_values.json", "r") as f:
        polytype_emb_dict = json.load(f)

    with open("../output/method_emb_pca_values.json", "r") as f:
        method_emb_dict = json.load(f)

    # Function to map a column to its PCA values
    def apply_pca_embeddings(df, source_col, emb_dict, prefix):
        df[f"{prefix}_pca_1"] = df[source_col].map(
            lambda x: emb_dict.get(x, {}).get("pca_1", np.nan)
        )
        df[f"{prefix}_pca_2"] = df[source_col].map(
            lambda x: emb_dict.get(x, {}).get("pca_2", np.nan)
        )
        return df

    # Replace previous process_embeddings calls
    df = apply_pca_embeddings(df, "polymerization_type", polytype_emb_dict, "polytype")
    df = apply_pca_embeddings(df, "method", method_emb_dict, "method")

    # Create reaction IDs before flipping
    print("🔑 Creating unique reaction IDs...")
    df["reaction_id"] = df.index
    print(f"✅ Created {df['reaction_id'].nunique()} unique reaction IDs")

    # Create flipped dataset
    print("🔄 Creating flipped dataset...")
    df_flipped = create_flipped_dataset(df)
    print(f"✅ Final dataset shape: {df_flipped.shape}")

    return df_flipped


# --- Main Execution ---
def main():
    print("📥 Loading input files...")
    df_temp_only = pd.read_csv(TEMP_ONLY_INPUT, sep=";", engine="python")
    df_solv_and_temp = pd.read_csv(SOLV_AND_TEMP_INPUT, sep=";", engine="python")

    print("🔥 Generating temperature-only variants...")
    df_aug_temp = augment_dataset(df_temp_only, vary_solvent=False)
    df_aug_temp.to_csv(TEMP_ONLY_OUTPUT, index=False)
    print(f"✅ Saved temperature-only variants to {TEMP_ONLY_OUTPUT}")

    print("🧪 Generating solvent + temperature variants...")
    df_aug_solv = augment_dataset(df_solv_and_temp, vary_solvent=True)
    df_aug_solv.to_csv(SOLV_AND_TEMP_OUTPUT, index=False)
    print(f"✅ Saved solvent+temperature variants to {SOLV_AND_TEMP_OUTPUT}")

    df_combined = pd.concat([df_aug_temp, df_aug_solv], ignore_index=True)
    print(f"🧪 Total combined datapoints: {len(df_combined)}")

    # === 4. Save combined file before preprocessing (optional backup) ===
    df_combined.to_csv("combined_augmented_raw.csv", index=False)

    # === 5. Save to temporary file path for processing ===
    combined_path = "combined_augmented_for_processing.csv"
    df_combined.to_csv(combined_path, index=False)
    combined_df = pd.read_csv(combined_path)

    # === 6. Run your preprocessing pipeline ===
    processed_df = preprocess_data(combined_df)

    # === 7. Save final processed file ===
    if processed_df is not None:
        processed_df.to_csv("processed_combined_augmented.csv", index=False)
        print("✅ Final processed dataset saved to: processed_combined_augmented.csv")
    else:
        print("❌ No processed data created.")


if __name__ == "__main__":
    main()
