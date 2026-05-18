#!/usr/bin/env python3
"""
Script to create individual JSON files for each reaction for database upload.
Each JSON contains all extracted/preprocessed data except monomer xtb features.
Monomer data only includes name and SMILES.
Includes PCA-reduced features for method and polymerization_type, and solvent features.
"""

import hashlib
import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available. Solvent features will not be calculated.")

try:
    import sys

    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from copolextractor import utils as copol_utils

    COPOL_UTILS_AVAILABLE = True
except ImportError:
    COPOL_UTILS_AVAILABLE = False
    print(
        "Warning: copolextractor.utils not available. IUPAC names will use monomer names from reactions."
    )


def get_smiles_md5(smiles: str) -> str:
    """Create MD5 hash from SMILES string for consistent filename."""
    return hashlib.md5(smiles.encode("utf-8")).hexdigest()


def get_iupac_name(smiles: str, fallback_name: Optional[str] = None) -> Optional[str]:
    """
    Get IUPAC name from SMILES. Falls back to monomer name from reactions if available.
    """
    if COPOL_UTILS_AVAILABLE:
        try:
            iupac_name = copol_utils.smiles_to_name(smiles)
            if iupac_name:
                return iupac_name
        except Exception:
            pass

    # Fallback to provided name if available
    if fallback_name:
        return fallback_name

    return None


def sanitize_filename(name: str) -> str:
    """
    Sanitize IUPAC name for use in filename.
    Replaces problematic characters with underscores.
    """
    if not name:
        return "unknown"

    # Replace problematic characters
    name = re.sub(r'[<>:"|?*\\/]', "_", name)
    name = re.sub(r"[^\w._-]", "_", name)
    name = re.sub(r"_+", "_", name)  # Replace multiple underscores with single
    name = name.strip("_")  # Remove leading/trailing underscores

    # Limit length
    if len(name) > 200:
        name = name[:200]

    return name if name else "unknown"


def normalize_doi(source: str) -> str:
    """
    Normalize DOI from source field to plain format.
    Returns filesystem-safe string.
    """
    if not isinstance(source, str) or not source.strip():
        return "unknown"

    s = source.strip()
    lowered = s.lower()

    # Remove common DOI prefixes
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:", "doi "):
        if lowered.startswith(prefix):
            s = s[len(prefix) :]
            break

    # Make filesystem-safe: replace / with _ and remove other problematic chars
    s = s.strip().lower()
    s = re.sub(r'[<>:"|?*\\/]', "_", s)
    s = re.sub(r"[^\w._-]", "_", s)
    s = re.sub(r"_+", "_", s)  # Replace multiple underscores with single
    s = s.strip("_")  # Remove leading/trailing underscores

    if not s:
        return "unknown"

    return s


def calculate_solvent_features(solvent_smiles: Optional[str]) -> Dict[str, Optional[float]]:
    """
    Calculate solvent features from SMILES string.
    """
    if not RDKIT_AVAILABLE:
        return {}

    def is_invalid(smiles):
        if not smiles or not isinstance(smiles, str):
            return True
        smiles_clean = smiles.strip().lower()
        return smiles_clean in {"", "na", "nan", "none"}

    if is_invalid(solvent_smiles):
        return {
            "solvent_logP": None,
            "solvent_TPSA": None,
            "solvent_HBA": None,
            "solvent_HBD": None,
            "solvent_FractionCSP3": None,
            "solvent_MolMR": None,
            "solvent_LabuteASA": None,
            "solvent_NumRotatableBonds": None,
            "solvent_RingCount": None,
            "solvent_HeavyAtomCount": None,
        }

    try:
        mol = Chem.MolFromSmiles(solvent_smiles)
        if mol is None:
            return {
                "solvent_logP": None,
                "solvent_TPSA": None,
                "solvent_HBA": None,
                "solvent_HBD": None,
                "solvent_FractionCSP3": None,
                "solvent_MolMR": None,
                "solvent_LabuteASA": None,
                "solvent_NumRotatableBonds": None,
                "solvent_RingCount": None,
                "solvent_HeavyAtomCount": None,
            }

        return {
            "solvent_logP": float(Descriptors.MolLogP(mol)),
            "solvent_TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
            "solvent_HBA": float(rdMolDescriptors.CalcNumHBA(mol)),
            "solvent_HBD": float(rdMolDescriptors.CalcNumHBD(mol)),
            "solvent_FractionCSP3": float(Descriptors.FractionCSP3(mol)),
            "solvent_MolMR": float(Descriptors.MolMR(mol)),
            "solvent_LabuteASA": float(rdMolDescriptors.CalcLabuteASA(mol)),
            "solvent_NumRotatableBonds": float(Descriptors.NumRotatableBonds(mol)),
            "solvent_RingCount": float(Descriptors.RingCount(mol)),
            "solvent_HeavyAtomCount": float(Descriptors.HeavyAtomCount(mol)),
        }
    except Exception:
        return {
            "solvent_logP": None,
            "solvent_TPSA": None,
            "solvent_HBA": None,
            "solvent_HBD": None,
            "solvent_FractionCSP3": None,
            "solvent_MolMR": None,
            "solvent_LabuteASA": None,
            "solvent_NumRotatableBonds": None,
            "solvent_RingCount": None,
            "solvent_HeavyAtomCount": None,
        }


def load_pca_mappings(polytype_path: str, method_path: str) -> tuple:
    """Load PCA value mappings for polymerization_type and method."""
    polytype_map = {}
    method_map = {}

    try:
        with open(polytype_path, "r") as f:
            polytype_map = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load polytype PCA mappings: {e}")

    try:
        with open(method_path, "r") as f:
            method_map = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load method PCA mappings: {e}")

    return polytype_map, method_map


def is_valid_smiles(smiles: Any) -> bool:
    """
    Check if SMILES is valid (not None, not 'none', not empty).
    """
    if smiles is None:
        return False
    if isinstance(smiles, str):
        smiles = smiles.strip().lower()
        return smiles not in ("", "none", "na", "nan", "null")
    return True


def is_valid_monomer_data(monomer_name: Any, monomer_smiles: Any) -> bool:
    """
    Check if monomer has valid SMILES (required - name alone is not enough).
    """
    return is_valid_smiles(monomer_smiles)


def clean_reaction_data(
    reaction_data: Dict[str, Any],
    polytype_pca_map: Dict[str, Dict],
    method_pca_map: Dict[str, Dict],
) -> Optional[Dict[str, Any]]:
    """
    Clean reaction data by removing xtb features from monomers.
    Keeps only name and SMILES for monomers.
    Adds PCA features for method and polymerization_type.
    Adds solvent features.
    Removes 'file' field.
    Returns None if reaction is invalid (missing monomer data).
    """
    if not isinstance(reaction_data, dict):
        return None

    cleaned = {}

    # Copy all top-level fields except monomer_data fields and 'file'
    # Skip None values and invalid strings for monomer fields
    invalid_values = ("none", "na", "nan", "null", "")
    for key, value in reaction_data.items():
        if key not in ["monomer1_data", "monomer2_data", "file"]:
            # Skip None values and invalid strings for monomer fields
            if key in ["monomer1_s", "monomer2_s", "monomer1", "monomer2"]:
                if value is not None:
                    if isinstance(value, str):
                        if value.strip().lower() not in invalid_values:
                            cleaned[key] = value
                    else:
                        cleaned[key] = value
            else:
                cleaned[key] = value

    # Get monomer names
    monomer1_name = cleaned.get("monomer1") or reaction_data.get("monomer1")
    monomer2_name = cleaned.get("monomer2") or reaction_data.get("monomer2")

    # Get monomer SMILES (prefer monomer1_s/monomer2_s, fallback to monomer_data)
    monomer1_smiles = None
    if "monomer1_s" in reaction_data and reaction_data["monomer1_s"] is not None:
        monomer1_smiles = reaction_data["monomer1_s"]
    elif "monomer1_data" in reaction_data and isinstance(reaction_data["monomer1_data"], dict):
        if "smiles" in reaction_data["monomer1_data"]:
            monomer1_smiles = reaction_data["monomer1_data"]["smiles"]

    monomer2_smiles = None
    if "monomer2_s" in reaction_data and reaction_data["monomer2_s"] is not None:
        monomer2_smiles = reaction_data["monomer2_s"]
    elif "monomer2_data" in reaction_data and isinstance(reaction_data["monomer2_data"], dict):
        if "smiles" in reaction_data["monomer2_data"]:
            monomer2_smiles = reaction_data["monomer2_data"]["smiles"]

    # Validate that both monomers have valid SMILES (required - not optional)
    if not is_valid_smiles(monomer1_smiles):
        return None  # Invalid: missing monomer1 SMILES

    if not is_valid_smiles(monomer2_smiles):
        return None  # Invalid: missing monomer2 SMILES

    # Add validated monomer data (only add if not None/empty)
    # Overwrite with validated values
    if monomer1_name:
        cleaned["monomer1"] = monomer1_name
    elif "monomer1" in cleaned:
        # Keep existing if valid
        pass
    else:
        # Remove if invalid
        cleaned.pop("monomer1", None)

    if monomer2_name:
        cleaned["monomer2"] = monomer2_name
    elif "monomer2" in cleaned:
        # Keep existing if valid
        pass
    else:
        # Remove if invalid
        cleaned.pop("monomer2", None)

    # Remove any old SMILES fields first
    for field in ["monomer1_smiles", "monomer2_smiles", "monomer1_s", "monomer2_s"]:
        cleaned.pop(field, None)

    # Add validated SMILES (both are guaranteed to be valid at this point)
    cleaned["monomer1_smiles"] = monomer1_smiles
    cleaned["monomer2_smiles"] = monomer2_smiles

    # Add PCA features for polymerization_type
    polytype = cleaned.get("polymerization_type")
    if polytype and polytype in polytype_pca_map:
        pca_data = polytype_pca_map[polytype]
        cleaned["polytype_emb_1"] = pca_data.get("pca_1")
        cleaned["polytype_emb_2"] = pca_data.get("pca_2")

    # Add PCA features for method
    method = cleaned.get("method")
    if method and method in method_pca_map:
        pca_data = method_pca_map[method]
        cleaned["method_emb_1"] = pca_data.get("pca_1")
        cleaned["method_emb_2"] = pca_data.get("pca_2")

    # Add solvent features
    solvent_smiles = cleaned.get("solvent")
    if solvent_smiles:
        solvent_features = calculate_solvent_features(solvent_smiles)
        cleaned.update(solvent_features)

    # If logP exists but solvent_logP doesn't, use it (for backward compatibility)
    if "logP" in cleaned and "solvent_logP" not in cleaned:
        cleaned["solvent_logP"] = cleaned.get("logP")

    return cleaned


def copy_monomer_files(
    reactions_by_doi: Dict[str, List],
    monomer_source_dir: Path,
    monomer_output_dir: Path,
    reaction_output_dir: Path,
):
    """
    Copy and rename monomer JSON files from source to output directory.
    Uses IUPAC names from reactions or calculates them from SMILES.
    """
    monomer_output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all unique monomer SMILES and their names from reactions
    monomer_info = {}  # {smiles: {'name': name, 'iupac': iupac}}

    print("\nCollecting monomer information from reactions...")
    for doi, reactions in reactions_by_doi.items():
        for reaction_info in reactions:
            reaction_data = reaction_info["data"]

            # Get monomer1 info
            m1_smiles = None
            if "monomer1_s" in reaction_data and reaction_data["monomer1_s"]:
                m1_smiles = reaction_data["monomer1_s"]
            elif "monomer1_data" in reaction_data and isinstance(
                reaction_data["monomer1_data"], dict
            ):
                m1_smiles = reaction_data["monomer1_data"].get("smiles")

            m1_name = reaction_data.get("monomer1")

            if m1_smiles and is_valid_smiles(m1_smiles):
                if m1_smiles not in monomer_info:
                    monomer_info[m1_smiles] = {"name": m1_name, "iupac": None}
                elif not monomer_info[m1_smiles]["name"] and m1_name:
                    monomer_info[m1_smiles]["name"] = m1_name

            # Get monomer2 info
            m2_smiles = None
            if "monomer2_s" in reaction_data and reaction_data["monomer2_s"]:
                m2_smiles = reaction_data["monomer2_s"]
            elif "monomer2_data" in reaction_data and isinstance(
                reaction_data["monomer2_data"], dict
            ):
                m2_smiles = reaction_data["monomer2_data"].get("smiles")

            m2_name = reaction_data.get("monomer2")

            if m2_smiles and is_valid_smiles(m2_smiles):
                if m2_smiles not in monomer_info:
                    monomer_info[m2_smiles] = {"name": m2_name, "iupac": None}
                elif not monomer_info[m2_smiles]["name"] and m2_name:
                    monomer_info[m2_smiles]["name"] = m2_name

    print(f"Found {len(monomer_info)} unique monomers")

    # Also check created reaction files for monomer names
    print("Reading monomer names from created reaction files...")
    for json_file in reaction_output_dir.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            m1_smiles = data.get("monomer1_smiles")
            m1_name = data.get("monomer1")
            if m1_smiles and m1_smiles in monomer_info:
                if not monomer_info[m1_smiles]["name"] and m1_name:
                    monomer_info[m1_smiles]["name"] = m1_name

            m2_smiles = data.get("monomer2_smiles")
            m2_name = data.get("monomer2")
            if m2_smiles and m2_smiles in monomer_info:
                if not monomer_info[m2_smiles]["name"] and m2_name:
                    monomer_info[m2_smiles]["name"] = m2_name
        except Exception:
            continue

    # Get IUPAC names for monomers (use name from reactions as primary, only lookup if missing)
    print("Preparing monomer names...")
    for smiles, info in monomer_info.items():
        if info["name"]:
            # Use name from reactions directly (often already IUPAC or common name)
            info["iupac"] = info["name"]
        else:
            # Only try to get IUPAC name if we don't have a name
            # This is slow, so we skip it for now and use SMILES-based name
            info["iupac"] = f"monomer_{smiles[:20]}"  # Use SMILES as fallback

    # Copy and rename monomer files
    print(f"\nCopying monomer files from {monomer_source_dir}...")
    copied_count = 0
    not_found_count = 0

    for smiles, info in monomer_info.items():
        # Find source file by MD5 hash
        md5_hash = get_smiles_md5(smiles)
        source_file = monomer_source_dir / f"{md5_hash}.json"

        if not source_file.exists():
            not_found_count += 1
            continue

        # Create target filename
        iupac_name = sanitize_filename(info["iupac"])
        target_file = monomer_output_dir / f"monomer_{iupac_name}.json"

        # Handle duplicate names (add index)
        if target_file.exists():
            counter = 1
            base_name = iupac_name
            while target_file.exists():
                target_file = monomer_output_dir / f"monomer_{base_name}_{counter}.json"
                counter += 1

        # Copy file
        try:
            shutil.copy2(source_file, target_file)
            copied_count += 1
        except Exception as e:
            print(f"Error copying {source_file} to {target_file}: {e}")

    print(f"Copied {copied_count} monomer files")
    if not_found_count > 0:
        print(f"Warning: {not_found_count} monomers not found in source directory")

    return copied_count


def process_reactions(
    input_dir: Path,
    output_dir: Path,
    polytype_pca_path: Optional[str] = None,
    method_pca_path: Optional[str] = None,
    monomer_source_dir: Optional[Path] = None,
    create_monomer_files: bool = True,
):
    """
    Process all reaction JSON files and create cleaned versions for database.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load PCA mappings
    if polytype_pca_path is None:
        polytype_pca_path = "copol_prediction/api/data/polytype_emb_pca_values.json"
    if method_pca_path is None:
        method_pca_path = "copol_prediction/api/data/method_emb_pca_values.json"

    print(f"Loading PCA mappings...")
    polytype_pca_map, method_pca_map = load_pca_mappings(polytype_pca_path, method_pca_path)
    print(f"  Loaded {len(polytype_pca_map)} polytype mappings")
    print(f"  Loaded {len(method_pca_map)} method mappings")

    # Group reactions by DOI to assign indices
    reactions_by_doi = defaultdict(list)

    # First pass: collect all reactions grouped by DOI
    json_files = list(input_dir.glob("*.json"))
    print(f"\nFound {len(json_files)} JSON files to process")

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Handle case where file contains a list of reactions
            if isinstance(data, list):
                for reaction_data in data:
                    if isinstance(reaction_data, dict):
                        source = reaction_data.get("source", "")
                        normalized_doi = normalize_doi(source)
                        reactions_by_doi[normalized_doi].append(
                            {"file": json_file, "data": reaction_data}
                        )
            elif isinstance(data, dict):
                source = data.get("source", "")
                normalized_doi = normalize_doi(source)
                reactions_by_doi[normalized_doi].append({"file": json_file, "data": data})
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
            continue

    print(f"Found {len(reactions_by_doi)} unique DOIs")

    # Second pass: create cleaned JSON files
    total_created = 0
    total_skipped = 0
    skipped_reactions = []

    for normalized_doi, reactions in reactions_by_doi.items():
        for idx, reaction_info in enumerate(reactions, start=1):
            reaction_data = reaction_info["data"]
            cleaned_data = clean_reaction_data(reaction_data, polytype_pca_map, method_pca_map)

            # Skip invalid reactions
            if cleaned_data is None:
                total_skipped += 1
                source = reaction_data.get("source", "unknown")
                skipped_reactions.append(
                    {
                        "doi": normalized_doi,
                        "source": source,
                        "reason": "Missing monomer name or SMILES",
                    }
                )
                continue

            # Create filename: polymerization_<doi>_<index>.json
            if len(reactions) == 1:
                filename = f"polymerization_{normalized_doi}.json"
            else:
                filename = f"polymerization_{normalized_doi}_{idx}.json"

            output_path = output_dir / filename

            # Save cleaned JSON
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

            total_created += 1

    print(f"\nCreated {total_created} cleaned JSON files in {output_dir}")
    if total_skipped > 0:
        print(f"Skipped {total_skipped} invalid reactions (missing monomer data)")
        if len(skipped_reactions) <= 10:
            print("\nSkipped reactions:")
            for rxn in skipped_reactions:
                print(f"  - {rxn['doi']}: {rxn['reason']}")
        else:
            print(f"\nFirst 10 skipped reactions:")
            for rxn in skipped_reactions[:10]:
                print(f"  - {rxn['doi']}: {rxn['reason']}")

    # Validate all created files
    print("\nValidating created files...")
    validation_errors = validate_created_files(output_dir)
    if validation_errors:
        print(f"\n⚠️  Found {len(validation_errors)} files with validation errors:")
        for error in validation_errors[:10]:
            print(f"  - {error}")
        if len(validation_errors) > 10:
            print(f"  ... and {len(validation_errors) - 10} more")
    else:
        print("✅ All files validated successfully!")

    # Copy monomer files if requested
    if create_monomer_files and monomer_source_dir and monomer_source_dir.exists():
        monomer_output_dir = output_dir / "monomers"
        copy_monomer_files(reactions_by_doi, monomer_source_dir, monomer_output_dir, output_dir)
    elif create_monomer_files:
        print(f"\n⚠️  Monomer source directory not found: {monomer_source_dir}")
        print("   Skipping monomer file creation.")

    return total_created, reactions_by_doi


def validate_created_files(output_dir: Path) -> List[str]:
    """
    Validate all created JSON files to ensure they have valid monomer data.
    Returns list of files with errors.
    """
    errors = []
    json_files = list(output_dir.glob("*.json"))

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Check monomer1 SMILES (required)
            monomer1_smiles = data.get("monomer1_smiles") or data.get("monomer1_s")
            if not is_valid_smiles(monomer1_smiles):
                errors.append(f"{json_file.name}: Missing or invalid monomer1 SMILES")

            # Check monomer2 SMILES (required)
            monomer2_smiles = data.get("monomer2_smiles") or data.get("monomer2_s")
            if not is_valid_smiles(monomer2_smiles):
                errors.append(f"{json_file.name}: Missing or invalid monomer2 SMILES")

            # Check for None or invalid values in SMILES fields (required)
            for field in ["monomer1_smiles", "monomer2_smiles", "monomer1_s", "monomer2_s"]:
                if field in data:
                    value = data[field]
                    if value is None:
                        errors.append(f"{json_file.name}: {field} is None")
                    elif isinstance(value, str) and value.strip().lower() in (
                        "none",
                        "na",
                        "nan",
                        "null",
                        "",
                    ):
                        errors.append(f"{json_file.name}: {field} is invalid: '{value}'")

        except Exception as e:
            errors.append(f"{json_file.name}: Error reading file: {e}")

    return errors


if __name__ == "__main__":
    # Set paths (relative to project root, script is in database/ folder)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    input_directory = project_root / "dump/processed_reactions"
    output_directory = project_root / "dump/database_json"
    polytype_pca_path = str(project_root / "copol_prediction/api/data/polytype_emb_pca_values.json")
    method_pca_path = str(project_root / "copol_prediction/api/data/method_emb_pca_values.json")
    monomer_source_directory = project_root / "copol_prediction/api/molecule_properties"

    print("Starting database JSON creation...")
    print(f"Input directory: {input_directory}")
    print(f"Output directory: {output_directory}")
    print(f"Polytype PCA path: {polytype_pca_path}")
    print(f"Method PCA path: {method_pca_path}")
    print(f"Monomer source directory: {monomer_source_directory}")
    print()

    process_reactions(
        input_directory,
        output_directory,
        polytype_pca_path,
        method_pca_path,
        monomer_source_directory,
        create_monomer_files=True,
    )
    print("\nDone!")
