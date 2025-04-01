import os
import json
from tqdm import tqdm
from morfeus.conformer import ConformerEnsemble
from morfeus import XTB
from copolextractor.mongodb_storage import CoPolymerDB
from pymongo import MongoClient


def calculate_missing_properties(smiles, existing_properties):
    """Calculate only missing properties for a molecule."""
    missing_properties = {}

    if (
            "best_conformer_elements" in existing_properties
            and "best_conformer_coordinates" in existing_properties
    ):
        elements = existing_properties["best_conformer_elements"]
        coordinates = existing_properties["best_conformer_coordinates"]
    else:
        print(f"Best conformer data is missing for {smiles}. Calculating it.")
        elements, coordinates, energy = conformer_opt(smiles)
        missing_properties.update(
            {
                "best_conformer_elements": elements,
                "best_conformer_coordinates": coordinates,
                "best_conformer_energy": energy,
            }
        )

    xtb = XTB(elements, coordinates)

    # Calculate and store missing properties
    xtb_properties = [
        ("ip", xtb.get_ip),
        ("ip_corrected", lambda: xtb.get_ip(corrected=True)),
        ("ea", xtb.get_ea),
        ("homo", xtb.get_homo),
        ("lumo", xtb.get_lumo),
        ("charges", xtb.get_charges),
        ("dipole", lambda: xtb.get_dipole().tolist()),
        (
            "global_electrophilicity",
            lambda: xtb.get_global_descriptor("electrophilicity", corrected=True),
        ),
        (
            "global_nucleophilicity",
            lambda: xtb.get_global_descriptor("nucleophilicity", corrected=True),
        ),
        ("fukui_electrophilicity", lambda: xtb.get_fukui("electrophilicity")),
        ("fukui_nucleophilicity", lambda: xtb.get_fukui("nucleophilicity")),
        ("fukui_radical", lambda: xtb.get_fukui("radical")),
    ]

    for prop, func in xtb_properties:
        if prop not in existing_properties:
            try:
                missing_properties[prop] = func()
            except Exception as e:
                print(f"Error calculating {prop} for {smiles}: {e}")

    return missing_properties


def conformer_opt(smiles):
    """Optimize conformer for a given SMILES."""
    ce = ConformerEnsemble.from_rdkit(smiles, optimize="MMFF94")
    ce.prune_rmsd()
    ce.sort()

    try:
        print("Started GFN-FF optimization.")
        ce.optimize_qc_engine(
            program="xtb", model={"method": "GFN-FF"}, procedure="geometric"
        )
    except Exception as e:
        print(f"GFN-FF optimization failed for {smiles}: {e}")
        ce = ConformerEnsemble.from_rdkit(smiles, optimize="MMFF94")
        ce.prune_rmsd()
        ce.sort()

    try:
        print("Started GFN2-xTB optimization.")
        ce.optimize_qc_engine(
            program="xtb", model={"method": "GFN2-xTB"}, procedure="geometric"
        )
    except Exception as e:
        print(f"GFN2-xTB optimization failed for {smiles}: {e}")

    ce.sp_qc_engine(program="xtb", model={"method": "GFN2-xTB"})
    best_conformer = ce.conformers[0]

    elements = best_conformer.elements.tolist()
    coordinates = best_conformer.coordinates.tolist()
    energy = best_conformer.energy

    return elements, coordinates, energy


def calculate_property(smiles):
    """Calculate all properties for a given SMILES."""
    elements, coordinates, energy = conformer_opt(smiles)
    xtb = XTB(elements, coordinates)

    properties = {
        "smiles": smiles,
        "best_conformer_coordinates": coordinates,
        "best_conformer_elements": elements,
        "best_conformer_energy": energy,
        "ip": xtb.get_ip(),
        "ip_corrected": xtb.get_ip(corrected=True),
        "ea": xtb.get_ea(),
        "homo": xtb.get_homo(),
        "lumo": xtb.get_lumo(),
        "charges": xtb.get_charges(),
        "dipole": xtb.get_dipole().tolist(),
        "global_electrophilicity": xtb.get_global_descriptor(
            "electrophilicity", corrected=True
        ),
        "global_nucleophilicity": xtb.get_global_descriptor(
            "nucleophilicity", corrected=True
        ),
        "fukui_electrophilicity": xtb.get_fukui("electrophilicity"),
        "fukui_nucleophilicity": xtb.get_fukui("nucleophilicity"),
        "fukui_radical": xtb.get_fukui("radical"),
    }

    return properties


def sanitize_for_mongodb(data):
    """
    Recursively convert a dictionary's numeric keys to strings and handle numpy arrays
    to make it MongoDB compatible.
    """
    if isinstance(data, dict):
        return {
            str(key): sanitize_for_mongodb(value)
            for key, value in data.items()
        }
    elif isinstance(data, list):
        return [sanitize_for_mongodb(item) for item in data]
    elif hasattr(data, 'tolist'):  # Handle numpy arrays
        return data.tolist()
    return data


def update_mongodb_entries(db: CoPolymerDB, updated_data: dict):
    """Update MongoDB entries with calculated molecular properties."""
    updated_count = 0
    for entry in db.collection.find():
        monomer1_smiles = entry.get("monomer1_s")
        monomer2_smiles = entry.get("monomer2_s")

        update_fields = {}

        # Add monomer1 properties if available
        if monomer1_smiles and monomer1_smiles in updated_data:
            update_fields["monomer1_data"] = sanitize_for_mongodb(
                updated_data[monomer1_smiles]
            )

        # Add monomer2 properties if available
        if monomer2_smiles and monomer2_smiles in updated_data:
            update_fields["monomer2_data"] = sanitize_for_mongodb(
                updated_data[monomer2_smiles]
            )

        if update_fields:
            try:
                db.collection.update_one(
                    {"_id": entry["_id"]},
                    {"$set": update_fields}
                )
                updated_count += 1
            except Exception as e:
                print(f"Error updating entry {entry['_id']}: {e}")
                print(f"Problematic update fields: {update_fields}")
                continue

    print(f"Updated {updated_count} entries in MongoDB")


def save_properties(properties, filename):
    """Save properties to a file."""
    with open(filename, "w") as f:
        json.dump(properties, f, indent=4)


def analyze_mongodb_updates(db: CoPolymerDB, updated_data: dict):
    """
    Analyze the MongoDB update process to verify property assignment.
    """
    analysis = {
        "total_entries": 0,
        "entries_with_monomer1": 0,
        "entries_with_monomer2": 0,
        "entries_with_both": 0,
        "monomer1_with_properties": 0,
        "monomer2_with_properties": 0,
        "missing_properties": {
            "monomer1": [],
            "monomer2": []
        },
        "smiles_not_in_updated_data": {
            "monomer1": set(),
            "monomer2": set()
        }
    }

    # Analyze all entries
    for entry in db.collection.find():
        analysis["total_entries"] += 1

        monomer1_smiles = entry.get("monomer1_s")
        monomer2_smiles = entry.get("monomer2_s")

        # Count entries with monomers
        if monomer1_smiles:
            analysis["entries_with_monomer1"] += 1
        if monomer2_smiles:
            analysis["entries_with_monomer2"] += 1
        if monomer1_smiles and monomer2_smiles:
            analysis["entries_with_both"] += 1

        # Check if properties were calculated
        if monomer1_smiles:
            if monomer1_smiles in updated_data:
                analysis["monomer1_with_properties"] += 1
            else:
                analysis["smiles_not_in_updated_data"]["monomer1"].add(monomer1_smiles)

        if monomer2_smiles:
            if monomer2_smiles in updated_data:
                analysis["monomer2_with_properties"] += 1
            else:
                analysis["smiles_not_in_updated_data"]["monomer2"].add(monomer2_smiles)

        # Check for missing properties in the database
        if "monomer1_data" in entry:
            properties_to_check = [
                "best_conformer_elements",
                "best_conformer_coordinates",
                "best_conformer_energy",
                "ip",
                "ip_corrected",
                "ea",
                "homo",
                "lumo",
                "charges",
                "dipole",
                "global_electrophilicity",
                "global_nucleophilicity",
                "fukui_electrophilicity",
                "fukui_nucleophilicity",
                "fukui_radical"
            ]

            for prop in properties_to_check:
                if prop not in entry.get("monomer1_data", {}):
                    analysis["missing_properties"]["monomer1"].append({
                        "smiles": monomer1_smiles,
                        "missing_prop": prop
                    })

            if "monomer2_data" in entry:
                for prop in properties_to_check:
                    if prop not in entry.get("monomer2_data", {}):
                        analysis["missing_properties"]["monomer2"].append({
                            "smiles": monomer2_smiles,
                            "missing_prop": prop
                        })

    # Print analysis
    print("\n=== MongoDB Update Analysis ===")
    print(f"Total entries in database: {analysis['total_entries']}")
    print(f"Entries with monomer1: {analysis['entries_with_monomer1']}")
    print(f"Entries with monomer2: {analysis['entries_with_monomer2']}")
    print(f"Entries with both monomers: {analysis['entries_with_both']}")
    print(f"\nMonomer1 entries with properties: {analysis['monomer1_with_properties']}")
    print(f"Monomer2 entries with properties: {analysis['monomer2_with_properties']}")

    print("\nSMILES strings not found in updated_data:")
    print(f"Monomer1: {len(analysis['smiles_not_in_updated_data']['monomer1'])}")
    print(f"Monomer2: {len(analysis['smiles_not_in_updated_data']['monomer2'])}")

    print("\nEntries with missing properties:")
    print(f"Monomer1: {len(analysis['missing_properties']['monomer1'])}")
    print(f"Monomer2: {len(analysis['missing_properties']['monomer2'])}")

    return analysis


def update_mongodb_entries_safe(db: CoPolymerDB, updated_data: dict):
    """Update MongoDB entries with calculated molecular properties with additional safety checks."""
    updated_count = 0
    skipped_count = 0
    error_count = 0

    for entry in db.collection.find():
        monomer1_smiles = entry.get("monomer1_s")
        monomer2_smiles = entry.get("monomer2_s")

        update_fields = {}

        # Check monomer1
        if monomer1_smiles:
            if monomer1_smiles in updated_data:
                data = updated_data[monomer1_smiles]
                if all(key in data for key in
                       ["best_conformer_elements", "best_conformer_coordinates", "best_conformer_energy"]):
                    update_fields["monomer1_data"] = sanitize_for_mongodb(data)
                else:
                    print(f"Warning: Incomplete data for monomer1 {monomer1_smiles}")
                    skipped_count += 1
            else:
                print(f"Warning: No data found for monomer1 {monomer1_smiles}")
                skipped_count += 1

        # Check monomer2
        if monomer2_smiles:
            if monomer2_smiles in updated_data:
                data = updated_data[monomer2_smiles]
                if all(key in data for key in
                       ["best_conformer_elements", "best_conformer_coordinates", "best_conformer_energy"]):
                    update_fields["monomer2_data"] = sanitize_for_mongodb(data)
                else:
                    print(f"Warning: Incomplete data for monomer2 {monomer2_smiles}")
                    skipped_count += 1
            else:
                print(f"Warning: No data found for monomer2 {monomer2_smiles}")
                skipped_count += 1

        # Update database if we have fields to update
        if update_fields:
            try:
                db.collection.update_one(
                    {"_id": entry["_id"]},
                    {"$set": update_fields}
                )
                updated_count += 1
            except Exception as e:
                print(f"Error updating entry {entry['_id']}: {e}")
                error_count += 1

    print(f"\nUpdate Statistics:")
    print(f"Successfully updated: {updated_count}")
    print(f"Skipped due to missing data: {skipped_count}")
    print(f"Failed updates: {error_count}")


def main(output_folder="output/molecule_properties", smiles_error_path="output/smiles_error.json"):
    # Initialize MongoDB connection
    db = CoPolymerDB()

    # Load or initialize error log
    try:
        with open(smiles_error_path, "r") as f:
            smiles_error = json.load(f)
    except FileNotFoundError:
        smiles_error = []

    # Extract unique SMILES from MongoDB
    unique_smiles = set()
    for entry in db.collection.find():
        if entry.get("monomer1_s"):
            unique_smiles.add(entry["monomer1_s"])
        if entry.get("monomer2_s"):
            unique_smiles.add(entry["monomer2_s"])

    print(f"Found {len(unique_smiles)} unique SMILES strings")

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    updated_data = {}

    # Process each unique SMILES
    for smiles in tqdm(sorted(unique_smiles), desc="Processing SMILES"):
        print(f"\nProcessing {smiles}")
        filename = os.path.join(output_folder, f"{smiles.replace('/', '_')}.json")

        if os.path.exists(filename):
            print("loading ", filename)
            try:
                # Load existing properties and check if file is empty
                with open(filename, "r") as f:
                    file_content = f.read()
                    if not file_content.strip():
                        print(f"Empty file found for {smiles}, calculating properties")
                        raise ValueError("Empty file")

                    existing_properties = json.loads(file_content)

                missing_properties = calculate_missing_properties(smiles, existing_properties)
                existing_properties.update(missing_properties)
                save_properties(existing_properties, filename)
                updated_data[smiles] = existing_properties

            except (json.JSONDecodeError, ValueError) as e:
                print(f"Error loading properties for {smiles}: {e}")
                try:
                    # Calculate all properties if file is empty or corrupt
                    properties = calculate_property(smiles)
                    save_properties(properties, filename)
                    updated_data[smiles] = properties
                except Exception as calc_error:
                    print(f"Error calculating properties for {smiles}: {calc_error}")
                    smiles_error.append({"smiles": smiles, "error": str(calc_error)})
                    with open(smiles_error_path, "w") as f:
                        json.dump(smiles_error, f, indent=4)

        elif smiles not in [error["smiles"] for error in smiles_error]:
            try:
                # Calculate all properties
                properties = calculate_property(smiles)
                save_properties(properties, filename)
                updated_data[smiles] = properties
            except Exception as e:
                print(f"Error processing {smiles}: {e}")
                smiles_error.append({"smiles": smiles, "error": str(e)})
                with open(smiles_error_path, "w") as f:
                    json.dump(smiles_error, f, indent=4)

    print("\nAnalyzing MongoDB state before update...")
    before_analysis = analyze_mongodb_updates(db, updated_data)

    print("\nUpdating MongoDB entries with new properties...")
    update_mongodb_entries_safe(db, updated_data)

    print("\nAnalyzing MongoDB state after update...")
    after_analysis = analyze_mongodb_updates(db, updated_data)

    # Save analyses to file
    analysis_results = {
        "before_update": before_analysis,
        "after_update": after_analysis,
        "total_unique_smiles": len(unique_smiles),
        "successfully_calculated": len(updated_data),
        "calculation_errors": len(smiles_error)
    }

    with open(os.path.join(output_folder, "mongodb_analysis.json"), "w") as f:
        json.dump(analysis_results, f, indent=4)


if __name__ == "__main__":
    main()