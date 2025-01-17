import os
import json
from tqdm import tqdm
from morfeus.conformer import ConformerEnsemble
from morfeus import XTB
from copolextractor.mongodb_storage import CoPolymerDB
from copolextractor.utils import name_to_smiles
from pymongo import MongoClient


class SmilesCache:
    def __init__(self):
        """Initialize connection to MongoDB and create smiles_cache collection."""
        self.client = MongoClient("mongodb://localhost:27017/")
        self.db = self.client.co_polymer_database
        self.cache = self.db.smiles_cache

        # Create index for faster lookups
        self.cache.create_index("monomer_name", unique=True)

    def get_smiles(self, monomer_name: str) -> str:
        """Get SMILES from cache or compute and store it."""
        # Check cache first
        cached = self.cache.find_one({"monomer_name": monomer_name})
        if cached:
            return cached["smiles"]

        # If not in cache, compute and store
        try:
            smiles = name_to_smiles(monomer_name)
            self.cache.insert_one({
                "monomer_name": monomer_name,
                "smiles": smiles
            })
            return smiles
        except Exception as e:
            print(f"Error converting {monomer_name} to SMILES: {e}")
            return None


def convert_monomers_to_smiles(db: CoPolymerDB):
    """Convert monomer names to SMILES and update database using cache."""
    print("Converting monomer names to SMILES...")

    # Initialize SMILES cache
    smiles_cache = SmilesCache()

    # Get all entries
    entries = list(db.collection.find())
    total_entries = len(entries)

    # Keep track of successful and failed conversions
    conversion_stats = {
        "success": 0,
        "failed": 0,
        "cached": 0,
        "new": 0
    }

    # Process each entry
    for i, entry in enumerate(tqdm(entries, desc="Converting monomers")):
        update_fields = {}

        # Convert first monomer
        if "monomers" in entry and len(entry["monomers"]) > 0:
            monomer1_name = entry["monomers"][0]

            # Skip if SMILES already exists
            if not entry.get("monomer1_s"):
                monomer1_smiles = smiles_cache.get_smiles(monomer1_name)
                if monomer1_smiles:
                    update_fields["monomer1_s"] = monomer1_smiles
                    conversion_stats["success"] += 1
                    if "cached" in smiles_cache.cache.find_one({"monomer_name": monomer1_name}):
                        conversion_stats["cached"] += 1
                    else:
                        conversion_stats["new"] += 1
                else:
                    conversion_stats["failed"] += 1

        # Convert second monomer
        if "monomers" in entry and len(entry["monomers"]) > 1:
            monomer2_name = entry["monomers"][1]

            # Skip if SMILES already exists
            if not entry.get("monomer2_s"):
                monomer2_smiles = smiles_cache.get_smiles(monomer2_name)
                if monomer2_smiles:
                    update_fields["monomer2_s"] = monomer2_smiles
                    conversion_stats["success"] += 1
                    if "cached" in smiles_cache.cache.find_one({"monomer_name": monomer2_name}):
                        conversion_stats["cached"] += 1
                    else:
                        conversion_stats["new"] += 1
                else:
                    conversion_stats["failed"] += 1

        # Update database if we have new SMILES
        if update_fields:
            db.collection.update_one(
                {"_id": entry["_id"]},
                {"$set": update_fields}
            )

        # Print progress periodically
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{total_entries} entries")

    # Print final statistics
    print("\nConversion Statistics:")
    print(f"Total successful conversions: {conversion_stats['success']}")
    print(f"Retrieved from cache: {conversion_stats['cached']}")
    print(f"Newly computed: {conversion_stats['new']}")
    print(f"Failed conversions: {conversion_stats['failed']}")


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


def update_mongodb_entries(db: CoPolymerDB, updated_data: dict):
    """Update MongoDB entries with calculated molecular properties."""
    updated_count = 0
    for entry in db.collection.find():
        monomer1_smiles = entry.get("monomer1_s")
        monomer2_smiles = entry.get("monomer2_s")

        update_fields = {}

        # Add monomer1 properties if available
        if monomer1_smiles and monomer1_smiles in updated_data:
            update_fields["monomer1_data"] = updated_data[monomer1_smiles]

        # Add monomer2 properties if available
        if monomer2_smiles and monomer2_smiles in updated_data:
            update_fields["monomer2_data"] = updated_data[monomer2_smiles]

        if update_fields:
            db.collection.update_one(
                {"_id": entry["_id"]},
                {"$set": update_fields}
            )
            updated_count += 1

    print(f"Updated {updated_count} entries in MongoDB")


def save_properties(properties, filename):
    """Save properties to a file."""
    with open(filename, "w") as f:
        json.dump(properties, f, indent=4)


def main(output_folder="output/molecule_properties", smiles_error_path="output/smiles_error.json"):
    # Initialize MongoDB connection
    db = CoPolymerDB()

    # First, convert all monomer names to SMILES
    convert_monomers_to_smiles(db)

    # Load or initialize error log
    try:
        with open(smiles_error_path, "r") as f:
            smiles_error = json.load(f)
    except FileNotFoundError:
        smiles_error = []

    # Now extract unique SMILES from MongoDB
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
            # Load existing properties and calculate missing ones
            with open(filename, "r") as f:
                existing_properties = json.load(f)
            missing_properties = calculate_missing_properties(smiles, existing_properties)
            existing_properties.update(missing_properties)
            save_properties(existing_properties, filename)
            updated_data[smiles] = existing_properties
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

    # Update MongoDB entries with new properties
    update_mongodb_entries(db, updated_data)


if __name__ == "__main__":
    main()

