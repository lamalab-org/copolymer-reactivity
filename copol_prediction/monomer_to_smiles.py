from copolextractor.mongodb_storage import CoPolymerDB
import asyncio
from collections import defaultdict
from tqdm import tqdm
from copolextractor.utils import name_to_smiles


async def convert_name_to_smiles(name: str) -> str:
    """Wrapper function to convert chemical names to SMILES using name_to_smiles"""
    try:
        smiles = name_to_smiles(name)
        if smiles:
            print(f"Successfully converted '{name}' to SMILES: {smiles}")
            return smiles
        else:
            print(f"Failed to convert '{name}' to SMILES - no SMILES returned")
            return None
    except Exception as e:
        print(f"Error converting '{name}' to SMILES: {e}")
        return None


async def update_reactions_with_smiles():
    db = CoPolymerDB()

    # Get all entries needing SMILES conversion
    none_entries = list(db.collection.find({
        "$or": [
            {"monomer1_s": None},
            {"monomer2_s": None},
            {"monomer1_s": {"$exists": False}},
            {"monomer2_s": {"$exists": False}}
        ]
    }))

    unique_monomers = {}  # {name: smiles}
    conversion_stats = {
        "total_entries": len(none_entries),
        "updated_reactions": 0,
        "successful_conversions": 0,
        "failed_conversions": 0,
        "skipped_entries": 0,
        "empty_monomers": 0,
        "invalid_format": 0,
        "monomer_formats": {"dict": 0, "marker": 0, "simple": 0}
    }

    # Process each entry
    for entry in tqdm(none_entries, desc="Processing entries"):
        update_fields = {}
        monomers = entry.get('monomers', [])

        print(f"\nProcessing entry {entry.get('_id')}:")
        print(f"Original source: {entry.get('original_source', 'Not specified')}")  # Neue Zeile
        print(f"Current monomer1_s: {entry.get('monomer1_s')}")
        print(f"Current monomer2_s: {entry.get('monomer2_s')}")
        print(f"Monomers data: {monomers}")

        if not monomers:
            conversion_stats["empty_monomers"] += 1
            continue

        try:
            # Format 1: List of dictionaries
            if isinstance(monomers[0], dict):
                conversion_stats["monomer_formats"]["dict"] += 1
                print("Processing Format 1: Dictionary format")
                for monomer_dict in monomers[:1]:  # Verarbeite das erste Dictionary
                    # Prüfe beide Monomere im Dictionary
                    for i in range(1, 3):
                        name = monomer_dict.get(f'monomer{i}')
                        if name and (entry.get(f"monomer{i}_s") is None):
                            print(f"Processing monomer{i}: {name}")
                            smiles = await process_monomer(name, unique_monomers)
                            if smiles:
                                update_fields[f"monomer{i}_s"] = smiles
                                conversion_stats["successful_conversions"] += 1
                            else:
                                conversion_stats["failed_conversions"] += 1

            # Format 2: List with monomer1/monomer2 markers
            elif "monomer1" in monomers or "monomer2" in monomers:
                conversion_stats["monomer_formats"]["marker"] += 1
                print("Processing Format 2: Marker format")
                for i in range(1, 3):
                    marker = f"monomer{i}"
                    if marker in monomers:
                        idx = monomers.index(marker)
                        if idx + 1 < len(monomers):
                            name = monomers[idx + 1]
                            if name and (entry.get(f"monomer{i}_s") is None):
                                print(f"Processing monomer{i}: {name}")
                                smiles = await process_monomer(name, unique_monomers)
                                if smiles:
                                    update_fields[f"monomer{i}_s"] = smiles
                                    conversion_stats["successful_conversions"] += 1
                                else:
                                    conversion_stats["failed_conversions"] += 1

            # Format 3: Simple list of two monomers
            else:
                conversion_stats["monomer_formats"]["simple"] += 1
                print("Processing Format 3: Simple list format")
                for i, name in enumerate(monomers[:2], 1):
                    if name and (entry.get(f"monomer{i}_s") is None):
                        print(f"Processing monomer{i}: {name}")
                        smiles = await process_monomer(name, unique_monomers)
                        if smiles:
                            update_fields[f"monomer{i}_s"] = smiles
                            conversion_stats["successful_conversions"] += 1
                        else:
                            conversion_stats["failed_conversions"] += 1

        except Exception as e:
            conversion_stats["invalid_format"] += 1
            print(f"Error processing entry: {e}")
            continue

        if update_fields:
            print(f"\nUpdating document {entry.get('_id')}:")
            print("Original monomer1_s:", entry.get('monomer1_s'))
            print("Original monomer2_s:", entry.get('monomer2_s'))
            print("Monomer entry: ", monomers)
            print("Update fields:", update_fields)

            result = db.collection.update_one(
                {"_id": entry["_id"]},
                {"$set": update_fields}
            )

            # Verify update
            updated_doc = db.collection.find_one({"_id": entry["_id"]})
            print("Updated monomer1_s:", updated_doc.get('monomer1_s'))
            print("Updated monomer2_s:", updated_doc.get('monomer2_s'))

            if result.modified_count > 0:
                conversion_stats["updated_reactions"] += 1
                print("Update successful!")
            else:
                print("Warning: Update didn't modify document!")
                print("No changes detected in database update")
        else:
            conversion_stats["skipped_entries"] += 1

    print(f"\nDetailed Conversion Summary:")
    print(f"Total entries processed: {conversion_stats['total_entries']}")
    print(f"Updated reactions: {conversion_stats['updated_reactions']}")
    print(f"Successful conversions: {conversion_stats['successful_conversions']}")
    print(f"Failed conversions: {conversion_stats['failed_conversions']}")
    print(f"Skipped entries: {conversion_stats['skipped_entries']}")
    print(f"Empty monomers: {conversion_stats['empty_monomers']}")
    print(f"Invalid format: {conversion_stats['invalid_format']}")
    print(f"Monomer formats found: {conversion_stats['monomer_formats']}")


async def process_monomer(name: str, unique_monomers: dict) -> str:
    """Process a single monomer name and return its SMILES notation"""
    if name in unique_monomers:
        return unique_monomers[name]

    try:
        smiles = await convert_name_to_smiles(name)
        if smiles:
            unique_monomers[name] = smiles
            return smiles
    except Exception as e:
        print(f"Error converting {name}: {e}")

    return None


async def main():
    await update_reactions_with_smiles()


if __name__ == "__main__":
    asyncio.run(main())