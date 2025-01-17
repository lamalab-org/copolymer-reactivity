import json
import os
import re
from typing import Dict, List
from urllib.parse import urlparse
from copolextractor.mongodb_storage import CoPolymerDB
import copolextractor.utils as utils


def process_reaction(reaction: Dict, reaction_index: int, source: str, base_filename: str,
                     original_source: str) -> Dict:
    """Process a single reaction entry from the data."""
    # Generate filename with reaction index if needed
    filename = f"{base_filename}_{reaction_index}.json" if reaction_index > 0 else f"{base_filename}.json"

    # Get first reaction condition
    reaction_condition = reaction["reaction_conditions"][0]

    # Build reaction conditions with required fields
    reaction_conditions = {
        "temperature": reaction_condition["temperature"],
        "temperature_unit": reaction_condition["temperature_unit"],
        "solvent": reaction_condition["solvent"],
        "method": reaction_condition["method"],
        "polymerization_type": reaction_condition["polymerization_type"],
        "reaction_constants": reaction_condition["reaction_constants"],
        "reaction_constant_conf": reaction_condition["reaction_constant_conf"],
        "determination_method": reaction_condition["determination_method"]
    }

    # Add optional fields only if they exist
    for optional_field, source_field in [
        ("Q_value", "Q-value"),
        ("e_value", "e-Value"),
        ("r_product", "r_product")
    ]:
        if source_field in reaction_condition:
            reaction_conditions[optional_field] = reaction_condition[source_field]

    processed_data = {
        "filename": filename,
        "monomers": reaction["monomers"],
        "source": source,
        "original_source": original_source,
        "reaction_conditions": reaction_conditions
    }
    return processed_data


def load_data_from_directory(db: CoPolymerDB, directory: str, original_source: str) -> List[Dict]:
    """Load and process all JSON files from a specific directory."""
    results = []

    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return results

    # Process each JSON file
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            try:
                print(f"\nProcessing file: {filename}")

                # Load JSON file
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Extract source and create base filename
                source = data.get("source", "")
                if source and "doi.org/" in source:
                    base_filename = utils.sanitize_filename(source.split("doi.org/")[-1])
                else:
                    base_filename = filename.replace('.json', '')

                print(f"Found source: {source}")
                print(f"Base filename: {base_filename}")

                # Process each reaction
                reactions = data.get("reactions", [])
                print(f"Found {len(reactions)} reactions")

                for i, reaction in enumerate(reactions):
                    try:
                        print(f"Processing reaction {i + 1}/{len(reactions)}")
                        processed_data = process_reaction(
                            reaction=reaction,
                            reaction_index=i,
                            source=source,
                            base_filename=base_filename,
                            original_source=original_source
                        )

                        # Save to database
                        result = db.save_data(processed_data)
                        result['filename'] = processed_data['filename']
                        results.append(result)
                        print(f"Processed reaction {i + 1}: {result['message']} "
                              f"with filename {processed_data['filename']}")

                    except Exception as e:
                        print(f"Error processing reaction {i + 1} in {filename}: {str(e)}")

            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")
                continue

    return results


def main(input_path_new_data):
    # Initialize database
    db = CoPolymerDB()

    # Process data from both directories
    print("\nProcessing crossref data...")
    crossref_results = load_data_from_directory(
        db,
        directory=input_path_new_data,
        original_source="crossref"
    )

    print("\nProcessing copol database data...")
    copol_results = load_data_from_directory(
        db,
        directory="./data_extraction_GPT-4o/output/copol_database/model_output_extraction",
        original_source="copol database"
    )

    # Print summary
    successful_crossref = sum(1 for r in crossref_results if r['success'])
    successful_copol = sum(1 for r in copol_results if r['success'])

    print(f"\nProcessing summary:")
    print(f"Crossref data:")
    print(f"Successfully processed: {successful_crossref}")
    print(f"Failed: {len(crossref_results) - successful_crossref}")

    print(f"\nCopol database data:")
    print(f"Successfully processed: {successful_copol}")
    print(f"Failed: {len(copol_results) - successful_copol}")


if __name__ == "__main__":
    main()