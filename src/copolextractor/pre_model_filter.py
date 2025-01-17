import json
from copolextractor.mongodb_storage import CoPolymerDB
from typing import List, Dict, Optional


def convert_to_float(value):
    """Convert string values to float."""
    if value == "na" or value is None:
        return None
    try:
        return float(value)
    except ValueError:
        print(f"Warning: Could not convert {value} to float. Setting it to None.")
        return None


def preprocess_data(data: List[Dict]):
    """Preprocess data and convert relevant values to float."""
    for entry in data:
        reaction_conditions = entry.get("reaction_conditions", {})

        # Extract reaction constants
        reaction_constants = reaction_conditions.get("reaction_constants", {})
        if reaction_constants:
            reaction_constants["constant_1"] = convert_to_float(reaction_constants.get("constant_1"))
            reaction_constants["constant_2"] = convert_to_float(reaction_constants.get("constant_2"))

        # Extract confidence intervals
        reaction_constant_conf = reaction_conditions.get("reaction_constant_conf", {})
        if reaction_constant_conf:
            reaction_constant_conf["constant_conf_1"] = convert_to_float(reaction_constant_conf.get("constant_conf_1"))
            reaction_constant_conf["constant_conf_2"] = convert_to_float(reaction_constant_conf.get("constant_conf_2"))

        # Extract r_product
        reaction_conditions["r_product"] = convert_to_float(reaction_conditions.get("r_product"))


def is_within_deviation(actual_product, expected_product, deviation=0.10):
    """Check if actual product is within allowed deviation."""
    if expected_product == 0:
        return actual_product == 0
    return abs(actual_product - expected_product) / abs(expected_product) <= deviation


def apply_r_product_filter(data: List[Dict]):
    """Apply the r-product filter to the data."""
    for entry in data:
        reaction_conditions = entry.get("reaction_conditions", {})
        reaction_constants = reaction_conditions.get("reaction_constants", {})

        r1 = reaction_constants.get("constant_1")
        r2 = reaction_constants.get("constant_2")
        r_product = reaction_conditions.get("r_product")

        if r_product is None or r1 is None or r2 is None:
            entry["r-product_filter"] = True
            continue

        actual_product = r1 * r2
        if is_within_deviation(actual_product, r_product):
            entry["r-product_filter"] = True
        else:
            entry["r-product_filter"] = False


def apply_r_conf_filter(data: List[Dict]):
    """Apply the confidence interval filter to the data."""
    for entry in data:
        reaction_conditions = entry.get("reaction_conditions", {})
        reaction_constants = reaction_conditions.get("reaction_constants", {})
        reaction_constant_conf = reaction_conditions.get("reaction_constant_conf", {})

        r1 = reaction_constants.get("constant_1")
        r2 = reaction_constants.get("constant_2")
        conf_1 = reaction_constant_conf.get("constant_conf_1")
        conf_2 = reaction_constant_conf.get("constant_conf_2")

        if r1 is not None and r2 is not None:
            if conf_1 is not None and conf_2 is not None:
                entry["r_conf_filter"] = (conf_1 <= r1) and (conf_2 <= r2)
            else:
                entry["r_conf_filter"] = True
        else:
            entry["r_conf_filter"] = True


from datetime import datetime


def convert_mongo_document(doc):
    """Convert MongoDB document to JSON serializable format."""
    if isinstance(doc, dict):
        return {key: convert_mongo_document(value) for key, value in doc.items()}
    elif isinstance(doc, list):
        return [convert_mongo_document(item) for item in doc]
    elif str(type(doc)) == "<class 'bson.objectid.ObjectId'>":
        return str(doc)
    elif isinstance(doc, datetime):
        return doc.isoformat()  # Konvertiert datetime zu ISO-Format String
    else:
        return doc


def load_and_filter_data(source: Optional[str] = None, output_file: Optional[str] = None):
    """
    Load data from MongoDB, optionally filter by source, apply filters and save results.

    Args:
        source: Optional filter for original_source ('crossref' or 'copol database')
        output_file: Optional file path to save the filtered results
    """
    # Initialize database connection
    db = CoPolymerDB()

    # Prepare query
    query = {}
    if source:
        query["original_source"] = source

    # Load data from MongoDB
    data = db.query_data(query)
    print(f"Loaded {len(data)} entries from database")

    # Convert MongoDB documents to JSON serializable format
    data = convert_mongo_document(data)

    # Preprocess and apply filters
    preprocess_data(data)
    apply_r_product_filter(data)
    apply_r_conf_filter(data)

    # Save results if output file is specified
    if output_file:
        with open(output_file, "w") as file:
            json.dump(data, file, indent=4)
        print(f"Results saved to {output_file}")

    return data


def main():
    # Example usage
    # Load all data
    all_data = load_and_filter_data(output_file="all_data_filtered.json")
    print(f"Total entries processed: {len(all_data)}")

    # Load only crossref data
    crossref_data = load_and_filter_data(source="crossref", output_file="crossref_data_filtered.json")
    print(f"Crossref entries processed: {len(crossref_data)}")

    # Load only copol database data
    copol_data = load_and_filter_data(source="copol database", output_file="copol_data_filtered.json")
    print(f"Copol database entries processed: {len(copol_data)}")


if __name__ == "__main__":
    main()