import json
import os
import csv
from copolextractor import utils

data_path = "../model_output_GPT4-o"


def load_data():
    combined_data = []
    # Loop through all JSON files in the directory
    for filename in os.listdir(data_path):
        if filename.endswith('.json'):
            file_path = os.path.join(data_path, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)

                # Check if data has a "reactions" key
                if 'reactions' in data and isinstance(data['reactions'], list):
                    # For each reaction in the file
                    for reaction in data['reactions']:
                        # Add file metadata to each reaction
                        reaction['source_filename'] = filename
                        if 'source' in data:
                            reaction['original_source'] = data['source']
                        if 'PDF_name' in data:
                            reaction['PDF_name'] = data['PDF_name']
                        combined_data.append(reaction)
                else:
                    # Old format: just add the filename to the data
                    data['source_filename'] = filename
                    combined_data.append(data)
    return combined_data


def extract_monomers(monomers):
    monomer1_name = None
    monomer2_name = None

    # list with [monomer1, monomer2]
    if isinstance(monomers, list) and len(monomers) == 2:
        monomer1_name = monomers[0]
        monomer2_name = monomers[1]

    # list with [monomer1, name_monomer1, monomer2, name_monomer2]
    elif isinstance(monomers, list) and len(monomers) == 4:
        monomer1_name = monomers[1]
        monomer2_name = monomers[3]

    # dict with monomer1 and monomer2
    elif isinstance(monomers, dict):
        if 'monomer1' in monomers:
            monomer1_name = monomers['monomer1']
        if 'monomer2' in monomers:
            monomer2_name = monomers['monomer2']

    # List of dictionaries [{'monomer1': '...', 'monomer2': '...'}]
    elif isinstance(monomers, list) and len(monomers) == 1 and isinstance(monomers[0], dict):
        monomer_dict = monomers[0]
        if 'monomer1' in monomer_dict:
            monomer1_name = monomer_dict['monomer1']
        if 'monomer2' in monomer_dict:
            monomer2_name = monomer_dict['monomer2']

    print(f"Original format: {monomers}, Output: {monomer1_name, monomer2_name}")

    return monomer1_name, monomer2_name


def unnest_data(combined_data):
    result = []

    # Fields to be converted to float
    float_fields = [
        'temperature', 'r_product',
        'constant_1', 'constant_2',
        'constant_conf_1', 'constant_conf_2',
        'q_value_1', 'q_value_2',
        'e_value_1', 'e_value_2'
    ]

    # Helper function for safe conversion to float
    def safe_float(value):
        if value is None or value == "" or (isinstance(value, str) and value.lower() == "na"):
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    for reaction in combined_data:
        # Check if monomers key exists
        if 'monomers' not in reaction:
            print(f"Warning: No monomers key in {reaction.get('source_filename', 'unknown')}")
            monomer1, monomer2 = None, None
            monomer1_smiles, monomer2_smiles = None, None
        else:
            # Extract monomer information
            monomer1, monomer2 = extract_monomers(reaction['monomers'])

        # Extract reaction conditions
        if 'reaction_conditions' in reaction and isinstance(reaction['reaction_conditions'], list):
            for condition in reaction['reaction_conditions']:
                # Create new data point
                data_point = {
                    'source_filename': reaction.get('source_filename', 'unknown'),
                    'monomer1_name': monomer1,
                    'monomer2_name': monomer2,
                    'polymerization_type': condition.get('polymerization_type'),
                    'solvent': condition.get('solvent'),
                    'method': condition.get('method'),
                    'temperature': safe_float(condition.get('temperature')),
                    'temperature_unit': condition.get('temperature_unit'),
                    'determination_method': condition.get('determination_method'),
                    'r_product': safe_float(condition.get('r_product'))
                }

                # Add reaction constants
                if 'reaction_constants' in condition:
                    constants = condition['reaction_constants']
                    data_point['constant_1'] = safe_float(constants.get('constant_1'))
                    data_point['constant_2'] = safe_float(constants.get('constant_2'))

                # Add confidence values
                if 'reaction_constant_conf' in condition:
                    conf = condition['reaction_constant_conf']
                    data_point['constant_conf_1'] = safe_float(conf.get('constant_conf_1'))
                    data_point['constant_conf_2'] = safe_float(conf.get('constant_conf_2'))

                # Add Q-values - safely handle None case
                if 'Q-value' in condition and condition['Q-value'] is not None:
                    q_values = condition['Q-value']
                    data_point['q_value_1'] = safe_float(q_values.get('constant_1'))
                    data_point['q_value_2'] = safe_float(q_values.get('constant_2'))
                else:
                    data_point['q_value_1'] = None
                    data_point['q_value_2'] = None

                # Add e-values - safely handle None case
                if 'e-Value' in condition and condition['e-Value'] is not None:
                    e_values = condition['e-Value']
                    data_point['e_value_1'] = safe_float(e_values.get('constant_1'))
                    data_point['e_value_2'] = safe_float(e_values.get('constant_2'))
                else:
                    data_point['e_value_1'] = None
                    data_point['e_value_2'] = None

                # Calculate r_product_filter
                data_point['r_product_filter'] = r_product_filter(data_point)

                # Calculate conf_filter
                data_point['conf_filter'] = conf_filter(data_point)

                # Calculate actual r-product
                if data_point['constant_1'] is not None and data_point['constant_2'] is not None:
                    data_point['actual_r_product'] = data_point['constant_1'] * data_point['constant_2']
                else:
                    data_point['actual_r_product'] = None

                # Add other important information from the reaction
                for key, value in reaction.items():
                    if key not in ['monomers', 'reaction_conditions', 'source_filename']:
                        data_point[key] = value

                # Add data point to result
                result.append(data_point)
        else:
            print(f"Warning: No valid reaction_conditions in {reaction.get('source_filename', 'unknown')}")

    return result


def process_chemicals(data):
    for entry in data:
        monomer1_smiles = utils.name_to_smiles(entry['monomer1_name'])
        monomer2_smiles = utils.name_to_smiles(entry['monomer2_name'])
        solvent_smiles = utils.name_to_smiles(entry['solvent'])
        solvent_logp = utils.calculate_logP(solvent_smiles)

        entry.update({
            'monomer1_smiles': monomer1_smiles,
            'monomer2_smiles': monomer2_smiles,
            'solvent_smiles': solvent_smiles,
            'solvent_logp': solvent_logp
        })

    return data


def is_within_deviation(calc_product, expected_product, deviation=0.10):
    """Check if actual product is within allowed deviation."""
    if expected_product == 0:
        return calc_product == 0
    return abs(calc_product - expected_product) / abs(expected_product) <= deviation


def r_product_filter(entry):
    """
    Check if the product of reaction constants is within allowed deviation of the reported r-product.
    Returns True if within deviation, False otherwise.
    """
    # Check if r_product exists (note: key is 'r_product' not 'r-product')
    if 'r_product' in entry and entry['r_product'] is not None:
        # Check if both constants exist and are not None
        if ('constant_1' in entry and entry['constant_1'] is not None and
                'constant_2' in entry and entry['constant_2'] is not None):

            calc_product = entry['constant_1'] * entry['constant_2']
            # Return True if within deviation
            return is_within_deviation(calc_product, entry['r_product'])
        else:
            # Missing constants
            return False
    else:
        # Missing r_product
        return False


def conf_filter(entry):
    """
    Also checks if confidence values are less than or equal to the actual constants.
    Returns True if valid confidence values exist and satisfy the condition, False otherwise.
    """
    # Check if all necessary values exist and are not None
    if ('constant_conf_1' in entry and entry['constant_conf_1'] is not None and
            'constant_conf_2' in entry and entry['constant_conf_2'] is not None and
            'constant_1' in entry and entry['constant_1'] is not None and
            'constant_2' in entry and entry['constant_2'] is not None):

        # Try to convert to numeric values if they're strings but represent numbers
        try:
            conf_1 = float(entry['constant_conf_1']) if entry['constant_conf_1'] != "na" else float('inf')
            conf_2 = float(entry['constant_conf_2']) if entry['constant_conf_2'] != "na" else float('inf')
            r1 = float(entry['constant_1'])
            r2 = float(entry['constant_2'])

            # Check if confidence values are less than or equal to the constants
            return (conf_1 <= r1) and (conf_2 <= r2)
        except (ValueError, TypeError):
            # If conversion fails, return False
            return False

    # If any required value is missing
    return False


def filter_data(data):
    for entry in data:
        r_product_filter_value = r_product_filter(entry)
        conf_filter_value = conf_filter(entry)
        entry.update({
            'r_product_filter': r_product_filter_value,
            'conf_filter': conf_filter_value
        })
    return data


def write_to_csv(data, output_file="output.csv"):
    if not data:
        print("No data available to write.")
        return

    # Collect all possible field names
    fieldnames = set()
    for entry in data:
        fieldnames.update(entry.keys())

    # Write to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=sorted(fieldnames))
        writer.writeheader()
        writer.writerows(data)

    print(f"Data successfully written to {output_file}.")


def main():
    # Load data
    combined_data = load_data()

    # Process data
    unnested_data = unnest_data(combined_data)

    print("Processing the data...")
    # add smiles and lop
    processed_data = process_chemicals(unnested_data)

    print("Filtering the data...")
    # filter data
    filtered_data = filter_data(processed_data)

    # Write to CSV
    write_to_csv(filtered_data, "extracted_reactions.csv")


if __name__ == "__main__":
    main()