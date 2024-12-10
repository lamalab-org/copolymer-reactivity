import json
import os


# Function to check if the actual product is within the allowed deviation
def is_within_deviation(actual_product, expected_product, deviation=0.10):
    if expected_product == 0:
        return actual_product == 0
    return abs(actual_product - expected_product) / abs(expected_product) <= deviation


# Function to apply the r-product filter
def apply_r_product_filter(data):
    for entry in data:
        r1 = entry['r_values'].get('constant_1')
        r2 = entry['r_values'].get('constant_2')
        r_product = entry.get('r-product')

        if r_product is None or r1 is None or r2 is None:
            entry['r-product_filter'] = False
            continue

        actual_product = r1 * r2
        if is_within_deviation(actual_product, r_product):
            entry['r-product_filter'] = True  # Keep reaction
        else:
            entry['r-product_filter'] = False  # Filter out reaction


# Function to apply the confidence interval filter
def apply_r_conf_filter(data):
    for entry in data:
        r1 = entry.get('r_values', {}).get('constant_1')
        r2 = entry.get('r_values', {}).get('constant_2')
        conf_1 = entry.get('conf_intervals', {}).get('constant_conf_1')
        conf_2 = entry.get('conf_intervals', {}).get('constant_conf_2')

        if (
            r1 is not None and
            r2 is not None and
            conf_1 is not None and
            conf_2 is not None
        ):
            # Confidence intervals should not exceed the corresponding r-values
            entry['r_conf_filter'] = (conf_1 <= r1) and (conf_2 <= r2)
        else:
            entry['r_conf_filter'] = False  # Filter out due to missing/conflicting data


# Main function
def main(input_file, output_file):
    # Load data from JSON file
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist.")
        return

    with open(input_file, 'r') as file:
        data = json.load(file)

    # Apply filters
    apply_r_product_filter(data)
    apply_r_conf_filter(data)

    # Save updated data to output file
    with open(output_file, 'w') as file:
        json.dump(data, file, indent=4)

    print(f"Filters applied and results saved to {output_file}")


if __name__ == "__main__":
    input_file = "output/processed_reactions/all_reactions.json"
    output_file = "output/processed_reactions/filtered_reactions.json"

    main(input_file, output_file)
