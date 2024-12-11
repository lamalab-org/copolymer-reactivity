import json
import pandas as pd


# Function: Filter data based on the filters
def filter_data(data):
    """
    Filters the input data to include only valid entries where both
    'r-product_filter' and 'r_conf_filter' are True.
    """
    filtered_data = [
        entry for entry in data if entry["r-product_filter"] and entry["r_conf_filter"]
    ]
    print(f"Filtered datapoints: {len(filtered_data)}")
    return filtered_data


# Function: Extract numerical features and molecular properties from the JSON data
def extract_features(df):
    """
    Extracts features from 'monomer1_data' and 'monomer2_data' JSON fields
    and adds them as new columns to the input DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame with 'monomer1_data' and 'monomer2_data' JSON fields.

    Returns:
        pd.DataFrame: DataFrame with additional columns for extracted features.
    """
    # Keys to process with min, max, mean statistics
    keys_to_process = [
        "charges",
        "fukui_electrophilicity",
        "fukui_nucleophilicity",
        "fukui_radical",
    ]

    # Loop through each monomer and process its features
    for monomer_key in ["monomer1_data", "monomer2_data"]:
        # Process specific keys for min, max, mean statistics
        for key in keys_to_process:
            df[f"{monomer_key}_{key}_min"] = df[monomer_key].apply(
                lambda x: min(x[key].values())
                if isinstance(x, dict) and key in x
                else None
            )
            df[f"{monomer_key}_{key}_max"] = df[monomer_key].apply(
                lambda x: max(x[key].values())
                if isinstance(x, dict) and key in x
                else None
            )
            df[f"{monomer_key}_{key}_mean"] = df[monomer_key].apply(
                lambda x: sum(x[key].values()) / len(x[key].values())
                if isinstance(x, dict) and key in x
                else None
            )

        # Process dipole components
        df[f"{monomer_key}_dipole_x"] = df[monomer_key].apply(
            lambda x: x["dipole"][0]
            if isinstance(x, dict) and "dipole" in x and isinstance(x["dipole"], list)
            else None
        )
        df[f"{monomer_key}_dipole_y"] = df[monomer_key].apply(
            lambda x: x["dipole"][1]
            if isinstance(x, dict) and "dipole" in x and isinstance(x["dipole"], list)
            else None
        )
        df[f"{monomer_key}_dipole_z"] = df[monomer_key].apply(
            lambda x: x["dipole"][2]
            if isinstance(x, dict) and "dipole" in x and isinstance(x["dipole"], list)
            else None
        )

    return df


# Function: Create a flipped dataset with reversed monomer order
def create_flipped_dataset(df_features):
    """
    Creates a flipped dataset where the order of monomer1 and monomer2 is swapped.

    Args:
        df_features (pd.DataFrame): The input DataFrame with extracted features.

    Returns:
        pd.DataFrame: A new DataFrame with flipped monomer order.
    """
    flipped_rows = []
    for _, row in df_features.iterrows():
        flipped_row = row.copy()
        flipped_row["monomer1_s"], flipped_row["monomer2_s"] = (
            row["monomer2_s"],
            row["monomer1_s"],
        )
        flipped_rows.append(flipped_row)
    return pd.DataFrame(flipped_rows)


# Main function
def main(input_file, output_file):
    # Load JSON data
    with open(input_file, "r") as file:
        data = json.load(file)
    print(f"Initial datapoints: {len(data)}")

    # Filter data
    filtered_data = filter_data(data)

    # Convert to DataFrame
    df = pd.DataFrame(filtered_data)

    # Extract relevant features and molecular properties
    df = extract_features(df)

    # Add extracted columns directly into the DataFrame
    df["r1"] = df["r_values"].apply(
        lambda x: x["constant_1"] if isinstance(x, dict) and "constant_1" in x else None
    )
    df["r2"] = df["r_values"].apply(
        lambda x: x["constant_2"] if isinstance(x, dict) and "constant_2" in x else None
    )

    # Replace missing values with defaults
    df["determination_method"] = df["determination_method"].fillna("None")
    df["polymerization_type"] = df["polymerization_type"].fillna("None")

    # Drop rows with missing required values
    df = df.dropna(
        subset=[
            "r1",
            "r2",
            "monomer1_s",
            "monomer2_s",
            "temperature",
            "determination_method",
            "polymerization_type",
        ]
    )

    # Set solvent to "bulk" if polymerization type is "bulk"
    df.loc[df["polymerization_type"] == "bulk", "solvent"] = "bulk"

    # Drop rows with missing solvent
    df = df.dropna(subset=["solvent"])

    print(f"Filtered and processed datapoints: {len(df)}")

    # Create flipped dataset
    df_flipped = create_flipped_dataset(df)

    # Save the results
    df.to_csv(output_file, index=False)
    print(df.columns)
    df_flipped.to_csv(output_file.replace(".csv", "_flipped.csv"), index=False)

    print(
        f"Processed data saved to {output_file} and flipped data to {output_file.replace('.csv', '_flipped.csv')}."
    )
    print(f"Number of datapoints in {output_file}: {len(df)}")
    print(
        f"Number of datapoints in {output_file.replace('.csv', '_flipped.csv')}: {len(df_flipped)}"
    )


if __name__ == "__main__":
    input_file = "../../copol_prediction/output/extracted_data_w_features_filtered.json"  # Input file
    output_file = "../../copol_prediction/output/processed_data.csv"  # Output CSV file
    main(input_file, output_file)
