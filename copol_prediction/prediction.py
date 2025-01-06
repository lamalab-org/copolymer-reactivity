from copolextractor.pre_model_filter import main as pre_model_filter
from copolextractor.preprocess_data import main as pre_process
from copolextractor.model import main as model


def combine_csv_files(csv_files, output_file_combined):
    """
    Combines multiple CSV files into a single CSV file.

    Parameters:
    - csv_files: List of paths to the CSV files to combine.
    - output_file_combined: Path to the output combined CSV file.
    """
    import pandas as pd

    # Combine all CSV files
    combined_df = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)

    # Save combined DataFrame to the output file
    combined_df.to_csv(output_file_combined, index=False)
    print(f"Combined {len(csv_files)} files into {output_file_combined} with {len(combined_df)} rows.")


def preprocess_data(
        input_file_filter,
        output_file_filter,
        output_file_processing,
        training_data,
        training_data_flipped,
        add_flipped_data,
        combine_data_files_flag,
        data_files_to_combine):

    # Pre modeling filter
    pre_model_filter(input_file_filter, output_file_filter)

    # Pre process data
    pre_process(output_file_filter, output_file_processing)

    # If combining CSV files is enabled
    if combine_data_files_flag:
        # Combine the specified CSV files
        combined_csv_file = "./output/combined_data.csv"
        combine_csv_files(data_files_to_combine, combined_csv_file)
        training_data = combined_csv_file  # Use the combined CSV file as the input data

    # model
    model(training_data, training_data_flipped, add_flipped_data)


def main():
    import os

    # Print the current working directory
    print("Current Working Directory:", os.getcwd())

    # Pre modeling filter
    input_file_filter = "./output/extracted_data_w_features_copol.json"
    output_file_filter = "./output/extracted_data_w_features_filtered_copol.json"

    # Pre-processing
    output_file_processing = "./output/processed_data_copol.csv"

    # Model
    training_data = "./output/processed_data_copol.csv"
    training_data_flipped = "./output/processed_data_copol_flipped.csv"
    add_flipped_data = False  # Set True to incorporate flipped data to training and test dataset
    # Combine CSV files
    combine_data_files_flag = True  # Set to True to enable CSV file combination
    data_files_to_combine = [
        "./output/processed_data_copol.csv",
        "./output/processed_data.csv",
    ]

    preprocess_data(
        input_file_filter,
        output_file_filter,
        output_file_processing,
        training_data,
        training_data_flipped,
        add_flipped_data,
        combine_data_files_flag,
        data_files_to_combine
    )


if __name__ == "__main__":
    main()
