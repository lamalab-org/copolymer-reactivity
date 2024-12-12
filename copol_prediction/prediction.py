from copolextractor.pre_model_filter import main as pre_model_filter
from copolextractor.preprocess_data import main as pre_process
from copolextractor.model import main as model


def preprocess_data(input_file_filter, output_file_filter, output_file_processing, data, data_flipped):
    # Pre modeling filter
    pre_model_filter(input_file_filter, output_file_filter)

    # Pre process data
    pre_process(output_file_filter, output_file_processing)

    # model
    model(data, data_flipped)


def main():
    import os

    # Print the current working directory
    print("Current Working Directory:", os.getcwd())

    # Pre modeling filter
    input_file_filter = "./output/extracted_data_w_features_copol.json"
    output_file_filter = (
        "./output/extracted_data_w_features_filtered_copol.json"
    )

    # Pre-processing
    output_file_processing = "./output/processed_data_copol.csv"

    # Model
    data = "./output/processed_data_copol.csv"
    data_flipped = "./output/processed_data_copol_flipped.csv"

    preprocess_data(input_file_filter, output_file_filter, output_file_processing)


if __name__ == "__main__":

    main()