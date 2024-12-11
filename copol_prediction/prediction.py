from copolextractor.pre_model_filter import main as pre_model_filter
from copolextractor.preprocess_data import main as pre_process
from copolextractor.model import main as model


def preprocess_data(input_file_filter, output_file_filter, output_file_processing):

    # Pre modeling filter
    pre_model_filter(input_file_filter, output_file_filter)

    # Pre process data
    pre_process(output_file_filter, output_file_processing)

    # model
    #model()


def main():

    # Pre modeling filter
    input_file_filter = "../../copol_prediction/output/extracted_data_w_features.json"
    output_file_filter = "../../copol_prediction/output/extracted_data_w_features_filtered.json"

    # Pre-processing
    output_file_processing = "../../copol_prediction/output/processed_data.csv"

    preprocess_data(input_file_filter, output_file_filter, output_file_processing)