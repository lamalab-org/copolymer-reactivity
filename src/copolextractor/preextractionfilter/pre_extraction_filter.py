import os
from copolextractor.preextractionfilter.precision_prediction_randomforest import main as rf_main
from copolextractor.preextractionfilter.extract_PDF_quality_GPT4 import main as pdf_main


def run_combined_pipeline(
    training_file,
    input_file_xgboost_filter,
    output_file,
    pdf_input_folder,
    output_folder_images,
    output_folder_LLM_score,
    seed,
    threshold,
):
    """
    Combined pipeline that first processes PDFs for scoring and then performs RF filtering.

    Parameters:
        pdf_input_folder (str): Folder containing the PDF files to be processed.
        output_folder_images (str): Folder to store processed PDF images.
        output_folder_LLM_score (str): Folder to save the PDF processing results.
        selected_entries_path (str): JSON file containing the entries to be scored.
        rf_output_path (str): Output path for the RF filtering module results.
    """
    print("Starting PDF processing and quality scoring...")
    pdf_main(
        input_folder=pdf_input_folder,
        output_folder_images=output_folder_images,
        output_folder=output_folder_LLM_score,
        selected_entries_path=input_file_xgboost_filter,
        output_file=output_file,
    )
    print(f"PDF processing and scoring completed. Results saved to {output_file}")

    print("Starting Random Forest filtering...")
    rf_main(
        training_file=training_file,
        scoring_file=output_file,
        output_file=output_file,
        seed_rf=seed,
        threshold=threshold,
    )
    print(f"RF filtering completed. Results saved to {output_file}.")


def main(seed_xgboost_model, threshold_xgboost_model, pdf_input_folder, output_folder_images, output_folder_LLM_score, training_file_xgboost_model, input_file_xgboost_filter, output_file_xgboost_filter):

    # Ensure output directories exist
    os.makedirs(output_folder_images, exist_ok=True)
    os.makedirs(output_folder_LLM_score, exist_ok=True)

    # Run the combined pipeline
    run_combined_pipeline(
        training_file_xgboost_model,
        input_file_xgboost_filter,
        output_file_xgboost_filter,
        pdf_input_folder,
        output_folder_images,
        output_folder_LLM_score,
        seed_xgboost_model,
        threshold_xgboost_model
    )

