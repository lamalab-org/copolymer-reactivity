import os
from random_forest_filter.precision_prediction_randomforest import main as rf_main
from random_forest_filter.extract_PDF_quality_GPT4 import main as pdf_main


def run_combined_pipeline(filter_input_path, filter_output_path, pdf_input_folder, output_folder_images, output_folder, selected_entries_path):
    """
    Combined pipeline that first performs RF filtering and then processes PDFs.

    Parameters:
        filter_input_path (str): Input path for the RF filtering module.
        filter_output_path (str): Output path for the RF filtering module results.
        pdf_input_folder (str): Folder containing the PDF files to be processed.
        output_folder_images (str): Folder to store processed PDF images.
        output_folder (str): Folder to save the PDF processing results.
        selected_entries_path (str): JSON file containing the filtered entries.
    """
    print("Starting Random Forest filtering...")
    rf_main(input_path=filter_input_path, output_path=filter_output_path)
    print(f"RF filtering completed. Results saved to {filter_output_path}.")

    print("Starting PDF processing and quality scoring...")
    pdf_main(
        input_folder=pdf_input_folder,
        output_folder_images=output_folder_images,
        output_folder=output_folder,
        selected_entries_path=filter_output_path  # Use the output file from the RF module
    )
    print("PDF processing and scoring completed.")


def main():
    # Paths for the RF filtering module
    filter_input_path = "../collected_data/enhanced_doi_list_unique.json"
    filter_output_path = "../collected_data/enhanced_doi_list_filtered.json"

    # Paths for the PDF processing module
    pdf_input_folder = "./PDF"
    output_folder_images = "./processed_images"
    output_folder = "./model_output_score"
    selected_entries_path = filter_output_path  # Use RF output here

    # Ensure output directories exist
    os.makedirs(output_folder_images, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    # Run the combined pipeline
    run_combined_pipeline(filter_input_path, filter_output_path, pdf_input_folder, output_folder_images, output_folder, selected_entries_path)


if __name__ == "__main__":
    main()
