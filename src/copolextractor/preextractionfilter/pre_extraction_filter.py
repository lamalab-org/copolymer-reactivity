"""Helpers for filtering downloaded PDFs prior to LLM extraction."""

from __future__ import annotations

import os
from typing import Union

from copolextractor.preextractionfilter.extract_PDF_quality_GPT4 import (
    main as pdf_main,
)
from copolextractor.preextractionfilter.precision_prediction_randomforest import (
    main as rf_main,
)


def run_combined_pipeline(
    training_file,
    input_file_xgboost_filter,
    output_file,
    pdf_input_folder,
    output_folder_images,
    output_folder_LLM_score,
    seed,
    threshold,
    enable_pdf_processing,
):
    """
    Combined pipeline that first processes PDFs for scoring and then performs RF filtering.

    Parameters:
        pdf_input_folder (str): Folder containing the PDF files to be processed.
        output_folder_images (str): Folder to store processed PDF images.
        output_folder_LLM_score (str): Folder to save the PDF processing results.
        selected_entries_path (str): JSON file containing the entries to be scored.
        rf_output_path (str): Output path for the RF filtering module results.
        enable_pdf_processing (bool): Whether to run the optional GPT-4o based
            quality scoring stage before the RandomForest filter.  The stage is
            disabled by default because it is comparatively expensive.
    """
    print("Starting PDF processing and quality scoring...")

    if enable_pdf_processing:
        pdf_main(
            input_folder=pdf_input_folder,
            output_folder_images=output_folder_images,
            output_folder=output_folder_LLM_score,
            selected_entries_path=input_file_xgboost_filter,
            output_file=output_file,
        )
        print(
            "PDF processing and scoring completed. Results saved to "
            f"{output_file}"
        )
    else:
        print("Skipping PDF quality scoring (disabled via configuration).")

    print("Starting Random Forest filtering...")
    rf_main(
        training_file=training_file,
        scoring_file=output_file,
        output_file=output_file,
        seed_rf=seed,
        threshold=threshold,
    )
    print(f"RF filtering completed. Results saved to {output_file}.")


def main(
    seed_xgboost_model: int,
    threshold_xgboost_model: float,
    pdf_input_folder: Union[str, os.PathLike],
    output_folder_images: Union[str, os.PathLike],
    output_folder_LLM_score: Union[str, os.PathLike],
    training_file_xgboost_model: Union[str, os.PathLike],
    input_file_xgboost_filter: Union[str, os.PathLike],
    output_file_xgboost_filter: Union[str, os.PathLike],
    *,
    enable_pdf_processing: bool = False,
) -> None:
    """Entry point that wraps the optional PDF scoring and the RF filter.

    Args:
        seed_xgboost_model: Seed used by the RandomForest classifier.
        threshold_xgboost_model: Minimum probability required to keep a PDF.
        pdf_input_folder: Directory holding the candidate PDFs.
        output_folder_images: Directory used to cache intermediate page images.
        output_folder_LLM_score: Directory storing GPT-4o quality annotations.
        training_file_xgboost_model: JSON file with labelled training examples.
        input_file_xgboost_filter: Source JSON with PDF metadata to score.
        output_file_xgboost_filter: Destination JSON storing the scored subset.
        enable_pdf_processing: When ``True`` runs the optional GPT-4o quality
            scoring prior to the RandomForest filter.
    """

    os.makedirs(output_folder_images, exist_ok=True)
    os.makedirs(output_folder_LLM_score, exist_ok=True)

    run_combined_pipeline(
        training_file_xgboost_model,
        input_file_xgboost_filter,
        output_file_xgboost_filter,
        pdf_input_folder,
        output_folder_images,
        output_folder_LLM_score,
        seed_xgboost_model,
        threshold_xgboost_model,
        enable_pdf_processing,
    )

