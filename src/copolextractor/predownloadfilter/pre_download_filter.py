"""Coordinate keyword scoring and embedding filtering prior to downloading PDFs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Sequence

from src.copolextractor.predownloadfilter.embedding_filter import main as embedding_filter
from src.copolextractor.predownloadfilter.keyword_filter import main as keyword_filter


def run_combined_pipeline(
    input_file: str,
    journal_file: str,
    keywords: Mapping[str, int],
    score_limit: int,
    output_dir: str,
    selected_papers_path: str,
    number_of_selected_papers: int,
    key_embedding_filter: str,
    values_embedding_filter: Sequence[str],
    scoring_file_embedding_filter: str,
    existing_doi_csv: str,
) -> None:
    """
    Combined pipeline that first scores the papers and then processes embeddings.

    Parameters:
        input_file: Path to the JSON file containing collected DOI metadata.
        journal_file: JSON file containing the supported journal names.
        keywords: Dictionary of keywords with their weights for scoring.
        score_limit: Minimum score threshold for embedding generation.
        output_dir: Directory to store embeddings and processed data.
        selected_papers_path: Path to save the selected top papers.
        number_of_selected_papers: Number of nearest papers to select based on embeddings.
    """
    print("Step 1: Scoring papers...")
    scoring_output_path = os.path.join(output_dir, "scored_doi.json")
    keyword_filter(
        input_file=input_file,
        journal_file=journal_file,
        keywords=keywords,
        output_file=scoring_output_path,
        existing_doi_csv=existing_doi_csv,
    )

    print("Scoring completed.")

    print("Step 2: Generating embeddings...")
    embedding_filter(
        file_path=scoring_output_path,
        output_dir=output_dir,
        doi_list_path=os.path.join(output_dir, "embeddings/existing_embeddings.json"),
        selected_papers_path=selected_papers_path,
        score_limit=score_limit,
        number_of_selected_paper=number_of_selected_papers,
        key=key_embedding_filter,
        values=values_embedding_filter,
        new_papers_path=scoring_file_embedding_filter,
    )

    print("Embedding generation completed.")


def main(
    keywords: Mapping[str, int],
    score_limit: int,
    number_of_selected_papers: int,
    crossref_metadata_input_file: str,
    output_file_pre_download_filter: str,
    key_embedding_filter: str,
    values_embedding_filter: Sequence[str],
    scoring_file_embedding_filter: str,
    existing_doi_csv: str,
    *,
    journal_file: str | os.PathLike | None = None,
    output_dir: str | os.PathLike | None = None,
) -> None:
    """Apply keyword scoring followed by the embedding based filter.

    Args:
        keywords: Keyword/weight mapping used by the heuristics.
        score_limit: Minimum score a paper requires to enter the embedding step.
        number_of_selected_papers: Number of papers to retain after the embedding
            similarity search.
        crossref_metadata_input_file: Metadata JSON produced by ``crossref_search``.
        output_file_pre_download_filter: Final JSON containing the top-N papers.
        key_embedding_filter: Metadata key that is required to match *values*.
        values_embedding_filter: Allowed values for *key_embedding_filter*.
        scoring_file_embedding_filter: CSV file with existing training examples.
        existing_doi_csv: CSV used to avoid reprocessing known DOIs.
        journal_file: Optional override for the journal whitelist.
        output_dir: Optional override for the working directory that stores
            intermediate results.
    """

    project_root = Path(__file__).resolve().parents[3]
    data_root = project_root / "data_extraction"

    if journal_file is None:
        journal_file = data_root / "artifacts" / "metadata" / "output" / "journals.json"

    if output_dir is None:
        output_dir = data_root / "artifacts" / "metadata" / "output"

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    run_combined_pipeline(
        str(crossref_metadata_input_file),
        str(journal_file),
        keywords,
        score_limit,
        str(output_dir_path),
        str(output_file_pre_download_filter),
        number_of_selected_papers,
        key_embedding_filter,
        values_embedding_filter,
        str(scoring_file_embedding_filter),
        str(existing_doi_csv),
    )
