import os
from keyword_filter import main as keyword_filter
from embedding_filter import main as embedding_filter


def run_combined_pipeline(
    input_file,
    journal_file,
    keywords,
    score_limit,
    output_dir,
    selected_papers_path,
    number_of_selected_papers,
):
    """
    Combined pipeline that first scores the papers and then processes embeddings.

    Parameters:
        input_file (str): Path to the JSON file containing collected DOI metadata.
        journal_file (str): JSON file containing the supported journal names.
        keywords (dict): Dictionary of keywords with their weights for scoring.
        score_limit (int): Minimum score threshold for embedding generation.
        output_dir (str): Directory to store embeddings and processed data.
        selected_papers_path (str): Path to save the selected top papers.
        number_of_selected_papers (int): Number of nearest papers to select based on embeddings.
    """
    print("Step 1: Scoring papers...")
    scoring_output_path = os.path.join(output_dir, "scored_doi.json")
    keyword_filter(
        input_file=input_file,
        journal_file=journal_file,
        keywords=keywords,
        output_file=scoring_output_path,
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
    )

    print("Embedding generation completed.")


def main(keywords, score_limit, number_of_selected_papers):
    # Define paths and parameters
    input_file = (
        "output/collected_doi_metadata.json"  # Input file containing DOI metadata
    )
    journal_file = "output/journals.json"  # JSON file with supported journal names
    output_dir = "output"  # Directory to store results
    selected_papers_path = os.path.join(output_dir, "selected_200_papers.json")

    # Ensure output directories exist
    os.makedirs(output_dir, exist_ok=True)

    # Run the combined pipeline
    run_combined_pipeline(
        input_file,
        journal_file,
        keywords,
        score_limit,
        output_dir,
        selected_papers_path,
        number_of_selected_papers,
    )
