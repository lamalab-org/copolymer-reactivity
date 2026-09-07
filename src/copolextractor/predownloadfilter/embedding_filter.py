import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import requests
from openai import OpenAI
from scipy.spatial.distance import cdist

import copolextractor.utils as utils


def save_embedding_file(output_dir: str, paper: dict, doi_list_path: str) -> None:
    """Save individual embedding JSON files and update DOI list.

    Args:
        output_dir: Base directory; embeddings are written to its "embeddings"
            subdirectory (created if needed).
        paper: Paper dict with "DOI", "Title", "Abstract" and "Embedding" keys.
        doi_list_path: Path to the JSON file tracking DOIs that have an embedding;
            updated in place with `paper["DOI"]` if not already present.
    """
    embeddings_dir = os.path.join(output_dir, "embeddings")
    os.makedirs(embeddings_dir, exist_ok=True)

    sanitized_doi = utils.sanitize_filename(paper["DOI"])
    filename = os.path.join(embeddings_dir, f"{sanitized_doi}.json")
    data = {
        "DOI": paper["DOI"],
        "Title": paper["Title"],
        "Abstract": paper["Abstract"],
        "Embedding": paper["Embedding"],
    }
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    # Update DOI list
    if os.path.exists(doi_list_path):
        with open(doi_list_path, "r") as f:
            doi_list = json.load(f)
    else:
        doi_list = []

    if paper["DOI"] not in doi_list:
        doi_list.append(paper["DOI"])

    with open(doi_list_path, "w") as f:
        json.dump(doi_list, f, indent=2)


def load_failed_crossref(failed_path: str = "output/failed_crossref.json") -> set:
    """Load the set of DOIs for which CrossRef lookups previously failed.

    Args:
        failed_path: Path to the JSON file with a list of failed DOIs.

    Returns:
        The set of failed DOIs, or an empty set if the file does not exist.
    """
    if os.path.exists(failed_path):
        with open(failed_path, "r") as f:
            return set(json.load(f))
    return set()


def save_failed_crossref(failed_dois: set, failed_path: str = "output/failed_crossref.json") -> None:
    """Persist the set of DOIs for which CrossRef lookups failed.

    Args:
        failed_dois: DOIs to persist.
        failed_path: Destination path for the JSON file.
    """
    with open(failed_path, "w") as f:
        json.dump(list(failed_dois), f, indent=2)


def load_existing_doi_list(doi_list_path: str) -> list:
    """Load the existing DOI list from file.

    Args:
        doi_list_path: Path to the JSON file with a list of DOIs.

    Returns:
        The list of DOIs, or an empty list if the file does not exist.
    """
    if os.path.exists(doi_list_path):
        with open(doi_list_path, "r") as f:
            return json.load(f)
    return []


def load_existing_embedding(output_dir: str, doi: str) -> Tuple[Optional[list], str]:
    """Check if an embedding already exists for a given DOI and load it.

    Args:
        output_dir: Base directory whose "embeddings" subdirectory is searched.
        doi: DOI to look up.

    Returns:
        A (embedding, filename) tuple. `embedding` is the stored embedding vector,
        or None if no cached file exists for this DOI. `filename` is the path the
        embedding is (or would be) stored at.
    """
    embeddings_dir = os.path.join(output_dir, "embeddings")
    print(doi)
    sanitized_doi = utils.sanitize_filename(doi)
    filename = os.path.join(embeddings_dir, f"{sanitized_doi}.json")
    if os.path.exists(filename):
        with open(filename, "r") as f:
            data = json.load(f)
            return data.get("Embedding"), filename
    return None, filename


def process_embeddings(
    file_path: str, output_dir: str, client: OpenAI, score: float, doi_list_path: str
) -> None:
    """Process and save embeddings for each paper.

    Args:
        file_path: Path to a JSON file containing a list of scored paper dicts.
        output_dir: Base directory embeddings are stored under.
        client: OpenAI client used to create new embeddings.
        score: Minimum "Score" value a paper needs to be embedded.
        doi_list_path: Path to the JSON file tracking DOIs that already have an
            embedding.
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {file_path}. Details: {e}")
        return

    if not isinstance(data, list):
        print("Error: Expected a list of papers in the JSON file. Exiting.")
        return

    # Filter out papers that have already been extracted
    filtered_data = [paper for paper in data if not paper.get("already_extracted", False)]

    existing_dois = load_existing_doi_list(doi_list_path)
    total_tokens = 0

    for paper in filtered_data:  # Iteriere nur über die gefilterten Papers
        print(f"Processing embedding of {paper}.")
        if paper.get("DOI") in existing_dois:
            continue

        title = paper.get("Title")
        abstract = paper.get("Abstract")
        score_value = paper.get("Score", 0)
        doi = paper.get("DOI", "Unknown")

        if score_value < score:
            continue

        if not title and not abstract:
            continue

        text = f"{title if title else ''}. {abstract if abstract else ''}".strip()

        # Check if embedding already exists
        existing_embedding, filename = load_existing_embedding(output_dir, paper.get("DOI"))
        if existing_embedding is not None:
            paper["Embedding"] = existing_embedding
            save_embedding_file(output_dir, paper, doi_list_path)
            print(f"Skipping embedding creation for {paper}.")
            continue

        try:
            embedding, token_usage = get_embedding(client, text)
            total_tokens += token_usage
            paper["Embedding"] = embedding
            save_embedding_file(output_dir, paper, doi_list_path)
        except Exception as e:
            print(f"Error embedding paper {paper.get('DOI', 'Unknown')}: {e}")
            continue

    print(f"Total tokens used: {total_tokens}")


def embed_filtered_papers(
    new_papers_path: str,
    output_dir: str,
    client: OpenAI,
    key: str,
    values: Sequence[Any],
    failed_path: str = "failed_crossref.json",
) -> List[dict]:
    """Embed papers from a CSV, filtered by a column value, fetching missing metadata from CrossRef.

    Args:
        new_papers_path: Path to a CSV with at least an "original_source" (DOI)
            column.
        output_dir: Base directory embeddings are stored under.
        client: OpenAI client used to create new embeddings.
        key: Column name to filter rows on.
        values: Allowed values for `key`; rows are kept only if `key` is missing
            from the CSV or its value is one of `values`.
        failed_path: Path to the JSON file tracking DOIs with failed CrossRef
            lookups (loaded before processing, updated afterward).

    Returns:
        List of paper dicts augmented with "Embedding", "filename" and "DOI" keys.
        Papers without title/abstract that CrossRef also cannot resolve are skipped.
    """
    print(f"Loading new papers from CSV: {new_papers_path}")
    failed_dois = load_failed_crossref(failed_path)
    try:
        # Load the CSV file
        df = pd.read_csv(new_papers_path)

        # Check if required columns exist
        required_cols = ["original_source"]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            print(f"Error: Missing required columns in CSV: {missing_cols}")
            return []

        # Filter papers based on the key and values
        if key in df.columns:
            filtered_df = df[df[key].isin(values)]
        else:
            print(f"Warning: Key '{key}' not found in CSV. Processing all papers.")
            filtered_df = df

        # Convert DataFrame to list of dictionaries
        new_papers = filtered_df.to_dict("records")

    except Exception as e:
        print(f"Error loading CSV file: {new_papers_path}. Details: {e}")
        return []

    print(f"Found {len(new_papers)} papers after filtering")

    embedded_papers = []
    for paper in new_papers:
        doi = paper.get("original_source")
        if not doi:
            print(f"Skipping paper without DOI (original_source)")
            continue

        if doi in failed_dois:
            print(f"Skipping previously failed DOI: {doi}")
            continue

        print(f"Processing embedding for paper with DOI: {doi}")
        title = paper.get("Title")
        abstract = paper.get("Abstract")

        if not title or not abstract:
            print(f"Missing Title or Abstract for paper with DOI: {doi}. Fetching from CrossRef...")
            crossref_data = get_crossref_data(
                doi, paper.get("Source", "Unknown"), paper.get("Format", "Unknown")
            )

            title = crossref_data.get("Title")
            abstract = crossref_data.get("Abstract")
            paper["Title"] = title
            paper["Abstract"] = abstract

            if title == "No title" and abstract == "No abstract available":
                print(f"CrossRef could not provide Title and Abstract for DOI: {doi}.")
                failed_dois.add(doi)
                continue

        text = f"{title if title else ''}. {abstract if abstract else ''}".strip()

        # Check if embedding already exists
        existing_embedding, current_filename = load_existing_embedding(output_dir, doi)
        if existing_embedding is not None:
            paper["Embedding"] = existing_embedding
            paper["filename"] = current_filename
            paper["DOI"] = doi  # Ensure DOI is in the standard key
            embedded_papers.append(paper)
            print(f"Using existing embedding for {doi}")
            continue

        try:
            embedding, _ = get_embedding(client, text)
            paper["Embedding"] = embedding
            paper["filename"] = current_filename
            paper["DOI"] = doi  # Ensure DOI is in the standard key
            embedded_papers.append(paper)
            print(f"Created new embedding for {doi}")
        except Exception as e:
            print(f"Error embedding paper {doi}: {e}")
            continue

    save_failed_crossref(failed_dois, failed_path)
    return embedded_papers


def get_crossref_data(doi: str, source: str, format_type: str) -> Dict[str, Any]:
    """
    Fetch metadata from CrossRef API for a given DOI.

    Args:
        doi: DOI to look up.
        source: Value stored under the "Source" key of the returned dict.
        format_type: Value stored under the "Format" key of the returned dict.

    Returns:
        A dict with "DOI", "Title", "Abstract", "Keywords", "Journal", "Source",
        "Format" (with placeholder values such as "No title" if unavailable), or an
        "Error" key describing the failed HTTP status if the request did not
        succeed.
    """
    url = f"https://api.crossref.org/works/{doi}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        item = data.get("message", {})
        title = item.get("title", ["No title"])[0]
        abstract = item.get("abstract", "No abstract available")
        keywords = item.get("subject", "No keywords available")
        journal = item.get("container-title", ["No journal title"])[0]

        return {
            "DOI": doi,
            "Title": title,
            "Abstract": abstract,
            "Keywords": keywords,
            "Journal": journal,
            "Source": source,
            "Format": format_type,
        }
    else:
        return {
            "DOI": doi,
            "Title": "No title",
            "Abstract": "No abstract available",
            "Error": f"Unable to fetch data (Status Code: {response.status_code})",
        }


def find_nearest_paper_with_new(
    output_dir: str,
    selected_papers_path: str,
    key: str,
    values: Sequence[Any],
    number_of_selected_paper: int,
    new_papers_path: str,
    client: OpenAI,
) -> None:
    """Select the previously-processed papers most similar (by embedding) to new candidate papers.

    Args:
        output_dir: Base directory containing "embeddings/embedded_papers.json".
        selected_papers_path: Destination path for the selected-papers JSON file.
        key: Metadata key copied into each selected paper's result entry.
        values: Candidate filter values, forwarded to `embed_filtered_papers`.
        number_of_selected_paper: Maximum number of nearest papers to select.
        new_papers_path: Path to the CSV of new candidate papers, forwarded to
            `embed_filtered_papers`.
        client: OpenAI client used to embed new candidate papers.
    """
    embeddings_path = os.path.join(output_dir, "embeddings/embedded_papers.json")

    # Load processed data
    with open(embeddings_path, "r") as f:
        processed_data = json.load(f)

    if not processed_data:
        print("Error: No processed papers available.")
        return

    # Embed only filtered new papers
    new_papers = embed_filtered_papers(new_papers_path, output_dir, client, key, values)

    if not new_papers:
        print("Error: No new papers found with the specified filter.")
        return

    # Extract embeddings for the two groups
    processed_embeddings = np.array([entry["Embedding"] for entry in processed_data])
    new_embeddings = np.array([entry["Embedding"] for entry in new_papers])

    # Compute cosine distances between processed papers and new papers
    distances = cdist(processed_embeddings, new_embeddings, metric="cosine")

    if distances.size == 0:
        print("Error: Distance matrix is empty.")
        return

    # Find the minimum distance for each processed paper
    min_distances = distances.min(axis=1)

    # Get indices of the processed papers sorted by distance
    sorted_indices = np.argsort(min_distances)

    output_folder = "./model_output_GPT4-o"

    # Generate filename for each processed paper
    def get_filename_for_paper(doi: str) -> str:
        """Return the sanitized JSON filename used for a paper's DOI in the database."""
        sanitized_doi = utils.sanitize_filename(doi)
        return f"{sanitized_doi}.json"

    # Select papers, checking if they already exist in the database
    nearest_papers = []
    papers_checked = 0

    for i in sorted_indices:
        if len(nearest_papers) >= number_of_selected_paper:
            break

        paper_doi = processed_data[i].get("DOI", "Unknown")
        filename = get_filename_for_paper(paper_doi)
        json_file_path = os.path.join(output_folder, filename)

        # Check if this paper already exists in the database
        if os.path.exists(json_file_path):
            print(f"Skipping {filename}: JSON file already exists in the database.")
            continue

        # Add paper to the selection
        nearest_papers.append(
            {
                "Title": processed_data[i]["Title"],
                "Abstract": processed_data[i]["Abstract"],
                "Score": processed_data[i].get("Score", 0),
                "DOI": paper_doi,
                "Source": processed_data[i].get("Source", "Unknown"),
                key: processed_data[i].get(key, "Unknown"),
                "Similarity": 1 - min_distances[i],
                "filename": filename,
            }
        )
        papers_checked += 1

        if papers_checked >= len(sorted_indices):
            print(
                f"Warning: Checked all {papers_checked} papers but only found {len(nearest_papers)} new papers."
            )
            break

    # Save the results to the specified JSON file
    with open(selected_papers_path, "w") as outfile:
        json.dump(nearest_papers, outfile, indent=2)

    print(
        f"Found {len(nearest_papers)} new papers out of {number_of_selected_paper} requested. Results saved to {selected_papers_path}."
    )


def get_embedding(client: OpenAI, text: str, model: str = "text-embedding-3-small") -> Tuple[list, int]:
    """Get embeddings for a given text.

    Args:
        client: OpenAI client used to create the embedding.
        text: Text to embed (newlines are replaced with spaces).
        model: OpenAI embedding model name.

    Returns:
        A (embedding, token_usage) tuple with the embedding vector and the total
        number of tokens billed for the request.
    """
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    embedding = response.data[0].embedding
    token_usage = response.usage.total_tokens
    return embedding, token_usage


def main(
    file_path: str,
    output_dir: str,
    doi_list_path: str,
    selected_papers_path: str,
    score_limit: float,
    number_of_selected_paper: int,
    key: str,
    values: Sequence[Any],
    new_papers_path: str,
) -> None:
    """Embed scored papers and select the ones most similar to new candidate papers.

    Args:
        file_path: Path to the JSON file with scored papers to embed.
        output_dir: Base directory embeddings and results are stored under.
        doi_list_path: Path to the JSON file tracking DOIs that already have an
            embedding.
        selected_papers_path: Destination path for the selected-papers JSON file.
        score_limit: Minimum "Score" value a paper needs to be embedded.
        number_of_selected_paper: Maximum number of nearest papers to select.
        key: Metadata key used to filter/annotate candidate papers.
        values: Allowed values for `key`.
        new_papers_path: Path to the CSV of new candidate papers.
    """
    # Ensure output_2 directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Step 1: Process embeddings and save them in individual files
    process_embeddings(file_path, output_dir, client, score_limit, doi_list_path)

    # Step 2: Find and save the x nearest papers including new papers
    find_nearest_paper_with_new(
        output_dir,
        selected_papers_path,
        key,
        values,
        number_of_selected_paper,
        new_papers_path,
        client,
    )
