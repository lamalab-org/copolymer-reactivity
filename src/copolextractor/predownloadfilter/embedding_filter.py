import os
import json
import numpy as np
from scipy.spatial.distance import cdist
from openai import OpenAI
import copolextractor.utils as utils


def save_embedding_file(output_dir, paper, doi_list_path):
    """Save individual embedding JSON files and update DOI list."""
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


def load_existing_doi_list(doi_list_path):
    """Load the existing DOI list from file."""
    if os.path.exists(doi_list_path):
        with open(doi_list_path, "r") as f:
            return json.load(f)
    return []


def process_embeddings(file_path, output_dir, client, score, doi_list_path):
    """Process and save embeddings for each paper."""
    # Load all papers
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {file_path}. Details: {e}")
        return

    # Validate loaded data
    if not isinstance(data, list):
        print("Error: Expected a list of papers in the JSON file. Exiting.")
        return

    # Load existing DOI list
    existing_dois = load_existing_doi_list(doi_list_path)

    # Initialize total token counter
    total_tokens = 0
    skipped_count = 0

    for i, paper in enumerate(data, start=1):
        if not isinstance(paper, dict):
            print(f"Warning: Skipped entry at index {i} due to invalid format.")
            skipped_count += 1
            continue

        doi = paper.get("DOI", "Unknown")
        title = paper.get("Title")
        abstract = paper.get("Abstract")
        score_value = paper.get("Score", 0)

        if not title or not abstract:
            print(
                f"Warning: Skipped paper with DOI {doi} due to missing Title or Abstract."
            )
            skipped_count += 1
            continue

        if score_value < score:
            print(
                f"Warning: Skipped paper with DOI {doi} because its score ({score_value}) is below the threshold ({score})."
            )
            skipped_count += 1
            continue

        if doi in existing_dois:
            print(f"Embedding already exists for DOI: {doi}. Skipping.")
            continue

        print(f"Processing paper {i}: {title}")
        text = f"{title}. {abstract}"

        try:
            embedding, token_usage = get_embedding(client, text)
            total_tokens += token_usage
            paper["Embedding"] = embedding

            # Save embedding to a separate JSON file and update DOI list
            save_embedding_file(output_dir, paper, doi_list_path)
            existing_dois.append(doi)
            utils.save_json(existing_dois, doi_list_path)

        except Exception as e:
            print(f"Error processing paper with DOI {doi}. Details: {e}")
            skipped_count += 1
            continue

    print(f"Total tokens used: {total_tokens}")
    if skipped_count > 0:
        print(f"Total papers skipped: {skipped_count}. See warnings above.")


def find_nearest_papers(output_dir, selected_papers_path, number_of_selected_paper=200):
    """
    Find and save the nearest papers based on embeddings, focusing on 'copol database'.
    """
    embeddings_path = os.path.join(output_dir, "embeddings/embedded_papers.json")

    # Load processed data
    with open(embeddings_path, "r") as f:
        processed_data = json.load(f)

    # Extract embeddings and sources
    embeddings = np.array([entry["Embedding"] for entry in processed_data])
    sources = [entry.get("Source", "") for entry in processed_data]

    # Identify indices for "copol database" papers and other papers
    copol_indices = []
    other_indices = []

    for i, source in enumerate(sources):
        if source == "copol database":
            copol_indices.append(i)
        else:
            other_indices.append(i)

    if not copol_indices or not other_indices:
        print("Error: Either 'copol database' or other sources are empty.")
        return

    # Extract embeddings for the two groups
    copol_embeddings = embeddings[copol_indices]
    other_embeddings = embeddings[other_indices]

    # Compute cosine distances between "copol database" papers and other papers
    distances = cdist(other_embeddings, copol_embeddings, metric="cosine")

    if distances.size == 0:
        print("Error: Distance matrix is empty.")
        return

    # Find the minimum distance for each non-"copol database" paper
    min_distances = distances.min(axis=1)

    # Get indices of the `number_of_selected_paper` papers with the smallest distances
    nearest_indices = np.argsort(min_distances)[:number_of_selected_paper]

    # Select the nearest papers, including the similarity scores
    nearest_papers = [
        {
            "Title": processed_data[other_indices[i]]["Title"],
            "Abstract": processed_data[other_indices[i]]["Abstract"],
            "Score": processed_data[other_indices[i]]["Score"],
            "DOI": processed_data[other_indices[i]].get("DOI"),
            "Source": processed_data[other_indices[i]].get("Source", "Unknown"),
            "Similarity": 1 - min_distances[i],  # Cosine similarity (1 - cosine distance)
        }
        for i in nearest_indices
    ]

    # Save the results to the specified JSON file
    with open(selected_papers_path, "w") as outfile:
        json.dump(nearest_papers, outfile, indent=2)

    print(
        f"The {number_of_selected_paper} nearest papers have been saved to {selected_papers_path}."
    )


def get_embedding(client, text, model="text-embedding-3-small"):
    """Get embeddings for a given text."""
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    embedding = response.data[0].embedding
    token_usage = response.usage.total_tokens
    return embedding, token_usage


def main(file_path, output_dir, doi_list_path, selected_papers_path, score_limit, number_of_selected_paper):

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Step 1: Process embeddings and save them in individual files
    process_embeddings(file_path, output_dir, client, score_limit, doi_list_path)

    # Step 2: Find and save the x nearest papers
    find_nearest_papers(output_dir, selected_papers_path, number_of_selected_paper)

