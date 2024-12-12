import os
import json
import numpy as np
from scipy.spatial.distance import cdist
from openai import OpenAI
import re


def sanitize_filename(filename):
    """Replace invalid characters in filename with underscores."""
    return re.sub(r'[<>:"/\\|?*]', "_", filename)


def save_embedding_file(output_dir, paper, doi_list_path):
    """Save individual embedding JSON files and update DOI list."""
    embeddings_dir = os.path.join(output_dir, "embeddings")
    os.makedirs(embeddings_dir, exist_ok=True)

    sanitized_doi = sanitize_filename(paper["DOI"])
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
    with open(file_path, "r") as f:
        data = json.load(f)

    # Load existing DOI list
    existing_dois = load_existing_doi_list(doi_list_path)

    # Initialize total token counter
    total_tokens = 0

    for paper in data:
        if paper.get("Score", 0) >= score and "Title" in paper and "Abstract" in paper:
            doi = paper.get("DOI")
            if doi in existing_dois:
                print(f"Embedding already exists for DOI: {doi}. Skipping.")
                continue

            print(f"Processing paper: {paper['Title']}")
            text = f"{paper['Title']}. {paper['Abstract']}"

            # Generate embedding only if DOI is not in the list
            embedding, token_usage = get_embedding(client, text)
            total_tokens += token_usage

            # Save embedding to a separate JSON file and update DOI list
            paper["Embedding"] = embedding
            save_embedding_file(output_dir, paper, doi_list_path)
            existing_dois.append((doi))
            save_json(existing_dois, doi_list_path)

    print(f"Total tokens used: {total_tokens}")


def find_nearest_papers(output_dir, selected_papers_path, number_of_selected_paper=200):
    """Find and save the nearest papers based on embeddings, focusing on 'copol database'."""
    embeddings_path = os.path.join(output_dir, "embeddings/embedded_papers.json")

    # Load processed data
    with open(embeddings_path, "r") as f:
        processed_data = json.load(f)

    # Extract embeddings and sources
    embeddings = np.array([entry["Embedding"] for entry in processed_data])
    sources = [entry.get("Source", "") for entry in processed_data]

    # Identify indices for "copol database" papers and other papers
    copol_indices = [
        i for i, source in enumerate(sources) if source == "copol database"
    ]
    other_indices = [
        i for i, source in enumerate(sources) if source != "copol database"
    ]

    if not copol_indices or not other_indices:
        print("Error: Either 'copol database' or other sources are empty.")
        return

    # Extract embeddings for the two groups
    copol_embeddings = embeddings[copol_indices]
    other_embeddings = embeddings[other_indices]

    # Compute distances between "copol database" papers and other papers
    distances = cdist(other_embeddings, copol_embeddings, metric="euclidean")

    if distances.size == 0:
        print("Error: Distance matrix is empty.")
        return

    # Find the minimum distance for each non-"copol database" paper
    min_distances = distances.min(axis=1)

    # Get indices of the `number_of_selected_paper` papers with the smallest distances
    nearest_indices = np.argsort(min_distances)[:number_of_selected_paper]

    # Select the nearest papers
    nearest_papers = [
        {
            "Title": processed_data[other_indices[i]]["Title"],
            "Abstract": processed_data[other_indices[i]]["Abstract"],
            "Score": processed_data[other_indices[i]]["Score"],
            "DOI": processed_data[other_indices[i]].get("DOI"),
            "Source": processed_data[other_indices[i]].get("Source", "Unknown"),
        }
        for i in nearest_indices
    ]

    # Save the results to the specified JSON file
    with open(selected_papers_path, "w") as outfile:
        json.dump(nearest_papers, outfile, indent=2)

    print(
        f"The {number_of_selected_paper} nearest papers have been saved to {selected_papers_path}."
    )


def save_json(data, path):
    """Save JSON data to a file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def get_embedding(client, text, model="text-embedding-3-small"):
    """Get embeddings for a given text."""
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    embedding = response.data[0].embedding
    token_usage = response.usage.total_tokens
    return embedding, token_usage


def save_json(data, path):
    """Save JSON data to a file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    # Define input and output paths
    input_file_path = "../../../data_extraction/obtain_data/output/scored_doi.json"
    output_dir = "../../../data_extraction/obtain_data/output"
    doi_list_path = os.path.join(output_dir, "embeddings/existing_embeddings.json")
    selected_papers_path = os.path.join(output_dir, "selected_200_papers.json")
    number_of_selected_paper = 200
    score_limit = 65

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Step 1: Process embeddings and save them in individual files
    process_embeddings(input_file_path, output_dir, client, score_limit, doi_list_path)

    # Step 2: Find and save the x nearest papers
    find_nearest_papers(output_dir, selected_papers_path, number_of_selected_paper)


if __name__ == "__main__":
    main()
