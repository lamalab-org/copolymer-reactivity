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

def load_existing_embedding(output_dir, doi):
    """Check if an embedding already exists for a given DOI and load it."""
    embeddings_dir = os.path.join(output_dir, "embeddings")
    print(doi)
    sanitized_doi = utils.sanitize_filename(doi)
    filename = os.path.join(embeddings_dir, f"{sanitized_doi}.json")
    if os.path.exists(filename):
        with open(filename, "r") as f:
            data = json.load(f)
            return data.get("Embedding")
    return None

def process_embeddings(file_path, output_dir, client, score, doi_list_path):
    """Process and save embeddings for each paper."""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {file_path}. Details: {e}")
        return

    if not isinstance(data, list):
        print("Error: Expected a list of papers in the JSON file. Exiting.")
        return

    existing_dois = load_existing_doi_list(doi_list_path)
    total_tokens = 0

    for paper in data:
        print(f"Processing embedding of {paper}.")
        if paper.get("DOI") in existing_dois:
            continue

        title = paper.get("Title")
        abstract = paper.get("Abstract")
        score_value = paper.get("Score", 0)
        doi = paper.get("DOI", "Unknown")

        if score_value < score:
            print(
                f"Warning: Skipped paper with DOI {doi} because its score ({score_value}) is below the threshold ({score})."
            )
            continue

        if not title and not abstract:
            continue

        text = f"{title if title else ''}. {abstract if abstract else ''}".strip()

        # Check if embedding already exists
        existing_embedding = load_existing_embedding(output_dir, paper.get("DOI"))
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

import os
import json
import requests
from scipy.spatial.distance import cdist
from openai import OpenAI
import copolextractor.utils as utils

def embed_filtered_papers(new_papers_path, output_dir, client, key, values):
    """Embed only filtered new papers from a given JSON file."""
    print(new_papers_path)

    try:
        with open(new_papers_path, "r") as f:
            new_papers = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {new_papers_path}. Details: {e}")
        return []

    print(new_papers_path)

    filtered_papers = [
        paper for paper in new_papers if paper.get(key) in values
    ]

    embedded_papers = []
    for paper in filtered_papers:
        print(f"Processing embedding of {paper}.")
        title = paper.get("Title")
        abstract = paper.get("Abstract")

        # Check for missing Title and Abstract and use CrossRef if needed
        if not title and not abstract:
            print(f"Missing Title and Abstract for paper with DOI: {paper.get('source')}. Fetching from CrossRef...")
            crossref_data = get_crossref_data(paper.get("source"), paper.get("Source", "Unknown"), paper.get("Format", "Unknown"))
            title = crossref_data.get("Title")
            abstract = crossref_data.get("Abstract")

            if title == "No title" and abstract == "No abstract available":
                print(f"CrossRef could not provide Title and Abstract for DOI: {paper.get('source')}.")
                continue
            else:
                paper["Title"] = title
                paper["Abstract"] = abstract

        text = f"{title if title else ''}. {abstract if abstract else ''}".strip()

        # Check if embedding already exists
        existing_embedding = load_existing_embedding(output_dir, paper.get("source"))
        if existing_embedding is not None:
            paper["Embedding"] = existing_embedding
            embedded_papers.append(paper)
            continue

        try:
            embedding, _ = get_embedding(client, text)
            paper["Embedding"] = embedding
            embedded_papers.append(paper)
        except Exception as e:
            print(f"Error embedding paper {paper.get('source', 'Unknown')}: {e}")
            continue

    return embedded_papers

def get_crossref_data(doi, source, format_type):
    """
    Fetch metadata from CrossRef API for a given DOI.
    """
    url = f"https://api.crossref.org/works/{doi}"
    print(type(url))
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
            "Error": f"Unable to fetch data (Status Code: {response.status_code})",
        }


def find_nearest_paper_with_new(output_dir, selected_papers_path, key, values, number_of_selected_paper, new_papers_path, client):
    """
    Find and save the nearest papers based on embeddings, where new papers are compared to processed papers.
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

    # Get indices of the `number_of_selected_paper` processed papers with the smallest distances
    nearest_indices = np.argsort(min_distances)[:number_of_selected_paper]

    # Select the nearest papers, including the similarity scores
    nearest_papers = [
        {
            "Title": processed_data[i]['Title'],
            "Abstract": processed_data[i]['Abstract'],
            "Score": processed_data[i].get("Score", 0),
            "DOI": processed_data[i].get("DOI"),
            "Source": processed_data[i].get("Source", "Unknown"),
            key: processed_data[i].get(key, "Unknown"),
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


def main(file_path, output_dir, doi_list_path, selected_papers_path, score_limit, number_of_selected_paper, key, values, new_papers_path):

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Step 1: Process embeddings and save them in individual files
    process_embeddings(file_path, output_dir, client, score_limit, doi_list_path)

    # Step 2: Find and save the x nearest papers including new papers
    find_nearest_paper_with_new(output_dir, selected_papers_path, key, values, number_of_selected_paper, new_papers_path, client)
