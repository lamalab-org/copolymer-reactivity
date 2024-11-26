import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import mpld3  # For saving interactive plots as HTML
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI()
file_path = "scored_doi.json"
output_dir = 'output'
output_path = os.path.join(output_dir, 'embedded_papers.json')
html_path = os.path.join(output_dir, 'interactive_embeddings_plot.html')

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load all papers
with open(file_path, "r") as f:
    data = json.load(f)

# Load already processed papers (if the output JSON exists)
try:
    with open(output_path, "r") as f:
        processed_data = json.load(f)
except FileNotFoundError:
    processed_data = []

# Create a set of processed DOIs and Titles to avoid reprocessing
processed_identifiers = {(paper.get("DOI"), paper['Title']) for paper in processed_data}

# Function to compute the embedding for a single text and track token usage
def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")  # Clean up newline characters
    response = client.embeddings.create(input=[text], model=model)
    embedding = response.data[0].embedding
    token_usage = response.usage.total_tokens  # Access total tokens as an attribute
    return embedding, token_usage

# Initialize total token counter
total_tokens = 0

# Filter all papers with Score >= 65 that haven't been processed
filtered_data_high_score = [
    paper for paper in data
    if paper.get('Score', 0) >= 65 and 'Title' in paper and 'Abstract' in paper and
       (paper.get("DOI"), paper['Title']) not in processed_identifiers
]

# Process each high-score paper and embed
for paper in filtered_data_high_score:
    print(f"Processing high-score paper: {paper['Title']}")

    # Combine Title and Abstract for embedding
    text = f"{paper['Title']}. {paper['Abstract']}"
    embedding, token_usage = get_embedding(text)
    total_tokens += token_usage

    # Append the paper details to processed_data
    processed_data.append({
        "Title": paper['Title'],
        "Abstract": paper['Abstract'],
        "Score": paper['Score'],
        "DOI": paper.get("DOI"),
        "Source": paper.get("Source", "High Score"),  # Default source for high-score papers if missing
        "Embedding": embedding
    })

    # Save the updated list to JSON after each entry
    with open(output_path, 'w') as outfile:
        json.dump(processed_data, outfile, indent=2)

print("All high-score embeddings computed and stored.")

# Process 100 additional papers with Score 0
filtered_data_score_0 = [
    paper for paper in data
    if paper.get('Score') == 0 and 'Title' in paper and 'Abstract' in paper and
       (paper.get("DOI"), paper['Title']) not in processed_identifiers
][:0]

for paper in filtered_data_score_0:
    print(f"Processing score 0 paper: {paper['Title']}")

    # Combine Title and Abstract for embedding
    text = f"{paper['Title']}. {paper['Abstract']}"
    embedding, token_usage = get_embedding(text)
    total_tokens += token_usage

    # Append the paper details to processed_data
    processed_data.append({
        "Title": paper['Title'],
        "Abstract": paper['Abstract'],
        "Score": paper['Score'],
        "DOI": paper.get("DOI"),
        "Source": paper.get("Source", "Score 0"),  # Default source for Score 0 papers if missing
        "Embedding": embedding
    })

    # Save the updated list to JSON after each entry
    with open(output_path, 'w') as outfile:
        json.dump(processed_data, outfile, indent=2)

print("All score 0 embeddings computed and stored.")
print(f"Total tokens used: {total_tokens}")

# Extract embeddings, titles, and separate "copol database" entries for PCA
embeddings = np.array([entry["Embedding"] for entry in processed_data])
titles = [entry["Title"] for entry in processed_data]  # Ensure lowercase `titles`
scores = np.array([entry["Score"] for entry in processed_data])
sources = [entry.get("Source") for entry in processed_data]

# Reduce embeddings to 2D using PCA
# Reduce embeddings to 2D using PCA
if embeddings.shape[0] > 1:
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    # Separate "copol database" entries
    copol_indices = [i for i, source in enumerate(sources) if source == 'copol database']
    other_indices = [i for i in range(len(sources)) if i not in copol_indices]

    # Function to check for abstract presence
    def has_abstract(paper):
        # Check if the abstract exists and is not explicitly "No abstract available"
        abstract = paper.get('Abstract', '')
        return 'No' if abstract.strip().lower() == 'no abstract available' or not abstract.strip() else 'Yes'

    # Generate tooltips including title and abstract presence
    other_tooltips = [
        f"Title: {titles[i]}<br>Abstract: {has_abstract(processed_data[i])}"
        for i in other_indices
    ]
    copol_tooltips = [
        f"Title: {titles[i]}<br>Abstract: {has_abstract(processed_data[i])}"
        for i in copol_indices
    ]

    # Plot non-"copol database" entries with color mapped by score
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        reduced_embeddings[other_indices, 0],
        reduced_embeddings[other_indices, 1],
        c=scores[other_indices],
        cmap='viridis',
        alpha=0.7,
        label="High Score/Score 0 Papers"
    )

    # Overlay "copol database" entries with dark red color
    copol_scatter = ax.scatter(
        reduced_embeddings[copol_indices, 0],
        reduced_embeddings[copol_indices, 1],
        color='#8B0000',
        alpha=0.7,
        label="'copol database' Papers"
    )

    # Add color bar and labels
    plt.colorbar(scatter, label='Score')
    plt.title("2D Plot of Embeddings Colored by Score")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()

    # Use mpld3 to add hover text and save the interactive plot as HTML
    other_tooltip_plugin = mpld3.plugins.PointLabelTooltip(scatter, labels=other_tooltips)
    copol_tooltip_plugin = mpld3.plugins.PointLabelTooltip(copol_scatter, labels=copol_tooltips)
    mpld3.plugins.connect(fig, other_tooltip_plugin)
    mpld3.plugins.connect(fig, copol_tooltip_plugin)

    # Save the interactive plot as HTML
    mpld3.save_html(fig, html_path)

    print(f"Interactive plot saved as {html_path}")

else:
    print("Insufficient embeddings for PCA visualization.")
