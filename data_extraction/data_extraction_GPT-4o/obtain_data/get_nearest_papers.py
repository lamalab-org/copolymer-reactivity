import numpy as np
import json
import os
from scipy.spatial.distance import cdist

# Paths
output_dir = 'output'
output_path = os.path.join(output_dir, 'embedded_papers.json')
selected_papers_path = os.path.join(output_dir, 'next_200_nearest_papers.json')

# Load the processed data
with open(output_path, "r") as f:
    processed_data = json.load(f)

# Extract embeddings and sources
embeddings = np.array([entry["Embedding"] for entry in processed_data])
sources = [entry.get("Source", "") for entry in processed_data]

# Identify indices for "copol database" papers and other papers
copol_indices = [i for i, source in enumerate(sources) if source == 'copol database']
other_indices = [i for i, source in enumerate(sources) if source != 'copol database']

copol_embeddings = embeddings[copol_indices]
other_embeddings = embeddings[other_indices]

# Compute distances between "copol database" papers and other papers
distances = cdist(other_embeddings, copol_embeddings, metric='euclidean')

# Find the minimum distance for each non-"copol database" paper
min_distances = distances.min(axis=1)

# Get indices of the 200 papers with the smallest distances
nearest_indices = np.argsort(min_distances)[:200]

# Select the 200 nearest papers
nearest_papers = [
    {
        "Title": processed_data[other_indices[i]]["Title"],
        "Abstract": processed_data[other_indices[i]]["Abstract"],
        "Score": processed_data[other_indices[i]]["Score"],
        "DOI": processed_data[other_indices[i]].get("DOI"),
        "Source": processed_data[other_indices[i]].get("Source", "Unknown")
    }
    for i in nearest_indices
]

# Save the results to a new JSON file
with open(selected_papers_path, 'w') as outfile:
    json.dump(nearest_papers, outfile, indent=2)

print(f"The 200 nearest papers with required fields have been saved to {selected_papers_path}.")
