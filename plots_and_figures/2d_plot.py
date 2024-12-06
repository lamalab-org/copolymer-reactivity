import json
import numpy as np
import umap
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

# Load the JSON data
with open('../data_extraction/data_extraction_GPT-4o/output/copol_database/copol_extracted_data.json', 'r') as file:
    data = json.load(file)


# Function to sort and combine fingerprints
def sort_and_combine_fingerprints(fingerprint_1, fingerprint_2, fingerprint_solvent):
    if sum(fingerprint_1) < sum(fingerprint_2):
        fingerprint_1, fingerprint_2 = fingerprint_2, fingerprint_1
    combined_fingerprint = (np.array(fingerprint_1) + np.array(fingerprint_2) + np.array(fingerprint_solvent))
    return combined_fingerprint


# Prepare data for UMAP and collect r1*r2 products for coloring
fingerprints = []
colors = []
color_labels = []
for entry in data:
    fingerprint_1 = entry['fingerprint_1']
    fingerprint_2 = entry['fingerprint_2']
    fingerprint_solvent = entry['solvent_fingerprint']
    combined_fingerprint = sort_and_combine_fingerprints(fingerprint_1, fingerprint_2, fingerprint_solvent)

    r1 = entry['r_values'].get('constant_1')
    r2 = entry['r_values'].get('constant_2')

    if r1 is not None and r2 is not None:
        fingerprints.append(combined_fingerprint)
        product = r1 * r2
        if product <= 0.1:
            colors.append('blue')
            color_labels.append('0')
        elif 0.1 < product <= 1:
            colors.append('green')
            color_labels.append('0 < r1*r2 < 1')
        elif 1 < product <= 1.5:
            colors.append('purple')
            color_labels.append('r1*r2 = 1')
        elif product > 1.5:
            colors.append('orange')
            color_labels.append('> 1')

# Ensure fingerprints and colors have the same length
fingerprints = np.array(fingerprints)
colors = np.array(colors)

# Verify the lengths are the same
print(f'Number of fingerprints: {len(fingerprints)}')
print(f'Number of colors: {len(colors)}')

# Scale the data
scaled_fingerprints = StandardScaler().fit_transform(fingerprints)

# Initialize UMAP and fit_transform the data
reducer = umap.UMAP()
embedding = reducer.fit_transform(scaled_fingerprints)

# Verify the lengths are the same after transformation
print(f'Number of embedding points: {len(embedding)}')
print(f'Number of colors: {len(colors)}')

# Create a 2D scatter plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, alpha=0.6)
plt.gca().set_aspect('equal', 'datalim')

# Create legend
unique_labels = list(set(color_labels))
patches = [mpatches.Patch(color=color, label=label) for color, label in
           zip(['blue', 'green', 'purple', 'orange'], unique_labels)]
plt.legend(handles=patches, title='Classes')

plt.title('UMAP projection of the Molecule Fingerprints', fontsize=24)
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')

plt.savefig('2D_plot.png', bbox_inches='tight', dpi=300)
plt.show()