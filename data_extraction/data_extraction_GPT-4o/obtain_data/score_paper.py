import json
from collections import Counter

# Load your journal list
with open("journals.json", "r") as f:
    journal_list = json.load(f)

# Load your original data JSON
with open("../collected_data/doi_with_metadata.json", "r") as f:
    data = json.load(f)

# Define your keyword list
keywords = {
    "copolymerization": 10,
    "polymerization": 5,
    "monomers": 5,
    "copolymers": 5,
    "ratios": 20,
    "reactivity ratios": 40
}

def calculate_score(entry):

    #if entry.get("Source") == "copol database":
        #return None

    score = 0
    # Check if 'Journal' key exists and if journal is in the list
    if "Journal" in entry:
        journal_in_list = any(journal.lower() in entry["Journal"].lower() for journal in journal_list)
        if journal_in_list:
            score += 40

    # Check for weighted keywords in title and abstract
    title_abstract = entry.get("Title", "") + " " + entry.get("Abstract", "")
    for word, weight in keywords.items():
        if word.lower() in title_abstract.lower():
            score += weight

    entry["Score"] = score
    return entry


# Process entries and filter out None values
scored_data = [calculate_score(entry) for entry in data if calculate_score(entry) is not None]

# Sort by score and get top 50
top_50_papers = sorted(scored_data, key=lambda x: x["Score"], reverse=True)[:50]

# Print top 50 papers with title and score
for paper in top_50_papers:
    print(f"Title: {paper['Title']}, Score: {paper['Score']}")

score_counts = Counter(entry["Score"] for entry in scored_data)
print("\nScore distribution:")
for score, count in sorted(score_counts.items(), reverse=True):
    print(f"Score {score}: {count} papers")

print('Total number of papers: ', len(scored_data))


# Save the updated data to a new JSON file
with open("scored_doi.json", "w") as f:
    json.dump(scored_data, f, indent=2)
