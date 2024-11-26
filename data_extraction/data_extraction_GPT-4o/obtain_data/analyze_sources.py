import json
from collections import Counter
import re


file_path = '../collected_data/doi_with_metadata.json'
with open(file_path, 'r') as file:
    data = json.load(file)

# Filter entries with source 'copol database'
filtered_entries = [entry for entry in data if entry.get('Source', '').lower() == 'copol database']
print('Number of analyzed papers: ', len(filtered_entries))


# Count how often each journal occurs
journal_counter = Counter([entry['Journal'] for entry in filtered_entries])

# Analyze the words in the titles
word_counter = Counter()

# Iterate through filtered entries and count words in titles
for entry in filtered_entries:
    # Get the title and split into words (ignoring case and special characters)
    title = entry['Title']
    words = re.findall(r'\b\w+\b', title.lower())  # Extract words and convert to lowercase
    word_counter.update(words)

# Print the journal frequency
print("Journal frequencies in 'copol database':")
for journal, count in journal_counter.most_common():
    print(f"{journal}: {count}")

# Print the most common words in the titles
print("\nMost common words in titles:")
for word, count in word_counter.most_common(30):
    print(f"{word}: {count}")
