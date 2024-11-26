import requests
import json
import os


# Function to get metadata from CrossRef API for a given DOI
def get_crossref_data(doi, source, format):
    url = f"https://api.crossref.org/works/{doi}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        item = data.get('message', {})

        # Retrieve title, abstract, keywords, and journal
        title = item.get('title', [])
        title = title[0] if title else 'No title'

        abstract = item.get('abstract', 'No abstract available')
        keywords = item.get('subject', 'No keywords available')

        journal = item.get('container-title', [])
        journal = journal[0] if journal else 'No journal title'

        return {
            'DOI': doi,
            'Title': title,
            'Abstract': abstract,
            'Keywords': keywords,
            'Journal': journal,
            'Source': source,
            'Format': format
        }
    else:
        return {
            'DOI': doi,
            'Error': f"Unable to fetch data (Status Code: {response.status_code})"
        }


# File paths for input and output JSON files
file_path = 'obtain_data/unique_collected_doi.json'
output_file_path = 'collected_data/doi_with_metadata.json'

# Load already processed entries from the JSON file, if it exists
if os.path.exists(output_file_path):
    with open(output_file_path, 'r') as output_file:
        results = json.load(output_file)
else:
    results = []

# Load the list of DOIs from the input file
with open(file_path, 'r') as file:
    doi_list = json.load(file)


# Function to check if a DOI has already been processed
def is_doi_processed(doi, results):
    for entry in results:
        if entry.get('DOI') == doi:
            return True
    return False


for entry in doi_list:
    if entry['source'] == 'copol database':
        doi_url = entry['doi']
        doi = doi_url.split("doi.org/")[-1]  # Extract DOI from the URL
        source = entry['source']
        format = entry['format']

        # Check if the DOI has already been processed
        if not is_doi_processed(doi, results):
            result = get_crossref_data(doi, source, format)
            print(result)
            results.append(result)

            # Save the results after each new entry
            with open(output_file_path, 'w') as output_file:
                json.dump(results, output_file, indent=4)
        else:
            print(f'DOI {doi} from copol database is already processed.')



# Iterate through the DOI list and fetch metadata if the DOI has not been processed yet
for entry in doi_list:
    doi_url = entry['doi']
    doi = doi_url.split("doi.org/")[-1]  # Extract DOI from the URL

    source = entry['source']
    format = entry['format']

    # Check if the DOI has already been processed
    if not is_doi_processed(doi, results):
        result = get_crossref_data(doi, source, format)
        print(result)
        results.append(result)

        # Save the results after each new entry
        with open(output_file_path, 'w') as output_file:
            json.dump(results, output_file, indent=4)
    else:
        print(f'DOI {doi} is already processed.')

# Final save to ensure all data is written
with open(output_file_path, 'w') as output_file:
    json.dump(results, output_file, indent=4)
