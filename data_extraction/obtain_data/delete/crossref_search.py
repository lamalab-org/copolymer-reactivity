import json
import os
from crossref.restful import Works


def add_to_database(doi, source, format_type, extracted_data):
    """
    Add a DOI entry to the database if it doesn't already exist.
    """
    if not any(item['doi'] == doi for item in extracted_data):
        extracted_data.append({
            "doi": doi,
            "source": source,
            "format": format_type
        })
    else:
        print(f"DOI {doi} is already in the database.")


def process_copol_database(copol_file_path, extracted_data):
    """
    Process copol database papers and add them to the database.
    """
    print('Processing copol database papers...')
    with open(copol_file_path, 'r') as file:
        copol_data = json.load(file)

    for entry in copol_data:
        if entry["paper"]:
            doi = entry["paper"]
            source = "copol database"
            format_type = "pdf"
            add_to_database(doi, source, format_type, extracted_data)


def process_crossref(query, output_crossref_file, extracted_data):
    """
    Query CrossRef for papers and add them to the database.
    """
    print('Processing CrossRef papers...')
    works = Works(timeout=60)
    query_result = (
        works.query(bibliographic=query)
        .select("DOI")
    )
    results = [item for item in query_result]

    print('Crossref search resulted in ', len(results), ' doi.')

    with open(output_crossref_file, 'w') as file:
        json.dump(results, file, indent=4)

    for entry in results:
        if "DOI" in entry:
            doi = entry["DOI"]
            source = 'crossref'
            format_type = None
            add_to_database(doi, source, format_type, extracted_data)


def save_extracted_data(output_file_path, extracted_data):
    """
    Save the extracted data to a JSON file.
    """
    with open(output_file_path, 'w') as output_file:
        json.dump(extracted_data, output_file, indent=4)
    print(f"Extracted data saved to {output_file_path}.")


def main():
    # Define input and output file paths
    copol_file_path = '../../data_extraction_GPT-4o/collected_data/enhanced_doi_list_unique.json'
    output_crossref_file = 'output/crossref_search.json'
    output_file_path = 'output/collected_doi.json'

    # Define the CrossRef query
    crossref_query = "'copolymerization' AND 'reactivity ratio'"

    # Initialize extracted data list
    extracted_data = []

    # Process sources
    process_copol_database(copol_file_path, extracted_data)
    process_crossref(crossref_query, output_crossref_file, extracted_data)

    # Save the final combined dataset
    save_extracted_data(output_file_path, extracted_data)


if __name__ == "__main__":
    main()
