import os
import json
import requests
from crossref.restful import Works


def add_to_database(doi, source, format_type, extracted_data):
    """
    Add a DOI entry to the database if it doesn't already exist.
    """
    if not any(item["doi"] == doi for item in extracted_data):
        extracted_data.append({"doi": doi, "source": source, "format": format_type})
    else:
        print(f"DOI {doi} is already in the database.")


def process_copol_database(copol_file_path, extracted_data):
    """
    Process copol database papers and add them to the database.
    """
    print("Processing copol database papers...")
    with open(copol_file_path, "r") as file:
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
    print("Processing CrossRef papers...")
    works = Works(timeout=60)
    query_result = works.query(bibliographic=query).select(
        "DOI", "title", "author", "type", "publisher", "issued"
    )
    results = [item for item in query_result]

    with open(output_crossref_file, "w") as file:
        json.dump(results, file, indent=4)

    for entry in results:
        if "DOI" in entry:
            add_to_database(entry["DOI"], "crossref", None, extracted_data)


def get_crossref_data(doi, source, format_type):
    """
    Fetch metadata from CrossRef API for a given DOI.
    """
    url = f"https://api.crossref.org/works/{doi}"
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


def is_doi_processed(doi, results):
    """
    Check if a DOI has already been processed.
    """
    return any(entry.get("DOI") == doi for entry in results)


def fetch_and_save_metadata(input_file, metadata_output_file, extracted_data):
    """
    Fetch metadata for DOIs from the database and save to a file.
    """
    print("Fetching metadata for DOIs...")
    if os.path.exists(metadata_output_file):
        with open(metadata_output_file, "r") as output_file:
            results = json.load(output_file)
    else:
        results = []

    for entry in extracted_data:
        doi_url = entry["doi"]
        doi = doi_url.split("doi.org/")[-1]  # Extract DOI from the URL
        source = entry["source"]
        format_type = entry["format"]

        if not is_doi_processed(doi, results):
            result = get_crossref_data(doi, source, format_type)
            print(result)
            results.append(result)

            # Save after each processed entry
            with open(metadata_output_file, "w") as output_file:
                json.dump(results, output_file, indent=4)
        else:
            print(f"DOI {doi} is already processed.")

    # Final save
    with open(metadata_output_file, "w") as output_file:
        json.dump(results, output_file, indent=4)
    print(f"Metadata saved to {metadata_output_file}.")


def save_extracted_data(output_file_path, extracted_data):
    """
    Save the extracted data to a JSON file.
    """
    with open(output_file_path, "w") as output_file:
        json.dump(extracted_data, output_file, indent=4)
    print(f"Extracted data saved to {output_file_path}.")


def main(crossref_query):
    # Define file paths
    copol_file_path = (
        "../../data_extraction/data_extraction_GPT-4o/output/copol_paper_list.json"
    )
    output_crossref_file = (
        "../../data_extraction/obtain_data/output/crossref_search.json"
    )
    extracted_doi_file = "../../data_extraction/obtain_data/output/collected_doi.json"
    metadata_output_file = (
        "../../data_extraction/obtain_data/collected_doi_metadata.json"
    )

    # Initialize extracted data list
    extracted_data = []

    # Process data sources
    process_copol_database(copol_file_path, extracted_data)
    process_crossref(crossref_query, output_crossref_file, extracted_data)

    # Save combined DOIs
    save_extracted_data(extracted_doi_file, extracted_data)

    # Fetch and save metadata
    fetch_and_save_metadata(extracted_doi_file, metadata_output_file, extracted_data)


if __name__ == "__main__":
    # Define CrossRef query
    crossref_query = "'copolymerization' AND 'reactivity ratio'"

    main(crossref_query)
