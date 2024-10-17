import json
import os


def add_to_database(doi,source, format_type, extracted_data):
    if not any(item['doi'] == doi for item in extracted_data):
        extracted_data.append({
            "doi": doi,
            "source": source,
            "format": format_type
        })
        return extracted_data
    else:
        print(f"DOI {doi} is already in database.")


extracted_data = []

file_path = './collected_data/enhanced_doi_list_unique.json'

with open(file_path, 'r') as file:
    copol_data = json.load(file)

# copol database
print('processing copol database papers.')
for entry in copol_data:
    if entry["paper"]:
        doi = entry["paper"]
        source = "copol database"
        format_type = "pdf"

        add_to_database(doi, source, format_type, extracted_data)


# pygetpaper
print('processing pygetpaper papers.')
input_dir = './files_pygetpaper'
# Iterate over all subdirectories in the input directory
for subdir in os.listdir(input_dir):

    subdir_path = os.path.join(input_dir, subdir)
    # Ensure it's a directory before proceeding
    if os.path.isdir(subdir_path):
        # Iterate over all files in the subdirectory
        for filename in os.listdir(subdir_path):
            if filename.endswith(".json"):
                file_path = os.path.join(subdir_path, filename)
                with open(file_path, 'r') as file:
                    pyget_data = json.load(file)
                if "doi" in pyget_data:
                    doi = pyget_data["doi"]
                    source = 'pygetpaper'
                    format_type = 'XML'
                    add_to_database(doi, source, format_type, extracted_data)


# crossref
print('processing crossref papers.')
file_path = 'copol_crossref.json'

with open(file_path, 'r') as file:
    crossref_data = json.load(file)
for entry in crossref_data:
        if "DOI" in entry:
            doi = entry["DOI"]
            source = 'crossref'
            format_type = None
            add_to_database(doi, source, format_type, extracted_data)


print(extracted_data)
print(len(extracted_data))

output_file_path = 'unique_collected_doi.json'

with open(output_file_path, 'w') as output_file:
    json.dump(extracted_data, output_file, indent=4)


