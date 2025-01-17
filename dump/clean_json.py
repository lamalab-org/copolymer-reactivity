import json
import os
import shutil
import re
import ast


def sanitize_filename(filename: str) -> str:
    """Replace invalid characters in filename with underscores."""
    return re.sub(r'[<>:"/\\|?*]', "_", filename)


def process_manual_dois(input_dir: str, output_dir: str, manual_doi_file: str):
    """Process files using DOIs from source field."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load manual mappings
    try:
        with open(manual_doi_file, 'r', encoding='utf-8') as f:
            manual_mappings = json.load(f)

        print(f"Loaded {len(manual_mappings)} mappings")

        # Create filename to source mapping
        source_mapping = {
            entry['filename']: entry.get('source', '')
            for entry in manual_mappings
            if 'source' in entry
        }

        # Process each file that has a source
        for filename, source in source_mapping.items():
            if not source:
                print(f"No source provided for {filename}, skipping...")
                continue

            input_path = os.path.join(input_dir, filename)
            if not os.path.exists(input_path):
                print(f"File not found: {input_path}")
                continue

            try:
                # Generate new filename from source (DOI)
                if "doi.org/" in source:
                    new_filename = sanitize_filename(source.split("doi.org/")[-1]) + ".json"
                else:
                    new_filename = sanitize_filename(source) + ".json"

                # Read and parse the original JSON
                with open(input_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Save with new filename
                output_path = os.path.join(output_dir, new_filename)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)

                print(f"Processed {filename} -> {new_filename}")

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue

    except Exception as e:
        print(f"Error loading mapping file: {str(e)}")
        return


def create_filename_mapping(mapping_file: str, manual_mapping_file: str) -> dict:
    """Create mapping from json filename to DOI using both mapping sources."""
    mapping = {}

    # Load original mapping
    try:
        with open(mapping_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for entry in data:
            if "out" in entry and "paper" in entry:
                pdf_path = entry["out"]
                json_name = os.path.basename(pdf_path).replace('.pdf', '.json')
                mapping[json_name] = entry["paper"]

        print(f"Created mapping for {len(mapping)} files from original mapping")

        # Load manual mapping
        if os.path.exists(manual_mapping_file):
            with open(manual_mapping_file, 'r', encoding='utf-8') as f:
                manual_data = json.load(f)

            manual_count = 0
            for entry in manual_data:
                if "filename" in entry and "source" in entry:
                    filename = entry["filename"]
                    if filename not in mapping:  # Only add if not in original mapping
                        mapping[filename] = entry["source"]
                        manual_count += 1

            print(f"Added {manual_count} entries from manual mapping")

    except Exception as e:
        print(f"Error creating mapping: {str(e)}")

    return mapping


def clean_json_file(input_dir: str, output_dir: str, mapping_file: str, manual_mapping_file: str):
    """Process all JSON files in the input directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load combined mapping
    filename_mapping = create_filename_mapping(mapping_file, manual_mapping_file)
    not_found_papers = []

    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            input_path = os.path.join(input_dir, filename)
            try:
                print(f"\nProcessing file: {filename}")

                # Read and parse the JSON
                with open(input_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()

                try:
                    json_str = ast.literal_eval(content)
                    data = json.loads(json_str)
                except (SyntaxError, ValueError) as e:
                    print(f"Error evaluating string: {e}")
                    continue

                # Try to get DOI from combined mapping
                new_filename = filename
                if filename in filename_mapping:
                    doi = filename_mapping[filename]
                    print(f"Found DOI in mapping: {doi}")
                    source = filename_mapping[filename]  # Store original mapping source

                    if "doi.org/" in doi:
                        new_filename = sanitize_filename(doi.split("doi.org/")[-1]) + ".json"
                    else:
                        new_filename = sanitize_filename(doi) + ".json"
                else:
                    print(f"No mapping found for {filename}")
                    # Save info for papers without any mapping
                    paper_info = {
                        "filename": filename,
                        "title": data.get("PDF_name", ""),
                        "year": data.get("year")
                    }
                    not_found_papers.append(paper_info)

                # Save cleaned JSON
                output_path = os.path.join(output_dir, new_filename)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)

                print(f"Successfully cleaned and saved as: {new_filename}")

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue

    # Save information about papers without any mapping
    if not_found_papers:
        new_not_found_file = "./papers_still_without_mapping.json"
        with open(new_not_found_file, 'w', encoding='utf-8') as f:
            json.dump(not_found_papers, f, indent=4, ensure_ascii=False)
        print(f"\nSaved information about {len(not_found_papers)} papers without mapping to {new_not_found_file}")


def main():
    input_dir = "../data_extraction/data_extraction_GPT-4o/output/copol_database/model_output_extraction"  # Directory with original files
    output_dir = "../data_extraction/data_extraction_GPT-4o/output/copol_database/model_output_extraction"  # Directory for cleaned files

    mapping_file = "../data_extraction/data_extraction_GPT-4o/output/copol_paper_list.json"  # File with DOI mappings
    manual_mapping_file = "papers_without_mapping.json"

    print(f"\nProcessing files from {input_dir} to {output_dir}")
    clean_json_file(input_dir, output_dir, mapping_file, manual_mapping_file)
    print("\nProcessing complete!")


if __name__ == "__main__":
    main()
