import json
from scidownl.core.task import ScihubTask
import os
import copolextractor.utils as utils


def generate_filename(base_name, output_folder, extension=".pdf"):
    """
    Generate a sanitized filename and check if it exists in the output folder.
    """
    sanitized_name = utils.sanitize_filename(base_name)
    unique_name = sanitized_name + extension

    # If the file exists, return None to indicate skipping
    if os.path.exists(os.path.join(output_folder, unique_name)):
        return None

    return unique_name


def download_papers_and_update_file(input_file, output_folder):
    """
    Download papers based on the DOIs in the input JSON file and update the file with download status.
    """
    data = utils.load_json(input_file)
    paper_count = 0
    failed_download_count = 0
    downloaded_paper_count = 0

    for index, entry in enumerate(data):
        doi = entry.get("DOI", "").strip()
        if not doi:
            print(f"Skipping entry {index + 1}: No DOI found.")
            continue

        doi_url = (
            f"https://doi.org/{doi}" if not doi.startswith("https://doi.org/") else doi
        )
        paper_count += 1

        # Generate a sanitized filename for the PDF
        base_name = f"paper_{index + 1}" if not doi else utils.sanitize_filename(doi)
        pdf_name = generate_filename(base_name, output_folder)

        # Check if the file already exists
        if pdf_name is None:
            print(f"Skipping paper {index + 1}/{len(data)}: File already exists.")
            entry["downloaded"] = True
            continue

        output_path = os.path.join(output_folder, pdf_name)
        print(f"Processing paper {index + 1}/{len(data)}: {doi_url}")

        try:
            task = ScihubTask(source_keyword=doi_url, out=output_path)
            task.run()

            print(f"Download successful: PDF saved in {task.context.get('out')}")
            entry["downloaded"] = True
            entry["pdf_name"] = pdf_name  # Save the PDF name in the JSON
            downloaded_paper_count += 1

        except Exception as e:
            print(f"Error downloading DOI {doi}: {e}")
            entry["downloaded"] = False
            failed_download_count += 1

        # Update the JSON file
        with open(input_file, "w") as file:
            json.dump(data, file, indent=4)

    print(
        f"Out of {paper_count} papers, {downloaded_paper_count} were successfully downloaded, "
        f"{failed_download_count} downloads failed."
    )


def main(input_file_paper, output_folder):
    """
    Main function to handle the download process and update the JSON file.
    """

    # Check if the output folder exists, and create it if not
    os.makedirs(output_folder, exist_ok=True)
    print(f"Ensured folder exists: {output_folder}")

    print("Starting the paper download process...")
    download_papers_and_update_file(input_file_paper, output_folder)

    pdf_files = [f for f in os.listdir(output_folder) if f.endswith(".pdf")]
    pdf_count = len(pdf_files)
    print(f"There are {pdf_count} PDFs in the folder.")
