import json
import os
from copolextractor.doi2pdf import doi2pdf
import copolextractor.utils as utils


def is_valid_pdf(file_path):
    """
    Check if a PDF file is valid and not corrupted.
    Returns True if the file is a valid PDF, False otherwise.
    """
    try:
        # Try to read the first few bytes to check for PDF signature
        with open(file_path, "rb") as f:
            header = f.read(4)
            if header != b"%PDF":
                print(f"Invalid PDF header in {file_path}")
                return False

            # Try to read the end of the file to check for EOF marker
            # Move to the end of the file minus 1024 bytes (or beginning if small file)
            f.seek(max(0, os.path.getsize(file_path) - 1024))
            footer = f.read().lower()
            if b"%%eof" not in footer:
                print(f"Missing EOF marker in {file_path}")
                return False

            return True
    except Exception as e:
        print(f"Error checking PDF validity for {file_path}: {str(e)}")
        return False


def generate_filename(base_name, output_folder, extension=".pdf"):
    """
    Generate a sanitized filename and check if it exists in the output folder.
    If it exists, check if it's a valid PDF.
    """
    sanitized_name = utils.sanitize_filename(base_name)
    unique_name = sanitized_name + extension
    file_path = os.path.join(output_folder, unique_name)

    # If the file doesn't exist, return the new filename
    if not os.path.exists(file_path):
        return unique_name

    # If the file exists, check if it's a valid PDF
    if is_valid_pdf(file_path):
        # Valid PDF exists, return None to indicate skipping
        return None
    else:
        # Corrupted PDF, delete it and return the filename for re-download
        print(f"Found corrupted PDF: {file_path}. Will re-download.")
        try:
            os.remove(file_path)
            print(f"Deleted corrupted file: {file_path}")
        except Exception as e:
            print(f"Error deleting corrupted file {file_path}: {str(e)}")

        return unique_name


def download_papers(input_file, output_folder):
    """
    Download papers based on the DOIs in the input JSON file using doi2pdf
    and update the file with download status.
    """
    data = utils.load_json(input_file)
    paper_count = 0
    failed_download_count = 0
    downloaded_paper_count = 0
    redownloaded_count = 0

    for index, entry in enumerate(data):
        doi = entry.get("DOI", "").strip()
        if not doi:
            print(f"Skipping entry {index + 1}: No DOI found.")
            continue

        paper_count += 1

        # Generate a sanitized filename for the PDF
        base_name = f"paper_{index + 1}" if not doi else utils.sanitize_filename(doi)
        pdf_name = generate_filename(base_name, output_folder)

        # Check if the file already exists and is valid
        if pdf_name is None:
            print(f"Skipping paper {index + 1}/{len(data)}: Valid file already exists.")
            entry["downloaded"] = True
            continue
        elif entry.get("downloaded", False) and entry.get("pdf_name", "") == pdf_name:
            # This is a redownload case
            redownloaded_count += 1
            print(f"Re-downloading paper {index + 1}/{len(data)}: Previous file was corrupted.")
        else:
            print(f"Processing paper {index + 1}/{len(data)}: {doi}")

        output_path = os.path.join(output_folder, pdf_name)

        # Extract just the DOI part if it's a full URL
        if doi.startswith('https://doi.org/'):
            doi = doi.replace('https://doi.org/', '')

        try:
            # Download using doi2pdf
            doi2pdf(doi, output=output_path)

            # Check if file was created and is valid
            if os.path.exists(output_path) and is_valid_pdf(output_path):
                print(f"Download successful: Valid PDF saved in {output_path}")
                entry["downloaded"] = True
                entry["pdf_name"] = pdf_name
                downloaded_paper_count += 1
            else:
                if os.path.exists(output_path):
                    print(f"Downloaded file is corrupted: {output_path}")
                    try:
                        os.remove(output_path)
                        print(f"Deleted corrupted download: {output_path}")
                    except Exception as e:
                        print(f"Error deleting corrupted download {output_path}: {str(e)}")
                else:
                    print(f"Failed to download DOI {doi}")

                entry["downloaded"] = False
                failed_download_count += 1
        except Exception as e:
            print(f"Error downloading DOI {doi}: {str(e)}")
            entry["downloaded"] = False
            failed_download_count += 1

        # Update the JSON file after each paper
        with open(input_file, "w") as file:
            json.dump(data, file, indent=4)

    print(
        f"Out of {paper_count} papers, {downloaded_paper_count} were successfully downloaded "
        f"({redownloaded_count} were re-downloaded due to corruption), "
        f"{failed_download_count} downloads failed."
    )


def main(input_file_paper, output_folder):
    """
    Main function to handle the download process and update the JSON file.
    Args:
        input_file_paper: Path to the JSON file containing paper information
        output_folder: Path to the folder where PDFs will be saved
    """
    # Check if the output folder exists, and create it if not
    os.makedirs(output_folder, exist_ok=True)
    print(f"Ensured folder exists: {output_folder}")

    print(f"Starting the paper download process using doi2pdf...")
    download_papers(input_file_paper, output_folder)

    pdf_files = [f for f in os.listdir(output_folder) if f.endswith(".pdf")]
    valid_pdf_count = sum(1 for f in pdf_files if is_valid_pdf(os.path.join(output_folder, f)))
    corrupted_pdf_count = len(pdf_files) - valid_pdf_count

    print(f"There are {len(pdf_files)} PDFs in the folder:")
    print(f"  - {valid_pdf_count} valid PDFs")
    print(f"  - {corrupted_pdf_count} corrupted PDFs (if any, these will be re-downloaded on next run)")


if __name__ == "__main__":
    input_file = "../../data_extraction/output/selected_papers.json"
    output_folder = "../../data_extraction/output_2/PDF2"

    main(input_file, output_folder)