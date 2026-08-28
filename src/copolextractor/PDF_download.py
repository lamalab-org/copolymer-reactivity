import json
import os
import time
from pathlib import Path

import copolextractor.utils as utils
from copolextractor.doi2pdf import doi2pdf
import requests
from dotenv import load_dotenv


REQUEST_TIMEOUT_SECONDS = 20
SLEEP_BETWEEN_REQUESTS_SECONDS = 0.2


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
        if doi.startswith("https://doi.org/"):
            doi = doi.replace("https://doi.org/", "")

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


def get_openalex_pdf_url(doi):
    """Return an open-access URL for a DOI from OpenAlex, or None if unavailable."""
    try:
        response = requests.get(
            f"https://api.openalex.org/works/https://doi.org/{doi}",
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException:
        return None

    metadata = response.json()
    best_location = metadata.get("best_oa_location") or {}
    return best_location.get("pdf_url") or (metadata.get("open_access") or {}).get("oa_url")


def get_unpaywall_pdf_url(doi, email):
    """Return an open-access URL for a DOI from Unpaywall, or None if unavailable."""
    try:
        response = requests.get(
            f"https://api.unpaywall.org/v2/{doi}",
            params={"email": email},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException:
        return None

    best_location = (response.json().get("best_oa_location") or {})
    return best_location.get("url_for_pdf") or best_location.get("url")


def get_semantic_scholar_pdf_url(doi):
    """Return an open-access URL for a DOI from Semantic Scholar, or None if unavailable."""
    try:
        response = requests.get(
            f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}",
            params={"fields": "openAccessPdf"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException:
        return None

    return (response.json().get("openAccessPdf") or {}).get("url")


def get_core_pdf_url(doi, api_key):
    """Return an open-access URL for a DOI from CORE, or None if unavailable."""
    if not api_key:
        return None

    try:
        response = requests.post(
            "https://api.core.ac.uk/v3/search/works",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"q": f'doi:"{doi}"'},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException:
        return None

    records = (response.json() or {}).get("results") or []
    return records[0].get("downloadUrl") if records else None


def download_open_access_papers(input_file, output_folder):
    """Download PDFs through legal open-access APIs and save unresolved DOIs separately."""
    input_path = Path(input_file)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    load_dotenv(input_path.parent / ".env")
    load_dotenv(Path(__file__).resolve().parents[2] / "data_extraction" / "notebooks" / ".env")

    papers = utils.load_json(str(input_path))
    downloadable_papers = [paper for paper in papers if paper.get("downloaded") is True]
    unresolved_dois = []
    email = os.environ.get("UNPAYWALL_EMAIL", "mara.wilhelmi@uni-jena.de")
    core_api_key = os.environ.get("CORE_API_KEY", "")

    for paper in downloadable_papers:
        doi = paper.get("DOI", "").strip()
        if not doi:
            continue
        doi = doi.removeprefix("https://doi.org/").removeprefix("http://doi.org/")
        target = output_path / f"{utils.sanitize_filename(doi)}.pdf"
        if target.exists() and is_valid_pdf(str(target)):
            continue

        pdf_url = get_openalex_pdf_url(doi)
        time.sleep(SLEEP_BETWEEN_REQUESTS_SECONDS)
        if not pdf_url and email:
            pdf_url = get_unpaywall_pdf_url(doi, email)
            time.sleep(SLEEP_BETWEEN_REQUESTS_SECONDS)
        if not pdf_url:
            pdf_url = get_semantic_scholar_pdf_url(doi)
            time.sleep(SLEEP_BETWEEN_REQUESTS_SECONDS)
        if not pdf_url:
            pdf_url = get_core_pdf_url(doi, core_api_key)
            time.sleep(SLEEP_BETWEEN_REQUESTS_SECONDS)

        content = None if not pdf_url else _download_open_access_pdf(pdf_url)
        if content is None:
            unresolved_dois.append(doi)
            continue
        target.write_bytes(content)

    unresolved_path = output_path / "unresolved_papers.json"
    unresolved_path.write_text(json.dumps(unresolved_dois, indent=2), encoding="utf-8")
    print(f"Open-access downloads complete: {len(downloadable_papers) - len(unresolved_dois)} successful")
    print(f"Unresolved papers: {len(unresolved_dois)}")
    print(f"Saved unresolved DOIs to {unresolved_path}")
    return unresolved_dois


def _download_open_access_pdf(pdf_url):
    """Return PDF bytes from a URL, or None when the response is not a valid PDF."""
    try:
        response = requests.get(
            pdf_url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; copolextractor/1.0)"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException:
        return None
    return response.content if response.content.startswith(b"%PDF") else None


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

    print("Starting the paper download process using open-access sources...")
    download_open_access_papers(input_file_paper, output_folder)

    pdf_files = [f for f in os.listdir(output_folder) if f.endswith(".pdf")]
    valid_pdf_count = sum(1 for f in pdf_files if is_valid_pdf(os.path.join(output_folder, f)))
    corrupted_pdf_count = len(pdf_files) - valid_pdf_count

    print(f"There are {len(pdf_files)} PDFs in the folder:")
    print(f"  - {valid_pdf_count} valid PDFs")
    print(
        f"  - {corrupted_pdf_count} corrupted PDFs (if any, these will be re-downloaded on next run)"
    )


if __name__ == "__main__":
    # Historical paths kept for one-off manual runs; the orchestrated pipeline
    # in `data_extraction/obtain_data.py` passes its own paths via
    # ExtractionConfig and does not rely on these defaults.
    input_file = "../../data_extraction/provenance/selected_200_papers.json"
    output_folder = "../../data_extraction/provenance/PDF2"

    main(input_file, output_folder)
