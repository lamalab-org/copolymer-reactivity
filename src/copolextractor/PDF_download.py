import json
import os
import time
from pathlib import Path
from typing import List, Optional, Union

import copolextractor.utils as utils
import requests
from dotenv import load_dotenv


REQUEST_TIMEOUT_SECONDS = 20
SLEEP_BETWEEN_REQUESTS_SECONDS = 0.2


def is_valid_pdf(file_path: Union[str, Path]) -> bool:
    """
    Check if a PDF file is valid and not corrupted.

    Args:
        file_path: Path to the PDF file to check.

    Returns:
        True if the file starts with the PDF signature and contains an EOF marker
        near the end, False otherwise (including on read errors).
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


def generate_filename(base_name: str, output_folder: Union[str, Path], extension: str = ".pdf") -> Optional[str]:
    """
    Generate a sanitized filename and check if it exists in the output folder.
    If it exists, check if it's a valid PDF.

    Args:
        base_name: Raw name to sanitize into a filename (e.g. a DOI).
        output_folder: Directory the file would be saved in.
        extension: File extension to append to the sanitized name.

    Returns:
        The sanitized filename if the file doesn't exist yet, or is corrupted and
        was deleted so it can be re-downloaded. None if a valid PDF already exists
        at that path (nothing to do).
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


def get_openalex_pdf_url(doi: str) -> Optional[str]:
    """Return an open-access URL for a DOI from OpenAlex, or None if unavailable.

    Args:
        doi: DOI to look up (without the "https://doi.org/" prefix).

    Returns:
        The open-access PDF URL, or None if OpenAlex has no known open-access
        location or the request failed.
    """
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


def get_unpaywall_pdf_url(doi: str, email: str) -> Optional[str]:
    """Return an open-access URL for a DOI from Unpaywall, or None if unavailable.

    Args:
        doi: DOI to look up (without the "https://doi.org/" prefix).
        email: Contact email required by the Unpaywall API's usage policy.

    Returns:
        The open-access PDF URL, or None if Unpaywall has no known open-access
        location or the request failed.
    """
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


def get_semantic_scholar_pdf_url(doi: str) -> Optional[str]:
    """Return an open-access URL for a DOI from Semantic Scholar, or None if unavailable.

    Args:
        doi: DOI to look up (without the "https://doi.org/" prefix).

    Returns:
        The open-access PDF URL, or None if Semantic Scholar has no known
        open-access PDF or the request failed.
    """
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


def get_core_pdf_url(doi: str, api_key: str) -> Optional[str]:
    """Return an open-access URL for a DOI from CORE, or None if unavailable.

    Args:
        doi: DOI to look up (without the "https://doi.org/" prefix).
        api_key: CORE API key. If falsy, the lookup is skipped and None is returned.

    Returns:
        The download URL of the best matching CORE record, or None if no API key
        was given, no record matched, or the request failed.
    """
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


def download_open_access_papers(input_file: Union[str, Path], output_folder: Union[str, Path]) -> List[str]:
    """Download PDFs through legal open-access APIs and save unresolved DOIs separately.

    For each paper marked "downloaded" in `input_file`, tries OpenAlex, then
    Unpaywall, then Semantic Scholar, then CORE (in that order) until an
    open-access PDF URL is found and successfully downloaded.

    Args:
        input_file: Path to the paper-list JSON file; a sibling/ancestor ".env"
            file (if present) is loaded for the `UNPAYWALL_EMAIL`/`CORE_API_KEY`
            environment variables.
        output_folder: Directory PDFs and "unresolved_papers.json" are written to.

    Returns:
        The list of DOIs that could not be resolved to a downloadable PDF.
    """
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


def _download_open_access_pdf(pdf_url: str) -> Optional[bytes]:
    """Return PDF bytes from a URL, or None when the response is not a valid PDF.

    Args:
        pdf_url: URL to download the PDF from.

    Returns:
        The raw PDF bytes, or None if the request failed or the content does not
        start with the PDF signature.
    """
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


def main(input_file_paper: Union[str, Path], output_folder: Union[str, Path]) -> None:
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
    input_file = "../../data_extraction/output/selected_papers.json"
    output_folder = "../../data_extraction/output_2/PDF2"

    main(input_file, output_folder)
