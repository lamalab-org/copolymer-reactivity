import json
import os
import asyncio
import logging
import aiohttp
import aiofiles
from enum import Enum
import copolextractor.utils as utils
from paperscraper import default_scraper
from paperscraper.log_formatter import CustomFormatter
from unpywall import Unpywall


class DownloadBackend(Enum):
    PAPER_SCRAPER = "paper_scraper"  # Uses paperscraper with all its backends including unpywall


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


def setup_logger():
    """Setup and return a logger with custom formatting."""
    logger = logging.getLogger("paper-scraper")
    if not logger.handlers:  # Only add handler if it doesn't already exist
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setFormatter(CustomFormatter())
        logger.addHandler(ch)
    return logger


async def download_file(url: str, path: str, session: aiohttp.ClientSession) -> bool:
    """Download a file from URL to path."""
    try:
        async with session.get(url) as response:
            if response.status != 200:
                return False
            async with aiofiles.open(path, 'wb') as f:
                await f.write(await response.read())
            return True
    except Exception:
        return False


async def download_paper_scraper_async(doi_url, output_path):
    """
    Download a single paper using paperscraper asynchronously.
    Returns True if successful, False otherwise.
    """
    try:
        # Extract just the DOI part if it's a full URL
        if doi_url.startswith('https://doi.org/'):
            doi = doi_url.replace('https://doi.org/', '')
        else:
            doi = doi_url

        # Create paper metadata
        paper_metadata = {
            "paperId": doi,
            "title": "Unknown Title",
            "externalIds": {"DOI": doi}
        }

        # Setup logger
        logger = setup_logger()
        logger.debug(f"Attempting to download DOI: {doi} to path: {output_path}")

        # Setup unpywall configuration
        if not os.getenv('UNPAYWALL_EMAIL'):
            os.environ['UNPAYWALL_EMAIL'] = "your.email@example.com"  # Ersetzen Sie dies durch Ihre E-Mail-Adresse

            # Create scraper with default configuration and add unpywall with high priority
            scraper = default_scraper()
            scraper.register_scraper(unpywall_scraper, priority=13, name="unpywall", attach_session=True)

            logger.debug("Created scraper with following backends:")
            for s in scraper.scrapers:
                logger.debug(f"  - {s.name} (priority: {s.priority})")

            # Try to download the paper
            success = await scraper.scrape(paper_metadata, output_path, logger=logger)

            # Cleanup
            await scraper.close()

            if success:
                logger.debug(f"Successfully downloaded paper to {output_path}")
            else:
                logger.debug("All scraper backends failed to download the paper")

            return success
    except Exception as e:
        print(f"Error using paperscraper: {e}")
        return False


async def download_paper_scraper_async(doi_url, output_path):
    """
    Download a single paper using paperscraper asynchronously.
    Returns True if successful, False otherwise.
    """
    try:
        # Extract just the DOI part if it's a full URL
        if doi_url.startswith('https://doi.org/'):
            doi = doi_url.replace('https://doi.org/', '')
        else:
            doi = doi_url

        # Create paper metadata
        paper_metadata = {
            "paperId": doi,
            "title": "Unknown Title",
            "externalIds": {"DOI": doi}
        }

        # Setup logger
        logger = setup_logger()
        logger.debug(f"Attempting to download DOI: {doi} to path: {output_path}")

        # Setup unpywall configuration
        if not os.getenv('UNPAYWALL_EMAIL'):
            os.environ['UNPAYWALL_EMAIL'] = "mara.wilhelmi@uni-jena.de"

            # Create scraper with default configuration and add unpywall with high priority
            scraper = default_scraper()
            scraper.register_scraper(unpywall_scraper, priority=13, name="unpywall", attach_session=True)

            logger.debug("Created scraper with following backends:")
            for s in scraper.scrapers:
                logger.debug(f"  - {s.name} (priority: {s.priority})")

            # Try to download the paper
            success = await scraper.scrape(paper_metadata, output_path, logger=logger)

            # Cleanup
            await scraper.close()

            if success:
                logger.debug(f"Successfully downloaded paper to {output_path}")
            else:
                logger.debug("All scraper backends failed to download the paper")

            return success
    except Exception as e:
        print(f"Error using paperscraper: {e}")
        return False


def download_paper_scraper(doi_url, output_path):
    """
    Synchronous wrapper for the async paperscraper download function.
    """
    return asyncio.run(download_paper_scraper_async(doi_url, output_path))


def download_papers(input_file, output_folder, backend=DownloadBackend.PAPER_SCRAPER):
    """
    Download papers based on the DOIs in the input JSON file and update the file with download status.
    """
    # Use the paper_scraper download function
    download_func = download_paper_scraper

    data = utils.load_json(input_file)
    paper_count = 0
    failed_download_count = 0
    downloaded_paper_count = 0

    for index, entry in enumerate(data):
        doi = entry.get("DOI", "").strip()
        if not doi:
            print(f"Skipping entry {index + 1}: No DOI found.")
            continue

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
        print(f"Processing paper {index + 1}/{len(data)}: {doi}")

        # Download using selected backend
        if download_func(doi, output_path):
            print(f"Download successful: PDF saved in {output_path}")
            entry["downloaded"] = True
            entry["pdf_name"] = pdf_name
            downloaded_paper_count += 1
        else:
            print(f"Failed to download DOI {doi}")
            entry["downloaded"] = False
            failed_download_count += 1

        # Update the JSON file
        with open(input_file, "w") as file:
            json.dump(data, file, indent=4)

    print(
        f"Out of {paper_count} papers, {downloaded_paper_count} were successfully downloaded, "
        f"{failed_download_count} downloads failed."
    )


def main(input_file_paper, output_folder, backend=DownloadBackend.PAPER_SCRAPER):
    """
    Main function to handle the download process and update the JSON file.
    Args:
        input_file_paper: Path to the JSON file containing paper information
        output_folder: Path to the folder where PDFs will be saved
        backend: DownloadBackend enum specifying which download method to use
    """
    # Check if the output folder exists, and create it if not
    os.makedirs(output_folder, exist_ok=True)
    print(f"Ensured folder exists: {output_folder}")

    print(f"Starting the paper download process using {backend.value}...")
    download_papers(input_file_paper, output_folder, backend)

    pdf_files = [f for f in os.listdir(output_folder) if f.endswith(".pdf")]
    pdf_count = len(pdf_files)
    print(f"There are {pdf_count} PDFs in the folder.")


if __name__ == "__main__":

    input_file = "../../data_extraction/output/selected_papers.json"
    output_folder = "../../data_extraction/output/PDF2"

    # Use paper-scraper backend with unpywall
    main(input_file, output_folder)