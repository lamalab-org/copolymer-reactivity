import argparse
from typing import Optional
import subprocess, os, platform
import bs4
import requests
import time


class NotFoundError(Exception):
    pass


# List of Sci-Hub URLs to try
SCI_HUB_URLS = [
    "https://sci-hub.se/",
    "https://sci-hub.st/",
    "https://sci-hub.yt/",
    "https://sci-hub.ren/",
    "https://sci-hub.hkvisa.net/",
]


def doi2pdf(
        doi: Optional[str] = None,
        *,
        output: Optional[str] = None,
        name: Optional[str] = None,
        url: Optional[str] = None,
        open_pdf: bool = False,
):
    """Retrieves the pdf file from DOI, name or URL of a research paper.
    Args:
        doi (Optional[str]): DOI of the paper. Defaults to None.
        output (str, optional): Path to save the PDF. Defaults to None.
        name (Optional[str], optional): Name of the paper. Defaults to None.
        url (Optional[str], optional): URL of the paper. Defaults to None.
        open_pdf (bool, optional): Whether to open the PDF after download. Defaults to False.
    """

    if len([arg for arg in (doi, name, url) if arg is not None]) > 1:
        raise ValueError("Only one of doi, name, url must be specified.")

    doi, title, pdf_url = get_paper_metadata(doi, name, url)

    # Try to get PDF from the original URL first
    pdf_content = None
    if pdf_url:
        try:
            pdf_content = get_pdf_from_url(pdf_url)
            print(f"Successfully retrieved PDF from original source")
        except NotFoundError:
            print(f"Failed to get PDF from original source, trying Sci-Hub mirrors")
            pdf_content = None

    # If we couldn't get the PDF from the original URL, try Sci-Hub
    if pdf_content is None:
        pdf_content = retrieve_from_scihub(doi)

    if pdf_content is None:
        raise NotFoundError("Could not retrieve PDF from any source")

    filename = title.replace(" ", "_") + ".pdf"
    if output is None:
        output = f"/tmp/{filename}"

    with open(output, "wb") as f:
        f.write(pdf_content)
        print(f"PDF saved to {output}")

    if open_pdf:
        if platform.system() == "Darwin":  # macOS
            subprocess.call(("open", output))
        elif platform.system() == "Windows":  # Windows
            os.startfile(output)
        else:  # linux variants
            subprocess.call(("xdg-open", output))


def get_paper_metadata(doi, name, url):
    """Returns metadata of a paper with http://openalex.org/"""
    if name:
        api_res = requests.get(
            f"https://api.openalex.org/works?search={name}&per-page=1&page=1&sort=relevance_score:desc"
        )
    if doi:
        api_res = requests.get(f"https://api.openalex.org/works/https://doi.org/{doi}")
    if url:
        api_res = requests.get(f"https://api.openalex.org/works/{url}")

    if api_res.status_code != 200:
        raise NotFoundError("Paper not found.")

    metadata = api_res.json()
    if metadata.get("results") is not None:
        metadata = metadata["results"][0]

    if metadata.get("doi") is not None:
        doi = metadata["doi"][len("https://doi.org/"):]
    title = metadata["display_name"]
    pdf_url = metadata["open_access"]["oa_url"]
    if pdf_url is None:
        if metadata.get("host_venue") is not None:
            pdf_url = metadata["host_venue"]["url"]
        elif metadata.get("primary_location") is not None:
            pdf_url = metadata["primary_location"]["landing_page_url"]
        else:
            raise NotFoundError("PDF URL not found.")

    print("Found paper: ", title)
    return doi, title, pdf_url


def get_html(url):
    """Returns bs4 object that you can iterate through based on html elements and attributes."""
    s = requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    }
    try:
        html = s.get(url, timeout=20, headers=headers, allow_redirects=True)
        html.encoding = "utf-8"
        html.raise_for_status()
        html = bs4.BeautifulSoup(html.text, "html.parser")
        return html
    except requests.exceptions.RequestException as e:
        print(f"Error accessing {url}: {str(e)}")
        return None


def retrieve_from_scihub(doi):
    """Tries all known Sci-Hub mirrors to retrieve the PDF for a DOI."""

    urls_to_try = [url.strip() for url in SCI_HUB_URLS if url.strip()]

    last_error = None
    for sci_hub_url in urls_to_try:
        print(f"[INFO] Trying Sci-Hub mirror: {sci_hub_url}")
        try:
            pdf_url = retrieve_scihub(doi, sci_hub_url)
            pdf_content = get_pdf_from_url(pdf_url)
            print(f"[INFO] Successfully retrieved PDF from {sci_hub_url}")
            return pdf_content
        except NotFoundError as e:
            print(f"[INFO] Failed to retrieve from {sci_hub_url}: {str(e)}")
            last_error = e
        except Exception as e:
            print(f"[INFO] Error with {sci_hub_url}: {str(e)}")
            last_error = e

    print("[INFO] Failed to retrieve PDF from any Sci-Hub mirror")
    if last_error:
        raise last_error
    return None


def retrieve_scihub(doi, sci_hub_url):
    """Returns the URL of the pdf file from the DOI of a research paper thanks to sci-hub."""
    full_url = f"{sci_hub_url}{doi}"
    print(f"[DEBUG] Accessing {full_url}")

    html_sci_hub = get_html(full_url)
    if html_sci_hub is None:
        raise NotFoundError(f"Failed to access Sci-Hub URL: {full_url}")

    # Look for PDF iframe
    iframe = html_sci_hub.find("iframe", {"id": "pdf"})

    # If iframe not found, try alternative methods
    if iframe is None:
        # Try finding download button (some sci-hub versions use this)
        download_button = html_sci_hub.find("a", {"id": "download"})
        if download_button and "href" in download_button.attrs:
            pdf_src = download_button["href"]
        else:
            # Try finding embed tag
            embed = html_sci_hub.find("embed")
            if embed and "src" in embed.attrs:
                pdf_src = embed["src"]
            else:
                # Try finding any PDF link
                pdf_link = html_sci_hub.find("a", href=lambda href: href and href.endswith(".pdf"))
                if pdf_link:
                    pdf_src = pdf_link["href"]
                else:
                    raise NotFoundError("PDF not found on Sci-Hub page")
    else:
        pdf_src = iframe["src"]

    # Handle relative URLs
    if pdf_src.startswith('//'):
        pdf_src = 'https:' + pdf_src
    elif pdf_src.startswith('/'):
        # Extract the base URL from sci_hub_url
        from urllib.parse import urlparse
        parsed_url = urlparse(sci_hub_url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        pdf_src = base_url + pdf_src

    print(f"[DEBUG] PDF URL: {pdf_src}")
    return pdf_src


def get_pdf_from_url(url):
    """Returns the content of a pdf file from a URL."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "application/pdf,*/*",
            "Referer": "https://scholar.google.com/"
        }

        print(f"[DEBUG] Downloading PDF from {url}")
        res = requests.get(url, headers=headers, timeout=30)

        # Check if response is successful
        res.raise_for_status()

        # Check if content is likely a PDF
        content_type = res.headers.get('Content-Type', '').lower()
        if 'application/pdf' not in content_type and not url.lower().endswith('.pdf'):
            # Check first few bytes for PDF signature
            if not res.content.startswith(b'%PDF'):
                print(f"[WARNING] Response may not be a PDF. Content-Type: {content_type}")

        return res.content
    except requests.exceptions.RequestException as e:
        print(f"Error downloading PDF from {url}: {str(e)}")
        raise NotFoundError(f"Bad PDF URL: {str(e)}")


def download_papers(papers, pdf_folder, max_retries=3):
    """
    Download PDFs for a list of papers.

    Args:
        papers: List of paper dictionaries, each containing at least a "DOI" key
        pdf_folder: Folder to save PDFs to
        max_retries: Maximum number of retry attempts per paper

    Returns:
        List of successfully downloaded papers
    """
    os.makedirs(pdf_folder, exist_ok=True)
    successful_papers = []

    for i, paper in enumerate(papers):
        doi = paper.get("DOI")
        if not doi:
            print(f"Skipping paper {i + 1}/{len(papers)}: No DOI available")
            continue

        print(f"Processing paper {i + 1}/{len(papers)}: {doi}")

        # Create output filename
        sanitized_doi = doi.replace("/", "_").replace(".", "_")
        output_path = os.path.join(pdf_folder, f"{sanitized_doi}.pdf")

        # Check if PDF already exists
        if os.path.exists(output_path):
            print(f"PDF for DOI {doi} already exists at {output_path}")
            successful_papers.append(paper)
            continue

        # Try to download with retries
        success = False
        for attempt in range(max_retries):
            try:
                doi2pdf(doi=doi, output=output_path)
                print(f"Successfully downloaded PDF for DOI {doi} to {output_path}")
                successful_papers.append(paper)
                success = True
                break
            except NotFoundError as e:
                print(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                if attempt < max_retries - 1:
                    print(f"Retrying in 2 seconds...")
                    time.sleep(2)
            except Exception as e:
                print(f"Unexpected error downloading DOI {doi}: {str(e)}")
                break

        if not success:
            print(f"Failed to download PDF for DOI {doi} after {max_retries} attempts")

    return successful_papers


def main():
    parser = argparse.ArgumentParser(
        description="Retrieves the pdf file from DOI of a research paper.", epilog=""
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Relative path of the target pdf file.",
        metavar="path",
    )

    parser.add_argument(
        "--doi", type=str, help="DOI of the research paper.", metavar="DOI"
    )

    parser.add_argument(
        "-n", "--name", type=str, help="Name of the research paper.", metavar="name"
    )

    parser.add_argument(
        "--url", type=str, help="URL of the research paper.", metavar="url"
    )

    parser.add_argument(
        "--open", action="store_true", help="Open the pdf file after downloading.",
    )

    parser.add_argument(
        "--batch", type=str, help="Path to JSON file with list of DOIs to download", metavar="path"
    )

    parser.add_argument(
        "--output-dir", type=str, help="Directory to save batch downloads", metavar="dir"
    )

    args = parser.parse_args()

    # Batch download mode
    if args.batch:
        if not args.output_dir:
            print("Error: --output-dir is required when using --batch")
            return

        try:
            import json
            with open(args.batch, 'r') as f:
                papers = json.load(f)

            successful = download_papers(papers, args.output_dir)
            print(f"Successfully downloaded {len(successful)} out of {len(papers)} papers")

            # Save successful downloads to a JSON file
            success_file = os.path.join(args.output_dir, "successful_downloads.json")
            with open(success_file, 'w') as f:
                json.dump(successful, f, indent=2)

        except Exception as e:
            print(f"Error in batch mode: {str(e)}")
        return

    # Single download mode
    if args.doi is None and args.name is None and args.url is None:
        parser.error("At least one of --doi, --name, --url must be specified.")
    if len([arg for arg in (args.doi, args.name, args.url) if arg is not None]) > 1:
        parser.error("Only one of --doi, --name, --url must be specified.")

    try:
        doi2pdf(
            args.doi, output=args.output, name=args.name, url=args.url, open_pdf=args.open)
    except NotFoundError as e:
        print(f"Error: {str(e)}")
        exit(1)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()