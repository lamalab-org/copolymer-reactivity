import argparse
from typing import Optional
import subprocess, os, platform
import bs4
import requests


class NotFoundError(Exception):
    pass


# List of Sci-Hub URLs to try
SCI_HUB_URLS = os.getenv("SCI_HUB_URLS",
                         "https://sci-hub.mksa.top/,https://sci-hub.ren/,http://sci-hub.ren/,http://sci-hub.red/,http://sci-hub.se/,https://sci-hub.se/,http://sci-hub.tw/").split(
    ",")


def doi2pdf(
        doi: Optional[str] = None,
        *,
        output: Optional[str] = None,
        name: Optional[str] = None,
        url: Optional[str] = None,
        open_pdf: bool = False,
        sci_hub_url: Optional[str] = None,
):
    """Retrieves the pdf file from DOI, name or URL of a research paper.
    Args:
        doi (Optional[str]): DOI of the paper. Defaults to None.
        output (str, optional): Path to save the PDF. Defaults to None.
        name (Optional[str], optional): Name of the paper. Defaults to None.
        url (Optional[str], optional): URL of the paper. Defaults to None.
        open_pdf (bool, optional): Whether to open the PDF after download. Defaults to False.
        sci_hub_url (Optional[str], optional): Specific Sci-Hub URL to use. Defaults to None.
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
        pdf_content = retrieve_from_scihub(doi, sci_hub_url)

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
        "User-Agent": "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36",
    }
    try:
        html = s.get(url, timeout=10, headers=headers, allow_redirects=True)
        html.encoding = "utf-8"
        html.raise_for_status()
        html = bs4.BeautifulSoup(html.text, "html.parser")
        return html
    except requests.exceptions.RequestException as e:
        print(f"Error accessing {url}: {str(e)}")
        return None


def retrieve_from_scihub(doi, specific_sci_hub_url=None):
    """Tries multiple Sci-Hub mirrors to retrieve the PDF for a DOI."""

    # If a specific URL is provided, try that one first
    if specific_sci_hub_url:
        print(f"[INFO] Trying specified Sci-Hub URL: {specific_sci_hub_url}")
        try:
            pdf_url = retrieve_scihub(doi, specific_sci_hub_url)
            pdf_content = get_pdf_from_url(pdf_url)
            return pdf_content
        except NotFoundError:
            print(f"[INFO] Failed to retrieve from specified Sci-Hub URL, trying others")

    # Try all the URLs in our list
    for sci_hub_url in SCI_HUB_URLS:
        sci_hub_url = sci_hub_url.strip()  # Remove any whitespace
        if not sci_hub_url:
            continue

        print(f"[INFO] Trying Sci-Hub mirror: {sci_hub_url}")
        try:
            pdf_url = retrieve_scihub(doi, sci_hub_url)
            pdf_content = get_pdf_from_url(pdf_url)
            print(f"[INFO] Successfully retrieved PDF from {sci_hub_url}")
            return pdf_content
        except NotFoundError:
            print(f"[INFO] Failed to retrieve from {sci_hub_url}")
        except Exception as e:
            print(f"[INFO] Error with {sci_hub_url}: {str(e)}")

    # If we get here, all URLs failed
    print("[INFO] Failed to retrieve PDF from any Sci-Hub mirror")
    return None


def retrieve_scihub(doi, sci_hub_url):
    """Returns the URL of the pdf file from the DOI of a research paper thanks to sci-hub."""
    html_sci_hub = get_html(f"{sci_hub_url}{doi}")
    if html_sci_hub is None:
        raise NotFoundError("Failed to access Sci-Hub")

    iframe = html_sci_hub.find("iframe", {"id": "pdf"})
    if iframe is None:
        raise NotFoundError("DOI not found or PDF iframe not present")

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

    return pdf_src


def get_pdf_from_url(url):
    """Returns the content of a pdf file from a URL."""
    try:
        res = requests.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36",
            },
            timeout=20,
        )
        res.raise_for_status()
        return res.content
    except requests.exceptions.RequestException as e:
        print(f"Error downloading PDF from {url}: {str(e)}")
        raise NotFoundError(f"Bad PDF URL: {str(e)}")


def main():
    parser = argparse.ArgumentParser(
        description="Retrieves the pdf file from DOI of a research paper.", epilog=""
    )

    parser.add_argument(
        "-o",
        "--output_2",
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
        "--sci-hub-url", type=str, help="Specific Sci-Hub URL to use", metavar="url"
    )

    args = parser.parse_args()

    if args.doi is None and args.name is None and args.url is None:
        parser.error("At least one of --doi, --name, --url must be specified.")
    if len([arg for arg in (args.doi, args.name, args.url) if arg is not None]) > 1:
        parser.error("Only one of --doi, --name, --url must be specified.")

    doi2pdf(
        args.doi, output=args.output, name=args.name, url=args.url, open_pdf=args.open,
        sci_hub_url=args.sci_hub_url
    )


if __name__ == "__main__":
    main()