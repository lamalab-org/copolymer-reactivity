"""Utilities around Crossref search and metadata aggregation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, MutableSequence, Sequence

import requests
from crossref.restful import Works


def add_to_database(doi: str, source: str, format_type: str | None, extracted_data: MutableSequence[dict]) -> None:
    """Add a DOI entry to the in-memory database if it is not present yet."""

    if not any(item["doi"] == doi for item in extracted_data):
        extracted_data.append({"doi": doi, "source": source, "format": format_type})
    else:
        print(f"DOI {doi} is already in the database.")


def process_copol_database(copol_file_path: Path, extracted_data: MutableSequence[dict]) -> None:
    """Seed the working set with DOIs from the curated copol database."""

    print("Processing copol database papers...")
    if not copol_file_path.exists():
        print(f"Warning: Copol database file not found at {copol_file_path}")
        return

    with copol_file_path.open("r", encoding="utf-8") as file:
        copol_data = json.load(file)

    for entry in copol_data:
        doi = entry.get("paper")
        if doi:
            add_to_database(doi, "copol database", "pdf", extracted_data)


def process_crossref(
    query: str,
    output_crossref_file: Path,
    extracted_data: MutableSequence[dict],
) -> None:
    """Query CrossRef for papers that match *query* and persist the raw response."""

    print("Processing CrossRef papers...")
    works = Works(timeout=120)
    query_result = works.query(bibliographic=query).select(
        "DOI", "title", "author", "type", "publisher", "issued"
    )
    results = list(query_result)

    output_crossref_file.parent.mkdir(parents=True, exist_ok=True)
    with output_crossref_file.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    for entry in results:
        if "DOI" in entry:
            add_to_database(entry["DOI"], "crossref", None, extracted_data)


def get_crossref_data(doi: str, source: str, format_type: str | None) -> dict:
    """Fetch metadata from CrossRef API for a given DOI."""

    url = f"https://api.crossref.org/works/{doi}"
    response = requests.get(url, timeout=30)

    if response.status_code == 200:
        data = response.json()
        item = data.get("message", {})
        
        # Safely extract title
        title_list = item.get("title", [])
        title = title_list[0] if title_list else "No title"
        
        # Safely extract journal
        journal_list = item.get("container-title", [])
        journal = journal_list[0] if journal_list else "No journal title"
        
        abstract = item.get("abstract", "No abstract available")
        keywords = item.get("subject", "No keywords available")

        return {
            "DOI": doi,
            "Title": title,
            "Abstract": abstract,
            "Keywords": keywords,
            "Journal": journal,
            "Source": source,
            "Format": format_type,
        }

    return {
        "DOI": doi,
        "Error": f"Unable to fetch data (Status Code: {response.status_code})",
    }


def is_doi_processed(doi: str, results: Sequence[dict]) -> bool:
    """Return ``True`` if *doi* already exists inside *results*."""

    return any(entry.get("DOI") == doi for entry in results)


def fetch_and_save_metadata(
    _input_file: Path,
    metadata_output_file: Path,
    extracted_data: Sequence[dict],
) -> None:
    """Fetch CrossRef metadata for all DOIs and write them to *metadata_output_file*."""

    print("Fetching metadata for DOIs...")
    if metadata_output_file.exists():
        with metadata_output_file.open("r", encoding="utf-8") as output_file:
            results: List[dict] = json.load(output_file)
    else:
        results = []

    for entry in extracted_data:
        doi_url = entry["doi"]
        doi = doi_url.split("doi.org/")[-1]  # Extract DOI from the URL
        source = entry["source"]
        format_type = entry.get("format")

        if not is_doi_processed(doi, results):
            result = get_crossref_data(doi, source, format_type)
            print(result)
            results.append(result)

            metadata_output_file.parent.mkdir(parents=True, exist_ok=True)
            with metadata_output_file.open("w", encoding="utf-8") as output_file:
                json.dump(results, output_file, indent=4)
        else:
            print(f"DOI {doi} is already processed.")

    with metadata_output_file.open("w", encoding="utf-8") as output_file:
        json.dump(results, output_file, indent=4)
    print(f"Metadata saved to {metadata_output_file}.")


def save_extracted_data(output_file_path: Path, extracted_data: Sequence[dict]) -> None:
    """Persist *extracted_data* to *output_file_path* as JSON."""

    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    with output_file_path.open("w", encoding="utf-8") as output_file:
        json.dump(list(extracted_data), output_file, indent=4)
    print(f"Extracted data saved to {output_file_path}.")


def _default_base_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data_extraction"


def main(
    crossref_query: str,
    output_file_crossref_search: str | Path,
    crossref_metadata_output_file: str | Path,
    *,
    copol_file_path: str | Path | None = None,
    collected_doi_output_file: str | Path | None = None,
    process_copol: bool = True,
    process_crossref_search: bool = True,
    fetch_metadata: bool = True,
) -> None:
    """Top-level helper that aggregates DOIs and metadata from various sources.
    
    Args:
        crossref_query: Query string for Crossref API search.
        output_file_crossref_search: Path to save raw Crossref search results.
        crossref_metadata_output_file: Path to save detailed metadata for all DOIs.
        copol_file_path: Optional path to copol database file.
        collected_doi_output_file: Optional path to save aggregated DOI list.
        process_copol: If True, process copol database DOIs (default: True).
        process_crossref_search: If True, perform Crossref search (default: True).
        fetch_metadata: If True, fetch detailed metadata for DOIs (default: True).
    """

    base_dir = _default_base_dir()
    metadata_root = base_dir / "artifacts" / "metadata" / "output"

    output_crossref_path = Path(output_file_crossref_search)
    metadata_output_path = Path(crossref_metadata_output_file)
    copol_path = (
        Path(copol_file_path)
        if copol_file_path
        else metadata_root / "copol_database" / "copol_paper_list.json"
    )
    all_doi_output_path = (
        Path(collected_doi_output_file)
        if collected_doi_output_file
        else metadata_root / "collected_doi.json"
    )

    extracted_data: List[dict] = []

    # Step 1: Process copol database (optional)
    if process_copol:
        process_copol_database(copol_path, extracted_data)
    
    # Step 2: Process Crossref search (optional)
    if process_crossref_search:
        process_crossref(crossref_query, output_crossref_path, extracted_data)
    
    # If we need to fetch metadata but haven't collected DOIs yet, load existing DOI file
    if fetch_metadata and not (process_copol or process_crossref_search):
        if all_doi_output_path.exists():
            print(f"Loading existing DOI list from {all_doi_output_path}")
            with all_doi_output_path.open("r", encoding="utf-8") as f:
                extracted_data = json.load(f)
        else:
            print(f"Warning: No DOI data available. Run with process_copol=True or process_crossref_search=True first.")
            return
    
    # Save aggregated DOI list if we collected any
    if extracted_data and (process_copol or process_crossref_search):
        save_extracted_data(all_doi_output_path, extracted_data)
    
    # Step 3: Fetch detailed metadata (optional)
    if fetch_metadata:
        fetch_and_save_metadata(all_doi_output_path, metadata_output_path, extracted_data)
