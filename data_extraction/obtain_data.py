"""Convenience entry point for orchestrating the data-extraction pipeline.

The module pulls together the individual building blocks that live in
``src/copolextractor``.  Lightweight data classes allow us to declare the
static configuration (paths, thresholds, prompts, …) and select which steps to
run.  This is useful because many steps are slow or depend on third-party
services and should therefore be executed on demand.

Typical usage::

    from data_extraction.obtain_data import (
        ExtractionConfig, ExtractionSteps, CrossrefSteps, obtain_data
    )

    config = ExtractionConfig(...)

    # Run full pipeline
    steps = ExtractionSteps(crossref_search=True, extraction=True)
    obtain_data(config, steps)

    # Run only metadata fetch (skip copol database and crossref search)
    substeps = CrossrefSteps(process_copol=False, process_crossref_search=False, fetch_metadata=True)
    steps = ExtractionSteps(crossref_search=True, crossref_substeps=substeps, extraction=False)
    obtain_data(config, steps)

The default ``main`` function keeps the behaviour users are familiar with by
only executing the expensive LLM-based extraction followed by persisting the
results.  All other steps can be enabled via the ``ExtractionSteps`` flags.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from copolextractor.crossref_search import main as crossref_search
from copolextractor.data_into_csv import main as save_data
from copolextractor.extraction_with_GPT_PDF import main as extractor
from copolextractor.PDF_download import main as pdf_download
from copolextractor.predownloadfilter.pre_download_filter import main as pre_download_filter
from copolextractor.preextractionfilter.pre_extraction_filter import main as pre_extraction_filter


@dataclass
class ExtractionConfig:
    """Holds the static configuration for the extraction pipeline.

    Paths are stored as :class:`~pathlib.Path` objects to avoid fragile
    string-based path juggling.  Downstream functions still expect strings, so
    we convert them just before invoking the respective modules.
    """

    crossref_keyword: str
    output_file_crossref_search: Path
    crossref_metadata_output_file: Path
    keywords_filter: Dict[str, int]
    output_file_pre_download_filter: Path
    pdf_download_input_file: Path
    score_limit: int
    number_of_selected_papers: int
    input_folder_images: Path
    output_folder_data_extraction: Path
    output_file_data_extraction: Path
    seed_xgboost_model: int
    threshold_xgboost_model: float
    pdf_folder: Path
    output_folder_images: Path
    output_folder_llm_score: Path
    training_file_xgboost_model: Path
    output_file_xgboost_filter: Path
    key_embedding_filter: str
    values_embedding_filter: List[str]
    scoring_file_embedding_filter: Path
    existing_doi_csv: Path


@dataclass
class CrossrefSteps:
    """Select which sub-steps of the Crossref search should be executed.

    This allows fine-grained control over the Crossref module, enabling scenarios
    like fetching only metadata for existing DOIs without re-running the search.
    """

    process_copol: bool = False
    process_crossref_search: bool = False
    fetch_metadata: bool = True


@dataclass
class ExtractionSteps:
    """Select which parts of the pipeline should be executed.

    The defaults mirror the previous behaviour: only the expensive extraction
    step and the persistence layer run automatically.  Enable additional steps
    when fresh metadata needs to be collected or the filters have to be
    re-computed.
    """

    crossref_search: bool = True
    crossref_substeps: Optional[CrossrefSteps] = None
    pre_download_filter: bool = False
    pdf_download: bool = False
    pdf_quality_filter: bool = False
    extraction: bool = True
    persist_results: bool = True


def _ensure_parent_dir(path: Path) -> None:
    """Create the parent directory for *path* if necessary."""

    if path.suffix:  # Treat items with a suffix as files
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        path.mkdir(parents=True, exist_ok=True)


def obtain_data(
    config: ExtractionConfig,
    steps: Optional[ExtractionSteps] = None,
) -> None:
    """Run the extraction pipeline.

    Args:
        config: Static configuration describing inputs, outputs and thresholds.
        steps: Optional selection of pipeline stages.  If omitted the default
            :class:`ExtractionSteps` instance is used (extraction + persistence).
    """

    steps = steps or ExtractionSteps()

    # Ensure that all folders we are about to touch exist.
    for folder in (
        config.input_folder_images,
        config.output_folder_data_extraction,
        config.pdf_folder,
        config.output_folder_images,
        config.output_folder_llm_score,
    ):
        folder.mkdir(parents=True, exist_ok=True)

    if steps.crossref_search:
        _ensure_parent_dir(config.output_file_crossref_search)
        _ensure_parent_dir(config.crossref_metadata_output_file)

        # Use substeps if provided, otherwise run all substeps
        substeps = steps.crossref_substeps or CrossrefSteps()

        crossref_search(
            config.crossref_keyword,
            str(config.output_file_crossref_search),
            str(config.crossref_metadata_output_file),
            process_copol=substeps.process_copol,
            process_crossref_search=substeps.process_crossref_search,
            fetch_metadata=substeps.fetch_metadata,
        )

    if steps.pre_download_filter:
        _ensure_parent_dir(config.output_file_pre_download_filter)
        pre_download_filter(
            config.keywords_filter,
            config.score_limit,
            config.number_of_selected_papers,
            str(config.crossref_metadata_output_file),
            str(config.output_file_pre_download_filter),
            config.key_embedding_filter,
            config.values_embedding_filter,
            str(config.scoring_file_embedding_filter),
            str(config.existing_doi_csv),
        )

    if steps.pdf_download:
        pdf_download(
            str(config.pdf_download_input_file),
            str(config.pdf_folder),
        )

    if steps.pdf_quality_filter:
        _ensure_parent_dir(config.output_file_xgboost_filter)
        pre_extraction_filter(
            config.seed_xgboost_model,
            config.threshold_xgboost_model,
            str(config.pdf_folder),
            str(config.output_folder_images),
            str(config.output_folder_llm_score),
            str(config.training_file_xgboost_model),
            str(config.output_file_pre_download_filter),
            str(config.output_file_xgboost_filter),
            enable_pdf_processing=True,
        )

    if steps.extraction:
        _ensure_parent_dir(config.output_file_data_extraction)
        extractor(
            str(config.input_folder_images),
            str(config.output_folder_data_extraction),
            str(config.output_file_xgboost_filter),
            str(config.pdf_folder),
            str(config.output_file_data_extraction),
        )

    if steps.persist_results:
        save_data(str(config.output_folder_data_extraction))


def main() -> None:
    """Execute the default extraction pipeline configuration.

    The configuration mirrors the previous hard-coded behaviour and keeps all
    paths relative to the project root.  Only the extraction and CSV export run
    by default to avoid accidental API calls or large downloads.  Toggle
    additional steps via :class:`ExtractionSteps`.
    """

    base_dir = Path(__file__).resolve().parent
    artifacts_dir = base_dir / "artifacts"
    metadata_dir = artifacts_dir / "metadata" / "output"
    llm_dir = artifacts_dir / "llm"
    datasets_dir = artifacts_dir / "datasets"

    llm_extractions_dir = llm_dir / "extractions" / "model_output_GPT4-o"
    llm_scores_dir = llm_dir / "model_output_score"
    llm_images_dir = llm_dir / "processed_images"

    config = ExtractionConfig(
        crossref_keyword='copolymerization AND "reactivity ratio"',
        output_file_crossref_search=metadata_dir / "crossref_search.json",
        crossref_metadata_output_file=metadata_dir / "collected_doi_metadata.json",
        keywords_filter={
            "copolymerization": 10,
            "polymerization": 5,
            "monomers": 5,
            "copolymers": 5,
            "ratios": 20,
            "reactivity ratios": 40,
        },
        output_file_pre_download_filter=metadata_dir / "selected_papers.json",
        pdf_download_input_file=metadata_dir / "selected_papers_merged.json",
        score_limit=65,
        number_of_selected_papers=2000,
        input_folder_images=llm_images_dir,
        output_folder_data_extraction=llm_extractions_dir,
        output_file_data_extraction=llm_dir / "extractions" / "extracted_data.json",
        seed_xgboost_model=22,
        threshold_xgboost_model=0.7,
        pdf_folder=metadata_dir / "PDF",
        output_folder_images=llm_images_dir,
        output_folder_llm_score=llm_scores_dir,
        training_file_xgboost_model=metadata_dir / "copol_database" / "copol_paper_list.json",
        output_file_xgboost_filter=metadata_dir / "paper_list.json",
        key_embedding_filter="polymerization_type",
        values_embedding_filter=[
            "free radical",
            "Free radical",
            "Free Radical",
            "atom transfer radical polymerization",
            "atom-transfer radical polymerization",
            "nickel-mediated radical",
            "bulk",
            "Radical",
            "radical",
            "controlled radical",
            "controlled/living radical",
            "conventional radical polymerization",
            "reversible deactivation radical polymerization",
            "reversible addition-fragmentation chain transfer polymerization",
            "reversible addition-fragmentation chain transfer",
            "Homogeneous Radical",
            "Radiation-induced",
            "radiation-induced",
            "Radiation-Initiated",
            "photo-induced polymerization",
            "photopolymerization",
            "thermal polymerization",
            "thermal",
            "group transfer polymerization",
            "Emulsion",
            "Homogeneous Radical",
            "semicontinuous emulsion",
            "emulsion",
        ],
        scoring_file_embedding_filter=datasets_dir / "extracted_reactions.csv",
        existing_doi_csv=datasets_dir / "extracted_reactions.csv",
    )

    steps = ExtractionSteps()

    obtain_data(config, steps)


if __name__ == "__main__":
    main()
