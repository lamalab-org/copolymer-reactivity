"""Run only the metadata fetch step (skip copol database and crossref search)."""

import sys
from pathlib import Path

# Add src and data_extraction to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent / 'data_extraction'))

from obtain_data import ExtractionConfig, ExtractionSteps, CrossrefSteps, obtain_data

# Setup paths
base_dir = Path(__file__).parent / "data_extraction"
artifacts_dir = base_dir / "artifacts"
metadata_dir = artifacts_dir / "metadata" / "output"
llm_dir = artifacts_dir / "llm"
datasets_dir = artifacts_dir / "datasets"

llm_extractions_dir = llm_dir / "extractions" / "model_output_GPT4-o"
llm_scores_dir = llm_dir / "model_output_score"
llm_images_dir = llm_dir / "processed_images"

# Create config (same as in obtain_data.py main())
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
        "free radical", "Free radical", "Free Radical",
        "atom transfer radical polymerization",
        "atom-transfer radical polymerization",
        "nickel-mediated radical", "bulk", "Radical", "radical",
        "controlled radical", "controlled/living radical",
        "conventional radical polymerization",
        "reversible deactivation radical polymerization",
        "reversible addition-fragmentation chain transfer polymerization",
        "reversible addition-fragmentation chain transfer",
        "Homogeneous Radical", "Radiation-induced", "radiation-induced",
        "Radiation-Initiated", "photo-induced polymerization",
        "photopolymerization", "thermal polymerization", "thermal",
        "group transfer polymerization", "Emulsion", "Homogeneous Radical",
        "semicontinuous emulsion", "emulsion",
    ],
    scoring_file_embedding_filter=datasets_dir / "extracted_reactions.csv",
    existing_doi_csv=datasets_dir / "extracted_reactions.csv",
)

# Configure to run ONLY metadata fetch
substeps = CrossrefSteps(
    process_copol=False,              # Skip copol database
    process_crossref_search=False,    # Skip Crossref search
    fetch_metadata=True               # Only fetch metadata
)

steps = ExtractionSteps(
    crossref_search=True,             # Enable crossref module
    crossref_substeps=substeps,       # Use our custom substeps
    pre_download_filter=False,        # Skip all other steps
    pdf_download=False,
    pdf_quality_filter=False,
    extraction=False,
    persist_results=False
)

print("=" * 60)
print("Running: METADATA FETCH ONLY")
print("=" * 60)
print("\nConfiguration:")
print(f"  - process_copol: {substeps.process_copol}")
print(f"  - process_crossref_search: {substeps.process_crossref_search}")
print(f"  - fetch_metadata: {substeps.fetch_metadata}")
print("\nThis will:")
print("  1. Load DOIs from collected_doi.json")
print("  2. Fetch detailed metadata from Crossref API")
print("  3. Save to collected_doi_metadata.json")
print("\nStarting...\n")

obtain_data(config, steps)

print("\n" + "=" * 60)
print("✓ Metadata fetch completed!")
print("=" * 60)
