# `data_extraction/` — Pipeline 1: literature → reactions

Mines ~300 k Crossref records for copolymerisation papers, filters them down to ~1.8 k downloadable PDFs, and extracts reactivity-ratio measurements with a vision–language model. The output is the curated `extracted_reactions.csv` that Pipeline 2 (`copol_prediction/`) ingests.

## Stages

```
Crossref query ──┐
                 ▼
       keyword + journal scoring  →  ~300 k → ~1.8 k candidates
                 ▼
       embedding-based pre-download filter  (cached → provenance/embeddings/)
                 ▼
       PDF download
                 ▼
       per-PDF quality scoring  (LLM, gates whether to extract)
                 ▼
       vision-language extraction  (per-paper JSON → extracted_data.json)
                 ▼
       consolidate + canonicalise SMILES + assign reaction_id
                 ▼
       src/copolextractor/extracted_reactions.csv      ← Pipeline 2 reads this
```

Each stage is implemented as a module under [`src/copolextractor/`](../src/copolextractor/); [`obtain_data.py`](obtain_data.py) is the thin orchestration layer.

## Layout

```
data_extraction/
├── obtain_data.py           ← orchestrator: ExtractionConfig + ExtractionSteps
├── provenance/              ← intermediates of the frozen run that produced
│                              the shipped extracted_reactions.csv (kept for
│                              audit / re-use). See provenance/README.md.
├── archive/                 ← superseded evaluation harnesses
└── token_stats.json         ← cumulative LLM token usage log
```

`obtain_data.py`'s default `ExtractionConfig` writes a fresh run's intermediates under `data_extraction/artifacts/{metadata, llm, datasets, …}/` (created on first run). The directory is gitignored by convention; promote anything you want to keep into `provenance/` and update its README.

## Running the pipeline

```bash
# Default: extraction + CSV export only (skip the slow + paid Crossref +
# download + embedding steps). Useful for re-extracting from PDFs already on
# disk.
python data_extraction/obtain_data.py
```

Enable additional steps explicitly — most are expensive (network calls, paid
API usage):

```python
from data_extraction.obtain_data import ExtractionConfig, ExtractionSteps, obtain_data

config = ExtractionConfig(...)             # see the dataclass for every path / threshold
steps = ExtractionSteps(
    crossref_search=True,                  # Crossref sweep + metadata
    pre_download_filter=True,              # keyword + embedding score
    pdf_download=True,                     # actually download PDFs
    pre_extraction_filter=True,            # LLM quality gate
    extraction=True,                       # the big one — vision-LM extraction
    save_data=True,                        # consolidate to extracted_reactions.csv
)
obtain_data(config, steps)
```

## Key modules (under `src/copolextractor/`)

| Module | Stage |
|---|---|
| `crossref_search.py` | Crossref DOI sweep + metadata fetch |
| `predownloadfilter/` | keyword scoring + embedding-based relevance filter |
| `PDF_download.py` | downloads the candidate PDFs |
| `preextractionfilter/` | LLM-based per-PDF quality scoring |
| `extraction_with_GPT_PDF.py` | the actual vision-language extraction |
| `data_into_csv.py` | consolidation → `extracted_reactions.csv` |

## Output

The handoff to Pipeline 2 is a single file:

```
src/copolextractor/extracted_reactions.csv     # ~5 MB, ~5 k rows
```

Pipeline 2 (`copol_prediction/`) loads it via `data_processing.load_and_preprocess_data` to produce its own `processed_data.csv`. See [`../copol_prediction/README.md`](../copol_prediction/README.md).
