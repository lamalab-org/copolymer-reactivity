# `data_extraction/provenance/` — intermediate artefacts of the extraction run

The cached intermediates from the extraction run that produced the dataset shipped in `src/copolextractor/extracted_reactions.csv`. Nothing in the active pipeline reads from here at runtime; the directory is here so the path from "Crossref query" to "row in `extracted_reactions.csv`" is auditable, and so re-runs can reuse expensive intermediates (embeddings, LLM scores) instead of paying to compute them again.

## What's here

| File / directory | Size | Step | What it contains |
|---|---:|---|---|
| `embeddings/embedded_papers.json` | ~97 MB | Pre-download filter | OpenAI embeddings of Crossref title + abstract; consumed by `src/copolextractor/predownloadfilter/embedding_filter.py` to score relevance before paying to download the PDF |
| `scored_doi.json` | ~41 MB | Pre-download filter | DOI → relevance-score map; the Crossref sweep result, narrowed from ~300 k to ~1.8 k download candidates |
| `paper_list.json` | ~3 MB | Pre-download filter | XGBoost-filtered DOI list that proceeded to PDF download |
| `selected_200_papers.json` | ~600 KB | Pre-extraction filter | Manually-screened subset used as an evaluation harness for the extraction LLM |
| `copol_database/` | ~6 MB | Pre-extraction filter | Auxiliary copol-database lookups used during scoring |
| `model_output_score/` | ~7 MB | Pre-extraction filter | LLM-based PDF quality scores (gate on whether to attempt extraction) |
| `failed_crossref.json` | ~10 KB | Crossref search | DOIs that returned 404 / errored — recorded for audit |
| `journals.json` | ~19 KB | Crossref search | Per-journal hit counts (used for the keyword-scoring weights) |
| `error_log.txt` | ~8 KB | (all steps) | Pipeline error log |

## When to keep, when to wipe

Keep this directory if you want to:

- Audit *which* Crossref records were scored, embedded, or rejected.
- Resume the pipeline without recomputing embeddings or LLM scores.
- Spot-check per-paper LLM quality scores.

It's safe to wipe before a fresh end-to-end run. The active pipeline (`data_extraction/obtain_data.py`) writes new intermediates to a path configured in its `ExtractionConfig` — see [`../README.md`](../README.md) — and does not read from `provenance/` by default. Expect different DOIs on a re-run: Crossref changes over time and the LLM scoring is stochastic.

## What's *not* here

The extracted reactions themselves are at `src/copolextractor/extracted_reactions.csv` (~5 MB, ~5 k rows). That file feeds `copol_prediction/`'s `load_and_preprocess_data` and is the actual hand-off to Pipeline 2.
