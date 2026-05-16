# Copolymerization Reactivity Prediction

Machine learning system for extracting copolymerization reactivity ratios (r₁, r₂) from scientific literature and predicting the r-product class for new monomer pairs.

> 📄 **Paper:** _TODO: add citation, DOI and link to the manuscript here when published._ See [Citation](#citation) below for the BibTeX placeholder to fill in.

## Quick start

```bash
git clone https://github.com/lamalab-org/copolymer-reactivity
cd copolymer-reactivity/copol_prediction/api
docker compose up                    # pulls ghcr.io/lamalab-org/copolymer-reactivity:latest
```

Open <http://localhost:8000/docs> for the interactive API.

A SMILES → prediction round-trip in one shell snippet:

```bash
FEATURES=$(curl -sS -X POST http://localhost:8000/preprocess_all \
  -H 'Content-Type: application/json' \
  -d '{"monomer1_smiles":"C=COC(C)=O","monomer2_smiles":"C=COC(=O)c1ccccc1",
       "solvent_smiles":"c1ccccc1","method":"solvent",
       "polytype":"free radical","temperature":79.6}' | jq -c '.features')

curl -sS -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d "{\"features\": $FEATURES}"
```

To install as a library instead of running the API:

```bash
pip install -e .            # core deps; add [extraction]/[training]/[testing] as needed
```

## Where things live

| Component | Path | README |
|---|---|---|
| REST API (FastAPI) | `copol_prediction/api/` | [api/README](copol_prediction/api/README.md) |
| Training & analysis pipeline | `copol_prediction/` | [pipeline/README](copol_prediction/README.md) |
| Literature data extraction | `data_extraction/` | [extraction/README](data_extraction/README.md) |
| Baseline experiments | `experiments/` | [experiments/README](experiments/README.md) |
| Core libraries (`copolextractor`, `copolpredictor`) | `src/` | — |
| Trained model artifact | `copol_prediction/artifacts/model_bundle/` | — |

## Model facts

The model artifact is the source of truth for class definitions, feature schema, and metrics — query it instead of trusting prose:

```bash
curl http://localhost:8000/model/info                  # while the API is running
jq . copol_prediction/artifacts/model_bundle/meta.json # offline
cat copol_prediction/artifacts/model_bundle/all_metrics.txt
```

## Citation

```bibtex
@article{TODO_paper_key,
  title   = {TODO: paper title},
  author  = {TODO: full author list},
  journal = {TODO: journal / arXiv ID},
  year    = {TODO},
  doi     = {TODO},
}
```

If you reference the code separately from the paper:

```bibtex
@software{copolymer_reactivity_code,
  author = {Schilling-Wilhelmi, Mara and Jablonka, Kevin M.},
  title  = {Copolymerization Reactivity Prediction},
  url    = {https://github.com/lamalab-org/copolymer-reactivity},
  year   = {2025},
}
```

## License

MIT — see [LICENSE](LICENSE).
