# Database Scripts

Pipeline for converting the curated reaction dataset into NOMAD archive files for upload to the polymerization OASIS.

```
copol_prediction/processed_data.csv                              ← input
            │
            │ ↓ create_database_json.py
            ▼
dump/database_json/polymerization_*.json                         per-measurement JSONs
dump/database_json/monomers/*.json                               (schema-compliant with
            │                                                     PolymerizationReactionInput
            │                                                     from FAIRmat-NFDI/nomad-
            │                                                     polymerization-reactions)
            │
            │ ↓ convert_to_archives.py + convert_monomers.py
            ▼
database/output/polymerization/*.archive.json                    NOMAD archives
database/output/monomers/*.archive.json                          (ready to upload)
```

## Quickstart (clean rebuild)

From the repository root:

```bash
rm -rf dump/database_json
rm -rf database/output/polymerization database/output/monomers
mkdir -p database/output/polymerization database/output/monomers

# 1) processed_data.csv → schema-compliant per-reaction JSONs
python database/create_database_json.py

# 2) JSONs → NOMAD archives (needs the nomad-polymerization CLI, see below)
python database/convert_to_archives.py

# 3) Monomer property files → monomer archives
python database/convert_monomers.py --source copol_prediction/api/molecule_properties
```

## Scripts

### `create_database_json.py`

Reads `copol_prediction/processed_data.csv` and writes one schema-compliant JSON per measurement to `dump/database_json/`. The output validates against `PolymerizationReactionInput` in [FAIRmat-NFDI/nomad-polymerization-reactions](https://github.com/FAIRmat-NFDI/nomad-polymerization-reactions/blob/main/src/nomad_polymerization_reactions/models.py): reactivity ratios are nested under `r_values` and `conf_intervals`, the schema-canonical key names (`monomer1_smiles`, `polymerization_method`, `calculation_method`, `r-product`) are used, and PCA-reduced polymerisation-type / method embeddings + RDKit solvent descriptors are added.

DOIs in the `source` field are recovered from `source_filename` where the LLM-populated `source` column is empty (~half the rows) — every output JSON carries a canonical `https://doi.org/<DOI>` URL when one exists.

```bash
# Default: reads copol_prediction/processed_data.csv
python database/create_database_json.py

# Or point at the paper-frozen snapshot:
python database/create_database_json.py --input copol_prediction/paper_dataset/processed_data.csv

# Or use a legacy per-reaction JSON-directory input layout:
python database/create_database_json.py --input dump/processed_reactions/
```

**Output**:
- `dump/database_json/polymerization_<doi>_<idx>.json` — one per measurement row in the CSV.
- `dump/database_json/monomers/monomer_*.json` — copied from `copol_prediction/api/molecule_properties/` for every monomer that appears in the dataset.

### `convert_to_archives.py`
Converts JSON files to NOMAD archive files (`.archive.json`) using the `nomad-polymerization` CLI tool.

**⚠️ IMPORTANT: NumPy Compatibility**

The `nomad-polymerization` tool requires NumPy < 2.0. If you have NumPy 2.x installed, create a separate virtual environment:

```bash
# Create virtual environment
python -m venv nomad_env

# Activate (Linux/Mac)
source nomad_env/bin/activate

# Activate (Windows)
nomad_env\Scripts\activate

# Install dependencies
pip install 'numpy<2.0' git+https://github.com/FAIRmat-NFDI/nomad-polymerization-reactions.git
```

Then run the script (it should automatically find the correct environment, or specify with `--python-env`).

**Standard Installation (if NumPy < 2.0 is already installed):**
```bash
pip install git+https://github.com/FAIRmat-NFDI/nomad-polymerization-reactions.git
```

**Usage:**

Standard usage (converts all files from `dump/database_json`):
```bash
python database/convert_to_archives.py
```

Archive files are saved by default in `database/output/`:
- `database/output/polymerization/*.archive.json` - Polymerization reactions
- `database/output/monomers/*.archive.json` - Monomers

Convert single file:
```bash
python database/convert_to_archives.py dump/database_json/polymerization_10.1002_actp.1983.010340208_1.json
```

Different input directory:
```bash
python database/convert_to_archives.py dump/other_json_folder
```

Custom output directory:
```bash
python database/convert_to_archives.py --output dump/custom_archives
```

Save archive files in the same directory as JSON files:
```bash
python database/convert_to_archives.py --same-dir
```

**Modes:**
- `--mode polymerization` - For polymerization reactions
- `--mode monomer` - For monomers
- Without `--mode`, the mode is automatically detected from the filename

### `convert_monomers.py`
Converts monomer *property files* (molecule properties) to NOMAD monomer archive files.

**Usage:**

Standard usage (uses `copol_prediction/api/molecule_properties/`):
```bash
python database/convert_monomers.py
```

Monomer files are:
1. Renamed from MD5 hash to `monomer_<IUPAC_name>.json`
2. Converted to archive files
3. Saved in `database/output/monomers/`

Different source directory:
```bash
python database/convert_monomers.py --source copol_prediction/output/molecule_properties
```

Custom output directory:
```bash
python database/convert_monomers.py --output dump/monomer_archives
```

Reconvert all files (including existing ones):
```bash
python database/convert_monomers.py --no-skip
```

**Note:** The script uses `copol_utils.smiles_to_name()` to calculate IUPAC names. If not available, SMILES-based names are used.

**Fixing Failed Files:**

List failed files:
```bash
python database/convert_monomers.py --list-failed
```

Convert single file with custom name:
```bash
python database/convert_monomers.py --fix "C=COCC1CO1.OCCO.json" "2-(oxiran-2-ylmethoxy)ethanol"
```

The archive files can then be uploaded directly to NOMAD.

## Analysis (plots + dataset statistics)

### `analysis/plot_combined_database_figure.py` (dataset analysis)

Creates the combined multi-panel figure and prints basic dataset statistics (counts + ranges).

```bash
python database/analysis/plot_combined_database_figure.py
```

**Output:**
- `database/analysis/figures/dataset_analysis.pdf`
- `database/analysis/figures/dataset_analysis.png`

**Boiling points input:**
- `database/analysis/solvent_boiling_points_c.json` (curated solvent boiling points in °C; `null` means “exclude”)

### `analysis/plot_polymerization_trends.py`

Creates stacked-area temporal trend plots for polymerization types and methods.
Uses the local `publication_year` column (no Crossref calls).

```bash
python database/analysis/plot_polymerization_trends.py
```

**Output:**
- `database/analysis/figures/polymerization_trends.pdf`
- `database/analysis/figures/polymerization_trends.png`
