# Database Scripts

This folder contains all scripts for preparing and converting data for NOMAD upload.

## Quickstart (clean rebuild)

From the repository root:

```bash
# Remove previous outputs (optional but recommended for a clean rebuild)
rm -rf dump/database_json
rm -rf database/output/polymerization database/output/monomers
mkdir -p database/output/polymerization database/output/monomers

# 1) Create reaction + monomer JSON files
python database/create_database_json.py

# 2) Convert reaction JSONs to NOMAD archives
python database/convert_to_archives.py

# 3) Convert monomer *property files* to monomer archives
python database/convert_monomers.py --source copol_prediction/api/molecule_properties

# 4) Split archives into upload batches
python database/split_archives.py --both
```

## Scripts

### `create_database_json.py`
Creates cleaned JSON files from processed reaction data for database upload.

**Usage:**
```bash
python database/create_database_json.py
```

**Output:**
- `dump/database_json/polymerization_*.json` - Polymerization reactions
- `dump/database_json/monomers/*.json` - Monomers

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

### `split_archives.py`
Splits archive files into batches of maximum 1000 files each (for upload limits).

**Usage:**

Split polymerization archives:
```bash
python database/split_archives.py --polymerization
```

Split monomer archives:
```bash
python database/split_archives.py --monomers
```

Split both:
```bash
python database/split_archives.py --both
```

Custom batch size:
```bash
python database/split_archives.py --polymerization --batch-size 500
```

**Output:**
Files are moved into subdirectories:
- `database/output/polymerization/batch_1/` - first 1000 files
- `database/output/polymerization/batch_2/` - next 1000 files
- `database/output/polymerization/batch_3/` - remaining files
- etc.

Each batch can then be uploaded separately to NOMAD.

## Workflow

1. **Create JSON files:**
   ```bash
   python database/create_database_json.py
   ```

2. **Convert to archive files:**
   ```bash
   python database/convert_to_archives.py
   ```
   
   Archive files are saved in `database/output/`:
   - `database/output/polymerization/*.archive.json` - Polymerization reactions
   - `database/output/monomers/*.archive.json` - Monomers

3. **Convert monomer files:**
   ```bash
   python database/convert_monomers.py
   ```

4. **Split archives into batches (if needed):**
   ```bash
   python database/split_archives.py --polymerization
   python database/split_archives.py --monomers
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
