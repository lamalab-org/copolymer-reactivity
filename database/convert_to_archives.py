#!/usr/bin/env python3
"""Convert per-reaction / per-monomer JSON files to NOMAD ``.archive.json``.

Thin wrapper around the upstream ``nomad-polymerization archive`` CLI from
``FAIRmat-NFDI/nomad-polymerization-reactions``. Reads the JSONs produced by
``create_database_json.py`` (and ``convert_monomers.py``), normalises a small
set of legacy field names, then shells out to the CLI for each file.
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple

NOMAD_CLI = "nomad-polymerization"
DEFAULT_INPUT_DIR = Path("dump/database_json")
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "output"
INSTALL_HINT = "Install with:\n" "  pip install '.[database]'\n"


def nomad_cli_available() -> bool:
    """True iff the ``nomad-polymerization`` executable is on PATH."""
    return shutil.which(NOMAD_CLI) is not None


def normalize_json_fields(data: dict) -> dict:
    """Promote legacy field names to the canonical NOMAD schema keys.

    Upstream ``PolymerizationReactionInput`` (see
    https://github.com/FAIRmat-NFDI/nomad-polymerization-reactions/blob/main/src/nomad_polymerization_reactions/models.py)
    expects ``monomer{1,2}_smiles`` and ``calculation_method``; we adapt the
    legacy ``monomer{1,2}_s`` / ``determination_method`` aliases without
    overwriting an existing canonical value.
    """
    normalized = data.copy()
    for legacy, canonical in (
        ("monomer1_s", "monomer1_smiles"),
        ("monomer2_s", "monomer2_smiles"),
        ("determination_method", "calculation_method"),
    ):
        if legacy in normalized and canonical not in normalized:
            normalized[canonical] = normalized.pop(legacy)
    return normalized


def determine_mode_from_filename(filename: str) -> Optional[str]:
    """Polymerization vs monomer mode (``None`` == monomer, the CLI default)."""
    if filename.startswith("monomer_"):
        return None
    return "polymerization"


def convert_json_to_archive(
    json_file: Path,
    output_dir: Optional[Path],
    mode: Optional[str],
    same_dir: bool,
) -> bool:
    """Run the CLI on one JSON; place the resulting archive next to it (or in `output_dir`).

    The CLI only supports ``--same-dir``, so we feed it a normalised copy
    sitting next to the input, then move the produced archive to its final
    destination under the right ``polymerization/`` or ``monomers/`` subdir.
    """
    if same_dir:
        final_output_dir = json_file.parent
    else:
        base = output_dir if output_dir is not None else DEFAULT_OUTPUT_DIR
        final_output_dir = base / ("polymerization" if mode == "polymerization" else "monomers")
    final_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        original_data = json.loads(json_file.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Exception reading {json_file.name}: {e}")
        return False
    normalized_data = normalize_json_fields(original_data)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, dir=json_file.parent
    ) as tmp:
        json.dump(normalized_data, tmp, indent=2, ensure_ascii=False)
        tmp_path = Path(tmp.name)

    try:
        cmd = [NOMAD_CLI, "archive", str(tmp_path)]
        if mode:
            cmd.extend(["--mode", mode])
        cmd.append("--same-dir")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            print(f"Timeout converting {json_file.name}")
            return False

        if result.returncode != 0:
            print(f"Error converting {json_file.name}: {result.stderr}")
            return False

        # `with_suffix("")` is unsafe for stems containing dots — pathlib
        # treats whatever follows the last dot as the suffix. Build by stem.
        produced = tmp_path.parent / f"{tmp_path.stem}.archive.json"
        if not produced.exists():
            print(f"Warning: No archive file created for {json_file.name}")
            return False

        target = final_output_dir / (json_file.stem + ".archive.json")
        if produced != target:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(produced), str(target))
        return True
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def convert_directory(
    input_dir: Path,
    output_dir: Optional[Path],
    same_dir: bool,
    mode: Optional[str],
) -> Tuple[int, int]:
    """Convert every ``*.json`` under `input_dir`.

    When `mode` is ``None`` the per-file mode is inferred from the filename
    (``monomer_*`` → monomer, anything else → polymerization).
    """
    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return 0, 0

    print(f"Found {len(json_files)} JSON files to convert")
    if not same_dir:
        target_base = output_dir if output_dir is not None else DEFAULT_OUTPUT_DIR
        print(f"Output directory: {target_base}")
        print(f"  - Polymerization files → {target_base}/polymerization/")
        print(f"  - Monomer files → {target_base}/monomers/")
    else:
        print("Output mode: same directory as JSON files")
    print()

    successful = failed = 0
    for json_file in json_files:
        file_mode = mode if mode is not None else determine_mode_from_filename(json_file.name)
        print(f"  {json_file.name}...", end=" ", flush=True)
        if convert_json_to_archive(json_file, output_dir, file_mode, same_dir):
            print("ok")
            successful += 1
        else:
            print("fail")
            failed += 1
    return successful, failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert JSON files to NOMAD archive files")
    parser.add_argument(
        "input_path",
        type=Path,
        nargs="?",
        default=DEFAULT_INPUT_DIR,
        help=f"Path to a JSON file or directory of JSON files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help=f"Base output directory for archive files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--same-dir",
        action="store_true",
        help="Save archive files in the same directory as JSON files (overrides --output)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["polymerization", "monomer"],
        default=None,
        help="Conversion mode (auto-detected from filename if not specified)",
    )
    args = parser.parse_args()

    if not nomad_cli_available():
        print(
            f"WARNING: '{NOMAD_CLI}' not found on PATH.\n{INSTALL_HINT}\n"
            "Continuing anyway in case it's available via another mechanism...\n"
        )

    if not args.input_path.exists():
        print(f"ERROR: Path does not exist: {args.input_path}")
        sys.exit(1)

    # `--mode monomer` maps to the CLI's "no --mode" default.
    cli_mode = "polymerization" if args.mode == "polymerization" else None

    if args.input_path.is_file():
        mode = cli_mode if args.mode else determine_mode_from_filename(args.input_path.name)
        print(f"Converting single file: {args.input_path.name} (mode={mode or 'monomer'})")
        success = convert_json_to_archive(args.input_path, args.output, mode, args.same_dir)
        sys.exit(0 if success else 1)

    if args.input_path.is_dir():
        successful, failed = convert_directory(
            args.input_path, args.output, args.same_dir, cli_mode if args.mode else None
        )
        print(f"\nConversion complete: {successful} successful, {failed} failed")
        sys.exit(0 if failed == 0 else 1)

    print(f"ERROR: Path is neither a file nor a directory: {args.input_path}")
    sys.exit(1)


if __name__ == "__main__":
    main()
