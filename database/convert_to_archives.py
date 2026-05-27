#!/usr/bin/env python3
"""
Script to convert JSON files to NOMAD archive files using the nomad-polymerization CLI tool.
Converts all polymerization and monomer JSON files to archive.json files.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple


def check_nomad_polymerization_installed() -> tuple[bool, Optional[str]]:
    """
    Check if nomad-polymerization CLI tool is installed.
    Returns (is_installed, error_message).
    """
    try:
        # Try to find the executable first
        import shutil

        exe_path = shutil.which("nomad-polymerization")
        if not exe_path:
            return (
                False,
                "nomad-polymerization CLI not found in PATH. Please install it with:\n  pip install git+https://github.com/FAIRmat-NFDI/nomad-polymerization-reactions.git",
            )

        # Try to run it with a longer timeout (tool can be slow to start)
        result = subprocess.run(
            ["nomad-polymerization", "--help"],
            capture_output=True,
            text=True,
            timeout=30,  # Increased timeout
        )
        if result.returncode == 0:
            return True, None
        else:
            # Check for numpy compatibility issues
            error_output = result.stderr or result.stdout
            if (
                "numpy" in error_output.lower()
                or "np.round_" in error_output
                or "AttributeError" in error_output
            ):
                return False, (
                    "numpy_compatibility: nomad-polymerization requires numpy < 2.0. "
                    "Please use a virtual environment with numpy < 2.0, e.g.:\n"
                    "  python -m venv nomad_env\n"
                    "  source nomad_env/bin/activate  # or 'nomad_env\\Scripts\\activate' on Windows\n"
                    "  pip install 'numpy<2.0' git+https://github.com/FAIRmat-NFDI/nomad-polymerization-reactions.git"
                )
            return False, f"CLI returned error: {error_output[:500]}"
    except FileNotFoundError:
        return (
            False,
            "nomad-polymerization CLI not found. Please install it with:\n  pip install git+https://github.com/FAIRmat-NFDI/nomad-polymerization-reactions.git",
        )
    except subprocess.TimeoutExpired:
        # Even if timeout, the tool might still work - try a quick test conversion instead
        return True, None  # Assume it's installed, let the actual conversion fail if not
    except Exception as e:
        return False, f"Error checking installation: {e}"


def normalize_json_fields(data: dict) -> dict:
    """Adapt input keys to the names ``generate_pr_archive_from_json`` reads.

    The upstream ``PolymerizationReactionInput`` schema (see
    https://github.com/FAIRmat-NFDI/nomad-polymerization-reactions/blob/main/src/nomad_polymerization_reactions/models.py)
    accepts the canonical key ``monomer{1,2}_smiles`` and ``calculation_method``
    (with ``determination_method`` as a deprecated fallback). For inputs that
    only carry the legacy ``monomer{1,2}_s`` or only ``determination_method``,
    we promote them to the canonical names so the SMILES + calculation method
    survive into the generated archive.
    """
    normalized = data.copy()

    # Promote legacy SMILES keys to the canonical names if the canonical ones
    # are absent. Never overwrite an existing canonical key.
    if "monomer1_s" in normalized and "monomer1_smiles" not in normalized:
        normalized["monomer1_smiles"] = normalized.pop("monomer1_s")
    if "monomer2_s" in normalized and "monomer2_smiles" not in normalized:
        normalized["monomer2_smiles"] = normalized.pop("monomer2_s")

    # Promote the deprecated determination_method fallback to the canonical
    # calculation_method if no canonical value is present.
    if "determination_method" in normalized and "calculation_method" not in normalized:
        normalized["calculation_method"] = normalized.pop("determination_method")

    return normalized


def convert_json_to_archive(
    json_file: Path,
    output_dir: Optional[Path] = None,
    mode: Optional[str] = None,
    same_dir: bool = False,
) -> bool:
    """
    Convert a single JSON file to an archive file using nomad-polymerization CLI.

    Args:
        json_file: Path to the JSON file to convert
        output_dir: Base directory to save archive files (if not same_dir)
        mode: Mode for conversion ('polymerization' or None for monomer)
        same_dir: If True, save archive in same directory as JSON file

    Returns:
        True if conversion successful, False otherwise
    """
    import shutil
    import tempfile

    # Determine output directory based on mode
    if same_dir:
        # Use same directory as JSON file
        temp_output_dir = json_file.parent
        final_output_dir = json_file.parent
    else:
        # Use separate subdirectories based on mode
        if output_dir is None:
            output_dir = Path("database/output")

        if mode == "polymerization":
            subdir = "polymerization"
        else:  # monomer
            subdir = "monomers"

        temp_output_dir = output_dir / subdir
        final_output_dir = temp_output_dir

    # Create output directory
    final_output_dir.mkdir(parents=True, exist_ok=True)

    # Read and normalize JSON fields before conversion
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            original_data = json.load(f)

        # Normalize field names to match expected format
        normalized_data = normalize_json_fields(original_data)

        # Create temporary file with normalized data
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, dir=json_file.parent
        ) as tmp_file:
            json.dump(normalized_data, tmp_file, indent=2, ensure_ascii=False)
            tmp_file_path = Path(tmp_file.name)

        try:
            # The CLI tool only supports --same-dir, so we need to:
            # 1. Create archive in same directory as JSON file (temporarily)
            # 2. Move it to the final destination
            cmd = ["nomad-polymerization", "archive", str(tmp_file_path)]

            if mode:
                cmd.extend(["--mode", mode])

            # Always use --same-dir since CLI doesn't support --output
            cmd.append("--same-dir")

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120  # Increased timeout for conversion
            )

            if result.returncode == 0:
                # Find the created archive file (it will be in the same directory as temp file)
                archive_file = tmp_file_path.parent / (tmp_file_path.stem + ".archive.json")

                if archive_file.exists():
                    # Determine final archive filename (based on original filename, not temp file)
                    final_archive_name = json_file.stem + ".archive.json"

                    # Move to final destination if needed
                    if not same_dir:
                        target_file = final_output_dir / final_archive_name
                        # Create target directory if it doesn't exist
                        target_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(archive_file), str(target_file))
                    else:
                        # Rename to match original filename
                        final_archive = json_file.parent / final_archive_name
                        if archive_file != final_archive:
                            shutil.move(str(archive_file), str(final_archive))
                    return True
                else:
                    print(f"Warning: No archive file created for {json_file.name}")
                    return False
            else:
                print(f"Error converting {json_file.name}: {result.stderr}")
                return False
        finally:
            # Clean up temporary file
            if tmp_file_path.exists():
                tmp_file_path.unlink()

    except subprocess.TimeoutExpired:
        print(f"Timeout converting {json_file.name}")
        return False
    except Exception as e:
        print(f"Exception converting {json_file.name}: {e}")
        return False


def convert_directory(
    input_dir: Path,
    output_dir: Optional[Path] = None,
    same_dir: bool = False,
    mode: Optional[str] = None,
) -> tuple[int, int]:
    """
    Convert all JSON files in a directory to archive files.

    Args:
        input_dir: Directory containing JSON files
        output_dir: Directory to save archive files (if not same_dir)
        same_dir: If True, save archives in same directory as JSON files
        mode: Mode for conversion ('polymerization' or None for monomer)

    Returns:
        Tuple of (successful_count, failed_count)
    """
    json_files = list(input_dir.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return 0, 0

    print(f"Found {len(json_files)} JSON files to convert")

    successful = 0
    failed = 0

    for json_file in json_files:
        print(f"Converting {json_file.name}...", end=" ", flush=True)
        if convert_json_to_archive(json_file, output_dir, mode, same_dir):
            print("✓")
            successful += 1
        else:
            print("✗")
            failed += 1

    return successful, failed


def determine_mode_from_filename(filename: str) -> Optional[str]:
    """Determine conversion mode from filename."""
    if filename.startswith("polymerization_"):
        return "polymerization"
    elif filename.startswith("monomer_"):
        return None  # Default mode for monomers
    else:
        # Default to polymerization for unknown patterns
        return "polymerization"


def convert_all_files(
    json_dir: Path, output_dir: Optional[Path] = None, same_dir: bool = False
) -> None:
    """
    Convert all JSON files in directory, automatically determining mode from filename.

    Args:
        json_dir: Directory containing JSON files
        output_dir: Base directory to save archive files (default: database/output)
        same_dir: If True, save archives in same directory as JSON files
    """
    json_files = list(json_dir.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {json_dir}")
        return

    # Set default output directory
    if output_dir is None and not same_dir:
        script_dir = Path(__file__).parent
        output_dir = script_dir / "output"

    print(f"Found {len(json_files)} JSON files to convert")
    if same_dir:
        print(f"Output mode: same directory as JSON files")
    else:
        print(f"Output directory: {output_dir}")
        print(f"  - Polymerization files → {output_dir}/polymerization/")
        print(f"  - Monomer files → {output_dir}/monomers/")
    print()

    successful = 0
    failed = 0

    # Group by mode for better progress reporting
    polymerization_files = []
    monomer_files = []
    unknown_files = []

    for json_file in json_files:
        mode = determine_mode_from_filename(json_file.name)
        if mode == "polymerization":
            polymerization_files.append(json_file)
        elif mode is None:
            monomer_files.append(json_file)
        else:
            unknown_files.append(json_file)

    # Convert polymerization files
    if polymerization_files:
        print(f"Converting {len(polymerization_files)} polymerization files...")
        for json_file in polymerization_files:
            print(f"  {json_file.name}...", end=" ", flush=True)
            if convert_json_to_archive(json_file, output_dir, "polymerization", same_dir):
                print("✓")
                successful += 1
            else:
                print("✗")
                failed += 1

    # Convert monomer files
    if monomer_files:
        print(f"\nConverting {len(monomer_files)} monomer files...")
        for json_file in monomer_files:
            print(f"  {json_file.name}...", end=" ", flush=True)
            if convert_json_to_archive(json_file, output_dir, None, same_dir):
                print("✓")
                successful += 1
            else:
                print("✗")
                failed += 1

    # Convert unknown files (default to polymerization mode)
    if unknown_files:
        print(
            f"\nConverting {len(unknown_files)} files (unknown type, using polymerization mode)..."
        )
        for json_file in unknown_files:
            print(f"  {json_file.name}...", end=" ", flush=True)
            if convert_json_to_archive(json_file, output_dir, "polymerization", same_dir):
                print("✓")
                successful += 1
            else:
                print("✗")
                failed += 1

    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(json_files)}")


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Convert JSON files to NOMAD archive files")
    parser.add_argument(
        "input_path",
        type=str,
        nargs="?",
        default="dump/database_json",
        help="Path to JSON file or directory containing JSON files (default: dump/database_json)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Base output directory for archive files (default: database/output)",
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
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip installation check (useful if check times out)",
    )

    args = parser.parse_args()

    # Check if nomad-polymerization is installed (non-blocking check)
    if not args.skip_check:
        is_installed, error_msg = check_nomad_polymerization_installed()
        if not is_installed:
            if error_msg and "Timeout" not in error_msg:
                # Only fail if it's a real error, not a timeout
                print("WARNING: Could not verify nomad-polymerization installation.")
                print(f"  {error_msg}")
                print("  Attempting to proceed anyway...")
                print("  (Use --skip-check to skip this check in the future)")
                print()
            # If timeout, assume it's installed and proceed
    else:
        print("Skipping installation check...")
        print()

    input_path = Path(args.input_path)

    if not input_path.exists():
        print(f"ERROR: Path does not exist: {input_path}")
        sys.exit(1)

    # Set output directory (default to database/output)
    if args.output:
        output_dir = Path(args.output)
    elif not args.same_dir:
        script_dir = Path(__file__).parent
        output_dir = script_dir / "output"
    else:
        output_dir = None

    same_dir = args.same_dir

    if input_path.is_file():
        # Single file conversion
        mode = args.mode or determine_mode_from_filename(input_path.name)
        print(f"Converting single file: {input_path.name}")
        print(f"Mode: {mode or 'monomer (default)'}")
        success = convert_json_to_archive(input_path, output_dir, mode, same_dir)
        sys.exit(0 if success else 1)
    elif input_path.is_dir():
        # Directory conversion
        if args.mode:
            # Use specified mode for all files
            successful, failed = convert_directory(input_path, output_dir, same_dir, args.mode)
            print(f"\nConversion complete: {successful} successful, {failed} failed")
            sys.exit(0 if failed == 0 else 1)
        else:
            # Auto-detect mode from filenames
            convert_all_files(input_path, output_dir, same_dir)
            sys.exit(0)
    else:
        print(f"ERROR: Path is neither a file nor a directory: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
