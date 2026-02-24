#!/usr/bin/env python3
"""
Script to convert monomer JSON files to NOMAD archive files.
- Reads monomer files from copol_prediction/api/molecule_properties/
- Renames them from MD5 hash to monomer_<IUPAC_name>.json
- Converts them to archive files
- Saves them in database/output/monomers/
"""

import json
import subprocess
import sys
import shutil
import re
from pathlib import Path
from typing import Optional, Dict, Any

# Import copolextractor utils for IUPAC name conversion
try:
    # Add project root to path
    project_root = Path(__file__).parent.parent
    src_path = project_root / "src"
    if src_path.exists():
        import sys
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        from copolextractor import utils as copol_utils
        COPOL_UTILS_AVAILABLE = True
        print("✓ copolextractor.utils loaded successfully")
    else:
        COPOL_UTILS_AVAILABLE = False
        print("Warning: src directory not found. Will use SMILES-based names.")
except ImportError as e:
    COPOL_UTILS_AVAILABLE = False
    print(f"Warning: copolextractor.utils not available ({e}). Will use SMILES-based names.")
except Exception as e:
    COPOL_UTILS_AVAILABLE = False
    print(f"Warning: Error loading copolextractor.utils ({e}). Will use SMILES-based names.")


def sanitize_filename(name: str) -> str:
    """
    Sanitize IUPAC name for use in filename.
    Replaces problematic characters with underscores.
    """
    if not name:
        return "unknown"
    
    # Replace problematic characters
    name = re.sub(r'[<>:"|?*\\/]', '_', name)
    name = re.sub(r'[^\w._-]', '_', name)
    name = re.sub(r'_+', '_', name)  # Replace multiple underscores with single
    name = name.strip('_')  # Remove leading/trailing underscores
    
    # Limit length
    if len(name) > 200:
        name = name[:200]
    
    return name if name else "unknown"


def get_iupac_name(smiles: str) -> Optional[str]:
    """
    Get IUPAC name from SMILES using copolextractor.utils.smiles_to_name.
    """
    if COPOL_UTILS_AVAILABLE:
        try:
            iupac_name = copol_utils.smiles_to_name(smiles)
            if iupac_name and iupac_name.strip():
                return iupac_name.strip()
        except Exception as e:
            # Silently fail and return None - will use SMILES-based name as fallback
            pass
    
    return None


def get_monomer_name_from_file(json_file: Path) -> tuple[Optional[str], Optional[str]]:
    """
    Extract SMILES and try to get IUPAC name from monomer JSON file.
    Returns (smiles, iupac_name).
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        smiles = data.get('smiles')
        if not smiles:
            return None, None
        
        iupac_name = get_iupac_name(smiles)
        return smiles, iupac_name
    except Exception as e:
        print(f"Error reading {json_file.name}: {e}")
        return None, None


def convert_monomer_file(
    source_file: Path,
    target_dir: Path,
    iupac_name: Optional[str] = None,
    smiles: Optional[str] = None
) -> bool:
    """
    Convert a monomer JSON file to archive format.
    
    Args:
        source_file: Path to source monomer JSON file
        target_dir: Directory to save the archive file
        iupac_name: IUPAC name for the filename (optional)
        smiles: SMILES string (optional, used as fallback)
    
    Returns:
        True if conversion successful, False otherwise
    """
    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Read source file and add 'name' field if missing
    try:
        with open(source_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Ensure 'name' field exists (required by Monomer schema)
        if 'name' not in data or not data['name']:
            if iupac_name:
                data['name'] = iupac_name
            elif smiles:
                data['name'] = smiles  # Fallback to SMILES
            else:
                data['name'] = source_file.stem  # Last resort
        
        # Ensure 'smiles' field exists
        if 'smiles' not in data and smiles:
            data['smiles'] = smiles
    except Exception as e:
        print(f"Error reading {source_file.name}: {e}")
        return False
    
    # Determine filename
    if iupac_name:
        sanitized_name = sanitize_filename(iupac_name)
        new_filename = f"monomer_{sanitized_name}.json"
    elif smiles:
        # Fallback to SMILES-based name
        sanitized_smiles = sanitize_filename(smiles[:50])  # Limit length
        new_filename = f"monomer_{sanitized_smiles}.json"
    else:
        # Last resort: use original filename
        new_filename = f"monomer_{source_file.stem}.json"
    
    # Handle duplicate names
    target_file = target_dir / new_filename
    counter = 1
    while target_file.exists():
        base_name = new_filename.replace('.json', '')
        target_file = target_dir / f"{base_name}_{counter}.json"
        counter += 1
    
    # Write modified data to target location with new name
    try:
        with open(target_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error writing {target_file.name}: {e}")
        return False
    
    # Convert to archive using nomad-polymerization
    cmd = ["nomad-polymerization", "archive", str(target_file), "--same-dir"]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode == 0:
            # Archive file should be created as target_file.archive.json
            archive_file = target_file.parent / (target_file.stem + ".archive.json")
            if archive_file.exists():
                # Move archive to final destination (target_dir)
                final_archive = target_dir / archive_file.name
                if archive_file != final_archive:
                    shutil.move(str(archive_file), str(final_archive))
                # Remove the temporary renamed JSON file (keep original)
                target_file.unlink()
                return True
            else:
                print(f"Warning: Archive file not created for {target_file.name}")
                return False
        else:
            print(f"Error converting {target_file.name}: {result.stderr}")
            # Clean up copied file on error
            if target_file.exists():
                target_file.unlink()
            return False
    except subprocess.TimeoutExpired:
        print(f"Timeout converting {target_file.name}")
        if target_file.exists():
            target_file.unlink()
        return False
    except Exception as e:
        print(f"Exception converting {target_file.name}: {e}")
        if target_file.exists():
            target_file.unlink()
        return False


def process_monomer_directory(
    source_dir: Path,
    target_dir: Path,
    skip_existing: bool = True
) -> tuple[int, int, int]:
    """
    Process all monomer JSON files in source directory.
    
    Args:
        source_dir: Directory containing monomer JSON files
        target_dir: Directory to save archive files
        skip_existing: If True, skip files that already have archive versions
    
    Returns:
        Tuple of (successful_count, failed_count, skipped_count)
    """
    json_files = list(source_dir.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {source_dir}")
        return 0, 0, 0
    
    print(f"Found {len(json_files)} monomer JSON files to process")
    print(f"Target directory: {target_dir}")
    print()
    
    successful = 0
    failed = 0
    skipped = 0
    
    for json_file in json_files:
        # Check if archive already exists
        if skip_existing:
            # Check if archive exists in target directory
            # We need to check by SMILES since we don't know the final name yet
            smiles, iupac_name = get_monomer_name_from_file(json_file)
            if smiles and iupac_name:
                sanitized_name = sanitize_filename(iupac_name)
                potential_archive = target_dir / f"monomer_{sanitized_name}.archive.json"
                if potential_archive.exists():
                    skipped += 1
                    continue
        
        print(f"Processing {json_file.name}...", end=" ", flush=True)
        
        # Get SMILES and IUPAC name
        smiles, iupac_name = get_monomer_name_from_file(json_file)
        
        if not smiles:
            print("✗ (no SMILES found)")
            failed += 1
            continue
        
        # Show what name will be used (for debugging)
        if iupac_name:
            name_preview = iupac_name[:30] + "..." if len(iupac_name) > 30 else iupac_name
            print(f"[IUPAC: {name_preview}]", end=" ", flush=True)
        
        # Convert file
        if convert_monomer_file(json_file, target_dir, iupac_name, smiles):
            print("✓")
            successful += 1
        else:
            print("✗")
            failed += 1
    
    return successful, failed, skipped


def convert_with_custom_name(
    source_file: Path,
    target_dir: Path,
    custom_name: str,
    smiles: Optional[str] = None
) -> bool:
    """Convert monomer file with a custom name."""
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Read source file
    try:
        with open(source_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except:
        # If not valid JSON, create minimal structure
        data = {'smiles': smiles or source_file.stem, 'name': custom_name}
    
    # Ensure required fields
    if 'name' not in data or not data['name']:
        data['name'] = custom_name
    if 'smiles' not in data and smiles:
        data['smiles'] = smiles
    
    # Create filename
    sanitized_name = sanitize_filename(custom_name)
    new_filename = f"monomer_{sanitized_name}.json"
    target_file = target_dir / new_filename
    
    # Handle duplicates
    counter = 1
    while target_file.exists():
        base_name = new_filename.replace('.json', '')
        target_file = target_dir / f"{base_name}_{counter}.json"
        counter += 1
    
    # Write file
    try:
        with open(target_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error writing {target_file.name}: {e}")
        return False
    
    # Convert to archive
    cmd = ["nomad-polymerization", "archive", str(target_file), "--same-dir"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            archive_file = target_file.parent / (target_file.stem + ".archive.json")
            if archive_file.exists():
                final_archive = target_dir / archive_file.name
                if archive_file != final_archive:
                    shutil.move(str(archive_file), str(final_archive))
                target_file.unlink()
                return True
            else:
                print(f"Warning: Archive not created")
                return False
        else:
            print(f"Error: {result.stderr}")
            if target_file.exists():
                target_file.unlink()
            return False
    except Exception as e:
        print(f"Exception: {e}")
        if target_file.exists():
            target_file.unlink()
        return False


def find_failed_files(source_dir: Path, target_dir: Path) -> list[tuple[Path, str]]:
    """Find files that failed to convert."""
    failed = []
    json_files = list(source_dir.glob("*.json"))
    
    for json_file in json_files:
        # Skip non-monomer files
        if json_file.name in ["mongodb_analysis.json", "calculation_summary.json"]:
            continue
        
        # Check if it's valid JSON
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, dict) or 'smiles' not in data:
                # Try to read SMILES from filename
                smiles = json_file.stem
                failed.append((json_file, smiles))
                continue
        except:
            # Invalid JSON, try to use filename as SMILES
            smiles = json_file.stem
            failed.append((json_file, smiles))
            continue
        
        # Check if archive exists
        try:
            smiles = data.get('smiles')
            if smiles:
                iupac_name = get_iupac_name(smiles)
                if iupac_name:
                    sanitized = sanitize_filename(iupac_name)
                    archive_file = target_dir / f"monomer_{sanitized}.archive.json"
                    if not archive_file.exists():
                        failed.append((json_file, smiles))
        except:
            failed.append((json_file, None))
    
    return failed


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert monomer JSON files to NOMAD archive files"
    )
    parser.add_argument(
        "--source",
        "-s",
        type=str,
        default="copol_prediction/api/molecule_properties",
        help="Source directory containing monomer JSON files (default: copol_prediction/api/molecule_properties)"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory for archive files (default: database/output/monomers)"
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Don't skip files that already have archive versions"
    )
    parser.add_argument(
        "--list-failed",
        action="store_true",
        help="List files that failed to convert and exit"
    )
    parser.add_argument(
        "--fix",
        type=str,
        nargs=2,
        metavar=("FILE", "NAME"),
        help="Fix a specific file with custom name: --fix <filename> <custom_name>"
    )
    parser.add_argument(
        "--smiles",
        type=str,
        help="SMILES string for --fix option (if not in file)"
    )
    
    args = parser.parse_args()
    
    # Set paths
    source_dir = Path(args.source)
    if args.output:
        target_dir = Path(args.output)
    else:
        script_dir = Path(__file__).parent
        target_dir = script_dir / "output" / "monomers"
    
    # Handle --fix option
    if args.fix:
        source_file = source_dir / args.fix[0]
        custom_name = args.fix[1]
        if not source_file.exists():
            print(f"ERROR: File not found: {source_file}")
            sys.exit(1)
        print(f"Converting {source_file.name} with custom name: {custom_name}")
        success = convert_with_custom_name(source_file, target_dir, custom_name, args.smiles)
        sys.exit(0 if success else 1)
    
    # Handle --list-failed option
    if args.list_failed:
        failed = find_failed_files(source_dir, target_dir)
        print(f"Found {len(failed)} files that need conversion:")
        print()
        for json_file, smiles in failed:
            print(f"  {json_file.name}")
            if smiles:
                print(f"    SMILES: {smiles}")
                if COPOL_UTILS_AVAILABLE:
                    try:
                        iupac = get_iupac_name(smiles)
                        if iupac:
                            print(f"    Suggested IUPAC: {iupac}")
                    except:
                        pass
            print()
        sys.exit(0)
    
    if not source_dir.exists():
        print(f"ERROR: Source directory does not exist: {source_dir}")
        sys.exit(1)
    
    if not source_dir.is_dir():
        print(f"ERROR: Source path is not a directory: {source_dir}")
        sys.exit(1)
    
    print("Starting monomer conversion...")
    print(f"Source directory: {source_dir}")
    print(f"Target directory: {target_dir}")
    print()
    
    successful, failed, skipped = process_monomer_directory(
        source_dir,
        target_dir,
        skip_existing=not args.no_skip
    )
    
    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Skipped: {skipped}")
    print(f"  Total: {successful + failed + skipped}")
    print(f"\nArchive files saved to: {target_dir}")
    
    if failed > 0:
        print(f"\nTo fix failed files, use:")
        print(f"  python database/convert_monomers.py --list-failed")
        print(f"  python database/convert_monomers.py --fix <filename> <custom_name>")


if __name__ == "__main__":
    main()
