#!/usr/bin/env python3
"""
Script to rename molecular property files from SMILES-based names to MD5 hashes.

This script:
1. Reads all JSON files in molecule_properties/
2. Extracts the SMILES string from each file
3. Creates MD5 hash of the SMILES
4. Renames the file to use the MD5 hash as filename
"""

import json
import hashlib
from pathlib import Path

def get_smiles_md5(smiles: str) -> str:
    """Create MD5 hash from SMILES string."""
    return hashlib.md5(smiles.encode('utf-8')).hexdigest()

def rename_molecule_properties(directory: Path):
    """Rename all molecule property files to use MD5 hashes."""
    renamed_count = 0
    skipped_count = 0
    error_count = 0
    
    for json_file in directory.glob("*.json"):
        try:
            # Read the JSON file
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract SMILES
            smiles = data.get("smiles")
            if not smiles:
                print(f"⚠ Skipping {json_file.name}: No 'smiles' field found")
                skipped_count += 1
                continue
            
            # Calculate MD5 hash
            md5_hash = get_smiles_md5(smiles)
            new_filename = f"{md5_hash}.json"
            new_path = directory / new_filename
            
            # Skip if already renamed
            if json_file.name == new_filename:
                print(f"✓ Already renamed: {json_file.name}")
                skipped_count += 1
                continue
            
            # Check if target file already exists
            if new_path.exists():
                print(f"⚠ Target already exists: {new_filename} (from {json_file.name})")
                # Compare SMILES to see if they match
                with open(new_path, 'r') as f:
                    existing_data = json.load(f)
                if existing_data.get("smiles") == smiles:
                    print(f"  → Files have same SMILES, removing duplicate {json_file.name}")
                    json_file.unlink()
                    skipped_count += 1
                    continue
                else:
                    print(f"  → Different SMILES! Keeping both files.")
                    error_count += 1
                    continue
            
            # Rename the file
            json_file.rename(new_path)
            print(f"✓ Renamed: {json_file.name} → {new_filename}")
            renamed_count += 1
            
        except json.JSONDecodeError as e:
            print(f"✗ Error reading {json_file.name}: Invalid JSON - {e}")
            error_count += 1
        except Exception as e:
            print(f"✗ Error processing {json_file.name}: {e}")
            error_count += 1
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Renamed: {renamed_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Errors:  {error_count}")
    print(f"{'='*60}")

if __name__ == "__main__":
    import sys
    
    # Get directory path
    if len(sys.argv) > 1:
        directory = Path(sys.argv[1])
    else:
        # Default to molecule_properties directory relative to this script
        directory = Path(__file__).parent / "molecule_properties"
    
    if not directory.exists():
        print(f"Error: Directory {directory} does not exist!")
        sys.exit(1)
    
    print(f"Renaming files in: {directory}")
    print(f"{'='*60}\n")
    
    # Ask for confirmation
    response = input("This will rename all JSON files. Continue? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Aborted.")
        sys.exit(0)
    
    rename_molecule_properties(directory)

