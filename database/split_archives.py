#!/usr/bin/env python3
"""
Script to split archive files into batches of maximum 1000 files each.
Creates subdirectories like polymerization_batch_1/, polymerization_batch_2/, etc.
"""

import shutil
from pathlib import Path
from typing import List


def split_archives(
    source_dir: Path,
    batch_size: int = 1000,
    prefix: str = "batch",
    clean_existing_batches: bool = True,
) -> None:
    """
    Split archive files into batches.
    
    Args:
        source_dir: Directory containing archive files
        batch_size: Maximum number of files per batch (default: 1000)
        prefix: Prefix for batch directory names (default: "batch")
        clean_existing_batches: If True, delete existing batch directories first
    """
    # Clean existing batch directories first to avoid mixing old/new files
    if clean_existing_batches:
        existing_batch_dirs = sorted([p for p in source_dir.glob(f"{prefix}_*") if p.is_dir()])
        if existing_batch_dirs:
            print(f"Cleaning {len(existing_batch_dirs)} existing batch directories in {source_dir}...")
            for d in existing_batch_dirs:
                try:
                    shutil.rmtree(d)
                except Exception as e:
                    print(f"  Warning: Could not remove {d.name}/: {e}")
            print()

    # Get all archive files
    archive_files = sorted(list(source_dir.glob("*.archive.json")))
    
    if not archive_files:
        print(f"No archive files found in {source_dir}")
        return
    
    total_files = len(archive_files)
    num_batches = (total_files + batch_size - 1) // batch_size  # Ceiling division
    
    print(f"Found {total_files} archive files")
    print(f"Will create {num_batches} batches of maximum {batch_size} files each")
    print()
    
    # Create batches
    for batch_num in range(1, num_batches + 1):
        batch_dir = source_dir / f"{prefix}_{batch_num}"
        batch_dir.mkdir(exist_ok=True)
        
        # Calculate range for this batch
        start_idx = (batch_num - 1) * batch_size
        end_idx = min(start_idx + batch_size, total_files)
        batch_files = archive_files[start_idx:end_idx]
        
        print(f"Creating {prefix}_{batch_num}/ with {len(batch_files)} files...", end=" ", flush=True)
        
        # Move files to batch directory
        moved = 0
        for archive_file in batch_files:
            try:
                target = batch_dir / archive_file.name
                shutil.move(str(archive_file), str(target))
                moved += 1
            except Exception as e:
                print(f"\nError moving {archive_file.name}: {e}")
        
        print(f"✓ ({moved}/{len(batch_files)} moved)")
    
    print()
    print(f"{'='*60}")
    print(f"Split complete!")
    print(f"  Total files: {total_files}")
    print(f"  Batches created: {num_batches}")
    print(f"  Files per batch: ~{batch_size}")
    print()
    print(f"Batch directories:")
    for batch_num in range(1, num_batches + 1):
        batch_dir = source_dir / f"{prefix}_{batch_num}"
        if batch_dir.exists():
            file_count = len(list(batch_dir.glob("*.archive.json")))
            print(f"  {batch_dir.name}/ - {file_count} files")


def split_monomers(
    source_dir: Path,
    batch_size: int = 1000,
    prefix: str = "batch",
    clean_existing_batches: bool = True,
) -> None:
    """Split monomer archive files into batches."""
    split_archives(source_dir, batch_size, prefix, clean_existing_batches=clean_existing_batches)


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Split archive files into batches for upload"
    )
    parser.add_argument(
        "--polymerization",
        action="store_true",
        help="Split polymerization archives (default)"
    )
    parser.add_argument(
        "--monomers",
        action="store_true",
        help="Split monomer archives"
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help="Split both polymerization and monomer archives"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Maximum number of files per batch (default: 1000)"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="batch",
        help="Prefix for batch directory names (default: 'batch')"
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not delete existing batch_* directories before splitting"
    )
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    output_dir = script_dir / "output"
    clean_existing_batches = not args.no_clean
    
    if args.monomers or args.both:
        monomers_dir = output_dir / "monomers"
        if monomers_dir.exists():
            print("Splitting monomer archives...")
            print(f"Source: {monomers_dir}")
            print()
            split_monomers(
                monomers_dir,
                args.batch_size,
                args.prefix,
                clean_existing_batches=clean_existing_batches,
            )
            print()
    
    if args.polymerization or (not args.monomers and not args.both):
        polymerization_dir = output_dir / "polymerization"
        if polymerization_dir.exists():
            print("Splitting polymerization archives...")
            print(f"Source: {polymerization_dir}")
            print()
            split_archives(
                polymerization_dir,
                args.batch_size,
                args.prefix,
                clean_existing_batches=clean_existing_batches,
            )
        else:
            print(f"ERROR: Directory not found: {polymerization_dir}")


if __name__ == "__main__":
    main()
