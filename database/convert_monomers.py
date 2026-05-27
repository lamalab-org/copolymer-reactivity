#!/usr/bin/env python3
"""Build NOMAD monomer archives covering every monomer used in the dataset.

Driven by the curated reaction table (default
``copol_prediction/processed_data.csv``): for every distinct
``monomer1_smiles``/``monomer2_smiles`` we

  1. look up the precomputed quantum-feature JSON under
     ``copol_prediction/api/molecule_properties/<md5(smiles)>.json``
     (rich entry: best conformer, charges, IP/EA, Fukui, …), or
  2. emit a minimal ``{smiles, name}`` stub when no feature file exists.

A stub is a valid ``MonomerInput`` record (only ``name`` + ``smiles`` are
required by the schema), so every monomer in the dataset ends up archived
even when its feature calculation hasn't been run yet.

A coverage report at the end prints how many archives carry rich features
vs. were stubbed, and which SMILES were stubbed.
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REACTIONS_CSV = REPO_ROOT / "copol_prediction" / "processed_data.csv"
DEFAULT_FEATURE_SOURCE_DIR = REPO_ROOT / "copol_prediction" / "api" / "molecule_properties"
DEFAULT_TARGET_DIR = Path(__file__).parent / "output" / "monomers"
NOMAD_CLI = "nomad-polymerization"
FIX_LOG_FILENAME = "monomer_fix_log.json"

# Optional dependency: copolextractor.utils.smiles_to_name for IUPAC fallback.
_src_path = REPO_ROOT / "src"
if _src_path.exists() and str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))
try:
    from copolextractor import utils as copol_utils  # type: ignore

    COPOL_UTILS_AVAILABLE = True
except Exception:
    copol_utils = None  # type: ignore
    COPOL_UTILS_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Small helpers                                                               #
# --------------------------------------------------------------------------- #


def smiles_md5(smiles: str) -> str:
    return hashlib.md5(smiles.encode("utf-8")).hexdigest()


def sanitize_filename(name: str) -> str:
    """Turn an arbitrary name into a filesystem-safe filename stem."""
    if not name:
        return "unknown"
    name = re.sub(r'[<>:"|?*\\/]', "_", name)
    name = re.sub(r"[^\w._-]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    if len(name) > 200:
        name = name[:200]
    return name or "unknown"


def _is_meaningful(value) -> bool:
    if not isinstance(value, str):
        return False
    return value.strip().lower() not in ("", "none", "na", "nan", "null")


def iupac_name(smiles: str) -> Optional[str]:
    """Look up an IUPAC name via copolextractor, returning ``None`` on failure."""
    if not COPOL_UTILS_AVAILABLE:
        return None
    try:
        name = copol_utils.smiles_to_name(smiles)
        return name.strip() if name and name.strip() else None
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Monomer universe                                                            #
# --------------------------------------------------------------------------- #


def collect_monomers_from_csv(csv_path: Path) -> Dict[str, Optional[str]]:
    """Return ``{smiles: best_name}`` from the curated reaction table.

    Best name = the longest non-empty value seen for that SMILES across
    ``monomer{1,2}_name``; longer strings tend to be properly capitalised
    full names ("N-Phenylmaleimide") rather than abbreviations.
    """
    df = pd.read_csv(csv_path)
    if "reaction_id" in df.columns:
        df = df[df["reaction_id"].notna()].drop_duplicates(subset="reaction_id")

    candidates: Dict[str, List[str]] = {}
    for smiles_col, name_col in (
        ("monomer1_smiles", "monomer1_name"),
        ("monomer2_smiles", "monomer2_name"),
    ):
        if smiles_col not in df.columns:
            continue
        for sm, nm in zip(df[smiles_col], df.get(name_col, [None] * len(df))):
            if not _is_meaningful(sm):
                continue
            sm = sm.strip()
            candidates.setdefault(sm, [])
            if _is_meaningful(nm):
                candidates[sm].append(nm.strip())

    return {sm: (max(names, key=len) if names else None) for sm, names in candidates.items()}


def resolve_display_name(smiles: str, csv_name: Optional[str]) -> str:
    """CSV name → copolextractor IUPAC → SMILES."""
    if csv_name:
        return csv_name
    name = iupac_name(smiles)
    if name:
        return name
    return smiles


def build_monomer_payload(
    smiles: str,
    name: str,
    feature_file: Optional[Path],
) -> Tuple[Dict, bool]:
    """Return ``(payload, has_features)`` ready to feed to ``nomad-polymerization``."""
    if feature_file is not None and feature_file.exists():
        try:
            data = json.loads(feature_file.read_text(encoding="utf-8"))
            has_features = True
        except Exception as e:
            print(f"  warning: could not parse {feature_file.name} ({e}); falling back to stub")
            data = {}
            has_features = False
    else:
        data = {}
        has_features = False

    data["smiles"] = smiles
    if not _is_meaningful(data.get("name")):
        data["name"] = name
    return data, has_features


# --------------------------------------------------------------------------- #
# Archive conversion                                                          #
# --------------------------------------------------------------------------- #


def _run_archive_cli(json_path: Path) -> Tuple[bool, str]:
    """Invoke ``nomad-polymerization archive --same-dir`` on `json_path`."""
    cmd = [NOMAD_CLI, "archive", str(json_path), "--same-dir"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except FileNotFoundError:
        return False, f"'{NOMAD_CLI}' executable not found on PATH"
    if result.returncode != 0:
        return False, (
            (result.stderr or result.stdout).strip().splitlines()[-1]
            if (result.stderr or result.stdout)
            else f"exit {result.returncode}"
        )
    return True, ""


def write_and_archive_monomer(
    payload: Dict,
    archive_stem: str,
    target_dir: Path,
) -> bool:
    """Write `payload` to ``target_dir`` and archive it via the CLI."""
    target_dir.mkdir(parents=True, exist_ok=True)

    # Resolve any IUPAC collisions deterministically with a numeric suffix.
    stem = archive_stem
    counter = 0
    while (target_dir / f"{stem}.json").exists() or (target_dir / f"{stem}.archive.json").exists():
        counter += 1
        stem = f"{archive_stem}_{counter}"
    json_path = target_dir / f"{stem}.json"
    archive_stem = stem

    try:
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        print(f"  error writing {json_path.name}: {e}")
        return False

    ok, err = _run_archive_cli(json_path)
    # NB: `with_suffix("")` is unsafe for names containing dots — pathlib
    # treats whatever follows the last dot as the suffix. Build the produced
    # archive path by concatenating the stem so names like
    # ``monomer_..._5.5_undecane.json`` survive.
    produced = json_path.parent / f"{json_path.stem}.archive.json"
    if ok and produced.exists():
        # The CLI writes <stem>.archive.json next to the input; clean up the
        # source JSON now that we have the archive.
        json_path.unlink(missing_ok=True)
        return True

    if err:
        print(f"  error archiving {json_path.name}: {err}")
    if json_path.exists():
        json_path.unlink()
    return False


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #


def smiles_already_archived(target_dir: Path) -> set:
    """SMILES strings already covered by ``monomer_*.archive.json`` in `target_dir`.

    We skip by SMILES (not by filename) because two distinct SMILES can
    legitimately resolve to the same display name; checking filename would
    silently drop the second one.
    """
    covered: set = set()
    if not target_dir.exists():
        return covered
    for archive in target_dir.glob("monomer_*.archive.json"):
        try:
            data = json.loads(archive.read_text(encoding="utf-8"))
        except Exception:
            continue
        sm = (data.get("data") or {}).get("smiles")
        if sm:
            covered.add(sm)
    return covered


def archive_monomer_universe(
    csv_path: Path,
    source_dir: Path,
    target_dir: Path,
    skip_existing: bool,
) -> None:
    print(f"Reading monomer universe from {csv_path}")
    universe = collect_monomers_from_csv(csv_path)
    print(f"  {len(universe):,} distinct monomer SMILES referenced by the dataset")

    print(f"Feature source dir: {source_dir}")
    print(f"Target dir:         {target_dir}")
    already_covered = smiles_already_archived(target_dir) if skip_existing else set()
    if already_covered:
        print(f"  {len(already_covered):,} SMILES already archived; will skip")
    print()

    written = skipped = failed = 0
    rich = stub = 0
    stubbed_smiles: List[str] = []
    failed_smiles: List[str] = []

    for smiles, csv_name in sorted(universe.items()):
        if smiles in already_covered:
            skipped += 1
            continue
        display_name = resolve_display_name(smiles, csv_name)
        archive_stem = f"monomer_{sanitize_filename(display_name)}"

        feature_file = source_dir / f"{smiles_md5(smiles)}.json"
        payload, has_features = build_monomer_payload(
            smiles, display_name, feature_file if feature_file.exists() else None
        )
        print(f"  {display_name[:60]:60s}  [{'rich' if has_features else 'stub'}]")

        if write_and_archive_monomer(payload, archive_stem, target_dir):
            written += 1
            if has_features:
                rich += 1
            else:
                stub += 1
                stubbed_smiles.append(smiles)
        else:
            failed += 1
            failed_smiles.append(smiles)

    print()
    print("=" * 70)
    print(
        f"Coverage: {written + skipped}/{len(universe)} archived  "
        f"(rich={rich}, stub={stub}, skipped-existing={skipped}, failed={failed})"
    )
    if stubbed_smiles:
        print(
            f"\nStub-only entries ({len(stubbed_smiles)} monomers with no precomputed "
            "feature file); rerun copol_prediction/monomer_feature_calculation.py "
            "to backfill quantum descriptors:"
        )
        for sm in stubbed_smiles:
            print(f"  {sm}")
    if failed_smiles:
        print(f"\nFailed to archive ({len(failed_smiles)}):")
        for sm in failed_smiles:
            print(f"  {sm}")


# --------------------------------------------------------------------------- #
# --fix workflow (manual single-file override)                                #
# --------------------------------------------------------------------------- #


def convert_with_custom_name(
    source_file: Path, target_dir: Path, custom_name: str, smiles: Optional[str]
) -> bool:
    """One-off: convert `source_file` to an archive with a user-supplied name."""
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        data = json.loads(source_file.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    data["name"] = custom_name
    if smiles:
        data["smiles"] = smiles
    if "smiles" not in data:
        data["smiles"] = source_file.stem

    archive_stem = f"monomer_{sanitize_filename(custom_name)}"
    ok = write_and_archive_monomer(data, archive_stem, target_dir)
    if ok:
        log_path = target_dir / FIX_LOG_FILENAME
        log = json.loads(log_path.read_text()) if log_path.exists() else {}
        log[source_file.name] = {
            "custom_name": custom_name,
            "archive_file": f"{archive_stem}.archive.json",
            "smiles": data.get("smiles"),
        }
        log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False))
    return ok


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build NOMAD monomer archives for every monomer in the dataset."
    )
    parser.add_argument(
        "--reactions-csv",
        type=Path,
        default=DEFAULT_REACTIONS_CSV,
        help=f"Curated reaction table that defines the monomer universe "
        f"(default: {DEFAULT_REACTIONS_CSV.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--source",
        "-s",
        type=Path,
        default=DEFAULT_FEATURE_SOURCE_DIR,
        help=f"Precomputed monomer feature JSONs, keyed by md5(SMILES) "
        f"(default: {DEFAULT_FEATURE_SOURCE_DIR.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_TARGET_DIR,
        help=f"Where to write the monomer archives (default: {DEFAULT_TARGET_DIR})",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Re-archive monomers even if a target archive already exists.",
    )
    parser.add_argument(
        "--fix",
        type=str,
        nargs=2,
        metavar=("FILE", "NAME"),
        help="One-off: re-archive a single source file under a custom name.",
    )
    parser.add_argument(
        "--smiles",
        type=str,
        help="SMILES for --fix when the source file doesn't carry one.",
    )
    args = parser.parse_args()

    if args.fix:
        source_file = args.source / args.fix[0]
        if not source_file.exists():
            print(f"ERROR: file not found: {source_file}")
            sys.exit(1)
        ok = convert_with_custom_name(source_file, args.output, args.fix[1], args.smiles)
        sys.exit(0 if ok else 1)

    if not args.reactions_csv.exists():
        print(f"ERROR: reactions CSV not found: {args.reactions_csv}")
        sys.exit(1)
    if not args.source.exists():
        print(f"ERROR: source directory not found: {args.source}")
        sys.exit(1)

    archive_monomer_universe(
        args.reactions_csv,
        args.source,
        args.output,
        skip_existing=not args.no_skip,
    )


if __name__ == "__main__":
    main()
