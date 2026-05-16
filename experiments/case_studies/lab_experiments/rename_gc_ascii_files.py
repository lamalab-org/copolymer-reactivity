from __future__ import annotations

import argparse
import datetime as dt
import os
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ParsedInfo:
    path: Path
    acquired: dt.datetime
    sample_name: str
    folder_mwh: str  # e.g. "017"


ACQUIRED_RE = re.compile(r"^Acquired\t(.+?)\s*$")
SAMPLE_NAME_RE = re.compile(r"^Sample Name\t(.+?)\s*$")
FOLDER_MWH_RE = re.compile(r"^MWH_(\d+)_ASCII$", re.IGNORECASE)
SAMPLE_PREFIX_RE = re.compile(r"^MWH_0*\d+_(.+)$", re.IGNORECASE)


def _parse_acquired(value: str) -> dt.datetime:
    # Example: "23.02.2026 09:08:14"
    value = value.strip()
    return dt.datetime.strptime(value, "%d.%m.%Y %H:%M:%S")


def _sanitize_sample_name(sample_name: str) -> str:
    s = sample_name.strip()
    m = SAMPLE_PREFIX_RE.match(s)
    if m:
        s = m.group(1).strip()
    # Keep it filename-safe and stable across OSes
    s = re.sub(r"\s+", "-", s)
    s = s.replace(os.sep, "-")
    # Remove characters that are annoying in filenames
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "unknown"


def _extract_folder_mwh(folder: Path) -> str:
    m = FOLDER_MWH_RE.match(folder.name)
    if not m:
        raise ValueError(f"Unexpected GC folder name: {folder.name!r}")
    digits = m.group(1)
    # Keep original digit width if possible; otherwise pad to 3 for consistency.
    return digits.zfill(3) if len(digits) < 3 else digits


def _parse_file(path: Path, folder_mwh: str) -> ParsedInfo:
    acquired: dt.datetime | None = None
    sample_name: str | None = None

    # The header is near the top; we don't need to read the whole file.
    # Use a permissive encoding to avoid choking on odd chars.
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for _ in range(400):
            line = f.readline()
            if not line:
                break
            line = line.rstrip("\n")

            if acquired is None:
                m = ACQUIRED_RE.match(line)
                if m:
                    acquired = _parse_acquired(m.group(1))
                    continue

            if sample_name is None:
                m = SAMPLE_NAME_RE.match(line)
                if m:
                    sample_name = m.group(1).strip()
                    continue

            if acquired is not None and sample_name is not None:
                break

    if acquired is None or sample_name is None:
        raise ValueError(
            f"Could not parse required fields in {path}: "
            f"acquired={acquired!r}, sample_name={sample_name!r}"
        )

    return ParsedInfo(path=path, acquired=acquired, sample_name=sample_name, folder_mwh=folder_mwh)


def _plan_renames(gc_folder: Path) -> list[tuple[Path, Path, ParsedInfo]]:
    folder_mwh = _extract_folder_mwh(gc_folder)

    # Support re-running the script after an earlier rename.
    files = sorted(set(gc_folder.glob("ASCII_*.txt")) | set(gc_folder.glob("MWH_*_gc_*.txt")))
    infos: list[ParsedInfo] = []
    for p in files:
        infos.append(_parse_file(p, folder_mwh=folder_mwh))

    # Group by sample name and only index duplicates within the same sample.
    groups: dict[str, list[ParsedInfo]] = {}
    for info in infos:
        key = _sanitize_sample_name(info.sample_name)
        groups.setdefault(key, []).append(info)

    plan: list[tuple[Path, Path, ParsedInfo]] = []
    for sample_key, group_infos in sorted(groups.items(), key=lambda kv: kv[0].lower()):
        # Sort oldest -> newest, and stabilize ties with the original filename.
        group_sorted = sorted(group_infos, key=lambda x: (x.acquired, x.path.name))
        n = len(group_sorted)

        if n == 1:
            info = group_sorted[0]
            new_name = f"MWH_{folder_mwh}_gc_{sample_key}.txt"
            plan.append((info.path, gc_folder / new_name, info))
            continue

        width = max(2, len(str(n)))
        for i, info in enumerate(group_sorted, start=1):
            suffix = f"{i:0{width}d}"
            new_name = f"MWH_{folder_mwh}_gc_{sample_key}_{suffix}.txt"
            plan.append((info.path, gc_folder / new_name, info))

    return plan


def _validate_no_collisions(plan: list[tuple[Path, Path, ParsedInfo]]) -> None:
    targets = [dst for _, dst, _ in plan]
    dupes = {p for p in targets if targets.count(p) > 1}
    if dupes:
        raise ValueError(f"Target collisions detected: {sorted({d.name for d in dupes})}")

    # Also ensure we don't overwrite an existing unrelated file
    for src, dst, _ in plan:
        if dst.exists() and dst.resolve() != src.resolve():
            raise FileExistsError(f"Target already exists: {dst}")


def _apply_renames(plan: list[tuple[Path, Path, ParsedInfo]]) -> None:
    # Two-phase rename to avoid edge collisions: rename to temp first, then final.
    tmp_pairs: list[tuple[Path, Path]] = []
    for src, dst, _ in plan:
        tmp = src.with_name(f".__renaming__{src.name}")
        if tmp.exists():
            raise FileExistsError(f"Temporary file already exists: {tmp}")
        tmp_pairs.append((src, tmp))

    for src, tmp in tmp_pairs:
        src.rename(tmp)

    tmp_to_final = []
    for (_, dst, _), (_, tmp) in zip(plan, tmp_pairs, strict=True):
        tmp_to_final.append((tmp, dst))

    for tmp, dst in tmp_to_final:
        tmp.rename(dst)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rename LabSolutions GC ASCII files based on Sample Name and Acquired timestamp."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("experiments/case_studies/lab_experiments/Experimental_data/GC_data"),
        help="Root directory containing MWH_*_ASCII folders.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually rename files (default is dry-run).",
    )
    args = parser.parse_args()

    root: Path = args.root
    if not root.exists():
        raise FileNotFoundError(root)

    folders = sorted([p for p in root.iterdir() if p.is_dir() and FOLDER_MWH_RE.match(p.name)])
    if not folders:
        raise FileNotFoundError(f"No MWH_*_ASCII folders found under: {root}")

    any_changes = False
    for folder in folders:
        plan = _plan_renames(folder)
        _validate_no_collisions(plan)

        print(f"\n== {folder} ==")
        for src, dst, info in plan:
            any_changes = any_changes or (src.name != dst.name)
            print(f"{info.acquired.isoformat(sep=' ')}\t{src.name}  ->  {dst.name}")

        if args.apply:
            _apply_renames(plan)

    if not any_changes:
        print("\nNo changes needed (files already match planned names).")
    elif not args.apply:
        print("\nDry-run only. Re-run with --apply to perform renames.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
