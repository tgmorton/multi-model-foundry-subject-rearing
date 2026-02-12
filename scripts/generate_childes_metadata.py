#!/usr/bin/env python
"""
Generate CHILDES age metadata from .train file boundary markers.

Scans a CHILDES .train file for file boundary markers of the form:
    = = = childes/CORPUS_PATH/FILENAME.cha = = =

Resolves child age in months via three tiers:
  1. Filename-encoded age (e.g., 010800.cha = 1yr 8mo = 20 months)
  2. Directory-level age (e.g., NewmanRatner/18/ = 18 months)
  3. Static corpus lookup table (documented age ranges)

Outputs a JSON file mapping each CHILDES transcript path to age info.

Usage
-----
    python scripts/generate_childes_metadata.py data/raw/train_90M/childes.train
    python scripts/generate_childes_metadata.py data/raw/test_10M/childes.test
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_BOUNDARY_PATTERN = re.compile(r"^= = = childes/(.+)\.cha = = =$")

# Tier 3: Static corpus age lookup (midpoint of documented range, in months)
# Sources: CHILDES database documentation, corpus README files
CORPUS_AGE_LOOKUP = {
    # EllisWeismer — number in path is age in months
    "EllisWeismer/TD/30": 30, "EllisWeismer/LT/30": 30,
    "EllisWeismer/TD/42": 42, "EllisWeismer/LT/42": 42,
    "EllisWeismer/TD/54": 54, "EllisWeismer/LT/54": 54,
    "EllisWeismer/TD/66": 66,
    # Bates — number is age in months
    "Bates/Free20": 20, "Bates/Free28": 28,
    "Bates/Snack28": 28, "Bates/Story28": 28,
    # Gillam — number is age in YEARS
    "Gillam/TD/5": 66, "Gillam/TD/6": 78, "Gillam/TD/7": 90,
    "Gillam/TD/8": 102, "Gillam/TD/9": 114, "Gillam/TD/10": 126,
    "Gillam/TD/11": 138,
    "Gillam/SLI/5": 66, "Gillam/SLI/6": 78, "Gillam/SLI/7": 90,
    "Gillam/SLI/8": 102, "Gillam/SLI/9": 114, "Gillam/SLI/10": 126,
    "Gillam/SLI/11": 138,
    # VanHouten — named by age group
    "VanHouten/Twos": 30, "VanHouten/Threes": 42,
    # Nippold — school age (7-15 years)
    "Nippold": 120,
    # HSLLD — Home-School Study, visits at specific ages
    "HSLLD/HV1": 36, "HSLLD/HV2": 48, "HSLLD/HV3": 60,
    "HSLLD/HV5": 72, "HSLLD/HV7": 84,
    # Hall — preschool (4 years)
    "Hall/BlackPro": 48, "Hall/WhitePro": 48,
    "Hall/BlackWork": 48, "Hall/WhiteWork": 48,
    # Conti-Ramsden
    "Conti1": 54, "Conti4": 66,
    # Edinburgh — 22-26 months
    "Edinburgh": 24,
    # Gelman studies
    "Gelman/2014": 48, "Gelman/2016": 48,
    "Gelman/2004": 30, "Gelman/1998": 30,
    # Warren — 18-36 months
    "Warren": 27,
    # Rollins — 9-24 months
    "Rollins": 16,
    # Valian — 22-34 months
    "Valian": 28,
    # Gleason — 2-5 years
    "Gleason/Mother": 42, "Gleason/Father": 42, "Gleason/Dinner": 42,
    # TD collection — mixed ages
    "CHILDES_NA/TD": 36,
    # Tardif — ~20 months
    "Tardif": 20,
    # Bohannon
    "Bohannon/Nat": 30, "Bohannon/Bax": 30,
    # ParentChild — 3-6 years
    "ParentChild": 54,
    # Narrative — 5-9 years
    "Narrative/SLI": 84, "Narrative/TD": 84,
    # Higginson — 20-22 months
    "Higginson": 21,
    # Morisset — 13-36 months
    "Morisset": 24,
    # MacWhinney — 24-60 months
    "MacWhinney": 42,
    # Cameron AAE/SAE — 4-5 years
    "Cameron/AAE": 54, "Cameron/SAE": 54,
    # DELV — 4-9 years
    "DELV": 78, "DSLT": 72,
    # Howe — 24-48 months
    "Howe": 36,
    # Tommerdahl — 5-10 years
    "Tommerdahl": 90,
    # Bliss — ~5 years
    "Bliss": 60,
    # Braunwald — 12-60 months
    "Braunwald": 36,
    # Twins — 30-36 months
    "Twins": 33,
    # Nadig — ~4 years
    "Nadig": 48,
    # Garvey — 3-5 years
    "Garvey": 48,
    # Barriere — AAE 3-5 years
    "Barriere": 48,
    # POLER — 3-8 years
    "POLER": 66,
    # Sawyer — 5 years
    "Sawyer": 60,
    # Sprott — 24-36 months
    "Sprott": 30,
    # OdiaMAIN — 5-6 years
    "OdiaMAIN": 66,
    # KellyQuigley — 7-8 years
    "KellyQuigley": 90,
    # McMillan
    "McMillan": 30,
    # Nuffield — UK SLI corpus (4-7 years)
    "Nuffield": 66,
    # Gathercole — bilingual study (7-9 years)
    "Gathercole": 96,
    # Evans — school age (~8 years)
    "Evans": 96,
    # Nicolopoulou — preschool narrative (3-5 years)
    "Nicolopoulou": 48,
    # EllisWeismer LT/66 — late talkers at 66 months
    "EllisWeismer/LT/66": 66,
    # Bernstein — ~4 years
    "Bernstein": 48,
    # Edwards — ~4 years
    "Edwards": 48,
    # NewmanRatner — longitudinal, directory = age in months
    "NewmanRatner/07": 7, "NewmanRatner/10": 10, "NewmanRatner/11": 11,
    "NewmanRatner/13": 13, "NewmanRatner/18": 18, "NewmanRatner/21": 21,
    "NewmanRatner/24": 24, "NewmanRatner/27": 27,
    # Brent — 9-15 months
    "Brent": 10,
    # Forrester — 12-36 months
    "Forrester": 12,
    # Gathburn — ~24 months
    "Gathburn": 24,
    # Korman — 9-15 months
    "Korman": 12,
    # Fletcher — number is age in years
    "Fletcher/3": 36, "Fletcher/5": 60, "Fletcher/7": 84,
    # VanKleeck — 3-5 years
    "VanKleeck": 42,
    # Belfast — 2-4 years
    "Belfast": 36,
}


def resolve_age(full_path: str) -> dict:
    """
    Resolve child age from a CHILDES transcript path.

    Args:
        full_path: Path relative to childes/, e.g.
            "CHILDES_NA/Brown/Eve/010800" (without .cha)

    Returns:
        Dict with age_months, age_source, child, corpus keys.
    """
    parts = full_path.split("/")
    filename = parts[-1]
    corpus_path = "/".join(parts[:-1])

    # Extract child and corpus names from path
    # Typical: CHILDES_NA/Brown/Eve/... → corpus=Brown, child=Eve
    child = parts[-2] if len(parts) >= 3 else None
    corpus = parts[-3] if len(parts) >= 4 else (parts[-2] if len(parts) >= 3 else None)

    result = {
        "path": full_path,
        "child": child,
        "corpus": corpus,
        "age_months": None,
        "age_source": None,
    }

    # Tier 1: Filename-encoded age (YYMMDD pattern)
    m = re.match(r"^(\d{2})(\d{2})(\d{2})", filename)
    if m:
        years, months = int(m.group(1)), int(m.group(2))
        if years <= 6:
            result["age_months"] = years * 12 + months
            result["age_source"] = "filename"
            return result

    # Tier 2: Static corpus lookup (longest match wins)
    # Run this BEFORE directory-level parsing, since the lookup table
    # has verified ages and avoids misinterpreting directory numbers
    # (e.g. Fletcher/3 = 3 years, not 3 months).
    best_match = None
    best_len = 0
    for pattern, age in CORPUS_AGE_LOOKUP.items():
        if pattern in full_path and len(pattern) > best_len:
            best_match = age
            best_len = len(pattern)

    if best_match is not None:
        result["age_months"] = best_match
        result["age_source"] = "corpus_lookup"
        return result

    # Tier 3: Directory-level age (e.g., EllisWeismer/TD/30/)
    # Only used as a fallback when no lookup match exists.
    # Require age >= 12 to avoid misinterpreting child/session IDs
    # as ages (e.g. NewmanRatner/07/ is a session, not 7 months).
    dir_m = re.match(r".*/(\d{1,2})$", corpus_path)
    if dir_m:
        dir_age = int(dir_m.group(1))
        if 12 <= dir_age <= 72:
            result["age_months"] = dir_age
            result["age_source"] = "directory"
            return result

    return result


def generate_metadata(train_path: Path) -> dict:
    """
    Scan a CHILDES .train/.test file and generate metadata for all transcripts.

    Returns:
        Dict mapping CHILDES transcript paths to age info.
    """
    metadata = {}

    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            m = _BOUNDARY_PATTERN.match(line.strip())
            if m:
                full_path = m.group(1)
                info = resolve_age(full_path)
                metadata[full_path] = info

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Generate CHILDES age metadata from .train file boundary markers."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to CHILDES .train or .test file",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output JSON path (default: same dir as input, childes_metadata.json)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    output = args.output or args.input.parent / "childes_metadata.json"

    print(f"Scanning {args.input}...", file=sys.stderr)
    metadata = generate_metadata(args.input)

    # Stats
    total = len(metadata)
    by_source = {}
    unresolved = 0
    for info in metadata.values():
        src = info["age_source"]
        if src:
            by_source[src] = by_source.get(src, 0) + 1
        else:
            unresolved += 1

    print(f"Transcripts found: {total}", file=sys.stderr)
    for src, count in sorted(by_source.items()):
        print(f"  {src}: {count}", file=sys.stderr)
    if unresolved:
        print(f"  unresolved: {unresolved}", file=sys.stderr)

    # Write output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metadata, indent=2))
    print(f"Written to {output}", file=sys.stderr)


if __name__ == "__main__":
    main()
