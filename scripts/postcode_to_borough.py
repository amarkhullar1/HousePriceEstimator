#!/usr/bin/env python3
"""
Postcode → Borough mapper using ONSPD (Aug 2025) file at:
  data/onspd/ONSPD_AUG_2025_UK.csv

What this script does
- Builds an in-memory index from ONSPD postcodes (pcds) to London borough names by
  reading lad25cd (Local Authority District) and mapping London codes to names.
- Exposes:
  - a Python API: postcode_to_borough(postcode)
  - a CLI: map single or multiple postcodes to boroughs

Assumptions
- We treat "boroughs" as Greater London local authorities (32 boroughs + City of London).
- ONSPD provides LAD codes in column lad25cd. London LAD codes start with E090000** and City of London is E09000001.
- The ONSPD file includes only codes, so we embed a static map of London LAD codes to names.

Usage
- Map a single postcode:
    python scripts/postcode_to_borough.py "SW1A 1AA"

- Map many postcodes from a file (one postcode per line):
    python scripts/postcode_to_borough.py --infile postcodes.txt --outfile results.csv

- Custom ONSPD path:
    python scripts/postcode_to_borough.py --onspd data/onspd/ONSPD_AUG_2025_UK.csv "SW1A 1AA"

Output
- Prints CSV to stdout (or writes to --outfile) with columns: postcode, borough_name, borough_code
  where borough_code is lad25cd and borough_name is a London borough name (or blank if not in London).

"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, Optional

# Embedded mapping of London LAD codes (E09000001..E09000033) to borough names
# City of London is included. This list is stable but review if ONS changes codes.
LONDON_LAD_CODE_TO_NAME: Dict[str, str] = {
    "E09000001": "City of London",
    "E09000002": "Barking and Dagenham",
    "E09000003": "Barnet",
    "E09000004": "Bexley",
    "E09000005": "Brent",
    "E09000006": "Bromley",
    "E09000007": "Camden",
    "E09000008": "Croydon",
    "E09000009": "Ealing",
    "E09000010": "Enfield",
    "E09000011": "Greenwich",
    "E09000012": "Hackney",
    "E09000013": "Hammersmith and Fulham",
    "E09000014": "Haringey",
    "E09000015": "Harrow",
    "E09000016": "Havering",
    "E09000017": "Hillingdon",
    "E09000018": "Hounslow",
    "E09000019": "Islington",
    "E09000020": "Kensington and Chelsea",
    "E09000021": "Kingston upon Thames",
    "E09000022": "Lambeth",
    "E09000023": "Lewisham",
    "E09000024": "Merton",
    "E09000025": "Newham",
    "E09000026": "Redbridge",
    "E09000027": "Richmond upon Thames",
    "E09000028": "Southwark",
    "E09000029": "Sutton",
    "E09000030": "Tower Hamlets",
    "E09000031": "Waltham Forest",
    "E09000032": "Wandsworth",
    "E09000033": "Westminster",
}

DEFAULT_ONSPD_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "onspd",
    "ONSPD_AUG_2025_UK.csv",
)

# Columns we need from ONSPD (based on header in your file)
REQUIRED_COLUMNS = ["pcds", "lad25cd"]


def normalize_postcode(pc: str) -> str:
    """Normalize a postcode to ONSPD pcds style (uppercase, single space before last 3 chars).
    Examples:
        "sw1a1aa" -> "SW1A 1AA"
        "SW1A  1AA" -> "SW1A 1AA"
    """
    s = (pc or "").strip().upper().replace(" ", "")
    if len(s) < 5:
        return pc.strip().upper()
    return f"{s[:-3]} {s[-3:]}"


class PostcodeToBoroughIndex:
    """Indexes London postcodes (pcds) -> (lad25cd, borough name) for fast lookup."""

    def __init__(self, onspd_csv_path: str = DEFAULT_ONSPD_PATH) -> None:
        self.onspd_csv_path = onspd_csv_path
        self._pcds_to_lad: Dict[str, str] = {}
        self._built = False

    def build(self) -> None:
        if self._built:
            return
        if not os.path.exists(self.onspd_csv_path):
            raise FileNotFoundError(f"ONSPD file not found: {self.onspd_csv_path}")

        with open(self.onspd_csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            missing = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
            if missing:
                raise ValueError(
                    "ONSPD file is missing required columns: " + ", ".join(missing)
                )

            for row in reader:
                lad = row.get("lad25cd", "").strip().strip('"')
                # Only index Greater London postcodes (boroughs + City of London)
                if lad in LONDON_LAD_CODE_TO_NAME:
                    pcds = row.get("pcds", "").strip().strip('"')
                    if pcds:
                        self._pcds_to_lad[pcds] = lad
        self._built = True

    def postcode_to_borough(self, postcode: str) -> tuple[Optional[str], Optional[str]]:
        """Return (borough_name, borough_code) for a given postcode, or (None, None) if not in London.
        Performs normalization to match ONSPD pcds formatting.
        """
        if not self._built:
            self.build()
        pcds = normalize_postcode(postcode)
        lad = self._pcds_to_lad.get(pcds)
        if not lad:
            return None, None
        return LONDON_LAD_CODE_TO_NAME.get(lad), lad


def postcode_to_borough(postcode: str, onspd_csv_path: str = DEFAULT_ONSPD_PATH) -> tuple[Optional[str], Optional[str]]:
    idx = PostcodeToBoroughIndex(onspd_csv_path)
    return idx.postcode_to_borough(postcode)


def describe_onspd_header(onspd_csv_path: str = DEFAULT_ONSPD_PATH) -> str:
    """Return a short description of the key columns found in the provided ONSPD file."""
    if not os.path.exists(onspd_csv_path):
        return f"ONSPD file not found: {onspd_csv_path}"
    with open(onspd_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, [])
    # Provide a concise description focusing on columns relevant to borough mapping
    cols = {c.strip() for c in header}
    pieces = [
        "Detected ONSPD columns:",
        "- pcds: Postcode (spaced)",
        "- lad25cd: Local Authority District code (used to identify London boroughs)",
    ]
    extras = [
        ("ctry25cd", "Country code"),
        ("rgn25cd", "Region (ITL1)"),
        ("pcon24cd", "Westminster constituency code"),
        ("wd25cd", "Electoral ward code"),
        ("lat", "Latitude"),
        ("long", "Longitude"),
    ]
    for c, desc in extras:
        if c in cols:
            pieces.append(f"- {c}: {desc}")
    return "\n".join(pieces)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Map UK postcodes to London boroughs using ONSPD")
    parser.add_argument("postcodes", nargs="*", help="Postcodes to map (space-separated)")
    parser.add_argument("--onspd", default=DEFAULT_ONSPD_PATH, help="Path to ONSPD CSV file")
    parser.add_argument("--infile", help="Path to a text file with one postcode per line")
    parser.add_argument("--outfile", help="CSV output path (defaults to stdout)")
    parser.add_argument("--describe", action="store_true", help="Print a short description of key ONSPD columns")

    args = parser.parse_args(argv)

    if args.describe:
        print(describe_onspd_header(args.onspd))
        # Continue to mapping if postcodes are also provided

    inputs: list[str] = []
    if args.infile:
        if not os.path.exists(args.infile):
            print(f"Input file not found: {args.infile}", file=sys.stderr)
            return 2
        with open(args.infile, encoding="utf-8") as f:
            inputs.extend([line.strip() for line in f if line.strip()])
    if args.postcodes:
        inputs.extend(args.postcodes)

    if not inputs:
        if args.describe:
            return 0
        parser.print_help()
        return 1

    index = PostcodeToBoroughIndex(args.onspd)
    index.build()

    out_fh = open(args.outfile, "w", encoding="utf-8", newline="") if args.outfile else sys.stdout
    writer = csv.writer(out_fh)
    writer.writerow(["postcode", "borough_name", "borough_code"])  # lad25cd

    for pc in inputs:
        borough_name, borough_code = index.postcode_to_borough(pc)
        writer.writerow([normalize_postcode(pc), borough_name or "", borough_code or ""])

    if args.outfile:
        out_fh.close()
        print(f"Wrote results to {args.outfile}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
