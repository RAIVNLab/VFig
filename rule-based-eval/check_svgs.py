#!/usr/bin/env python3
"""Check SVGs in a directory for validity.

Usage:
    python check_svgs.py <dir> [--verbose]
"""

import sys
import argparse
from pathlib import Path
from xml.etree import ElementTree as ET


def check_svg(path):
    """Return (ok, error_msg). Checks XML well-formedness and SVG root."""
    try:
        tree = ET.parse(path)
    except ET.ParseError as e:
        return False, f"XML parse error: {e}"

    root = tree.getroot()
    tag = root.tag
    # Strip namespace if present: {http://www.w3.org/2000/svg}svg -> svg
    if "}" in tag:
        tag = tag.split("}", 1)[1]
    if tag != "svg":
        return False, f"Root element is <{tag}>, expected <svg>"

    return True, None


def main():
    parser = argparse.ArgumentParser(description="Validate SVG files in a directory.")
    parser.add_argument("dir", help="Directory containing .svg files")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print OK files too")
    args = parser.parse_args()

    svg_dir = Path(args.dir)
    if not svg_dir.exists():
        print(f"ERROR: {svg_dir} does not exist")
        sys.exit(1)

    files = sorted(svg_dir.rglob("*.svg"))
    if not files:
        print(f"No .svg files found in {svg_dir}")
        sys.exit(0)

    ok_count = bad_count = 0
    for f in files:
        ok, err = check_svg(f)
        if ok:
            ok_count += 1
            if args.verbose:
                print(f"  OK  {f.name}")
        else:
            bad_count += 1
            print(f"  BAD {f.name}: {err}")

    total = ok_count + bad_count
    print(f"\n{ok_count}/{total} valid  |  {bad_count} invalid")
    sys.exit(0 if bad_count == 0 else 1)


if __name__ == "__main__":
    main()
