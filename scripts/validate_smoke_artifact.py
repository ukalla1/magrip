"""Validate MaGRIP smoke-test artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from magrip.validation import validate_smoke_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Path to summary.json or manifest.json.")
    parser.add_argument("--expected-topology", choices=("dense", "gated", "branched", "moe"))
    parser.add_argument("--strict", action="store_true", help="Treat warnings as failures.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_smoke_summary(args.path, expected_topology=args.expected_topology)
    for warning in result.warnings:
        print(f"WARNING: {warning}")
    for error in result.errors:
        print(f"ERROR: {error}")

    if result.ok and (not args.strict or not result.warnings):
        print(f"OK: {args.path}")
        return
    raise SystemExit(1)


if __name__ == "__main__":
    main()
