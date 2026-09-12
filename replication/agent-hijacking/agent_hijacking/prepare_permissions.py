"""Freeze the first paired permission unit. No Wikipedia fetch. No model."""

from __future__ import annotations

import argparse
from pathlib import Path

from .permissions import write_unit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=4100)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit("Use a new empty output directory; frozen inputs are not overwritten")
    unit = write_unit(out, args.seed)
    print({"unit": unit["unit_id"], "pairs": [row["id"] for row in unit["pairs"]], "out": str(out)})


if __name__ == "__main__":
    main()
