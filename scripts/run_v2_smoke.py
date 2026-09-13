"""Run the synthetic V2 engineering smoke harness."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.v2_smoke import run_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/v2-smoke"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    print(run_smoke(root, output_dir))


if __name__ == "__main__":
    main()
