"""Plan or run the gated V2 full benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.v2_execution import load_execution_context, run_v2_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/v2-full"))
    parser.add_argument("--dataset", dest="datasets", action="append", metavar="DATASET_ID")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report scope and blockers without starting any outcome-bearing work",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    context = load_execution_context(root, output_dir=output_dir)
    summary = run_v2_benchmark(
        context,
        selected_dataset_ids=tuple(args.datasets) if args.datasets else None,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
