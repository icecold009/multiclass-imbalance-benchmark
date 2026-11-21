"""Plan or run gated V2 Bayesian analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.v2_analysis import load_analysis_context, run_v2_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-run-dir", type=Path, default=Path("results/v2-full"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/v2-analysis"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    full_run_dir = args.full_run_dir if args.full_run_dir.is_absolute() else root / args.full_run_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    context = load_analysis_context(root, output_dir=full_run_dir)
    summary = run_v2_analysis(context, output_dir=output_dir, dry_run=args.dry_run)
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
