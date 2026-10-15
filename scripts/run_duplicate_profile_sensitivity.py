"""Run the frozen predictor-profile grouped-CV sensitivity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    from src.duplicate_profile_sensitivity import prepare_preflight, run_sensitivity

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/duplicate-profile-sensitivity"),
        help="separate ignored output directory for the sensitivity run",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="verify inputs and save all grouped splits without fitting models",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume an interrupted run from completed dataset checkpoints",
    )
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    result = (
        prepare_preflight(ROOT, output_dir)
        if args.preflight_only
        else run_sensitivity(ROOT, output_dir, resume=args.resume)
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
