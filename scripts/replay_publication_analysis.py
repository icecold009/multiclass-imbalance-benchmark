"""Replay packaged analysis from frozen aggregate records only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _from_root(root: Path, value: Path | None, default: str) -> Path:
    path = Path(default) if value is None else value
    return path if path.is_absolute() else root / path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from src.publication_evidence import replay_publication_analysis

    parser = argparse.ArgumentParser(
        description=(
            "Validate and replay the packaged V1 analysis from aggregate records. "
            "This does not run Gate D, Gate E, or model training."
        )
    )
    parser.add_argument("--run-dir", type=Path, help="aggregate-record directory")
    parser.add_argument("--analysis-dir", type=Path, help="frozen analysis directory")
    parser.add_argument("--output-dir", type=Path, help="separate replay output directory")
    args = parser.parse_args()
    receipt = replay_publication_analysis(
        root,
        run_dir=_from_root(root, args.run_dir, "results/full-run"),
        analysis_dir=_from_root(root, args.analysis_dir, "results/analysis"),
        output_dir=_from_root(root, args.output_dir, "results/analysis-portable-replay"),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
