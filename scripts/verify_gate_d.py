"""Command-line entry point for Gate D raw-output validation."""

import sys
from pathlib import Path


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.analysis import main as analysis_main

    sys.argv.extend(["--gate-d-only"])
    analysis_main()


if __name__ == "__main__":
    main()
