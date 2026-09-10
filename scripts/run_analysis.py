"""Command-line entry point for Gate D and the preregistered analysis."""

import sys
from pathlib import Path


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.analysis import main as analysis_main

    analysis_main()


if __name__ == "__main__":
    main()
