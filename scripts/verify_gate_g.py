"""Command-line entry point for Gate G technical release-snapshot validation."""

import sys
from pathlib import Path


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.submission import main as submission_main

    submission_main()


if __name__ == "__main__":
    main()
