"""Command-line entry point for Gate E robustness validation."""

import sys
from pathlib import Path


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.robustness import main as robustness_main

    robustness_main()


if __name__ == "__main__":
    main()
