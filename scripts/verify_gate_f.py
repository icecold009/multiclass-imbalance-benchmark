"""Command-line entry point for Gate F clean-checkout reproducibility."""

import sys
from pathlib import Path


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.reproducibility import main as reproducibility_main

    reproducibility_main()


if __name__ == "__main__":
    main()
