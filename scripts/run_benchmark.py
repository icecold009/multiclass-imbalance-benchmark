"""Command-line entry point for the locked full benchmark."""

import sys
from pathlib import Path


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.benchmark import main as benchmark_main

    benchmark_main()


if __name__ == "__main__":
    main()
