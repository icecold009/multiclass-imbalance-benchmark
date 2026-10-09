"""Build the current-commit source release and anonymous TMLR supplement."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from src.release_artifacts import build_release_artifacts

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "output",
        help="artifact output root inside the repository (defaults to ignored output)",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="build a clearly uncommitted preview; Gate G will reject its packages",
    )
    parser.add_argument(
        "--stage-only",
        action="store_true",
        help="stage the anonymous TeX source for compilation without creating release ZIPs",
    )
    args = parser.parse_args()
    output_root = args.output_root if args.output_root.is_absolute() else root / args.output_root
    payload = build_release_artifacts(
        root,
        output_root,
        allow_dirty=args.allow_dirty,
        stage_only=args.stage_only,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
