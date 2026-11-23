"""Build the source release and anonymous TMLR supplement."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def default_output_root(root: Path, *, anonymous_only: bool) -> Path:
    """Keep V4 anonymous-only builds separate from existing release artifacts."""

    if anonymous_only:
        return root / "output" / "publication-v4-anonymous-evidence"
    return root / "output"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from src.release_artifacts import build_release_artifacts

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "artifact output root inside the repository (defaults to the existing output "
            "root, or an isolated V4 root for --anonymous-only)"
        ),
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
    parser.add_argument(
        "--anonymous-only",
        action="store_true",
        help="build only the anonymous supplement and use an isolated V4 output root by default",
    )
    args = parser.parse_args()
    output_root = args.output_root or default_output_root(
        root,
        anonymous_only=args.anonymous_only,
    )
    if not output_root.is_absolute():
        output_root = root / output_root
    payload = build_release_artifacts(
        root,
        output_root,
        allow_dirty=args.allow_dirty,
        stage_only=args.stage_only,
        anonymous_only=args.anonymous_only,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
