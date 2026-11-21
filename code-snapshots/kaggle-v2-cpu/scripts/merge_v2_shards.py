"""Validate and merge completed V2 dataset-shard outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.v2_execution import load_execution_context
from src.v2_sharding import load_shard_plan, merge_v2_shards


def _parse_shard(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("shard must use SHARD_ID=OUTPUT_DIRECTORY")
    shard_id, output = value.split("=", 1)
    if not shard_id or not output:
        raise argparse.ArgumentTypeError("shard must use SHARD_ID=OUTPUT_DIRECTORY")
    return shard_id, Path(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--shard",
        action="append",
        type=_parse_shard,
        required=True,
        metavar="SHARD_ID=OUTPUT_DIRECTORY",
        help="completed output directory for one plan shard; repeat once per shard",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    plan_path = args.plan if args.plan.is_absolute() else root / args.plan
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    context = load_execution_context(root, output_dir=output_dir)
    plan = load_shard_plan(plan_path)
    shard_outputs = {
        shard_id: path if path.is_absolute() else root / path for shard_id, path in args.shard
    }
    manifest = merge_v2_shards(
        context,
        plan,
        shard_outputs,
        output_dir,
        plan_sha256=_sha256_file(plan_path),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
