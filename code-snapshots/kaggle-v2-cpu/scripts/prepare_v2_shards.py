"""Create a deterministic dataset-shard plan without starting the benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.v2_execution import load_execution_context
from src.v2_sharding import build_shard_plan, write_shard_plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=int, required=True, help="number of non-empty dataset shards")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/runs/v2-shard-plan.json"),
        help="path for the immutable shard plan",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output if args.output.is_absolute() else root / args.output
    context = load_execution_context(root)
    plan = build_shard_plan(context, args.shards)
    write_shard_plan(output, plan)
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
