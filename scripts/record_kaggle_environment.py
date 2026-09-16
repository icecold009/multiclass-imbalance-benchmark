"""Record a Kaggle Linux preflight without acquiring data or running the benchmark."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _memory_bytes() -> int | None:
    if hasattr(os, "sysconf"):
        try:
            return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
        except (ValueError, OSError):
            return None
    return None


def _command(*args: str) -> str:
    completed = subprocess.run(
        args,
        check=False,
        capture_output=True,
        text=True,
    )
    return (completed.stdout + completed.stderr).strip()


def _git_commit(root: Path) -> str:
    value = _command("git", "-C", str(root), "rev-parse", "HEAD")
    return value if value and "fatal:" not in value.lower() else "not-recorded"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/environment/v2"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    requirements = root / "requirements-v2.txt"
    config = root / "config/v2.yaml"
    registry = root / "data/dataset_registry_v2.csv"
    protocol = root / "docs/protocol-v2.md"
    required = (requirements, config, registry, protocol)
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("required package files are missing: " + ", ".join(missing))

    output_dir.mkdir(parents=True, exist_ok=True)
    memory = _memory_bytes()
    metadata: dict[str, Any] = {
        "schema_version": "v2-environment-v1",
        "recorded_at_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "purpose": "Kaggle CPU preflight; no data acquisition and no benchmark execution",
        "provider": "Kaggle",
        "os": platform.platform(),
        "python": sys.version,
        "cpu_count": os.cpu_count(),
        "memory_bytes": memory,
        "free_disk_bytes": shutil.disk_usage(root).free,
        "git_commit": _git_commit(root),
        "requirements_v2_sha256": _sha256(requirements),
        "config_v2_sha256": _sha256(config),
        "registry_v2_sha256": _sha256(registry),
        "protocol_v2_sha256": _sha256(protocol),
        "execution_policy": "cpu_only; workers_per_model=1; benchmark_started=false",
        "kaggle_kernel_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "unknown"),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "pip-freeze.txt").write_text(
        _command(sys.executable, "-m", "pip", "freeze") + "\n",
        encoding="utf-8",
    )
    (output_dir / "pip-check.txt").write_text(
        _command(sys.executable, "-m", "pip", "check") + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
