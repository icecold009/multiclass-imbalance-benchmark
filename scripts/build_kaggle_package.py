"""Build a reproducible, data-free package for the private Kaggle CPU notebook."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path


PACKAGE_FILES = (
    "README.md",
    "requirements-v2.txt",
    "requirements-v2.lock",
    "pyproject.toml",
    "config/v2.yaml",
    "data/dataset_registry_v2.csv",
    "data/dataset_feature_overrides_v2.csv",
    "docs/protocol-v2.md",
    "docs/analysis-plan-v2.md",
    "docs/decision-register-v2.md",
    "kaggle/README.md",
    "kaggle/v2_cpu_shard.ipynb",
    "scripts/run_v2_full.py",
    "scripts/run_v2_analysis.py",
    "scripts/prepare_v2_shards.py",
    "scripts/merge_v2_shards.py",
    "scripts/record_kaggle_environment.py",
    "src/__init__.py",
    "src/v2_analysis.py",
    "src/v2_engineering.py",
    "src/v2_execution.py",
    "src/v2_pipeline.py",
    "src/v2_registry.py",
    "src/v2_sharding.py",
    "src/v2_tuning.py",
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_commit(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    return value if completed.returncode == 0 else "not-recorded"


def build(root: Path, output: Path) -> dict[str, object]:
    root = root.resolve()
    output = output.resolve()
    entries: list[dict[str, object]] = []
    payloads: list[tuple[str, bytes]] = []
    for relative in PACKAGE_FILES:
        source = root / relative
        if not source.is_file():
            raise FileNotFoundError(f"required package file is missing: {relative}")
        data = source.read_bytes()
        archive_name = f"multiclass-imbalance-benchmark/{relative}"
        payloads.append((archive_name, data))
        entries.append(
            {
                "path": relative,
                "size": len(data),
                "sha256": _sha256_bytes(data),
            }
        )

    manifest: dict[str, object] = {
        "schema_version": "kaggle-v2-package-v1",
        "purpose": "data-free private Kaggle CPU package",
        "source_commit": _source_commit(root),
        "files": entries,
        "excluded": ["data/raw", "data/processed", "results", "output", "paper", ".git"],
    }
    manifest_data = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    payloads.append(("multiclass-imbalance-benchmark/kaggle/package-manifest.json", manifest_data))

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for archive_name, data in payloads:
            info = zipfile.ZipInfo(archive_name, date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/kaggle/v2-kaggle-cpu-package.zip"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    manifest = build(root, output)
    print(json.dumps({"output": str(output), **manifest}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
