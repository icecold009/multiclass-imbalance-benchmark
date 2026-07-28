"""Stage 0 validation and pilot utilities.

This module deliberately keeps data validation separate from model fitting. It
can be used to audit candidate CSV files before adding them to the committed
dataset registry. Full model execution will be added only after the registry
and applicability decisions are locked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it all in memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _feature_type(frame: pd.DataFrame, target: str) -> str:
    features = frame.drop(columns=[target])
    numeric = features.select_dtypes(include="number").shape[1]
    categorical = features.shape[1] - numeric
    if numeric and categorical:
        return "mixed"
    if numeric:
        return "numeric"
    return "categorical"


def audit_csv(path: Path, target: str) -> dict[str, Any]:
    """Audit a candidate CSV and return JSON-serialisable registry evidence."""

    if not path.is_file():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path)
    if target not in frame.columns:
        raise ValueError(f"Target column {target!r} is not present in {path}")

    labels = frame[target].dropna()
    counts = Counter(labels.tolist())
    ordered_counts = dict(sorted(counts.items(), key=lambda item: str(item[0])))
    n_classes = len(counts)
    n_rows = len(frame)
    min_count = min(counts.values(), default=0)
    max_count = max(counts.values(), default=0)

    evidence: dict[str, Any] = {
        "dataset_id": path.stem,
        "display_name": path.stem,
        "source_version": "local-unverified",
        "raw_sha256": sha256_file(path),
        "task_type": "single-label-classification",
        "target_column": target,
        "n_rows": n_rows,
        "n_classes": n_classes,
        "class_counts": ordered_counts,
        "ir_majority_minority": (
            max_count / min_count if min_count else None
        ),
        "n_min_class": min_count,
        "d_raw": frame.shape[1] - 1,
        "feature_type": _feature_type(frame, target),
        "has_missing": bool(frame.isna().any().any()),
        "has_groups": "unknown",
        "has_time_order": "unknown",
        "has_duplicates": bool(frame.duplicated().any()),
        "leakage_status": "not-audited",
        "eligible": False,
        "exclusion_reason": "registry metadata and split audit required",
        "applicable_families": [],
    }
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a Stage 0 CSV candidate")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    evidence = audit_csv(args.csv_path, args.target)
    rendered = json.dumps(evidence, indent=2, default=str)
    if args.json_out:
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
