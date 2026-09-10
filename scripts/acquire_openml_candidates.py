"""Acquire the fixed Stage 0 OpenML candidate pool.

The script downloads only the named public OpenML dataset versions listed in
``CANDIDATES`` and materializes their frames as CSV files under ``data/raw``.
It does not choose eligibility or inspect benchmark performance; those
decisions are recorded separately in the dataset registry.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from sklearn.datasets import fetch_openml


@dataclass(frozen=True)
class Candidate:
    dataset_id: str
    openml_data_id: int
    filename: str
    expected_target: str
    drop_columns: tuple[str, ...] = ()


CANDIDATES: tuple[Candidate, ...] = (
    Candidate("balance_scale", 11, "balance_scale.csv", "class"),
    Candidate("mfeat_factors", 12, "mfeat_factors.csv", "class"),
    Candidate("mfeat_fourier", 14, "mfeat_fourier.csv", "class"),
    Candidate("optdigits", 28, "optdigits.csv", "class"),
    Candidate("dermatology", 35, "dermatology.csv", "class"),
    Candidate("segment", 36, "segment.csv", "class"),
    Candidate("ecoli", 39, "ecoli.csv", "class"),
    Candidate("soybean", 42, "soybean.csv", "class"),
    Candidate("tae", 48, "tae.csv", "Class_attribute"),
    Candidate("iris", 61, "iris.csv", "class"),
    Candidate("zoo", 62, "zoo.csv", "type"),
    Candidate("wine", 187, "wine.csv", "class"),
    Candidate("satimage", 182, "satimage.csv", "class"),
    Candidate("hayes_roth", 329, "hayes_roth.csv", "class"),
    Candidate("cnae_9", 1468, "cnae_9.csv", "Class"),
    Candidate("seeds", 1499, "seeds.csv", "Class"),
    Candidate("splice", 46, "splice.csv", "Class", ("Instance_name",)),
    Candidate("wine_quality_red", 40691, "wine_quality_red.csv", "class"),
)


def acquire(output_dir: Path, selected: set[str] | None = None) -> list[dict[str, object]]:
    """Download selected candidates and return acquisition metadata."""

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for candidate in CANDIDATES:
        if selected is not None and candidate.dataset_id not in selected:
            continue
        dataset = fetch_openml(
            data_id=candidate.openml_data_id,
            as_frame=True,
            parser="auto",
        )
        frame = dataset.frame.copy()
        target = dataset.target.name
        if target != candidate.expected_target:
            raise ValueError(
                f"{candidate.dataset_id}: expected target {candidate.expected_target!r}, "
                f"received {target!r}"
            )
        present_drop_columns = [column for column in candidate.drop_columns if column in frame.columns]
        frame = frame.drop(columns=present_drop_columns)
        output_path = output_dir / candidate.filename
        frame.to_csv(output_path, index=False)
        records.append(
            {
                **asdict(candidate),
                "target": target,
                "source_name": dataset.details.get("name"),
                "source_version": dataset.details.get("version"),
                "source_url": dataset.details.get("url"),
                "source_license": dataset.details.get("licence"),
                "dropped_columns": present_drop_columns,
                "shape": list(frame.shape),
                "path": str(output_path),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--dataset-id", action="append", dest="dataset_ids")
    args = parser.parse_args()
    selected = set(args.dataset_ids) if args.dataset_ids else None
    unknown = selected.difference({candidate.dataset_id for candidate in CANDIDATES}) if selected else set()
    if unknown:
        parser.error(f"unknown dataset id(s): {', '.join(sorted(unknown))}")
    print(json.dumps(acquire(args.output_dir, selected), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
