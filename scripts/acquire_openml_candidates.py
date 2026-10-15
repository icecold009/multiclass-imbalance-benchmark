"""Acquire the fixed Stage 0 OpenML candidate pool.

The script downloads only the named public OpenML dataset versions listed in
``CANDIDATES`` and materializes their frames as CSV files under ``data/raw``.
It does not choose eligibility or inspect benchmark performance; those
decisions are recorded separately in the dataset registry.
"""

from __future__ import annotations

import argparse
import hashlib
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
    expected_source_version: int | None = None
    expected_sha256: str | None = None


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
    Candidate(
        "cmc",
        23,
        "cmc.csv",
        "Contraceptive_method_used",
        expected_source_version=1,
        expected_sha256="0a42de56db54f5e51830227dfc3efa2bee3120f2634dc623ed99dad0459ec5b3",
    ),
    Candidate(
        "glass",
        41,
        "glass.csv",
        "Type",
        drop_columns=("Id_number",),
        expected_source_version=1,
        expected_sha256="bb06702a7c96a667a0781150e3a70d5b3080a976cdccd7c7f242f78e23a7ad11",
    ),
    Candidate(
        "yeast",
        181,
        "yeast.csv",
        "class_protein_localization",
        drop_columns=("Sequence_Name",),
        expected_source_version=1,
        expected_sha256="743448d2ba31e10b538cf10ef03b03c6b04222ad6e32c4873ba0b07d252dcdc0",
    ),
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_local_materializations(
    output_dir: Path, selected: set[str] | None = None
) -> list[dict[str, str]]:
    """Verify recorded CMC, Glass, and Yeast CSV hashes without network access."""

    hash_pinned_ids = {
        candidate.dataset_id for candidate in CANDIDATES if candidate.expected_sha256 is not None
    }
    if selected is not None:
        unsupported = selected.difference(hash_pinned_ids)
        if unsupported:
            raise ValueError(
                "local hash verification is available only for hash-pinned datasets: "
                f"{', '.join(sorted(unsupported))}"
            )
    records: list[dict[str, str]] = []
    for candidate in CANDIDATES:
        if candidate.expected_sha256 is None:
            continue
        if selected is not None and candidate.dataset_id not in selected:
            continue
        path = output_dir / candidate.filename
        if not path.is_file():
            raise FileNotFoundError(f"{candidate.dataset_id}: materialized CSV is missing: {path}")
        actual_sha256 = _sha256(path.read_bytes())
        if actual_sha256 != candidate.expected_sha256:
            raise ValueError(
                f"{candidate.dataset_id}: CSV hash mismatch; expected "
                f"{candidate.expected_sha256}, got {actual_sha256}"
            )
        records.append(
            {
                "dataset_id": candidate.dataset_id,
                "path": str(path),
                "sha256": actual_sha256,
                "status": "verified_local_materialization",
            }
        )
    if not records:
        raise ValueError("no hash-pinned local materializations were selected for verification")
    return records


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
        source_version = dataset.details.get("version")
        if (
            candidate.expected_source_version is not None
            and str(source_version) != str(candidate.expected_source_version)
        ):
            raise ValueError(
                f"{candidate.dataset_id}: expected OpenML version "
                f"{candidate.expected_source_version}, received {source_version!r}"
            )
        present_drop_columns = [
            column for column in candidate.drop_columns if column in frame.columns
        ]
        frame = frame.drop(columns=present_drop_columns)
        output_path = output_dir / candidate.filename
        csv_bytes = frame.to_csv(index=False).encode("utf-8")
        actual_sha256 = _sha256(csv_bytes)
        if candidate.expected_sha256 is not None and actual_sha256 != candidate.expected_sha256:
            raise ValueError(
                f"{candidate.dataset_id}: materialized CSV hash mismatch; expected "
                f"{candidate.expected_sha256}, got {actual_sha256}"
            )
        if candidate.expected_sha256 is not None and output_path.exists():
            existing_sha256 = _sha256(output_path.read_bytes())
            if existing_sha256 != candidate.expected_sha256:
                raise FileExistsError(
                    f"{candidate.dataset_id}: refusing to overwrite existing CSV with "
                    f"unexpected hash {existing_sha256}"
                )
        else:
            output_path.write_bytes(csv_bytes)
        records.append(
            {
                **asdict(candidate),
                "target": target,
                "source_name": dataset.details.get("name"),
                "source_version": source_version,
                "source_url": dataset.details.get("url"),
                "source_license": dataset.details.get("licence"),
                "dropped_columns": present_drop_columns,
                "shape": list(frame.shape),
                "sha256": actual_sha256,
                "path": str(output_path),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--dataset-id", action="append", dest="dataset_ids")
    parser.add_argument(
        "--verify-local",
        action="store_true",
        help="verify recorded local CMC, Glass, and Yeast CSV hashes without network access",
    )
    args = parser.parse_args()
    selected = set(args.dataset_ids) if args.dataset_ids else None
    unknown = (
        selected.difference({candidate.dataset_id for candidate in CANDIDATES})
        if selected
        else set()
    )
    if unknown:
        parser.error(f"unknown dataset id(s): {', '.join(sorted(unknown))}")
    if args.verify_local:
        result = verify_local_materializations(args.output_dir, selected)
    else:
        result = acquire(args.output_dir, selected)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
