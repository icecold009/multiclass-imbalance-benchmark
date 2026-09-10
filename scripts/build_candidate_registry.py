"""Build Stage 0 registry and acquisition evidence for the candidate pool.

The structural metrics are calculated from the downloaded raw CSV files. Human
audit decisions are intentionally explicit in ``AUDIT_DECISIONS`` and are not
derived from model performance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import subprocess
from collections import Counter
from pathlib import Path

import pandas as pd
from acquire_openml_candidates import CANDIDATES

ROOT = Path(__file__).resolve().parents[1]
MATERIALIZED_DATE = "2026-09-08"
ACCESSED_AT_UTC = "2026-09-08T17:38:51Z"
LICENSE = "UCI CC BY 4.0; OpenML public"

FULL_NUMERIC = "numeric;raw;class_weighted;random_over;random_under;smote;adasyn;borderline_smote;smoteenn;smotetomek;balanced_random_forest"
FULL_MIXED = "mixed;raw;class_weighted;random_over;random_under;smotenc;smoteenn;smotetomek;balanced_random_forest"


AUDIT_DECISIONS: dict[str, dict[str, object]] = {
    "balance_scale": {
        "display_name": "Balance Scale",
        "source_url": "https://openml.org/data/v1/download/11/balance-scale.arff",
        "feature_type": "numeric",
        "has_groups": False,
        "has_time_order": False,
        "leakage_status": "pass; no identifier, group, or time field",
        "eligible": True,
        "exclusion_reason": "",
        "applicable_families": FULL_NUMERIC,
        "notes": "UCI Balance Scale; structural floor met and no duplicate feature profiles observed.",
    },
    "mfeat_factors": {
        "display_name": "Multiple Features (Factors)",
        "source_url": "https://openml.org/data/v1/download/12/mfeat-factors.arff",
        "feature_type": "numeric",
        "has_groups": False,
        "has_time_order": False,
        "leakage_status": "pass; no identifier, group, or time field",
        "eligible": True,
        "exclusion_reason": "",
        "applicable_families": FULL_NUMERIC,
        "notes": "One representation from the UCI Multiple Features source family; other mfeat representations are excluded as related blocks.",
    },
    "mfeat_fourier": {
        "display_name": "Multiple Features (Fourier)",
        "source_url": "https://openml.org/data/v1/download/14/mfeat-fourier.arff",
        "feature_type": "numeric",
        "has_groups": False,
        "has_time_order": False,
        "leakage_status": "related-source-risk; same underlying digit cases as retained mfeat_factors",
        "eligible": False,
        "exclusion_reason": "exclude: related feature representation from the same Multiple Features cases; retaining it would inflate correlated statistical blocks",
        "applicable_families": "numeric",
        "notes": "Kept in the candidate pool to make the related-source decision explicit; no model scores were inspected.",
    },
    "optdigits": {
        "display_name": "Optical Recognition of Handwritten Digits",
        "source_url": "https://openml.org/data/v1/download/28/optdigits.arff",
        "feature_type": "numeric",
        "has_groups": False,
        "has_time_order": False,
        "leakage_status": "pass; source identifier not retained",
        "eligible": True,
        "exclusion_reason": "",
        "applicable_families": FULL_NUMERIC,
        "notes": "UCI optical digit images; 5,620 rows are retained as the larger feasibility candidate and remain subject to the runtime gate.",
    },
    "dermatology": {
        "display_name": "Dermatology",
        "source_url": "https://openml.org/data/v1/download/35/dermatology.arff",
        "feature_type": "mixed",
        "has_groups": False,
        "has_time_order": False,
        "leakage_status": "pass; semantic ordinal signs restored as categorical and Age retained as numeric",
        "eligible": True,
        "exclusion_reason": "",
        "applicable_families": FULL_MIXED,
        "notes": "UCI Dermatology; 33 symptom/sign columns are semantic categorical values, Age is numeric, and eight missing values require fold-only imputation.",
    },
    "segment": {
        "display_name": "Image Segmentation",
        "source_url": "https://openml.org/data/v1/download/36/segment.arff",
        "feature_type": "numeric",
        "has_groups": True,
        "has_time_order": False,
        "leakage_status": "group-risk; regions originate from seven source images without a group column",
        "eligible": False,
        "exclusion_reason": "exclude: valid image-grouped cross-validation is unavailable because the source image identifier is absent",
        "applicable_families": "numeric",
        "notes": "UCI Statlog Image Segmentation; row-level random folds could share source-image information.",
    },
    "ecoli": {
        "display_name": "E. coli Protein Localization",
        "source_url": "https://openml.org/data/v1/download/39/ecoli.arff",
        "feature_type": "numeric",
        "has_groups": False,
        "has_time_order": False,
        "leakage_status": "support-fail; no group or time field, but minimum class support is below five",
        "eligible": False,
        "exclusion_reason": "exclude: two classes have only two observations, below the provisional n_min_class >= 5 feasibility floor",
        "applicable_families": "numeric",
        "notes": "UCI Ecoli; retained as an explicit support-failure candidate.",
    },
    "soybean": {
        "display_name": "Soybean Disease",
        "source_url": "https://openml.org/data/v1/download/42/soybean.arff",
        "feature_type": "categorical-only",
        "has_groups": False,
        "has_time_order": False,
        "leakage_status": "label-conflict; categorical-only and one conflicting duplicate feature profile",
        "eligible": False,
        "exclusion_reason": "exclude: categorical-only data requires SMOTEN, which is not implemented, and a duplicate feature profile has conflicting labels",
        "applicable_families": "categorical-only",
        "notes": "UCI Soybean; missing values and categorical semantics are preserved in the audit evidence.",
    },
    "tae": {
        "display_name": "Teaching Assistant Evaluation",
        "source_url": "https://openml.org/data/v1/download/48/tae.arff",
        "feature_type": "mixed",
        "has_groups": False,
        "has_time_order": False,
        "leakage_status": "label-conflict; four duplicate feature profiles have inconsistent targets",
        "eligible": False,
        "exclusion_reason": "exclude: unresolved duplicate feature profiles with conflicting labels prevent a defensible row-independent split",
        "applicable_families": "mixed",
        "notes": "UCI TAE; Course instructor, Course, and semester are semantic categorical fields even though the materialized CSV is numerically encoded.",
    },
    "iris": {
        "display_name": "Iris",
        "source_url": "https://openml.org/data/v1/download/61/iris.arff",
        "feature_type": "numeric",
        "has_groups": False,
        "has_time_order": False,
        "leakage_status": "pass; no identifier, group, or time field",
        "eligible": True,
        "exclusion_reason": "",
        "applicable_families": FULL_NUMERIC,
        "notes": "UCI Iris; balanced multiclass control with three exact duplicate rows and no conflicting duplicate labels.",
    },
    "zoo": {
        "display_name": "Zoo",
        "source_url": "https://openml.org/data/v1/download/52352/zoo.arff",
        "feature_type": "categorical-only",
        "has_groups": False,
        "has_time_order": False,
        "leakage_status": "support-fail; categorical-only and minimum class support is four",
        "eligible": False,
        "exclusion_reason": "exclude: categorical-only data requires SMOTEN and the smallest class has four observations",
        "applicable_families": "categorical-only",
        "notes": "UCI Zoo; the source animal-name identifier is not retained in the modelling table.",
    },
    "wine": {
        "display_name": "Wine",
        "source_url": "https://openml.org/data/v1/download/3624/wine.arff",
        "feature_type": "numeric",
        "has_groups": False,
        "has_time_order": False,
        "leakage_status": "pass; no identifier, group, or time field",
        "eligible": True,
        "exclusion_reason": "",
        "applicable_families": FULL_NUMERIC,
        "notes": "UCI Wine; compact three-class numeric candidate with all classes above the provisional support floor.",
    },
    "satimage": {
        "display_name": "Statlog Landsat Satellite",
        "source_url": "https://openml.org/data/v1/download/3619/satimage.arff",
        "feature_type": "numeric",
        "has_groups": True,
        "has_time_order": False,
        "leakage_status": "group-risk; pixel/neighborhood observations have no scene or spatial group identifier",
        "eligible": False,
        "exclusion_reason": "defer: spatial independence and a valid grouped split are unresolved; the 6,430-row size also exceeds the preferred Stage 0 range",
        "applicable_families": "numeric",
        "notes": "UCI Statlog Landsat Satellite; retained in the pool for an explicit spatial-dependence review.",
    },
    "hayes_roth": {
        "display_name": "Hayes-Roth",
        "source_url": "https://openml.org/data/v1/download/52233/hayes-roth.arff",
        "feature_type": "numeric",
        "has_groups": False,
        "has_time_order": False,
        "leakage_status": "label-conflict; nine duplicate feature profiles have inconsistent targets",
        "eligible": False,
        "exclusion_reason": "exclude: unresolved duplicate feature profiles with conflicting labels prevent a defensible row-independent split",
        "applicable_families": "numeric",
        "notes": "UCI Hayes-Roth; no identifier column is retained, but duplicate-profile label conflicts are disqualifying.",
    },
    "cnae_9": {
        "display_name": "CNAE-9",
        "source_url": "https://openml.org/data/v1/download/1586233/cnae-9.arff",
        "feature_type": "numeric",
        "has_groups": False,
        "has_time_order": False,
        "leakage_status": "pass; no identifier, group, or time field",
        "eligible": True,
        "exclusion_reason": "",
        "applicable_families": FULL_NUMERIC,
        "notes": "Public OpenML CNAE-9 candidate; 856 source features provide the high-dimensional feasibility case and 28 duplicate rows have consistent labels.",
    },
    "seeds": {
        "display_name": "Seeds",
        "source_url": "https://openml.org/data/v1/download/1592291/seeds.arff",
        "feature_type": "numeric",
        "has_groups": False,
        "has_time_order": False,
        "leakage_status": "pass; no identifier, group, or time field",
        "eligible": True,
        "exclusion_reason": "",
        "applicable_families": FULL_NUMERIC,
        "notes": "UCI Seeds; balanced three-class numeric candidate with 70 observations per class.",
    },
    "splice": {
        "display_name": "Molecular Biology Splice Junction",
        "source_url": "https://openml.org/data/v1/download/46/splice.arff",
        "feature_type": "categorical-only",
        "has_groups": False,
        "has_time_order": False,
        "leakage_status": "label-conflict; categorical-only and one conflicting duplicate feature profile",
        "eligible": False,
        "exclusion_reason": "exclude: categorical-only sequence data requires SMOTEN and one duplicate sequence profile has conflicting labels",
        "applicable_families": "categorical-only",
        "notes": "UCI Splice; the OpenML-documented Instance_name identifier is already absent from the downloaded modelling frame.",
    },
    "wine_quality_red": {
        "display_name": "Wine Quality Red",
        "source_url": "https://openml.org/data/v1/download/4965268/wine-quality-red.arff",
        "feature_type": "numeric",
        "has_groups": False,
        "has_time_order": False,
        "leakage_status": "pass; quality label treated as the declared single-label multiclass target",
        "eligible": True,
        "exclusion_reason": "",
        "applicable_families": FULL_NUMERIC,
        "notes": "UCI Wine Quality; quality scores 3-8 are retained as six declared classes, with duplicate feature profiles carrying consistent labels.",
    },
}


REGISTRY_COLUMNS = [
    "dataset_id", "display_name", "source", "source_id", "source_url", "license_or_terms",
    "source_version", "raw_sha256", "task_type", "target_column", "n_rows", "n_classes",
    "class_counts", "ir_majority_minority", "n_min_class", "d_raw", "d_encoded", "feature_type",
    "has_missing", "has_groups", "has_time_order", "has_duplicates", "leakage_status", "eligible",
    "exclusion_reason", "applicable_families", "notes",
]
MANIFEST_COLUMNS = [
    "dataset_id", "source", "source_id", "source_url", "source_version", "accessed_at_utc",
    "license_or_terms", "download_method", "source_file_name", "local_path", "raw_sha256",
    "file_size_bytes", "download_status", "notes",
]


def _class_counts(series: pd.Series) -> str:
    counts = Counter(series.dropna().tolist())
    return "|".join(f"{label}:{counts[label]}" for label in sorted(counts, key=lambda value: str(value)))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _conflicting_feature_groups(frame: pd.DataFrame, target: str) -> int:
    features = frame.drop(columns=[target])
    grouped = frame.groupby(list(features.columns), dropna=False, sort=False)[target].nunique()
    return int((grouped > 1).sum())


def _git_base_text(path: Path, base_ref: str) -> str:
    relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
    result = subprocess.run(
        ["git", "show", f"{base_ref}:{relative}"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))


def _append_rows(base_text: str, rows: list[dict[str, object]], columns: list[str]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\r\n")
    for row in rows:
        writer.writerow(row)
    if base_text and not base_text.endswith(("\n", "\r")):
        base_text += "\r\n"
    return base_text + output.getvalue()


def build(registry_path: Path, manifest_path: Path, raw_dir: Path, base_ref: str) -> None:
    candidate_ids = {candidate.dataset_id for candidate in CANDIDATES}
    base_registry_text = _git_base_text(registry_path, base_ref)
    base_manifest_text = _git_base_text(manifest_path, base_ref)
    existing_registry = _rows(base_registry_text)
    existing_manifest = _rows(base_manifest_text)
    registry_rows = [row for row in existing_registry if row["dataset_id"] not in candidate_ids]
    manifest_rows = [row for row in existing_manifest if row["dataset_id"] not in candidate_ids]

    for candidate in CANDIDATES:
        decision = AUDIT_DECISIONS[candidate.dataset_id]
        path = raw_dir / candidate.filename
        frame = pd.read_csv(path)
        target = candidate.expected_target
        if target not in frame.columns:
            raise ValueError(f"{candidate.dataset_id}: target column {target!r} is missing")
        labels = frame[target]
        counts = Counter(labels.dropna().tolist())
        feature_frame = frame.drop(columns=[target])
        duplicate_rows = int(frame.duplicated().sum())
        duplicate_features = int(feature_frame.duplicated().sum())
        conflict_groups = _conflicting_feature_groups(frame, target)
        source_id = f"openml:{candidate.openml_data_id}"
        source_version = f"OpenML dataset version 1; CSV materialized {MATERIALIZED_DATE}"
        metric_note = (
            f" Audit metrics: duplicate_rows={duplicate_rows}; duplicate_feature_rows={duplicate_features}; "
            f"conflicting_feature_groups={conflict_groups}."
        )
        registry_rows.append(
            {
                "dataset_id": candidate.dataset_id,
                "display_name": decision["display_name"],
                "source": "OpenML (original UCI)",
                "source_id": source_id,
                "source_url": decision["source_url"],
                "license_or_terms": LICENSE,
                "source_version": source_version,
                "raw_sha256": sha256_file(path),
                "task_type": "single-label-classification",
                "target_column": target,
                "n_rows": len(frame),
                "n_classes": len(counts),
                "class_counts": _class_counts(labels),
                "ir_majority_minority": max(counts.values()) / min(counts.values()),
                "n_min_class": min(counts.values()),
                "d_raw": len(feature_frame.columns),
                "d_encoded": "",
                "feature_type": decision["feature_type"],
                "has_missing": bool(frame.isna().any().any()),
                "has_groups": decision["has_groups"],
                "has_time_order": decision["has_time_order"],
                "has_duplicates": bool(duplicate_rows > 0),
                "leakage_status": decision["leakage_status"],
                "eligible": decision["eligible"],
                "exclusion_reason": decision["exclusion_reason"],
                "applicable_families": decision["applicable_families"],
                "notes": str(decision["notes"]) + metric_note,
            }
        )
        manifest_rows.append(
            {
                "dataset_id": candidate.dataset_id,
                "source": "OpenML (original UCI)",
                "source_id": source_id,
                "source_url": decision["source_url"],
                "source_version": source_version,
                "accessed_at_utc": ACCESSED_AT_UTC,
                "license_or_terms": LICENSE,
                "download_method": f"sklearn.datasets.fetch_openml(data_id={candidate.openml_data_id}; as_frame=True; parser=auto) via scripts/acquire_openml_candidates.py",
                "source_file_name": candidate.filename,
                "local_path": f"data/raw/{candidate.filename}",
                "raw_sha256": sha256_file(path),
                "file_size_bytes": path.stat().st_size,
                "download_status": "downloaded",
                "notes": "Raw CSV is ignored by Git; source identity and hash are recorded here.",
            }
        )

    new_registry_rows = registry_rows[len(existing_registry):]
    new_manifest_rows = manifest_rows[len(existing_manifest):]
    registry_path.write_text(
        _append_rows(base_registry_text, new_registry_rows, REGISTRY_COLUMNS),
        encoding="utf-8",
        newline="",
    )
    manifest_path.write_text(
        _append_rows(base_manifest_text, new_manifest_rows, MANIFEST_COLUMNS),
        encoding="utf-8",
        newline="",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=ROOT / "data/dataset_registry.csv")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/acquisition_manifest.csv")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument(
        "--base-ref",
        default="70ba5ae06b1709d1f703fce3d0d0b277656f81bd",
        help="Git ref containing the seven-row pre-expansion tables",
    )
    args = parser.parse_args()
    build(args.registry, args.manifest, args.raw_dir, args.base_ref)
    print(f"wrote {len(list(csv.DictReader(args.registry.open(encoding='utf-8'))))} registry rows")
    print(f"wrote {len(list(csv.DictReader(args.manifest.open(encoding='utf-8'))))} manifest rows")


if __name__ == "__main__":
    main()
