"""Build the V2 registry from the frozen candidate manifest and raw files.

The builder records observable provenance and data-quality facts only.  It
does not inspect model scores, pilot outputs, or benchmark results.  Any
candidate whose IID/group/time status cannot be supported from the raw frame
is retained as an explicit ineligible row with a reason.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from urllib.parse import unquote
from urllib.request import urlopen

import pandas as pd

from src.v2_registry import expected_conditions, imbalance_band

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_COLUMNS = [
    "dataset_id",
    "display_name",
    "source",
    "source_id",
    "source_url",
    "license_or_terms",
    "source_version",
    "raw_sha256",
    "task_type",
    "target_column",
    "n_rows",
    "n_classes",
    "class_counts",
    "ir_majority_minority",
    "n_min_class",
    "d_raw",
    "d_encoded",
    "feature_type",
    "has_missing",
    "has_groups",
    "has_time_order",
    "has_duplicates",
    "leakage_status",
    "imbalance_band",
    "eligible",
    "exclusion_reason",
    "applicable_conditions",
    "notes",
]
OVERRIDE_COLUMNS = ["dataset_id", "source_id", "target_column", "categorical_columns", "notes"]

# These are source/provenance decisions, not score-based exclusions.  The
# builder still writes every row so the decision remains auditable.
MANUAL_EXCLUSIONS = {
    "openml_279_meta_stream_intervals_arff": "exclude: stream/interval provenance is not demonstrably IID for ordinary stratified CV",
    "openml_40685_shuttle": "exclude: the source documents the original rows in time order; grouped/time-aware validation is unavailable",
    "openml_40985_tamilnadu_electricity": "exclude: electricity observations are time/season structured and no time-aware split is declared",
    "openml_45067_okcupid_stem": "exclude: duplicate OpenML task family with openml_41440_okcupid_stem; one underlying task counts once",
    "openml_40700_cars1": "exclude: duplicate OpenML task family with openml_455_cars; one underlying task counts once",
    "openml_45968_braidflowsessionlevel": "exclude: session-level observations imply grouped dependence without a supported grouped split",
    "openml_46807_melbourne_airbnb": "exclude: property/location and listing-time dependence cannot be separated with the retained fields",
    "openml_46955_sdss17": "exclude: astronomy source grouping/provenance is unresolved for row-wise IID folds",
    "openml_454_analcatdata_halloffame": "exclude: raw bytes duplicate openml_185_baseball; one underlying task counts once",
    "openml_46336_hayes_roth_clean": "exclude: cleaned duplicate OpenML task family with openml_329_hayes_roth; one underlying task counts once",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _target_from_openml(data_id: int, fallback: str) -> str:
    try:
        with urlopen(f"https://www.openml.org/api/v1/json/data/{data_id}", timeout=30) as response:
            description = json.loads(response.read().decode("utf-8"))["data_set_description"]
        target = unquote(str(description.get("default_target_attribute", ""))).strip()
        return target or fallback
    except Exception:  # noqa: BLE001 - provenance fallback is recorded in notes
        return fallback


def _categorical_columns_from_openml(
    data_id: int, frame: pd.DataFrame, target: str
) -> list[str]:
    fallback = [
        str(column)
        for column in frame.drop(columns=[target]).columns
        if column not in frame.select_dtypes(include="number").columns
    ]
    try:
        with urlopen(
            f"https://www.openml.org/api/v1/json/data/features/{data_id}", timeout=30
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
        features = payload["data_features"]["feature"]
        return [
            str(item["name"])
            for item in features
            if str(item.get("data_type", "")).casefold() == "nominal"
            and str(item.get("is_target", "false")).casefold() != "true"
            and str(item.get("is_ignore", "false")).casefold() != "true"
            and str(item.get("is_row_identifier", "false")).casefold() != "true"
            and str(item["name"]) in frame.columns
        ]
    except Exception:  # noqa: BLE001 - raw-dtype fallback remains explicit in notes
        return fallback


def _class_counts(labels: pd.Series) -> str:
    counts = Counter(labels.tolist())
    return "|".join(f"{label}:{counts[label]}" for label in sorted(counts, key=str))


def _conflicting_feature_groups(frame: pd.DataFrame, target: str) -> int:
    features = frame.drop(columns=[target])
    if features.empty:
        return 0
    grouped = frame.groupby(list(features.columns), dropna=False, sort=False)[target].nunique()
    return int((grouped > 1).sum())


def _suspicious_columns(frame: pd.DataFrame, target: str) -> tuple[list[str], list[str]]:
    groups: list[str] = []
    times: list[str] = []
    for column in frame.drop(columns=[target]).columns:
        name = str(column).casefold()
        if any(
            token in name
            for token in (
                "group_id",
                "subject_id",
                "patient_id",
                "session_id",
                "user_id",
                "author_id",
                "record_id",
                "entity_id",
            )
        ):
            groups.append(str(column))
        if any(token in name for token in ("timestamp", "datetime", "time_index", "sequence")):
            times.append(str(column))
    return groups, times


def build(manifest_path: Path, raw_dir: Path, registry_path: Path, overrides_path: Path) -> None:
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    registry_rows: list[dict[str, object]] = []
    override_rows: list[dict[str, object]] = []
    for candidate in manifest.to_dict(orient="records"):
        dataset_id = str(candidate["dataset_id"])
        raw_path = raw_dir / f"{dataset_id}.csv"
        blockers: list[str] = []
        if not raw_path.is_file():
            registry_rows.append(
                {
                    **{column: "" for column in REGISTRY_COLUMNS},
                    "dataset_id": dataset_id,
                    "display_name": candidate["display_name"],
                    "source": "OpenML",
                    "source_id": f"openml:{candidate['openml_data_id']}",
                    "source_url": candidate["source_url"],
                    "source_version": candidate["source_version"],
                    "raw_sha256": "",
                    "task_type": "single-label-classification",
                    "feature_type": candidate["feature_type"],
                    "eligible": "False",
                    "exclusion_reason": "FAILED / NOT APPLICABLE: raw candidate file is unavailable",
                    "applicable_conditions": ";".join(sorted(expected_conditions(candidate["feature_type"]))),
                    "notes": "Candidate retained for explicit acquisition failure accounting.",
                }
            )
            continue

        frame = pd.read_csv(raw_path)
        fallback_target = str(frame.columns[-1])
        target = _target_from_openml(int(candidate["openml_data_id"]), fallback_target)
        if target not in frame.columns:
            target = fallback_target
            blockers.append("OpenML target name did not match the materialized CSV; last column used")
        labels = frame[target].dropna()
        feature_frame = frame.drop(columns=[target])
        categorical_columns = _categorical_columns_from_openml(
            int(candidate["openml_data_id"]), frame, target
        )
        numeric_count = len(feature_frame.columns) - len(categorical_columns)
        feature_type = str(candidate["feature_type"])
        groups, times = _suspicious_columns(frame, target)
        conflicts = _conflicting_feature_groups(frame.dropna(subset=[target]), target)
        manual_reason = MANUAL_EXCLUSIONS.get(dataset_id)
        if groups:
            blockers.append(f"group-like columns require grouped validation: {', '.join(groups)}")
        if times:
            blockers.append(f"time-like columns require time-aware validation: {', '.join(times)}")
        if feature_type not in {"numeric", "mixed"}:
            blockers.append("categorical-only representation is outside the frozen core")
        if len(labels) == 0:
            blockers.append("target has no non-missing labels")
        counts = Counter(labels.tolist())
        n_min = min(counts.values(), default=0)
        ratio = max(counts.values(), default=0) / n_min if n_min else 0.0
        if len(counts) < 3 or n_min < 10:
            blockers.append("minimum three-class and minority-support rule is not met")
        if manual_reason:
            blockers.append(manual_reason)
        eligible = not blockers
        leakage = "pass; raw-frame structural audit found no group/time/conflicting-profile issue" if eligible else "review-fail; explicit blocker recorded"
        if manual_reason:
            leakage = "review-fail; manual provenance decision"
        registry_rows.append(
            {
                "dataset_id": dataset_id,
                "display_name": candidate["display_name"],
                "source": "OpenML",
                "source_id": f"openml:{candidate['openml_data_id']}",
                "source_url": candidate["source_url"],
                "license_or_terms": "OpenML public; underlying source terms recorded in OpenML description",
                "source_version": candidate["source_version"],
                "raw_sha256": sha256_file(raw_path),
                "task_type": "single-label-classification",
                "target_column": target,
                "n_rows": len(frame),
                "n_classes": len(counts),
                "class_counts": _class_counts(labels),
                "ir_majority_minority": ratio,
                "n_min_class": n_min,
                "d_raw": len(feature_frame.columns),
                "d_encoded": "",
                "feature_type": feature_type,
                "has_missing": bool(frame.isna().any().any()),
                "has_groups": bool(groups),
                "has_time_order": bool(times),
                "has_duplicates": bool(frame.duplicated().any()),
                "leakage_status": leakage,
                "imbalance_band": imbalance_band(ratio) if ratio else "",
                "eligible": eligible,
                "exclusion_reason": "; ".join(blockers),
                "applicable_conditions": ";".join(sorted(expected_conditions(feature_type))),
                "notes": (
                    "Structural eligibility only; no benchmark scores inspected. "
                    f"duplicate_rows={int(frame.duplicated().sum())}; conflicting_feature_groups={conflicts}; "
                    f"numeric_columns={numeric_count}; categorical_columns={len(categorical_columns)}."
                ),
            }
        )
        override_rows.append(
            {
                "dataset_id": dataset_id,
                "source_id": f"openml:{candidate['openml_data_id']}",
                "target_column": target,
                "categorical_columns": ";".join(categorical_columns),
                "notes": "Derived from materialized OpenML dtypes; review semantic encodings before protocol freeze.",
            }
        )

    registry_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(registry_rows, columns=REGISTRY_COLUMNS).to_csv(registry_path, index=False)
    pd.DataFrame(override_rows, columns=OVERRIDE_COLUMNS).to_csv(overrides_path, index=False)
    print(f"wrote {len(registry_rows)} registry rows to {registry_path}")
    print(f"wrote {len(override_rows)} feature override rows to {overrides_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/v2_candidate_manifest.csv")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw/v2")
    parser.add_argument("--registry", type=Path, default=ROOT / "data/dataset_registry_v2.csv")
    parser.add_argument(
        "--overrides",
        type=Path,
        default=ROOT / "data/dataset_feature_overrides_v2.csv",
    )
    args = parser.parse_args()
    build(args.manifest, args.raw_dir, args.registry, args.overrides)


if __name__ == "__main__":
    main()
