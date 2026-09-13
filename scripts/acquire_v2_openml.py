"""Freeze a score-free OpenML candidate manifest and materialize its CSVs.

Selection uses only public provenance and structural metadata from OpenML:
class count, minority support, row count, feature representation, and
imbalance band.  It never reads benchmark results or model scores.  The
resulting manifest is reviewed before it is promoted to the V2 registry.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

from sklearn.datasets import fetch_openml

from src.v2_registry import BAND_QUOTAS, FEATURE_QUOTAS, imbalance_band

ROOT = Path(__file__).resolve().parents[1]
CATALOG_URL = "https://www.openml.org/api/v1/json/data/list/limit/10000/status/active"
RETAINED_DATA_IDS = (11, 23, 28, 35, 41, 61, 181, 187, 1468, 1499, 40691)
RESERVE_DATA_IDS = (185, 40474, 40671, 40982, 41168, 41169, 41671, 46443, 46936, 46943)
EXCLUDED_DATA_IDS = frozenset({45067, 45938, 46446, 46520})
EXCLUDED_NAMES = frozenset(
    {
        "abalone",
        "eye_movements",
        "gesturephasesegmentationprocessed",
        "har",
        "japanesevowels",
        "letter-challenge-unlabeled.arff",
        "page-blocks",
        "rmftsa_sleepdata",
        "satimage",
        "segment",
        "vehicle",
        "wall-robot-navigation",
        "robot-failures-lp5",
        "allbp",
        "allrep",
        "calendardow",
        "indian_pines",
    }
)
EXCLUDED_NAME_PREFIXES = (
    "meta_album_",
    "jungle_chess",
    "volcanoes-",
    "autouniv-",
    "okcupid-stem_",
    "mabbob_ela_",
    "dgf_",
    "steel-plates-fault_seed_",
)
MANIFEST_COLUMNS = (
    "dataset_id",
    "openml_data_id",
    "display_name",
    "source_url",
    "source_version",
    "n_rows",
    "n_classes",
    "n_min_class",
    "ir_majority_minority",
    "d_raw",
    "n_numeric_features",
    "n_symbolic_features",
    "feature_type",
    "imbalance_band",
    "selection_rule",
)


@dataclass(frozen=True)
class Candidate:
    data_id: int
    name: str
    version: int
    n_rows: int
    n_classes: int
    n_min_class: int
    n_majority: int
    d_raw: int
    n_numeric: int
    n_symbolic: int

    @property
    def feature_type(self) -> str:
        if self.n_numeric and self.n_symbolic:
            return "mixed"
        if self.n_numeric:
            return "numeric"
        return "categorical-only"

    @property
    def ratio(self) -> float:
        return self.n_majority / self.n_min_class

    @property
    def band(self) -> str:
        return imbalance_band(self.ratio)

    @property
    def dataset_id(self) -> str:
        normalized = "".join(character.lower() if character.isalnum() else "_" for character in self.name)
        return f"openml_{self.data_id}_{normalized.strip('_')}"


def _quality(item: dict[str, object]) -> dict[str, float]:
    return {
        str(value["name"]): float(value["value"])
        for value in item.get("quality", [])
        if isinstance(value, dict) and "name" in value and "value" in value
    }


def load_catalog() -> list[Candidate]:
    with urlopen(CATALOG_URL, timeout=90) as response:
        payload = json.load(response)
    candidates: list[Candidate] = []
    for item in payload["data"]["dataset"]:
        quality = _quality(item)
        required = (
            "NumberOfInstances",
            "NumberOfClasses",
            "MinorityClassSize",
            "MajorityClassSize",
            "NumberOfFeatures",
            "NumberOfNumericFeatures",
            "NumberOfSymbolicFeatures",
        )
        if any(name not in quality for name in required):
            continue
        candidate = Candidate(
            data_id=int(item["did"]),
            name=str(item["name"]),
            version=int(item["version"]),
            n_rows=int(quality["NumberOfInstances"]),
            n_classes=int(quality["NumberOfClasses"]),
            n_min_class=int(quality["MinorityClassSize"]),
            n_majority=int(quality["MajorityClassSize"]),
            d_raw=max(1, int(quality["NumberOfFeatures"]) - 1),
            n_numeric=int(quality["NumberOfNumericFeatures"]),
            # OpenML's feature-quality counts include the nominal target.
            # Remove it so numeric datasets are not mislabeled as mixed.
            n_symbolic=max(0, int(quality["NumberOfSymbolicFeatures"]) - 1),
        )
        if (
            candidate.data_id in EXCLUDED_DATA_IDS
            or
            candidate.n_classes < 3
            or candidate.n_min_class < 10
            or not 100 <= candidate.n_rows <= 100_000
            or candidate.d_raw < 1
            or candidate.d_raw > 1_000
            or candidate.feature_type not in {"numeric", "mixed"}
            or candidate.name.casefold() in EXCLUDED_NAMES
            or candidate.name.casefold().startswith(EXCLUDED_NAME_PREFIXES)
            or (
                candidate.name.casefold().startswith("thyroid-")
                and candidate.data_id not in {40474, 40682}
            )
            or "_seed_" in candidate.name.casefold()
        ):
            continue
        candidates.append(candidate)
    return sorted(candidates, key=lambda candidate: (candidate.band, candidate.feature_type, candidate.data_id))


def _deduplicate(candidates: list[Candidate]) -> list[Candidate]:
    selected: list[Candidate] = []
    names: set[str] = set()
    for candidate in sorted(candidates, key=lambda item: item.data_id):
        name = candidate.name.casefold()
        if name in names:
            continue
        if name.startswith("mfeat-") and candidate.data_id != 12:
            continue
        selected.append(candidate)
        names.add(name)
    return selected


def select_candidates(catalog: list[Candidate], target_count: int = 45) -> list[Candidate]:
    """Select the smallest deterministic metadata-only set meeting quotas."""

    by_id = {candidate.data_id: candidate for candidate in catalog}
    selected: list[Candidate] = [
        by_id[data_id]
        for data_id in (*RETAINED_DATA_IDS, *RESERVE_DATA_IDS)
        if data_id in by_id
    ]
    selected = _deduplicate(selected)
    selected_ids = {candidate.data_id for candidate in selected}
    selected_names = {candidate.name.casefold() for candidate in selected}

    for band, minimum in BAND_QUOTAS.items():
        while sum(candidate.band == band for candidate in selected) < minimum:
            options = [
                candidate
                for candidate in catalog
                if (
                    candidate.data_id not in selected_ids
                    and candidate.name.casefold() not in selected_names
                    and candidate.band == band
                )
            ]
            options = _deduplicate(options)
            if not options:
                raise RuntimeError(f"OpenML catalog cannot meet {band} quota")
            numeric_deficit = FEATURE_QUOTAS["numeric"] - sum(
                item.feature_type == "numeric" for item in selected
            )
            mixed_deficit = FEATURE_QUOTAS["mixed"] - sum(
                item.feature_type == "mixed" for item in selected
            )
            preferred_type = "numeric" if numeric_deficit >= mixed_deficit else "mixed"
            typed_options = [item for item in options if item.feature_type == preferred_type]
            candidate = (typed_options or options)[0]
            selected.append(candidate)
            selected_ids.add(candidate.data_id)
            selected_names.add(candidate.name.casefold())

    for feature_type, minimum in FEATURE_QUOTAS.items():
        while sum(candidate.feature_type == feature_type for candidate in selected) < minimum:
            options = [
                candidate
                for candidate in catalog
                if (
                    candidate.data_id not in selected_ids
                    and candidate.name.casefold() not in selected_names
                    and candidate.feature_type == feature_type
                )
            ]
            options = _deduplicate(options)
            if not options:
                raise RuntimeError(f"OpenML catalog cannot meet {feature_type} quota")
            candidate = options[0]
            selected.append(candidate)
            selected_ids.add(candidate.data_id)
            selected_names.add(candidate.name.casefold())

    for candidate in catalog:
        if len(selected) >= target_count:
            break
        if candidate.data_id not in selected_ids and candidate.name.casefold() not in selected_names:
            selected.append(candidate)
            selected_ids.add(candidate.data_id)
            selected_names.add(candidate.name.casefold())
    if len(selected) < target_count:
        raise RuntimeError(f"OpenML catalog supplied only {len(selected)} unique candidates")
    return sorted(selected, key=lambda candidate: candidate.data_id)


def write_manifest(candidates: list[Candidate], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(
                {
                    "dataset_id": candidate.dataset_id,
                    "openml_data_id": candidate.data_id,
                    "display_name": candidate.name,
                    "source_url": f"https://openml.org/data/v1/download/{candidate.data_id}/{candidate.name}.arff",
                    "source_version": candidate.version,
                    "n_rows": candidate.n_rows,
                    "n_classes": candidate.n_classes,
                    "n_min_class": candidate.n_min_class,
                    "ir_majority_minority": f"{candidate.ratio:.12g}",
                    "d_raw": candidate.d_raw,
                    "n_numeric_features": candidate.n_numeric,
                    "n_symbolic_features": candidate.n_symbolic,
                    "feature_type": candidate.feature_type,
                    "imbalance_band": candidate.band,
                    "selection_rule": "OpenML structural metadata only; no benchmark scores",
                }
            )


def materialize(candidates: list[Candidate], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for candidate in candidates:
        output_path = output_dir / f"{candidate.dataset_id}.csv"
        if output_path.exists():
            continue
        dataset = fetch_openml(data_id=candidate.data_id, as_frame=True, parser="auto")
        frame = dataset.frame.copy()
        frame.to_csv(output_path, index=False)
        print(f"downloaded {candidate.data_id} -> {output_path} shape={frame.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/v2_candidate_manifest.csv")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw/v2")
    parser.add_argument("--target-count", type=int, default=55)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    selected = select_candidates(load_catalog(), target_count=args.target_count)
    write_manifest(selected, args.manifest)
    print(f"wrote {len(selected)} score-free candidates to {args.manifest}")
    if args.download:
        materialize(selected, args.raw_dir)


if __name__ == "__main__":
    main()
