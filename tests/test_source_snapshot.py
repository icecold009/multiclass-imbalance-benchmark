import pytest

from src.source_snapshot import disallowed_generated_paths


@pytest.mark.parametrize(
    "path",
    [
        "artifacts/environment/v2/metadata.json",
        "artifacts/environment/v2/pip-check.txt",
        "artifacts/environment/v2/pip-freeze.txt",
        "artifacts/environment/.gitkeep",
        "data/raw/.gitkeep",
        "results/.gitkeep",
        "docs/protocol.md",
    ],
)
def test_source_archive_accepts_frozen_provenance_and_source(path: str) -> None:
    assert disallowed_generated_paths([path]) == []


@pytest.mark.parametrize(
    "path",
    [
        "artifacts/environment/metadata.json",
        "artifacts/environment/v2/unapproved.json",
        "artifacts/environment/v2/metadata.json.bak",
        "artifacts/runs/scope-lock.json",
        "results/full-run/benchmark_results.csv",
        "data/raw/dataset.csv",
        "data/processed/dataset.csv",
    ],
)
def test_source_archive_rejects_generated_run_evidence(path: str) -> None:
    assert disallowed_generated_paths([path]) == [path]
