from pathlib import Path

import pytest

from src.reproducibility import (
    EXPECTED_SOURCE_FILES,
    _safe_extract,
    _validate_source_archive_paths,
)


def test_gate_f_source_inventory_includes_repository_v2_registry_test() -> None:
    assert "tests/test_v2_registry.py" in EXPECTED_SOURCE_FILES
    assert "paper/README-anonymous.md" in EXPECTED_SOURCE_FILES


@pytest.mark.parametrize(
    "path",
    [
        "artifacts/environment/v2/metadata.json",
        "artifacts/environment/v2/pip-check.txt",
        "artifacts/environment/v2/pip-freeze.txt",
    ],
)
def test_gate_f_archive_allows_committed_v2_provenance(path: str) -> None:
    _validate_source_archive_paths([path])


@pytest.mark.parametrize(
    "path",
    [
        "artifacts/environment/metadata.json",
        "artifacts/environment/v2/unapproved.json",
        "artifacts/runs/scope-lock.json",
        "results/full-run/benchmark_results.csv",
        "data/raw/dataset.csv",
        "data/processed/dataset.csv",
    ],
)
def test_gate_f_archive_rejects_other_generated_paths(path: str) -> None:
    with pytest.raises(RuntimeError, match="clean source archive contains generated data"):
        _validate_source_archive_paths([path])


def test_safe_extract_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.tar"
    import tarfile

    with tarfile.open(archive, "w") as handle:
        member = tarfile.TarInfo("checkout/../../outside.txt")
        member.size = 0
        handle.addfile(member)

    with pytest.raises(RuntimeError, match="escaping path"):
        _safe_extract(archive, tmp_path / "destination")
