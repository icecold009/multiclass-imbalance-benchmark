from pathlib import Path

import pytest

from src.reproducibility import (
    EXPECTED_SOURCE_FILES,
    _fresh_gate_e,
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


def test_gate_f_uses_current_interpreter_for_gate_e_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = {"status": "passed"}
    calls: list[tuple[Path, Path, Path]] = []

    def fake_gate_e(root: Path, run_dir: Path, analysis_dir: Path) -> dict[str, str]:
        calls.append((root, run_dir, analysis_dir))
        return expected

    monkeypatch.setattr("src.reproducibility.verify_gate_e", fake_gate_e)
    result = _fresh_gate_e(tmp_path, tmp_path / "run", tmp_path / "analysis", None)

    assert result == expected
    assert calls == [(tmp_path, tmp_path / "run", tmp_path / "analysis")]


def test_gate_f_can_run_gate_e_with_separate_analysis_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    interpreter = tmp_path / "analysis-python.exe"
    interpreter.write_bytes(b"test interpreter placeholder")
    calls: list[tuple[list[str | Path], Path, int]] = []

    def fake_run(
        command: list[str | Path], cwd: Path, timeout: int = 300
    ) -> dict[str, object]:
        calls.append((command, cwd, timeout))
        return {"command": [str(item) for item in command], "exit_code": 0, "stdout": "", "stderr": ""}

    monkeypatch.setattr("src.reproducibility._run", fake_run)
    monkeypatch.setattr(
        "src.reproducibility._read_json",
        lambda path, label: {"schema_version": "gate-e-review-v1", "status": "passed"},
    )

    result = _fresh_gate_e(tmp_path, tmp_path / "run", tmp_path / "analysis", interpreter)

    assert result["status"] == "passed"
    assert len(calls) == 1
    command, cwd, timeout = calls[0]
    assert command[0] == interpreter.resolve()
    assert command[1] == tmp_path / "scripts/verify_gate_e.py"
    assert "--run-dir" in command and tmp_path / "run" in command
    assert "--analysis-dir" in command and tmp_path / "analysis" in command
    assert "--output-dir" in command
    assert cwd == tmp_path
    assert timeout == 900


def test_gate_f_rejects_missing_analysis_interpreter(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Gate E analysis interpreter does not exist"):
        _fresh_gate_e(tmp_path, tmp_path / "run", tmp_path / "analysis", tmp_path / "missing.exe")


def test_gate_f_rejects_failed_separate_gate_e_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    interpreter = tmp_path / "analysis-python.exe"
    interpreter.write_bytes(b"test interpreter placeholder")
    monkeypatch.setattr(
        "src.reproducibility._run",
        lambda command, cwd, timeout=300: {
            "command": [str(item) for item in command],
            "exit_code": 1,
            "stdout": "",
            "stderr": "Gate E failed",
        },
    )

    with pytest.raises(RuntimeError, match="clean-checkout command failed"):
        _fresh_gate_e(tmp_path, tmp_path / "run", tmp_path / "analysis", interpreter)


def test_gate_f_rejects_nonpassing_separate_gate_e_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    interpreter = tmp_path / "analysis-python.exe"
    interpreter.write_bytes(b"test interpreter placeholder")
    monkeypatch.setattr(
        "src.reproducibility._run",
        lambda command, cwd, timeout=300: {
            "command": [str(item) for item in command],
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
        },
    )
    monkeypatch.setattr(
        "src.reproducibility._read_json",
        lambda path, label: {"schema_version": "gate-e-review-v1", "status": "failed"},
    )

    with pytest.raises(RuntimeError, match="fresh Gate E validation did not pass"):
        _fresh_gate_e(tmp_path, tmp_path / "run", tmp_path / "analysis", interpreter)
