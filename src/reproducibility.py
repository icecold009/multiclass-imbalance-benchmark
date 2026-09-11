"""Gate F clean-checkout and frozen-evidence reproducibility checks."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.benchmark import _read_json, sha256_file
from src.robustness import verify_gate_e

EXPECTED_SOURCE_FILES = (
    ".github/workflows/ci.yml",
    "AGENTS.md",
    "README.md",
    "config/stage0.yaml",
    "data/acquisition_manifest.csv",
    "data/dataset_feature_overrides.csv",
    "data/dataset_registry.csv",
    "docs/protocol.md",
    "docs/stage-0-runbook.md",
    "docs/timeline.md",
    "pyproject.toml",
    "requirements.txt",
    "scripts/verify_gate_e.py",
    "src/analysis.py",
    "src/benchmark.py",
    "src/pilot.py",
    "src/reproducibility.py",
    "src/robustness.py",
    "tests/test_reproducibility.py",
)
EVIDENCE_PATHS = (
    "artifacts/environment/metadata.json",
    "artifacts/runs/scope-lock.json",
    "results/full-run/benchmark_manifest.json",
    "results/full-run/gate-d-review.json",
    "results/analysis/analysis_manifest.json",
    "results/analysis/gate-e-review.json",
)
IGNORED_GENERATED_PREFIXES = (
    "artifacts/environment/",
    "artifacts/runs/",
    "results/",
    "data/raw/",
    "data/processed/",
)


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _run(command: Sequence[str | Path], cwd: Path, timeout: int = 300) -> dict[str, Any]:
    """Run one reproducibility check and retain its complete local evidence."""

    rendered = [str(part) for part in command]
    try:
        completed = subprocess.run(
            rendered,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        return {
            "command": rendered,
            "exit_code": None,
            "stdout": str(error.stdout or ""),
            "stderr": f"timed out after {timeout} seconds",
        }
    return {
        "command": rendered,
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _require_passed(result: dict[str, Any]) -> None:
    if result["exit_code"] != 0:
        command = " ".join(result["command"])
        detail = (result["stderr"] or result["stdout"]).strip()
        raise RuntimeError(f"clean-checkout command failed ({command}): {detail}")


def _tracked_files(root: Path) -> list[str]:
    result = _run(["git", "ls-files", "-z"], root)
    _require_passed(result)
    return sorted(path for path in result["stdout"].split("\x00") if path)


def _tracked_digest(paths: Sequence[str]) -> str:
    return hashlib.sha256("\x00".join(paths).encode("utf-8")).hexdigest()


def _safe_extract(archive: Path, destination: Path) -> list[str]:
    """Extract a locally-created Git archive without accepting path traversal."""

    names: list[str] = []
    with tarfile.open(archive, mode="r") as handle:
        for member in handle.getmembers():
            target = (destination / member.name).resolve()
            try:
                target.relative_to(destination.resolve())
            except ValueError as error:
                raise RuntimeError(f"source archive contains an escaping path: {member.name}") from error
            if member.issym() or member.islnk() or not (member.isdir() or member.isfile()):
                raise RuntimeError(f"source archive contains an unsupported member: {member.name}")
            handle.extract(member, destination)
            if member.isfile():
                names.append(member.name.removeprefix("checkout/"))
    return sorted(names)


def _archive_checkout(root: Path, temporary: Path) -> tuple[Path, Path, list[str]]:
    archive = temporary / "source.tar"
    result = _run(
        ["git", "archive", "--format=tar", "--prefix=checkout/", "--output", archive, "HEAD"],
        root,
    )
    _require_passed(result)
    checkout = temporary / "checkout"
    checkout.mkdir()
    members = _safe_extract(archive, temporary)
    return archive, checkout, members


def _reference(path: Path, root: Path) -> dict[str, str]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": sha256_file(path),
    }


def _validate_evidence(root: Path) -> dict[str, Any]:
    references: dict[str, dict[str, str]] = {}
    for relative in EVIDENCE_PATHS:
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"frozen evidence is missing: {relative}")
        references[relative] = _reference(path, root)

    expected_statuses = {
        "artifacts/runs/scope-lock.json": "ready",
        "results/full-run/benchmark_manifest.json": "complete",
        "results/full-run/gate-d-review.json": "passed",
        "results/analysis/analysis_manifest.json": "complete",
        "results/analysis/gate-e-review.json": "passed",
    }
    for relative, status in expected_statuses.items():
        payload = _read_json(root / relative, relative)
        if payload.get("status") != status:
            raise RuntimeError(f"{relative} does not have status {status!r}")
    gate_e = verify_gate_e(root, root / "results/full-run", root / "results/analysis")
    if gate_e.get("status") != "passed":
        raise RuntimeError("fresh Gate E validation did not pass")
    return {
        "references": references,
        "gate_e": {
            "status": gate_e["status"],
            "analysis_counts": gate_e["analysis_counts"],
            "replay": gate_e["deterministic_replay"],
        },
    }


def _validate_workflow(checkout: Path) -> dict[str, Any]:
    workflow = (checkout / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    required = {
        "pull_request": "pull_request:" in workflow,
        "python_3_11": '"3.11"' in workflow,
        "python_3_12": '"3.12"' in workflow,
        "pytest": "python -m pytest" in workflow,
        "ruff": "python -m ruff check src tests" in workflow,
        "read_only_contents": "contents: read" in workflow,
    }
    if not all(required.values()):
        missing = [name for name, passed in required.items() if not passed]
        raise RuntimeError(f"CI workflow is missing required clean-checkout controls: {missing}")
    return required


def verify_gate_f(root: Path, output_path: Path | None = None) -> dict[str, Any]:
    """Recheck frozen evidence and validate a clean Git source checkout."""

    root = root.resolve()
    status = _run(["git", "status", "--porcelain", "--untracked-files=all"], root)
    _require_passed(status)
    if status["stdout"].strip():
        raise RuntimeError("Gate F requires a clean tracked working tree")

    tracked = _tracked_files(root)
    evidence = _validate_evidence(root)
    with tempfile.TemporaryDirectory(prefix="stage0-gate-f-") as temporary_name:
        temporary = Path(temporary_name)
        archive, checkout, archived_files = _archive_checkout(root, temporary)
        if archived_files != tracked:
            raise RuntimeError("Git archive file set differs from git ls-files")
        if set(EXPECTED_SOURCE_FILES).difference(archived_files):
            missing = sorted(set(EXPECTED_SOURCE_FILES).difference(archived_files))
            raise RuntimeError(f"clean source archive is missing required files: {missing}")
        generated = [
            path
            for path in archived_files
            if any(path.startswith(prefix) for prefix in IGNORED_GENERATED_PREFIXES)
            and not path.endswith(".gitkeep")
        ]
        if generated:
            raise RuntimeError(f"clean source archive contains generated data: {generated}")
        workflow = _validate_workflow(checkout)
        python = Path(sys.executable)
        pytest_root = temporary / "pytest"
        cache_root = temporary / "cache"
        checks = [
            _run(
                [
                    python,
                    "-m",
                    "pytest",
                    "--override-ini",
                    "addopts=",
                    "--basetemp",
                    pytest_root,
                    "-o",
                    f"cache_dir={cache_root}",
                ],
                checkout,
            ),
            _run([python, "-m", "ruff", "check", "src", "tests", "scripts", "--no-cache"], checkout),
            _run([python, "-m", "src.stage0", "--help"], checkout),
            _run([python, "-m", "src.pilot", "--help"], checkout),
            _run([python, "scripts/verify_gate_e.py", "--help"], checkout),
            _run([python, "scripts/verify_gate_f.py", "--help"], checkout),
        ]
        for check in checks:
            _require_passed(check)
        snapshot = {
            "archive_sha256": sha256_file(archive),
            "tracked_file_count": len(tracked),
            "tracked_file_set_sha256": _tracked_digest(tracked),
            "checks": checks,
            "workflow": workflow,
        }

    payload: dict[str, Any] = {
        "schema_version": "gate-f-review-v1",
        "status": "passed",
        "reviewed_at_utc": _utc_now(),
        "git_head": _run(["git", "rev-parse", "HEAD"], root)["stdout"].strip(),
        "source_snapshot": snapshot,
        "frozen_evidence": evidence,
        "boundary": (
            "clean source, environment, and manifest reproducibility verified; "
            "no benchmark rerun, dataset expansion, protocol change, or paper claim is authorized"
        ),
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/analysis/gate-f-review.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output if args.output.is_absolute() else root / args.output
    payload = verify_gate_f(root, output)
    print(payload["status"])
    print(output)


if __name__ == "__main__":
    main()
