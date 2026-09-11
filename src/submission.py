"""Gate G technical submission-snapshot validation."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
import tarfile
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.benchmark import _read_json, sha256_file

GATE_F_REVIEW = Path("results/analysis/gate-f-review.json")
FROZEN_EVIDENCE = (
    "artifacts/environment/metadata.json",
    "artifacts/runs/scope-lock.json",
    "results/full-run/benchmark_manifest.json",
    "results/analysis/gate-d-review.json",
    "results/analysis/analysis_manifest.json",
    "results/analysis/gate-e-review.json",
)
REQUIRED_RELEASE_FILES = (
    "AGENTS.md",
    "README.md",
    "docs/data-acquisition.md",
    "docs/environment.md",
    "docs/protocol.md",
    "docs/release.md",
    "docs/stage-0-feasibility.md",
    "docs/stage-0-runbook.md",
    "docs/timeline.md",
    "requirements.txt",
)
DISALLOWED_TRACKED_PREFIXES = (
    "artifacts/environment/",
    "artifacts/runs/",
    "data/raw/",
    "data/processed/",
    "results/",
)


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _run(command: Sequence[str | Path], cwd: Path) -> dict[str, Any]:
    rendered = [str(part) for part in command]
    completed = subprocess.run(
        rendered,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return {
        "command": rendered,
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _require_passed(result: dict[str, Any], label: str) -> None:
    if result["exit_code"] != 0:
        detail = (result["stderr"] or result["stdout"]).strip()
        raise RuntimeError(f"{label} failed: {detail}")


def _tracked_files(root: Path) -> list[str]:
    result = _run(["git", "ls-files", "-z"], root)
    _require_passed(result, "git ls-files")
    return sorted(path for path in result["stdout"].split("\x00") if path)


def _tracked_digest(paths: Sequence[str]) -> str:
    return hashlib.sha256("\x00".join(paths).encode("utf-8")).hexdigest()


def _reference(path: Path, root: Path) -> dict[str, str]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": sha256_file(path),
    }


def _validate_clean_branch(root: Path) -> dict[str, str]:
    status = _run(["git", "status", "--porcelain", "--untracked-files=all"], root)
    _require_passed(status, "git status")
    if status["stdout"].strip():
        raise RuntimeError("Gate G requires a clean tracked working tree")

    branch = _run(["git", "branch", "--show-current"], root)
    _require_passed(branch, "git branch")
    branch_name = branch["stdout"].strip()
    if not branch_name or branch_name in {"main", "master"}:
        raise RuntimeError("Gate G requires a dedicated feature branch")

    head = _run(["git", "rev-parse", "HEAD"], root)
    _require_passed(head, "git rev-parse")
    return {"branch": branch_name, "head": head["stdout"].strip()}


def _validate_gate_f(root: Path, git_head: str) -> dict[str, Any]:
    path = root / GATE_F_REVIEW
    if not path.is_file():
        raise RuntimeError(f"Gate F review is missing: {GATE_F_REVIEW.as_posix()}")
    review = _read_json(path, GATE_F_REVIEW.as_posix())
    if review.get("status") != "passed":
        raise RuntimeError("Gate F review is not passed")
    if review.get("git_head") != git_head:
        raise RuntimeError("Gate F review does not describe the current commit")
    snapshot = review.get("source_snapshot")
    if not isinstance(snapshot, dict) or not snapshot.get("archive_sha256"):
        raise RuntimeError("Gate F review has no source archive hash")
    checks = snapshot.get("checks")
    if not isinstance(checks, list) or not checks or any(check.get("exit_code") != 0 for check in checks):
        raise RuntimeError("Gate F review contains a failed or incomplete clean-checkout check")
    return {
        "path": GATE_F_REVIEW.as_posix(),
        "sha256": sha256_file(path),
        "reviewed_at_utc": review.get("reviewed_at_utc"),
        "source_snapshot": {
            "archive_sha256": snapshot["archive_sha256"],
            "tracked_file_count": snapshot.get("tracked_file_count"),
            "tracked_file_set_sha256": snapshot.get("tracked_file_set_sha256"),
        },
    }


def _validate_frozen_evidence(root: Path) -> dict[str, dict[str, str]]:
    references: dict[str, dict[str, str]] = {}
    for relative in FROZEN_EVIDENCE:
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"frozen evidence is missing: {relative}")
        references[relative] = _reference(path, root)

    expected_statuses = {
        "artifacts/runs/scope-lock.json": "ready",
        "results/full-run/benchmark_manifest.json": "complete",
        "results/analysis/gate-d-review.json": "passed",
        "results/analysis/analysis_manifest.json": "complete",
        "results/analysis/gate-e-review.json": "passed",
    }
    for relative, status in expected_statuses.items():
        payload = _read_json(root / relative, relative)
        if payload.get("status") != status:
            raise RuntimeError(f"{relative} does not have status {status!r}")
    return references


def _archive_snapshot(root: Path, tracked: Sequence[str]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="stage0-gate-g-") as temporary_name:
        archive = Path(temporary_name) / "source.tar"
        result = _run(
            ["git", "archive", "--format=tar", "--prefix=checkout/", "--output", archive, "HEAD"],
            root,
        )
        _require_passed(result, "git archive")
        with tarfile.open(archive, mode="r") as handle:
            members = sorted(
                member.name.removeprefix("checkout/")
                for member in handle.getmembers()
                if member.isfile()
            )
        if members != list(tracked):
            raise RuntimeError("submission archive file set differs from git ls-files")
        generated = [
            path
            for path in members
            if any(path.startswith(prefix) for prefix in DISALLOWED_TRACKED_PREFIXES)
            and not path.endswith(".gitkeep")
        ]
        if generated:
            raise RuntimeError(f"submission archive contains generated data: {generated}")
        return {
            "archive_sha256": sha256_file(archive),
            "tracked_file_count": len(tracked),
            "tracked_file_set_sha256": _tracked_digest(tracked),
        }


def verify_gate_g(root: Path, output_path: Path | None = None) -> dict[str, Any]:
    """Validate and record the exact technical release snapshot for Gate G."""

    root = root.resolve()
    identity = _validate_clean_branch(root)
    tracked = _tracked_files(root)
    missing = [path for path in REQUIRED_RELEASE_FILES if path not in tracked]
    if missing:
        raise RuntimeError(f"release package is missing required tracked files: {missing}")
    frozen = _validate_frozen_evidence(root)
    gate_f = _validate_gate_f(root, identity["head"])
    snapshot = _archive_snapshot(root, tracked)
    if snapshot["tracked_file_set_sha256"] != gate_f["source_snapshot"]["tracked_file_set_sha256"]:
        raise RuntimeError("Gate F and Gate G tracked source snapshots differ")

    payload: dict[str, Any] = {
        "schema_version": "gate-g-review-v1",
        "status": "passed",
        "reviewed_at_utc": _utc_now(),
        "git": identity,
        "source_snapshot": snapshot,
        "gate_f": gate_f,
        "frozen_evidence": frozen,
        "submission": {
            "kind": "technical-release-snapshot",
            "paper_artifact": None,
            "venue": None,
            "external_submission": False,
        },
        "boundary": (
            "the exact feature-branch source snapshot and local environment/results manifests "
            "are archived for review; this payload does not claim a paper submission, venue "
            "acceptance, external review, deployment, or publication"
        ),
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/analysis/gate-g-review.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output if args.output.is_absolute() else root / args.output
    payload = verify_gate_g(root, output)
    print(payload["status"])
    print(output)


if __name__ == "__main__":
    main()
