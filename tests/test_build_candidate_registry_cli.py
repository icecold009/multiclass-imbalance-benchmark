"""Side-effect-free CLI checks for the candidate-registry builder."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_candidate_registry.py"


def test_candidate_registry_requires_an_explicit_base_ref() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "the following arguments are required: --base-ref" in result.stderr


def test_candidate_registry_accepts_an_explicit_base_ref_before_help_exit() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--base-ref", "test-ref", "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--base-ref BASE_REF" in result.stdout
