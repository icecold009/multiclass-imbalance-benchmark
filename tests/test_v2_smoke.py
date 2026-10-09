import json
from pathlib import Path

from src.v2_smoke import run_smoke


def test_synthetic_smoke_has_complete_valid_or_failure_accounting(tmp_path: Path) -> None:
    manifest = run_smoke(Path.cwd(), tmp_path)
    assert manifest["status"] == "complete"
    assert manifest["excluded_from_scientific_evidence"] is True
    assert manifest["counts"]["expected_cells"] == 110
    assert manifest["counts"]["valid_cells"] + manifest["counts"]["failure_cells"] == 110
    failures = json.loads((tmp_path / "smoke_manifest.json").read_text(encoding="utf-8"))
    assert failures["hpo_trials_contract"] == 20
