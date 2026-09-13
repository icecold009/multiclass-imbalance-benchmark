from pathlib import Path

from src.v2_tuning import contract_summary, load_tuning_contract


def test_frozen_hpo_contract_has_equal_twenty_trial_budget() -> None:
    contract = load_tuning_contract(Path("config/v2.yaml"))
    summary = contract_summary(contract)
    assert summary["trials_per_cell"] == 20
    assert summary["scoring"] == "macro_average_precision"
    assert summary["n_jobs"] == 1
    assert summary["equal_budget"] is True
    assert len(summary["models"]) == 6
