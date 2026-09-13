"""Validation of the frozen equal-budget V2 tuning contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

EXPECTED_MODELS = (
    "logistic_regression",
    "random_forest",
    "xgboost",
    "lightgbm",
    "catboost",
    "balanced_random_forest",
)


@dataclass(frozen=True)
class TuningContract:
    trials: int
    scoring: str
    random_state: int
    n_jobs: int
    spaces: dict[str, dict[str, Any]]


def load_tuning_contract(config_path: Path) -> TuningContract:
    """Load and validate the frozen HPO budget before any search starts."""

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    section = payload["hyperparameter_search"]
    spaces = section["spaces"]
    missing = sorted(set(EXPECTED_MODELS) - set(spaces))
    if missing:
        raise ValueError(f"HPO spaces are missing model(s): {missing}")
    contract = TuningContract(
        trials=int(section["trials_per_dataset_outer_fold_classifier_condition"]),
        scoring=str(section["scoring"]),
        random_state=int(section["random_state"]),
        n_jobs=int(section["n_jobs"]),
        spaces={str(model): dict(space) for model, space in spaces.items()},
    )
    if contract.trials != 20:
        raise ValueError("V2 HPO budget must be exactly 20 trials")
    if contract.scoring != "macro_average_precision":
        raise ValueError("V2 HPO scoring must be macro_average_precision")
    if contract.n_jobs != 1:
        raise ValueError("V2 HPO must use one worker")
    for model, space in contract.spaces.items():
        if not space:
            raise ValueError(f"HPO search space is empty for {model}")
        if any("sampler" in str(key).casefold() for key in space):
            raise ValueError(f"sampler parameter leaked into HPO space for {model}")
    return contract


def contract_summary(contract: TuningContract) -> dict[str, Any]:
    """Return a manifest-safe summary of the equal-budget tuning contract."""

    return {
        "trials_per_cell": contract.trials,
        "scoring": contract.scoring,
        "random_state": contract.random_state,
        "n_jobs": contract.n_jobs,
        "models": sorted(contract.spaces),
        "equal_budget": len({contract.trials}) == 1,
    }
