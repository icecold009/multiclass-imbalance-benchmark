"""Shared source-archive boundary for V1 releases and V2 provenance."""

from collections.abc import Iterable

TRACKED_V2_PROVENANCE = frozenset(
    {
        "artifacts/environment/v2/metadata.json",
        "artifacts/environment/v2/pip-check.txt",
        "artifacts/environment/v2/pip-freeze.txt",
    }
)
GENERATED_PREFIXES = (
    "artifacts/environment/",
    "artifacts/runs/",
    "results/",
    "data/raw/",
    "data/processed/",
)


def disallowed_generated_paths(paths: Iterable[str]) -> list[str]:
    """Permit the three frozen V2 host records, but never generated run data."""

    return [
        path
        for path in paths
        if path.startswith(GENERATED_PREFIXES)
        and not path.endswith(".gitkeep")
        and path not in TRACKED_V2_PROVENANCE
    ]
