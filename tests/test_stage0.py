from pathlib import Path

import pandas as pd

from src.stage0 import audit_csv, sha256_file


def test_audit_csv_records_multiclass_distribution(tmp_path: Path) -> None:
    path = tmp_path / "candidate.csv"
    pd.DataFrame(
        {
            "amount": [1, 2, 3, 4, 5, 6],
            "segment": ["a", "a", "b", "b", "c", "c"],
            "target": ["x", "x", "y", "y", "z", "z"],
        }
    ).to_csv(path, index=False)

    evidence = audit_csv(path, "target")

    assert evidence["n_rows"] == 6
    assert evidence["n_classes"] == 3
    assert evidence["class_counts"] == {"x": 2, "y": 2, "z": 2}
    assert evidence["feature_type"] == "mixed"
    assert len(evidence["raw_sha256"]) == 64


def test_sha256_file_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("stage0", encoding="utf-8")

    assert sha256_file(path) == sha256_file(path)
