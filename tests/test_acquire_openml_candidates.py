from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts import acquire_openml_candidates as acquisition


def _dataset(frame: pd.DataFrame, version: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        frame=frame,
        target=frame[frame.columns[-1]],
        details={"name": "fixture", "version": version, "url": "https://example.org"},
    )


def test_retained_candidates_match_recorded_sources_targets_and_hashes() -> None:
    expected = {
        "cmc": (
            23,
            "Contraceptive_method_used",
            1,
            "0a42de56db54f5e51830227dfc3efa2bee3120f2634dc623ed99dad0459ec5b3",
            (),
        ),
        "glass": (
            41,
            "Type",
            1,
            "bb06702a7c96a667a0781150e3a70d5b3080a976cdccd7c7f242f78e23a7ad11",
            ("Id_number",),
        ),
        "yeast": (
            181,
            "class_protein_localization",
            1,
            "743448d2ba31e10b538cf10ef03b03c6b04222ad6e32c4873ba0b07d252dcdc0",
            ("Sequence_Name",),
        ),
    }
    candidates = {candidate.dataset_id: candidate for candidate in acquisition.CANDIDATES}

    for dataset_id, (source_id, target, version, sha256, drops) in expected.items():
        candidate = candidates[dataset_id]
        assert candidate.openml_data_id == source_id
        assert candidate.expected_target == target
        assert candidate.expected_source_version == version
        assert candidate.expected_sha256 == sha256
        assert candidate.drop_columns == drops


def test_acquisition_applies_identity_drop_checks_version_and_hash_before_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = pd.DataFrame({"Id_number": [1, 2], "x": [1.5, 2.5], "Type": [1, 2]})
    transformed = frame.drop(columns=["Id_number"])
    serialized = transformed.to_csv(index=False).encode("utf-8")
    digest = hashlib.sha256(serialized).hexdigest()
    candidate = acquisition.Candidate(
        "glass_fixture",
        41,
        "glass.csv",
        "Type",
        drop_columns=("Id_number",),
        expected_source_version=1,
        expected_sha256=digest,
    )
    monkeypatch.setattr(acquisition, "CANDIDATES", (candidate,))
    monkeypatch.setattr(acquisition, "fetch_openml", lambda **_: _dataset(frame))

    records = acquisition.acquire(tmp_path, {"glass_fixture"})

    assert records[0]["dropped_columns"] == ["Id_number"]
    assert records[0]["sha256"] == digest
    assert (tmp_path / "glass.csv").read_bytes() == serialized


def test_version_or_hash_mismatch_does_not_create_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = pd.DataFrame({"x": [1, 2], "target": [0, 1]})
    candidate = acquisition.Candidate(
        "fixture", 23, "fixture.csv", "target", expected_source_version=1, expected_sha256="0" * 64
    )
    monkeypatch.setattr(acquisition, "CANDIDATES", (candidate,))
    monkeypatch.setattr(acquisition, "fetch_openml", lambda **_: _dataset(frame, version=2))
    with pytest.raises(ValueError, match="expected OpenML version"):
        acquisition.acquire(tmp_path, {"fixture"})
    assert not (tmp_path / "fixture.csv").exists()

    monkeypatch.setattr(acquisition, "fetch_openml", lambda **_: _dataset(frame, version=1))
    with pytest.raises(ValueError, match="materialized CSV hash mismatch"):
        acquisition.acquire(tmp_path, {"fixture"})
    assert not (tmp_path / "fixture.csv").exists()


def test_local_materialization_verification_is_read_only_and_checks_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"x,target\n1,0\n"
    digest = hashlib.sha256(payload).hexdigest()
    candidate = acquisition.Candidate(
        "fixture", 23, "fixture.csv", "target", expected_source_version=1, expected_sha256=digest
    )
    monkeypatch.setattr(acquisition, "CANDIDATES", (candidate,))
    path = tmp_path / candidate.filename
    path.write_bytes(payload)

    assert acquisition.verify_local_materializations(tmp_path) == [
        {
            "dataset_id": "fixture",
            "path": str(path),
            "sha256": digest,
            "status": "verified_local_materialization",
        }
    ]
    assert path.read_bytes() == payload


def test_local_verification_rejects_known_dataset_without_recorded_hash(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="only for hash-pinned datasets"):
        acquisition.verify_local_materializations(tmp_path, {"balance_scale"})
