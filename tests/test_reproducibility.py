from pathlib import Path

import pytest

from src.reproducibility import _safe_extract


def test_safe_extract_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.tar"
    import tarfile

    with tarfile.open(archive, "w") as handle:
        member = tarfile.TarInfo("checkout/../../outside.txt")
        member.size = 0
        handle.addfile(member)

    with pytest.raises(RuntimeError, match="escaping path"):
        _safe_extract(archive, tmp_path / "destination")
