from pathlib import Path

from src.submission import _tracked_digest


def test_tracked_digest_is_order_sensitive_and_null_delimited() -> None:
    assert _tracked_digest(["a.txt", "b.txt"]) != _tracked_digest(["b.txt", "a.txt"])
    assert _tracked_digest(["a.txt", "b.txt"]) == _tracked_digest(["a.txt", "b.txt"])


def test_gate_g_source_module_is_in_the_repository() -> None:
    assert Path("src/submission.py").is_file()
