import hashlib
import json
import zipfile
from io import BytesIO
from pathlib import Path
from zlib import compress

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, EncodedStreamObject, NameObject

from src.release_artifacts import (
    TMLR_SOURCE_FILES,
    _gate_f_review_for_source,
    _redact_git_revision_fields,
    _replace_hash_references,
    anonymize_gate_f_review_payload,
    anonymize_tmlr_source,
    author_identifiers,
    build_release_artifacts,
    expected_source_release_entries,
    validate_package_zip,
    validate_pdf,
    write_package_zip,
)
from src.submission import REQUIRED_RELEASE_FILES, _tracked_digest, _validate_publication_artifacts


@pytest.mark.parametrize("matching_commit", [True, False])
def test_expected_source_release_entries_bind_gate_f_to_current_commit(
    tmp_path, monkeypatch, matching_commit
) -> None:
    commit = "a" * 40
    (tmp_path / "paper").mkdir()
    (tmp_path / "paper/main.pdf").write_bytes(_pdf_bytes())
    (tmp_path / "README.md").write_bytes(b"Source snapshot\n")
    analysis_dir = tmp_path / "results/analysis"
    analysis_dir.mkdir(parents=True)
    (analysis_dir / "gate-f-review.json").write_text(
        json.dumps({"status": "passed", "git_head": commit if matching_commit else "b" * 40}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "src.release_artifacts._source_tree_state",
        lambda _root, allow_dirty: (commit, True, ["README.md", "paper/main.pdf"]),
    )

    entries = expected_source_release_entries(tmp_path)

    assert entries["README.md"] == b"Source snapshot\n"
    assert entries["paper/main.pdf"] == (tmp_path / "paper/main.pdf").read_bytes()
    evidence = json.loads(entries["results/analysis/gate-f-review.json"])
    assert evidence["status"] == ("passed" if matching_commit else "not_run")
    if matching_commit:
        assert evidence["git_head"] == commit


def _official_tmlr_styles() -> dict[str, bytes]:
    return {
        relative: Path(relative).read_bytes()
        for relative in ("paper/tmlr.sty", "paper/tmlr.bst", "paper/fancyhdr.sty")
    }


def _pdf_bytes(
    metadata: dict[str, str] | None = None,
    *,
    page_text: str | None = None,
    compress_text: bool = False,
) -> bytes:
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    if page_text is not None:
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        font_reference = writer._add_object(font)
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_reference})}
        )
        content = f"BT /F1 12 Tf 72 720 Td ({page_text}) Tj ET".encode("ascii")
        if compress_text:
            stream = EncodedStreamObject()
            stream._data = compress(content)
            stream[NameObject("/Filter")] = NameObject("/FlateDecode")
        else:
            stream = DecodedStreamObject()
            stream.set_data(content)
        page[NameObject("/Contents")] = writer._add_object(stream)
    if metadata:
        writer.add_metadata(metadata)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def _anonymous_pdf_bytes() -> bytes:
    return _pdf_bytes({"/Author": "Anonymous", "/Title": "Blinded paper"})


def test_tracked_digest_is_order_sensitive_and_null_delimited() -> None:
    assert _tracked_digest(["a.txt", "b.txt"]) != _tracked_digest(["b.txt", "a.txt"])
    assert _tracked_digest(["a.txt", "b.txt"]) == _tracked_digest(["a.txt", "b.txt"])


def test_gate_g_source_module_is_in_the_repository() -> None:
    assert Path("src/submission.py").is_file()
    assert "src/sensitivity.py" in TMLR_SOURCE_FILES
    assert "docs/reproducibility.md" in TMLR_SOURCE_FILES
    assert "docs/analysis-implementation-note.md" in TMLR_SOURCE_FILES
    assert "requirements-release.txt" in REQUIRED_RELEASE_FILES
    assert "requirements-release.txt" in TMLR_SOURCE_FILES


def test_anonymous_tmlr_source_uses_double_blind_style_mode() -> None:
    source = Path("paper/main.tex").read_text(encoding="utf-8")

    anonymous = anonymize_tmlr_source(source)

    assert r"\usepackage{tmlr}" in anonymous
    assert r"\@acceptedtrue" not in anonymous
    assert r"\author{" not in anonymous
    assert "Acknowledgments" not in anonymous
    assert "pdfauthor={Anonymous}" in anonymous
    assert (
        "pdftitle={An Auditable Benchmark of Imbalance Handling for Multiclass Tabular Classification}"
        in anonymous
    )
    assert "DE-ANONYMIZED AVAILABILITY" not in anonymous
    assert "github.com/icecold009" not in anonymous


def test_anonymous_evidence_redacts_revision_and_rebinds_manifest_hashes() -> None:
    payload = {
        "git_head": "a" * 40,
        "input": {"path": "results/full-run/benchmark_manifest.json", "sha256": "old"},
        "nested": {"source_commit": "b" * 40},
    }

    _redact_git_revision_fields(payload)
    _replace_hash_references(
        payload,
        {"results/full-run/benchmark_manifest.json": "c" * 64},
    )

    assert payload["git_head"] is None
    assert payload["nested"]["source_commit"] is None
    assert payload["input"]["sha256"] == "c" * 64


def test_anonymous_package_manifest_withholds_commit_and_checks_identity(tmp_path) -> None:
    commit = "a" * 40
    package = tmp_path / "supplement.zip"
    write_package_zip(
        package,
        {
            "README.md": b"Anonymous supplementary material.\n",
            "paper/main.tex": b"\\usepackage{tmlr}\n",
            "paper/main.pdf": _anonymous_pdf_bytes(),
            "paper/references.bib": b"",
            **_official_tmlr_styles(),
        },
        source_commit=None,
        source_tree_clean=True,
        anonymous=True,
    )

    evidence = validate_package_zip(package, expected_commit=commit, anonymous=True)

    assert evidence["size_bytes"] == package.stat().st_size
    with zipfile.ZipFile(package) as archive:
        expected_entries = {
            name: archive.read(name)
            for name in archive.namelist()
            if name != "package-manifest.json"
        }
    expected_entries["README.md"] += b"stale"
    with pytest.raises(RuntimeError, match="stale for current release input"):
        validate_package_zip(
            package,
            expected_commit=commit,
            anonymous=True,
            expected_entries=expected_entries,
        )
    with pytest.raises(ValueError, match="withhold the source commit"):
        write_package_zip(
            tmp_path / "leaky.zip",
            {
                "README.md": b"Anonymous supplementary material.\n",
                "paper/main.tex": b"\\usepackage{tmlr}\n",
                "paper/main.pdf": _anonymous_pdf_bytes(),
                "paper/references.bib": b"",
                **_official_tmlr_styles(),
            },
            source_commit=commit,
            source_tree_clean=True,
            anonymous=True,
        )

    with pytest.raises(RuntimeError, match="source commit"):
        leaked_package = tmp_path / "leaked.zip"
        write_package_zip(
            leaked_package,
            {
                "README.md": f"Anonymous package from {commit}.\n".encode(),
                "paper/main.tex": b"\\usepackage{tmlr}\n",
                "paper/main.pdf": _anonymous_pdf_bytes(),
                "paper/references.bib": b"",
                **_official_tmlr_styles(),
            },
            source_commit=None,
            source_tree_clean=True,
            anonymous=True,
        )
        validate_package_zip(
            leaked_package,
            expected_commit=commit,
            anonymous=True,
        )


def test_author_identifiers_are_extracted_for_anonymous_artifact_scans() -> None:
    source = Path("paper/main.tex").read_text(encoding="utf-8")

    identifiers = author_identifiers(source)

    assert len(identifiers) == 2
    assert all(identifiers)


def test_gate_f_blind_copy_removes_git_and_machine_identifiers() -> None:
    payload = {
        "git_head": "a" * 40,
        "status": "passed",
        "source_snapshot": {
            "archive_sha256": "b" * 64,
            "tracked_file_count": 12,
            "tracked_file_set_sha256": "c" * 64,
            "checks": [
                {
                    "command": [
                        chr(92).join(("C:", "Users", "private-reviewer", "Python", "python.exe")),
                        "verify_gate_f.py",
                    ],
                    "exit_code": 0,
                    "stderr": "",
                    "stdout": ("Contact " + "researcher@" + "example.org; revision " + "d" * 40),
                }
            ],
            "workflow": {"pytest": "passed"},
        },
    }

    anonymous = anonymize_gate_f_review_payload(payload)
    encoded = json.dumps(anonymous)

    assert anonymous["git_head"] is None
    assert anonymous["source_snapshot"]["archive_sha256"] is None
    assert anonymous["source_snapshot"]["tracked_file_set_sha256"] is None
    assert anonymous["source_snapshot"]["tracked_file_count"] == 12
    assert anonymous["source_snapshot"]["checks"][0]["exit_code"] == 0
    assert "private-reviewer" not in encoded
    assert ("researcher@" + "example.org") not in encoded
    assert "a" * 40 not in encoded
    assert "b" * 64 not in encoded
    assert "c" * 64 not in encoded
    assert "d" * 40 not in encoded


def test_dirty_preview_does_not_copy_a_historical_gate_f_pass(tmp_path) -> None:
    review_path = tmp_path / "results/analysis/gate-f-review.json"
    review_path.parent.mkdir(parents=True)
    review_path.write_text(
        json.dumps({"schema_version": "gate-f-review-v1", "status": "passed", "git_head": "a" * 40}),
        encoding="utf-8",
    )

    preview_review = _gate_f_review_for_source(tmp_path, "a" * 40, source_tree_clean=False)

    assert preview_review["status"] == "not_run"
    assert preview_review["source_tree_clean"] is False
    assert "dirty" in preview_review["boundary"]
    assert "git_head" not in preview_review


def test_anonymous_package_rejects_modified_tmlr_style_bytes(tmp_path) -> None:
    entries = {
        "README.md": b"Anonymous supplementary material.\n",
        "paper/main.tex": b"\\usepackage{tmlr}\n",
        "paper/main.pdf": _anonymous_pdf_bytes(),
        "paper/references.bib": b"",
        **_official_tmlr_styles(),
    }
    entries["paper/tmlr.sty"] += b"% modified\n"
    package = tmp_path / "changed-style.zip"
    write_package_zip(
        package,
        entries,
        source_commit=None,
        source_tree_clean=True,
        anonymous=True,
    )

    with pytest.raises(RuntimeError, match="style asset"):
        validate_package_zip(
            package,
            expected_commit="a" * 40,
            anonymous=True,
        )


def test_pdf_validator_rejects_incomplete_files(tmp_path) -> None:
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"not a PDF")

    with pytest.raises(RuntimeError, match="malformed or incomplete"):
        validate_pdf(path)

    path.write_bytes(_anonymous_pdf_bytes())
    assert validate_pdf(path)["size_bytes"] == path.stat().st_size


@pytest.mark.parametrize(
    "member",
    ("C:/escape.txt", "C:relative.txt", "../escape.txt", "/rooted.txt", "nested\\escape.txt"),
)
def test_package_writer_rejects_unsafe_member_paths(tmp_path, member) -> None:
    with pytest.raises(ValueError, match="unsafe package member path"):
        write_package_zip(
            tmp_path / "unsafe.zip",
            {member: b"no"},
            source_commit="a" * 40,
            source_tree_clean=True,
            anonymous=False,
        )


def test_package_validator_rejects_windows_drive_member_paths(tmp_path) -> None:
    package = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(package, mode="w") as archive:
        archive.writestr("C:/escape.txt", b"no")

    with pytest.raises(RuntimeError, match="unsafe member path"):
        validate_package_zip(package, expected_commit="a" * 40, anonymous=False)


def test_artifact_builder_rejects_output_outside_repository() -> None:
    root = Path(__file__).resolve().parents[1]
    outside = root.parent / "outside-artifact-output"

    with pytest.raises(ValueError, match="inside the repository"):
        build_release_artifacts(
            root,
            outside,
            allow_dirty=True,
            stage_only=True,
        )


def test_anonymous_pdf_validator_requires_blind_metadata(tmp_path) -> None:
    path = tmp_path / "paper.pdf"
    path.write_bytes(_anonymous_pdf_bytes())

    assert validate_pdf(path, anonymous=True)["size_bytes"] == path.stat().st_size
    path.write_bytes(_pdf_bytes({"/Author": "Some Author", "/Title": "A blinded paper"}))
    with pytest.raises(RuntimeError, match="author metadata"):
        validate_pdf(path, anonymous=True)


def test_anonymous_pdf_scans_text_inside_compressed_page_streams(tmp_path) -> None:
    path = tmp_path / "paper.pdf"
    identifying_name = "Review Author"
    pdf = _pdf_bytes(
        {"/Author": "Anonymous", "/Title": "Blinded paper"},
        page_text=identifying_name,
        compress_text=True,
    )
    assert identifying_name.encode() not in pdf
    path.write_bytes(pdf)

    with pytest.raises(RuntimeError, match="identifying author metadata"):
        validate_pdf(path, anonymous=True, forbidden_terms=(identifying_name,))


def test_anonymous_package_scans_bibliography_for_author_identifiers(tmp_path) -> None:
    package = tmp_path / "supplement.zip"
    author_name = "Jöhn Döe"
    write_package_zip(
        package,
        {
            "README.md": b"Anonymous supplementary material.\n",
            "paper/main.tex": b"\\usepackage{tmlr}\n",
            "paper/main.pdf": _anonymous_pdf_bytes(),
            "paper/references.bib": (f"@article{{self, author = {{{author_name}}}}}\n".encode()),
            **_official_tmlr_styles(),
        },
        source_commit=None,
        source_tree_clean=True,
        anonymous=True,
    )

    with pytest.raises(RuntimeError, match="identifying data in paper/references.bib"):
        validate_package_zip(
            package,
            expected_commit="a" * 40,
            anonymous=True,
            forbidden_terms=(author_name.upper(),),
        )


@pytest.mark.parametrize("artifact_kind", ("pdf", "zip"))
def test_artifact_validator_rejects_symlinked_parent_paths(
    tmp_path, monkeypatch, artifact_kind
) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    path = artifact_dir / ("paper.pdf" if artifact_kind == "pdf" else "source.zip")
    if artifact_kind == "pdf":
        path.write_bytes(_anonymous_pdf_bytes())
    else:
        write_package_zip(
            path,
            {"README.md": b"public release\n"},
            source_commit="a" * 40,
            source_tree_clean=True,
            anonymous=False,
        )

    actual_is_symlink = Path.is_symlink
    monkeypatch.setattr(
        Path,
        "is_symlink",
        lambda candidate: candidate == artifact_dir or actual_is_symlink(candidate),
    )

    with pytest.raises(RuntimeError, match="symbolic link in its path"):
        if artifact_kind == "pdf":
            validate_pdf(path)
        else:
            validate_package_zip(path, expected_commit="a" * 40, anonymous=False)


def test_gate_g_validates_and_hashes_all_four_publication_artifacts(tmp_path, monkeypatch) -> None:
    root = tmp_path / "repo"
    paper = root / "paper"
    paper.mkdir(parents=True)
    author_email = "test-author" + "@" + "example.org"
    author_source = r"\author{Test Author\\" + r"\texttt{" + author_email + "}}\n"
    (paper / "main.tex").write_text(
        author_source,
        encoding="utf-8",
    )
    commit = "a" * 40
    paths = {
        "deanonymized_release_pdf": paper / "main.pdf",
        "source_release_zip": root / "output/release" / f"paper-{commit[:7]}.zip",
        "tmlr_anonymous_pdf": root / "output/submission/tmlr-anonymous-submission.pdf",
        "tmlr_anonymous_supplement_zip": root / "output/submission/tmlr-anonymous-supplement.zip",
    }
    release_pdf = _pdf_bytes()
    anonymous_pdf = _anonymous_pdf_bytes()
    paths["deanonymized_release_pdf"].write_bytes(release_pdf)
    paths["tmlr_anonymous_pdf"].parent.mkdir(parents=True)
    paths["tmlr_anonymous_pdf"].write_bytes(anonymous_pdf)

    public_entries = {
        "README.md": b"Public release package.\n",
        "paper/main.pdf": release_pdf,
        **_official_tmlr_styles(),
    }
    write_package_zip(
        paths["source_release_zip"],
        public_entries,
        source_commit=commit,
        source_tree_clean=True,
        anonymous=False,
    )
    anonymous_entries = {
        "README.md": b"Anonymous supplementary material.\n",
        "paper/main.tex": b"\\usepackage{tmlr}\n",
        "paper/main.pdf": anonymous_pdf,
        "paper/references.bib": b"",
        **_official_tmlr_styles(),
    }
    write_package_zip(
        paths["tmlr_anonymous_supplement_zip"],
        anonymous_entries,
        source_commit=None,
        source_tree_clean=True,
        anonymous=True,
    )
    monkeypatch.setattr("src.submission.artifact_paths", lambda _root, _commit: paths)
    monkeypatch.setattr(
        "src.submission.expected_source_release_entries",
        lambda _root: public_entries,
    )
    monkeypatch.setattr(
        "src.submission.expected_tmlr_supplement_entries",
        lambda _root, _pdf: anonymous_entries,
    )

    evidence = _validate_publication_artifacts(root, commit)

    assert set(evidence) == set(paths)
    for name, path in paths.items():
        assert evidence[name]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert evidence[name]["path"] == path.relative_to(root).as_posix()

    paths["tmlr_anonymous_pdf"].unlink()
    with pytest.raises(RuntimeError, match="publication artifact is missing"):
        _validate_publication_artifacts(root, commit)
