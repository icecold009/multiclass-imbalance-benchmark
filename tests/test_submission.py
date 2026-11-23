import hashlib
import json
import shutil
import zipfile
from io import BytesIO
from pathlib import Path
from zlib import compress

import pandas as pd
import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, EncodedStreamObject, NameObject

from scripts import run_publication_addenda as publication_addenda
from scripts.build_release_artifacts import default_output_root
from src.release_artifacts import (
    PUBLICATION_EVIDENCE_FILES,
    TMLR_IDENTITY_SCAN_EXEMPT_FILES,
    TMLR_SOURCE_FILES,
    TMLR_STYLE_FIXTURE_ORIGINS,
    _anonymous_environment_and_benchmark_entries,
    _anonymous_exact_calibration_entries,
    _duplicate_profile_sensitivity_entries,
    _gate_f_review_for_source,
    _redact_git_revision_fields,
    _replace_hash_references,
    _supplement_entries,
    _validate_anonymous_tmlr_identity,
    _validate_tmlr_fixture_styles,
    anonymize_gate_f_review_payload,
    anonymize_tmlr_source,
    author_identifiers,
    build_release_artifacts,
    expected_source_release_entries,
    repository_identifiers,
    validate_package_zip,
    validate_pdf,
    write_package_zip,
)
from src.submission import (
    REQUIRED_RELEASE_FILES,
    _tracked_digest,
    _validate_publication_artifacts,
    _validate_submission_archive_paths,
)

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "submission"
ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "path",
    [
        "artifacts/environment/v2/metadata.json",
        "artifacts/environment/v2/pip-check.txt",
        "artifacts/environment/v2/pip-freeze.txt",
    ],
)
def test_gate_g_archive_allows_committed_v2_provenance(path: str) -> None:
    _validate_submission_archive_paths([path])


@pytest.mark.parametrize(
    "path",
    [
        "artifacts/environment/metadata.json",
        "artifacts/environment/v2/unapproved.json",
        "artifacts/runs/scope-lock.json",
        "results/full-run/benchmark_results.csv",
        "data/raw/dataset.csv",
        "data/processed/dataset.csv",
    ],
)
def test_gate_g_archive_rejects_other_generated_paths(path: str) -> None:
    with pytest.raises(RuntimeError, match="submission archive contains generated data"):
        _validate_submission_archive_paths([path])


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
    sensitivity_dir = tmp_path / "results/duplicate-profile-sensitivity"
    sensitivity_dir.mkdir(parents=True)
    sensitivity_output = b"dataset,score\ncmc,0.5\n"
    (sensitivity_dir / "grouped_results.csv").write_bytes(sensitivity_output)
    (sensitivity_dir / "sensitivity_manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "outputs": {
                    "grouped_results.csv": hashlib.sha256(sensitivity_output).hexdigest()
                },
            }
        ),
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
        f"paper/{name}": (FIXTURE_ROOT / "paper" / name).read_bytes()
        for name in ("tmlr.sty", "tmlr.bst", "fancyhdr.sty")
    }


@pytest.mark.parametrize(
    ("fixture_path", "official_path"), TMLR_STYLE_FIXTURE_ORIGINS.items()
)
def test_anonymous_package_exempts_only_hash_verified_style_fixtures(
    fixture_path: str, official_path: str
) -> None:
    fixture_content = (ROOT / fixture_path).read_bytes()

    assert fixture_path in TMLR_IDENTITY_SCAN_EXEMPT_FILES
    assert fixture_content == (ROOT / official_path).read_bytes()
    _validate_tmlr_fixture_styles({fixture_path: fixture_content})

    with pytest.raises(RuntimeError, match="reviewed official asset"):
        _validate_tmlr_fixture_styles({fixture_path: fixture_content + b"modified"})


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


def _write_synthetic_calibration_inputs(root: Path, monkeypatch) -> str:
    monkeypatch.setattr(
        publication_addenda,
        "EXPECTED_DATASETS",
        ("synthetic-dataset-a", "synthetic-dataset-b", "synthetic-dataset-c"),
    )
    analysis_dir = root / "results" / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    values = [0.5] * 7 + [0.1, 0.9]
    rows = [
        {
            "feature_type": "numeric",
            "classifier": "xgboost",
            "dataset": dataset,
            "condition": condition,
            "macro_f1": value,
        }
        for dataset in publication_addenda.EXPECTED_DATASETS
        for condition, value in zip(publication_addenda.CONDITIONS, values, strict=True)
    ]
    cell_means = analysis_dir / "cell_means.csv"
    pd.DataFrame(rows).to_csv(cell_means, index=False, lineterminator="\n")
    digest = hashlib.sha256(cell_means.read_bytes()).hexdigest()
    manifest = {
        "artifacts": {
            publication_addenda.CELL_MEANS.as_posix(): {
                "path": publication_addenda.CELL_MEANS.as_posix(),
                "sha256": digest,
            }
        }
    }
    (analysis_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(publication_addenda, "EXPECTED_CELL_MEANS_SHA256", digest)
    return digest


def _synthetic_supplement_entries(tmp_path: Path, monkeypatch) -> dict[str, bytes]:
    for relative in PUBLICATION_EVIDENCE_FILES:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    (tmp_path / "LICENSE").write_bytes((ROOT / "LICENSE").read_bytes())

    source_text = (FIXTURE_ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    anonymous_source = anonymize_tmlr_source(source_text)
    author_source = ROOT / "paper" / "main.tex"
    if author_source.is_file() and r"\author{" in author_source.read_text(encoding="utf-8"):
        identity_source = author_source.read_text(encoding="utf-8")
    else:
        identity_source = (
            r"\author{" + "Synthetic " + "Reviewer" + r"\\\texttt{"
            + "synthetic-reviewer"
            + "@"
            + "example.invalid"
            + "}}"
        )
    forbidden_terms = author_identifiers(identity_source) + repository_identifiers(ROOT)

    scope_lock = tmp_path / "artifacts" / "runs" / "scope-lock.json"
    scope_lock.parent.mkdir(parents=True, exist_ok=True)
    scope_lock.write_text("{}\n", encoding="utf-8")

    environment_name = "artifacts/environment/metadata.json"
    environment_path = tmp_path / environment_name
    environment_path.parent.mkdir(parents=True, exist_ok=True)
    environment_path.write_text(
        json.dumps({"git_commit": "a" * 40, "git_branch": "synthetic-branch"}) + "\n",
        encoding="utf-8",
    )
    run_manifest = tmp_path / "results" / "full-run" / "benchmark_manifest.json"
    run_manifest.parent.mkdir(parents=True, exist_ok=True)
    run_manifest.write_text(
        json.dumps(
            {
                "git_commit": "b" * 40,
                "frozen_inputs": {
                    environment_name: {
                        "path": environment_name,
                        "sha256": hashlib.sha256(environment_path.read_bytes()).hexdigest(),
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_manifest.parent / "benchmark_results.csv").write_text("dataset,score\nfixture,0.5\n")
    (run_manifest.parent / "benchmark_failures.csv").write_text("dataset,reason\n")

    analysis_dir = tmp_path / "results" / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    cell_means_sha256 = _write_synthetic_calibration_inputs(tmp_path, monkeypatch)
    cell_means_path = analysis_dir / "cell_means.csv"
    assert hashlib.sha256(cell_means_path.read_bytes()).hexdigest() == cell_means_sha256
    publication_addenda.run_addendum(
        tmp_path, tmp_path / "results" / "publication-addenda" / "v4"
    )

    for name in ("gate-d-review.json", "gate-e-review.json", "gate-f-review.json"):
        (analysis_dir / name).write_text('{"status": "passed", "git_head": "c"}\n')

    grouped_dir = tmp_path / "results" / "duplicate-profile-sensitivity"
    grouped_dir.mkdir(parents=True, exist_ok=True)
    grouped_output = b"dataset,score\nfixture,0.5\n"
    (grouped_dir / "grouped_results.csv").write_bytes(grouped_output)
    (grouped_dir / "sensitivity_manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "outputs": {
                    "grouped_results.csv": hashlib.sha256(grouped_output).hexdigest()
                },
            }
        ),
        encoding="utf-8",
    )

    entries = _supplement_entries(
        tmp_path,
        anonymous_source,
        forbidden_terms,
        _anonymous_pdf_bytes(),
        "d" * 40,
        False,
    )
    _validate_anonymous_tmlr_identity(entries, forbidden_terms)
    return entries


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
    assert "requirements-analysis.lock" in TMLR_SOURCE_FILES
    assert "src/publication_evidence.py" in TMLR_SOURCE_FILES
    assert "src/source_snapshot.py" in TMLR_SOURCE_FILES
    assert "src/source_snapshot.py" in PUBLICATION_EVIDENCE_FILES
    assert "paper/README-anonymous.md" in TMLR_SOURCE_FILES
    assert "paper/README.md" not in TMLR_SOURCE_FILES
    assert "tests/test_v2_registry.py" not in TMLR_SOURCE_FILES
    assert "scripts/replay_publication_analysis.py" in TMLR_SOURCE_FILES
    assert "scripts/run_publication_addenda.py" in TMLR_SOURCE_FILES
    assert "tests/test_sensitivity_exact.py" in TMLR_SOURCE_FILES
    assert "docs/duplicate-profile-policy.md" in TMLR_SOURCE_FILES
    assert "docs/duplicate-profile-sensitivity-results.md" in TMLR_SOURCE_FILES
    assert "scripts/run_duplicate_profile_sensitivity.py" in TMLR_SOURCE_FILES
    assert "src/duplicate_profile_sensitivity.py" in TMLR_SOURCE_FILES
    assert "tests/test_duplicate_profile_sensitivity.py" in TMLR_SOURCE_FILES
    assert "tests/fixtures/historical-v2/dataset_registry_v2.csv" in TMLR_SOURCE_FILES
    assert "tests/fixtures/submission/paper/main.tex" in TMLR_SOURCE_FILES
    assert not any(name.startswith(("data/raw/", "results/")) for name in TMLR_SOURCE_FILES)
    assert all(Path(name).is_file() for name in TMLR_SOURCE_FILES)
    assert all(Path(name).is_file() for name in PUBLICATION_EVIDENCE_FILES)


def test_anonymous_tmlr_source_uses_double_blind_style_mode() -> None:
    source = (FIXTURE_ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    source = source.replace("fixture at example dot org", "fixture@" + "example.org", 1)
    source = source.replace(
        r"\begin{document}",
        "The source and evidence package prepared for this draft is available from the author on request.\n"
        + r"\begin{document}",
        1,
    )

    anonymous = anonymize_tmlr_source(source)

    assert r"\usepackage{tmlr}" in anonymous
    assert r"\@acceptedtrue" not in anonymous
    assert r"\author{" not in anonymous
    assert "Acknowledgments" not in anonymous
    assert "pdfauthor={Anonymous}" in anonymous
    assert "pdftitle={Synthetic TMLR Test}" in anonymous
    assert "pdfauthor={Fixture Author}" not in anonymous
    assert "Fixture Author" not in anonymous
    assert "fixture@" + "example.org" not in anonymous
    assert "DE-ANONYMIZED AVAILABILITY" not in anonymous
    assert "available from the author on request" not in anonymous
    assert "included in the anonymous supplementary material" in anonymous


def test_repository_identifiers_are_host_qualified_and_credential_free(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.release_artifacts._git_output",
        lambda _root, *_args: "https://token:secret"
        + "@github.com/"
        + "synthetic"
        + "-owner/project.git",
    )
    identifiers = repository_identifiers(ROOT)
    assert "github.com/synthetic-owner" in identifiers
    assert "https://github.com/synthetic-owner" in identifiers
    assert "https://github.com/synthetic-owner/project.git" in identifiers
    assert not any("token" in identifier or "secret" in identifier for identifier in identifiers)
    assert "synthetic-owner" not in identifiers
    assert "project" not in identifiers

    monkeypatch.setattr(
        "src.release_artifacts._git_output",
        lambda _root, *_args: "git" + "@" + "github.com:" + "synthetic"
        + "-owner/project.git",
    )
    ssh_identifiers = repository_identifiers(ROOT)
    assert "github.com:synthetic-owner" in ssh_identifiers
    assert "git" + "@" + "github.com:" + "synthetic" + "-owner" in ssh_identifiers
    assert "synthetic-owner" not in ssh_identifiers

    with pytest.raises(RuntimeError, match="identifying data"):
        _validate_anonymous_tmlr_identity(
            {"README.md": b"See https://github.com/" + b"synthetic" + b"-owner for code."},
            ssh_identifiers,
        )


@pytest.mark.parametrize(
    "origin",
    (
        "https://github.com/" + "ml" + "/source.git",
        "/tmp/source.git",
        "file:///tmp/source.git",
        r"C:\\work\\source.git",
    ),
)
def test_repository_origin_identity_scan_is_safe_for_source_inventory(
    monkeypatch, origin: str
) -> None:
    monkeypatch.setattr(
        "src.release_artifacts._git_output", lambda _root, *_args: origin
    )
    identifiers = repository_identifiers(ROOT)
    if origin.startswith("https:"):
        assert "https://github.com/" + "ml" in identifiers
        assert "ml" not in identifiers
    else:
        assert identifiers == ()
    inventory = {
        relative: (ROOT / relative).read_bytes()
        for relative in PUBLICATION_EVIDENCE_FILES
    }
    _validate_anonymous_tmlr_identity(inventory, identifiers)


def test_supplement_inventory_omits_named_paper_readme() -> None:
    assert "paper/README-anonymous.md" in PUBLICATION_EVIDENCE_FILES
    assert "paper/README.md" not in PUBLICATION_EVIDENCE_FILES
    anonymous_readme = (ROOT / "paper/README-anonymous.md").read_bytes()
    named_pdf = f"main4{chr(46)}pdf"
    assert named_pdf.encode() not in anonymous_readme
    named_pdf_path = ROOT / "paper" / named_pdf
    if named_pdf_path.is_file():
        assert hashlib.sha256(named_pdf_path.read_bytes()).hexdigest().encode() not in anonymous_readme


def test_anonymous_only_staging_needs_no_named_pdf_or_release_zip(tmp_path, monkeypatch) -> None:
    root = tmp_path / "repo"
    paper = root / "paper"
    paper.mkdir(parents=True)
    shutil.copyfile(FIXTURE_ROOT / "paper" / "main.tex", paper / "main.tex")
    for relative in (
        "paper/references.bib",
        "paper/math_commands.tex",
        "paper/tmlr.sty",
        "paper/tmlr.bst",
        "paper/fancyhdr.sty",
    ):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"synthetic release fixture")

    monkeypatch.setattr(
        "src.release_artifacts._source_tree_state",
        lambda _root, _allow_dirty: ("a" * 40, False, []),
    )
    monkeypatch.setattr("src.release_artifacts._validate_tmlr_styles", lambda _entries: None)
    monkeypatch.setattr("src.release_artifacts.author_identifiers", lambda _source: ("Fixture Author",))
    monkeypatch.setattr("src.release_artifacts.repository_identifiers", lambda _root: ())

    output_root = default_output_root(root, anonymous_only=True)
    result = build_release_artifacts(
        root,
        output_root,
        allow_dirty=True,
        stage_only=True,
        anonymous_only=True,
    )

    assert output_root == root / "output" / "publication-v4-anonymous-evidence"
    assert result["anonymous_only"] is True
    assert Path(result["anonymous_source"]).is_file()
    assert not (paper / "main.pdf").exists()
    assert not (output_root / "release").exists()
    assert not (root / "output" / "submission").exists()


def test_anonymous_supplement_rebinds_exact_calibration_manifest(tmp_path, monkeypatch) -> None:
    _write_synthetic_calibration_inputs(tmp_path, monkeypatch)
    analysis_manifest = (tmp_path / "results/analysis/analysis_manifest.json").read_bytes()
    publication_addenda.run_addendum(
        tmp_path, tmp_path / "results/publication-addenda/v4"
    )
    entries = _anonymous_exact_calibration_entries(tmp_path, analysis_manifest)
    entries["results/analysis/analysis_manifest.json"] = analysis_manifest
    analysis_manifest = entries["results/analysis/analysis_manifest.json"]
    addendum_path = "results/publication-addenda/v4/publication_addendum_manifest.json"
    addendum = json.loads(entries[addendum_path])
    csv_path = "results/publication-addenda/v4/exact_friedman_permutation.csv"
    csv_bytes = entries[csv_path]
    cell_means = (tmp_path / "results/analysis/cell_means.csv").read_bytes()

    assert addendum["input"]["analysis_manifest"]["sha256"] == hashlib.sha256(
        analysis_manifest
    ).hexdigest()
    assert addendum["input"]["cell_means"]["sha256"] == hashlib.sha256(
        cell_means
    ).hexdigest()
    assert addendum["outputs"]["exact_friedman_permutation.csv"]["sha256"] == hashlib.sha256(
        csv_bytes
    ).hexdigest()
    assert addendum["result"]["exact_p_value_fraction"] == "1/5184"


def test_anonymous_supplement_identity_scan_excludes_named_pdf_and_owner(tmp_path, monkeypatch) -> None:
    entries = _synthetic_supplement_entries(tmp_path, monkeypatch)
    paper_source = (ROOT / "paper/main.tex").read_text(encoding="utf-8")
    if r"\author{" in paper_source:
        author_terms = author_identifiers(paper_source)
    else:
        synthetic_identity = (
            r"\author{" + "Synthetic " + "Reviewer" + r"\\\texttt{"
            + "synthetic-reviewer"
            + "@"
            + "example.invalid"
            + "}}"
        )
        author_terms = author_identifiers(synthetic_identity)
    forbidden = author_terms + repository_identifiers(ROOT)
    pdf_path = ROOT / "paper" / f"main4{chr(46)}pdf"
    pdf_digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest() if pdf_path.is_file() else None
    forbidden_name = f"main4{chr(46)}pdf"
    for member, content in entries.items():
        searchable = member.encode() + b"\n" + content
        assert forbidden_name.encode() not in searchable
        if pdf_digest is not None:
            assert pdf_digest.encode() not in searchable
        assert not any(term.encode().lower() in searchable.lower() for term in forbidden)
    _validate_anonymous_tmlr_identity(entries, forbidden)


def test_anonymous_environment_rebinds_nested_benchmark_hash(tmp_path) -> None:
    environment_name = "artifacts/environment/metadata.json"
    benchmark_name = "results/full-run/benchmark_manifest.json"
    environment_path = tmp_path / environment_name
    environment_path.parent.mkdir(parents=True)
    environment_path.write_text(
        json.dumps({"git_commit": "a" * 40, "git_branch": "synthetic-branch"}) + "\n",
        encoding="utf-8",
    )
    original_environment_sha = hashlib.sha256(environment_path.read_bytes()).hexdigest()
    benchmark_path = tmp_path / benchmark_name
    benchmark_path.parent.mkdir(parents=True)
    benchmark_path.write_text(
        json.dumps(
            {
                "git_commit": "b" * 40,
                "frozen_inputs": {
                    environment_name: {"path": environment_name, "sha256": original_environment_sha}
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    entries, _ = _anonymous_environment_and_benchmark_entries(tmp_path)
    environment_bytes = entries[environment_name]
    benchmark_payload = json.loads(entries[benchmark_name])

    assert json.loads(environment_bytes)["git_commit"] is None
    assert json.loads(environment_bytes)["git_branch"] is None
    assert benchmark_payload["git_commit"] is None
    assert benchmark_payload["frozen_inputs"][environment_name]["sha256"] == hashlib.sha256(
        environment_bytes
    ).hexdigest()
    assert benchmark_payload["frozen_inputs"][environment_name]["sha256"] != original_environment_sha


def test_duplicate_profile_sensitivity_package_entries_verify_manifest_hashes(tmp_path) -> None:
    output_dir = tmp_path / "results" / "duplicate-profile-sensitivity"
    output_dir.mkdir(parents=True)
    output = b"dataset,score\ncmc,0.5\n"
    (output_dir / "grouped_results.csv").write_bytes(output)
    manifest = {
        "status": "complete",
        "outputs": {"grouped_results.csv": hashlib.sha256(output).hexdigest()},
    }
    (output_dir / "sensitivity_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    entries = _duplicate_profile_sensitivity_entries(tmp_path)

    assert entries["results/duplicate-profile-sensitivity/grouped_results.csv"] == output
    assert "results/duplicate-profile-sensitivity/sensitivity_manifest.json" in entries

    (output_dir / "grouped_results.csv").write_bytes(output + b"stale")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        _duplicate_profile_sensitivity_entries(tmp_path)


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
    source = (FIXTURE_ROOT / "paper" / "main.tex").read_text(encoding="utf-8")

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
