"""Build and validate the V1 source release and anonymous TMLR supplement."""

from __future__ import annotations

import hashlib
import io
import json
import re
import stat
import subprocess
import zipfile
from collections.abc import Mapping
from io import BytesIO
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from pypdf import PdfReader

PACKAGE_MANIFEST = "package-manifest.json"
TMLR_MAX_SUPPLEMENT_BYTES = 100 * 1024 * 1024
MAX_PUBLIC_PACKAGE_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
EMAIL_PATTERN = re.compile(rb"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
REVISION_PATTERN = re.compile(rb"(?i)\b[0-9a-f]{40}\b")
REVISION_TEXT_PATTERN = re.compile(r"(?i)\b[0-9a-f]{40}\b")

LOCAL_USER_PATH_PATTERN = re.compile(
    rb"(?i)(?:[A-Z]:[\\/]+Us"
    rb"ers[\\/]+[^\s]+|/h"
    rb"ome/[^\s]+)"
)

TMLR_SOURCE_FILES = (
    "config/stage0.yaml",
    "data/acquisition_manifest.csv",
    "data/dataset_feature_overrides.csv",
    "data/dataset_registry.csv",
    "docs/data-acquisition.md",
    "docs/analysis-implementation-note.md",
    "docs/environment.md",
    "docs/protocol-status.md",
    "docs/protocol.md",
    "docs/reproducibility.md",
    "docs/stage-0-feasibility.md",
    "docs/stage-0-runbook.md",
    "requirements.txt",
    "scripts/build_release_artifacts.py",
    "scripts/run_analysis.py",
    "scripts/verify_gate_d.py",
    "scripts/verify_gate_e.py",
    "scripts/verify_gate_f.py",
    "scripts/verify_gate_g.py",
    "requirements-release.txt",
    "src/analysis.py",
    "src/benchmark.py",
    "src/pilot.py",
    "src/posthoc_sensitivity.py",
    "src/release_artifacts.py",
    "src/reproducibility.py",
    "src/robustness.py",
    "src/sensitivity.py",
    "src/source_snapshot.py",
    "src/stage0.py",
    "src/stats.py",
    "src/submission.py",
    "tests/test_analysis.py",
    "tests/test_benchmark.py",
    "tests/test_pilot.py",
    "tests/test_posthoc_sensitivity.py",
    "tests/test_reproducibility.py",
    "tests/test_robustness.py",
    "tests/test_source_snapshot.py",
    "tests/test_stage0.py",
    "tests/test_stats.py",
    "tests/test_submission.py",
)
TMLR_PAPER_SUPPORT_FILES = (
    "paper/references.bib",
    "paper/math_commands.tex",
    "paper/tmlr.sty",
    "paper/tmlr.bst",
    "paper/fancyhdr.sty",
)
TMLR_IDENTITY_SCAN_EXEMPT_FILES = frozenset(
    {"paper/tmlr.sty", "paper/tmlr.bst", "paper/fancyhdr.sty"}
)
TMLR_STYLE_SHA256 = {
    "paper/tmlr.sty": "5137789fbcfcf6188e67f3ccf28363472997a1a9498ce382e4cf4bee10d542e9",
    "paper/tmlr.bst": "e02e75749c78e47301c8dd346459bb34ec5efd08aa3ae99109a01e71bd42e107",
    "paper/fancyhdr.sty": "2bcdf00b7ff35411e1fc3dece5e1489b467e9e455a71e437bf07c41f359ab584",
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _contains_identifier(data: bytes, identifier: str) -> bool:
    if not identifier:
        return False
    decoded = data.decode("utf-8", errors="ignore").casefold()
    return identifier.casefold() in decoded or identifier.encode("utf-8").lower() in data.lower()


def anonymize_tmlr_source(source: str) -> str:
    """Convert the de-anonymized manuscript to TMLR's double-blind review mode."""

    style_block = (
        "\\usepackage[preprint]{tmlr}\n"
        "\\makeatletter\n"
        "\\@acceptedtrue\n"
        "\\@preprinttrue\n"
        "\\makeatother"
    )
    if source.count(style_block) != 1:
        raise RuntimeError("paper source no longer has the expected de-anonymized TMLR style block")
    anonymous = source.replace(style_block, r"\usepackage{tmlr}", 1)

    author_block = re.compile(
        r"(?m)^% De-anonymized local draft\.[^\r\n]*\r?\n"
        r"% before any external submission\.\r?\n"
        r"\\author\{[^\r\n]*\}\r?\n"
    )
    anonymous, author_count = author_block.subn("", anonymous, count=1)
    if author_count != 1:
        raise RuntimeError("paper source no longer has the expected author block")

    availability = re.compile(
        r"(?ms)^% BEGIN DE-ANONYMIZED AVAILABILITY\s*\r?\n"
        r".*?^% END DE-ANONYMIZED AVAILABILITY\s*\r?\n?"
    )
    anonymous, availability_count = availability.subn("", anonymous, count=1)
    if availability_count != 1:
        raise RuntimeError(
            "paper source no longer has the expected de-anonymized availability block"
        )

    acknowledgments = re.compile(r"(?ms)^\\section\*\{Acknowledgments\}\s+.*?(?=^\\bibliography\{)")
    anonymous, acknowledgment_count = acknowledgments.subn("", anonymous, count=1)
    if acknowledgment_count != 1:
        raise RuntimeError("paper source no longer has the expected acknowledgments section")

    metadata = r"\hypersetup{hidelinks}"
    anonymous_metadata = (
        r"\hypersetup{hidelinks,pdfauthor={Anonymous},pdfcreator={},"
        r"pdftitle={An Auditable Benchmark of Imbalance Handling for Multiclass Tabular Classification}}"
    )
    if anonymous.count(metadata) != 1:
        raise RuntimeError("paper source no longer has the expected PDF metadata declaration")
    anonymous = anonymous.replace(metadata, anonymous_metadata, 1)

    if re.search(r"(?m)^\\author\s*\{", anonymous):
        raise RuntimeError("anonymous paper source still declares an author")
    if re.search(r"(?im)^\\section\*\{Acknowledgments\}", anonymous):
        raise RuntimeError("anonymous paper source still contains acknowledgments")
    if EMAIL_PATTERN.search(anonymous.encode("utf-8")):
        raise RuntimeError("anonymous paper source still contains an email address")
    if "DE-ANONYMIZED AVAILABILITY" in anonymous:
        raise RuntimeError("anonymous paper source still contains de-anonymized markers")
    if "\\@acceptedtrue" in anonymous or "[preprint]" in anonymous:
        raise RuntimeError("anonymous paper source is not using TMLR double-blind review mode")
    return anonymous


def author_identifiers(source: str) -> tuple[str, ...]:
    """Return identifying values from the one-line author block for leak scans."""

    match = re.search(r"(?m)^\\author\{([^\\{}]+)\\\\\\texttt\{([^{}]+)\}\}", source)
    if match is None:
        raise RuntimeError("paper source no longer has the expected one-line author block")
    return tuple(value.strip() for value in match.groups() if value.strip())


def repository_identifiers(root: Path) -> tuple[str, ...]:
    """Return local origin URL spellings that must not enter blind material."""

    try:
        remote = _git_output(root, "remote", "get-url", "origin").strip()
    except subprocess.CalledProcessError:
        return ()
    if not remote:
        return ()
    variants = {remote, remote.removesuffix(".git")}
    if remote.startswith("git@") and ":" in remote:
        host, repository = remote.split(":", 1)
        variants.add(f"https://{host[4:]}/{repository}")
        variants.add(f"https://{host[4:]}/{repository.removesuffix('.git')}")
    return tuple(sorted(value for value in variants if value))


def _git_output(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout


def _source_tree_state(root: Path, allow_dirty: bool) -> tuple[str, bool, list[str]]:
    commit = _git_output(root, "rev-parse", "HEAD").strip()
    status = _git_output(root, "status", "--porcelain", "--untracked-files=all")
    clean = not status.strip()
    if not clean and not allow_dirty:
        raise RuntimeError(
            "artifact packages require a clean source tree; use --allow-dirty only for a preview"
        )
    tracked = [name for name in _git_output(root, "ls-files", "-z").split("\x00") if name]
    if allow_dirty:
        untracked = [
            name
            for name in _git_output(root, "ls-files", "--others", "--exclude-standard", "-z").split(
                "\x00"
            )
            if name
        ]
        tracked.extend(name for name in untracked if name not in tracked)
    return commit, clean, sorted(tracked)


def _zip_entries(entries: dict[str, bytes]) -> bytes:
    """Encode a deterministic ZIP with normalized order, timestamps, and modes."""

    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name, content in sorted(entries.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return buffer.getvalue()


def _unsafe_member_name(name: str) -> bool:
    windows_path = PureWindowsPath(name)
    segments = name.split("/")
    return (
        not name
        or "\\" in name
        or PurePosixPath(name).is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or any(segment in {"", ".", ".."} for segment in segments)
    )


def _path_has_symlink(path: Path) -> bool:
    current = path
    while True:
        if current.is_symlink():
            return True
        if current.parent == current:
            return False
        current = current.parent


def write_package_zip(
    archive_path: Path,
    entries: dict[str, bytes],
    *,
    source_commit: str | None,
    source_tree_clean: bool,
    anonymous: bool,
) -> dict[str, Any]:
    """Write a deterministic package with a per-file integrity manifest."""

    if anonymous and source_commit is not None:
        raise ValueError("anonymous packages must withhold the source commit")
    if not anonymous and not source_commit:
        raise ValueError("public release packages must identify their source commit")
    manifest_name = PACKAGE_MANIFEST
    if manifest_name in entries:
        raise ValueError(f"{manifest_name} is reserved for the package manifest")
    for name in entries:
        if _unsafe_member_name(name):
            raise ValueError(f"unsafe package member path: {name!r}")
    manifest = {
        "schema_version": "v1-artifact-package-v1",
        "source_commit": source_commit,
        "source_commit_withheld": anonymous,
        "source_tree_clean": source_tree_clean,
        "anonymous": anonymous,
        "files": {name: _sha256(data) for name, data in sorted(entries.items())},
    }
    complete = dict(entries)
    complete[manifest_name] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    if _path_has_symlink(archive_path):
        raise ValueError(f"refusing to write through a symbolic link: {archive_path}")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if _path_has_symlink(archive_path):
        raise ValueError(f"refusing to write through a symbolic link: {archive_path}")
    archive_path.write_bytes(_zip_entries(complete))
    return {
        "path": archive_path.as_posix(),
        "sha256": _sha256(archive_path.read_bytes()),
        "size_bytes": archive_path.stat().st_size,
        "source_commit": source_commit,
        "anonymous": anonymous,
    }


def validate_pdf(
    path: Path,
    *,
    anonymous: bool = False,
    forbidden_terms: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Check basic PDF framing before the artifact is included in Gate G."""

    if not path.is_file() or _path_has_symlink(path):
        raise RuntimeError(f"PDF artifact is missing or has a symbolic link in its path: {path}")
    data = path.read_bytes()
    return _validate_pdf_bytes(data, str(path), anonymous, forbidden_terms)


def _validate_pdf_bytes(
    data: bytes,
    label: str,
    anonymous: bool,
    forbidden_terms: tuple[str, ...],
) -> dict[str, Any]:
    if len(data) < 100 or not data.startswith(b"%PDF-") or b"%%EOF" not in data[-2048:]:
        raise RuntimeError(f"PDF artifact is malformed or incomplete: {label}")

    try:
        reader = PdfReader(BytesIO(data), strict=False)
        if reader.is_encrypted:
            raise RuntimeError(f"PDF artifact is encrypted and cannot be verified: {label}")
        pages = reader.pages
        if not pages:
            raise RuntimeError(f"PDF artifact has no pages: {label}")
        metadata = reader.metadata or {}
        metadata_text = "\n".join(str(value) for value in metadata.values())
        extracted_text = "\n".join(page.extract_text() or "" for page in pages)
        root_object = reader.trailer.get("/Root")
        if root_object is not None:
            root_object = root_object.get_object()
        xmp_reference = root_object.get("/Metadata") if root_object is not None else None
        xmp_data = xmp_reference.get_object().get_data() if xmp_reference is not None else b""
    except RuntimeError:
        raise
    except Exception as error:
        raise RuntimeError(f"PDF artifact cannot be parsed for validation: {label}") from error

    if anonymous:
        author = metadata.get("/Author")
        title = metadata.get("/Title")
        if author is None or str(author) != "Anonymous":
            raise RuntimeError("anonymous PDF must have exactly Anonymous as its author metadata")
        if title is None or not str(title).strip():
            raise RuntimeError("anonymous PDF is missing its title metadata")
        identity_material = b"\n".join(
            (
                data,
                metadata_text.encode("utf-8", errors="replace"),
                extracted_text.encode("utf-8", errors="replace"),
                xmp_data,
            )
        )
        if LOCAL_USER_PATH_PATTERN.search(identity_material) or any(
            _contains_identifier(identity_material, term) for term in forbidden_terms
        ):
            raise RuntimeError("anonymous PDF contains identifying author metadata")
    return {"sha256": _sha256(data), "size_bytes": len(data)}


def validate_package_zip(
    path: Path,
    *,
    expected_commit: str,
    anonymous: bool,
    forbidden_terms: tuple[str, ...] = (),
    require_clean_tree: bool = True,
    expected_entries: Mapping[str, bytes] | None = None,
) -> dict[str, Any]:
    """Validate ZIP structure, commit identity, member hashes, and anonymization."""

    if not path.is_file() or _path_has_symlink(path):
        raise RuntimeError(f"ZIP artifact is missing or has a symbolic link in its path: {path}")
    size = path.stat().st_size
    if anonymous and size > TMLR_MAX_SUPPLEMENT_BYTES:
        raise RuntimeError("TMLR supplementary ZIP exceeds the 100 MB limit")
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if any(info.is_dir() for info in infos):
                raise RuntimeError("ZIP artifact contains explicit directory entries")
            names = [info.filename for info in infos]
            if len(names) != len({name.casefold() for name in names}):
                raise RuntimeError("ZIP artifact contains duplicate or case-colliding member names")
            for name in names:
                if _unsafe_member_name(name):
                    raise RuntimeError(f"ZIP artifact contains an unsafe member path: {name}")
            if any(info.flag_bits & 0x1 for info in infos):
                raise RuntimeError("ZIP artifact contains encrypted members")
            if any(stat.S_ISLNK(info.external_attr >> 16) for info in infos):
                raise RuntimeError("ZIP artifact contains symbolic-link members")
            uncompressed_limit = (
                TMLR_MAX_SUPPLEMENT_BYTES if anonymous else MAX_PUBLIC_PACKAGE_UNCOMPRESSED_BYTES
            )
            if sum(info.file_size for info in infos) > uncompressed_limit:
                raise RuntimeError("ZIP artifact exceeds the allowed uncompressed size")
            corrupt = archive.testzip()
            if corrupt is not None:
                raise RuntimeError(f"ZIP artifact has a corrupt member: {corrupt}")
            if PACKAGE_MANIFEST not in names:
                raise RuntimeError("ZIP artifact has no integrity manifest")
            manifest = json.loads(archive.read(PACKAGE_MANIFEST))
            if manifest.get("schema_version") != "v1-artifact-package-v1":
                raise RuntimeError("ZIP artifact has an unsupported manifest schema")
            if anonymous:
                if (
                    manifest.get("source_commit") is not None
                    or manifest.get("source_commit_withheld") is not True
                ):
                    raise RuntimeError("anonymous TMLR ZIP exposes source commit metadata")
            elif manifest.get("source_commit") != expected_commit:
                raise RuntimeError("ZIP artifact was built from a different source commit")
            if require_clean_tree and manifest.get("source_tree_clean") is not True:
                raise RuntimeError("ZIP artifact was built from an uncommitted source tree")
            if manifest.get("anonymous") is not anonymous:
                raise RuntimeError("ZIP artifact anonymity flag does not match its intended use")
            if manifest.get("source_commit_withheld") is not anonymous:
                raise RuntimeError("ZIP artifact source-commit privacy flag is inconsistent")
            expected_files = manifest.get("files")
            if not isinstance(expected_files, dict) or set(expected_files) != set(names) - {
                PACKAGE_MANIFEST
            }:
                raise RuntimeError("ZIP artifact file list does not match its manifest")
            for name, digest in expected_files.items():
                if _sha256(archive.read(name)) != digest:
                    raise RuntimeError(f"ZIP artifact member hash mismatch: {name}")
            if expected_entries is not None:
                if set(expected_entries) != set(expected_files):
                    raise RuntimeError(
                        "ZIP artifact members do not match the current release inputs"
                    )
                for name, expected_content in expected_entries.items():
                    if archive.read(name) != expected_content:
                        raise RuntimeError(
                            f"ZIP artifact is stale for current release input: {name}"
                        )
            _validate_tmlr_styles(
                {name: archive.read(name) for name in TMLR_STYLE_SHA256 if name in names}
            )
            if anonymous:
                required = {
                    "README.md",
                    "paper/main.tex",
                    "paper/main.pdf",
                    "paper/references.bib",
                    "paper/tmlr.sty",
                }
                if not required.issubset(names):
                    raise RuntimeError(
                        "anonymous TMLR ZIP is missing required paper or style files"
                    )
                tex = archive.read("paper/main.tex")
                if re.search(rb"(?m)^\\author\s*\{", tex) or EMAIL_PATTERN.search(tex):
                    raise RuntimeError("anonymous TMLR ZIP contains identifying paper metadata")
                if b"\\@acceptedtrue" in tex or b"[preprint]" in tex:
                    raise RuntimeError("anonymous TMLR ZIP is not in double-blind review mode")
                _validate_pdf_bytes(
                    archive.read("paper/main.pdf"),
                    "anonymous TMLR ZIP",
                    anonymous=True,
                    forbidden_terms=forbidden_terms,
                )
                package_bytes = b"".join(
                    archive.read(name).lower() for name in names if name != PACKAGE_MANIFEST
                )
                commit_tokens = {
                    expected_commit.encode("ascii").lower(),
                    expected_commit[:7].encode("ascii").lower(),
                }
                if any(
                    token in package_bytes
                    or any(token.decode("ascii") in name.lower() for name in names)
                    for token in commit_tokens
                ):
                    raise RuntimeError("anonymous TMLR ZIP contains its source commit identifier")
                for name in names:
                    if name in TMLR_IDENTITY_SCAN_EXEMPT_FILES:
                        continue
                    member_data = archive.read(name)
                    identity_content = name.encode("utf-8") + b"\n" + member_data
                    if (
                        EMAIL_PATTERN.search(identity_content)
                        or LOCAL_USER_PATH_PATTERN.search(identity_content)
                        or any(
                            _contains_identifier(identity_content, term) for term in forbidden_terms
                        )
                    ):
                        raise RuntimeError(
                            f"anonymous TMLR ZIP contains identifying data in {name}"
                        )
            elif "paper/main.pdf" not in names:
                raise RuntimeError("public source ZIP is missing its compiled PDF")
    except zipfile.BadZipFile as error:
        raise RuntimeError(f"ZIP artifact is malformed: {path}") from error
    return {"sha256": _sha256(path.read_bytes()), "size_bytes": size}


def artifact_paths(root: Path, commit: str) -> dict[str, Path]:
    """Return the four Gate G artifact paths for one exact source commit."""

    short = commit[:7]
    stem = f"multiclass-imbalance-benchmark-paper-{short}"
    return {
        "deanonymized_release_pdf": root / "paper/main.pdf",
        "source_release_zip": root / f"output/release/{stem}.zip",
        "tmlr_anonymous_pdf": root / "output/submission/tmlr-anonymous-submission.pdf",
        "tmlr_anonymous_supplement_zip": root / "output/submission/tmlr-anonymous-supplement.zip",
    }


def _read_required(root: Path, relative: str) -> bytes:
    path = root / relative
    if not path.is_file():
        raise RuntimeError(f"supplementary source is missing: {relative}")
    return path.read_bytes()


def _validate_tmlr_styles(files: dict[str, bytes]) -> None:
    for relative, expected_hash in TMLR_STYLE_SHA256.items():
        content = files.get(relative)
        if content is None or _sha256(content) != expected_hash:
            raise RuntimeError(f"TMLR style asset is missing or changed: {relative}")


def _anonymous_license(license_text: bytes) -> bytes:
    text = license_text.decode("utf-8")
    lines = text.splitlines()
    copyright_lines = [
        index for index, line in enumerate(lines) if line.startswith("Copyright (c)")
    ]
    if len(copyright_lines) != 1:
        raise RuntimeError("the project license has an unexpected copyright notice")
    lines[copyright_lines[0]] = "Copyright (c) 2026 Anonymous Author"
    result = ("\n".join(lines) + "\n").encode("utf-8")
    if EMAIL_PATTERN.search(result):
        raise RuntimeError("anonymous license still contains an email address")
    return result


def _anonymous_readme() -> bytes:
    return (
        b"# Supplementary material\n\n"
        b"This anonymized archive accompanies the V1 benchmark manuscript. It contains "
        b"the paper source, analysis code, public dataset provenance, frozen result tables, "
        b"and figures that directly support the manuscript. The raw datasets are not "
        b"redistributed; their public source identifiers and acquisition instructions are "
        b"provided in `data/acquisition_manifest.csv` and `docs/data-acquisition.md`.\n\n"
        b"For source and scope details, see `data/dataset_registry.csv` and "
        b"`data/acquisition_manifest.csv`. In `results/analysis/`, `cell_means.csv` and "
        b"`macro_f1_figure_coverage.csv` locate complete-cell values and coverage; "
        b"`class_means.csv`, `per_class_recall_summary.csv`, and "
        b"`baseline_deltas_per_class.csv` contain class-wise summaries; "
        b"`failure_accounting.csv` and `failure_reason_counts.csv` separate unsupported "
        b"conditions from applicable failures. `adasyn_excluded_friedman.csv`, "
        b"`adasyn_excluded_pairwise.csv`, and "
        b"`adasyn_excluded_imbalance_descriptive.csv` report the post hoc macro-F1 "
        b"sensitivity, exact follow-ups, and descriptive imbalance strata.\n\n"
        b"Install the packages in `requirements.txt`, then follow `docs/reproducibility.md` "
        b"to regenerate the analysis from the supplied frozen result records; this does not "
        b"rerun model training. A fresh benchmark run requires separately obtaining the listed "
        b"public datasets and following `docs/stage-0-runbook.md`. In repository records, "
        b"Stage 0 means the pre-run feasibility and scope audit; Gate D accounts for raw "
        b"records, Gate E replays and validates the analysis, Gate F checks reproduction "
        b"from a clean checkout, and Gate G validates a release snapshot. These are internal "
        b"workflow labels, not venue certifications. The TMLR style "
        b"files in `paper/` are supplied unchanged. Install `requirements-release.txt` "
        b"as well when running the PDF validators. Source revision fields are redacted from "
        b"the anonymous evidence manifests; their hashes are updated to match the packaged "
        b"bytes. Per-file hashes in the ZIP manifest identify the exact review archive.\n"
    )


def anonymize_gate_f_review_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep Gate F results while withholding linkable identity and Git metadata."""

    payload = json.loads(json.dumps(payload))
    payload["git_head"] = None
    snapshot = payload.get("source_snapshot")
    if not isinstance(snapshot, dict):
        raise TypeError("Gate F review has no source snapshot to anonymize")
    snapshot["archive_sha256"] = None
    snapshot["tracked_file_set_sha256"] = None
    path_pattern = re.compile(LOCAL_USER_PATH_PATTERN.pattern.decode("ascii"))
    email_pattern = re.compile(EMAIL_PATTERN.pattern.decode("ascii"), re.IGNORECASE)
    checks = snapshot.get("checks")
    if not isinstance(checks, list):
        raise TypeError("Gate F source snapshot has no check list to anonymize")
    for check in checks:
        if not isinstance(check, dict):
            continue
        command = check.get("command")
        if isinstance(command, list):
            check["command"] = [
                email_pattern.sub(
                    "<email>",
                    REVISION_TEXT_PATTERN.sub(
                        "<revision>",
                        path_pattern.sub("<local-path>", argument),
                    ),
                )
                if isinstance(argument, str)
                else argument
                for argument in command
            ]
        for key in ("stdout", "stderr"):
            value = check.get(key)
            if isinstance(value, str):
                check[key] = email_pattern.sub(
                    "<email>",
                    REVISION_TEXT_PATTERN.sub(
                        "<revision>",
                        path_pattern.sub("<local-path>", value),
                    ),
                )
    payload["double_blind_redactions"] = [
        "git_head",
        "source_snapshot.archive_sha256",
        "source_snapshot.tracked_file_set_sha256",
        "local_user_paths",
        "email_addresses",
        "revision_hashes_in_gate_f_logs",
    ]
    return payload


def _gate_f_review_for_source(
    root: Path,
    source_commit: str,
    source_tree_clean: bool,
) -> dict[str, Any]:
    """Return only Gate F evidence that matches this exact clean source tree."""

    review = json.loads(_read_required(root, "results/analysis/gate-f-review.json"))
    if (
        source_tree_clean
        and review.get("status") == "passed"
        and review.get("git_head") == source_commit
    ):
        return review
    return {
        "schema_version": review.get("schema_version", "gate-f-review-v1"),
        "status": "not_run",
        "source_tree_clean": source_tree_clean,
        "boundary": (
            "No Gate F clean-checkout result is claimed for this package because its working "
            "tree is dirty or the stored Gate F result describes a different source commit."
        ),
    }


def _anonymous_gate_f_review(
    root: Path,
    source_commit: str,
    source_tree_clean: bool,
) -> bytes:
    payload = _gate_f_review_for_source(root, source_commit, source_tree_clean)
    if payload["status"] == "passed":
        payload = anonymize_gate_f_review_payload(payload)
    return _json_bytes(payload)


def _redact_git_revision_fields(value: Any) -> Any:
    if isinstance(value, dict):
        for key, item in list(value.items()):
            if str(key).casefold() in {
                "git_head",
                "git_commit",
                "git_branch",
                "source_commit",
                "source_branch",
            }:
                value[key] = None
            else:
                value[key] = _redact_git_revision_fields(item)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            value[index] = _redact_git_revision_fields(item)
    return value


def _replace_hash_references(value: Any, replacements: Mapping[str, str]) -> Any:
    if isinstance(value, dict):
        path = value.get("path")
        if isinstance(path, str) and path in replacements and "sha256" in value:
            value["sha256"] = replacements[path]
        for item in value.values():
            _replace_hash_references(item, replacements)
    elif isinstance(value, list):
        for item in value:
            _replace_hash_references(item, replacements)
    return value


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _supplement_entries(
    root: Path,
    anonymous_source: str,
    forbidden_terms: tuple[str, ...],
    anonymous_pdf: bytes,
    source_commit: str,
    source_tree_clean: bool,
) -> dict[str, bytes]:
    entries: dict[str, bytes] = {
        "README.md": _anonymous_readme(),
        "LICENSE": _anonymous_license(_read_required(root, "LICENSE")),
        "paper/main.tex": anonymous_source.encode("utf-8"),
        "paper/main.pdf": anonymous_pdf,
    }
    for relative in (*TMLR_SOURCE_FILES, *TMLR_PAPER_SUPPORT_FILES):
        entries[relative] = _read_required(root, relative)
    _validate_tmlr_styles(entries)

    run_manifest_name = "results/full-run/benchmark_manifest.json"
    run_manifest_payload = _redact_git_revision_fields(
        json.loads(_read_required(root, run_manifest_name))
    )
    run_manifest_bytes = _json_bytes(run_manifest_payload)
    entries[run_manifest_name] = run_manifest_bytes
    hash_replacements = {run_manifest_name: _sha256(run_manifest_bytes)}

    for path in sorted((root / "results/full-run").glob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_file() and path.suffix in {".csv", ".json"} and relative != run_manifest_name:
            entries[relative] = path.read_bytes()

    analysis_dir = root / "results/analysis"
    for path in sorted(analysis_dir.glob("*.csv")):
        entries[path.relative_to(root).as_posix()] = path.read_bytes()

    gate_d_name = "results/analysis/gate-d-review.json"
    gate_d_payload = _redact_git_revision_fields(json.loads(_read_required(root, gate_d_name)))
    _replace_hash_references(gate_d_payload, hash_replacements)
    gate_d_bytes = _json_bytes(gate_d_payload)
    entries[gate_d_name] = gate_d_bytes
    hash_replacements[gate_d_name] = _sha256(gate_d_bytes)

    analysis_manifest_name = "results/analysis/analysis_manifest.json"
    analysis_payload = _redact_git_revision_fields(
        json.loads(_read_required(root, analysis_manifest_name))
    )
    _replace_hash_references(analysis_payload, hash_replacements)
    analysis_bytes = _json_bytes(analysis_payload)
    entries[analysis_manifest_name] = analysis_bytes
    hash_replacements[analysis_manifest_name] = _sha256(analysis_bytes)

    gate_e_name = "results/analysis/gate-e-review.json"
    gate_e_payload = _redact_git_revision_fields(json.loads(_read_required(root, gate_e_name)))
    _replace_hash_references(gate_e_payload, hash_replacements)
    entries[gate_e_name] = _json_bytes(gate_e_payload)

    gate_f_name = "results/analysis/gate-f-review.json"
    gate_f_payload = _redact_git_revision_fields(
        json.loads(_anonymous_gate_f_review(root, source_commit, source_tree_clean))
    )
    _replace_hash_references(gate_f_payload, hash_replacements)
    entries[gate_f_name] = _json_bytes(gate_f_payload)
    for path in sorted((analysis_dir / "figures").glob("*.png")):
        entries[path.relative_to(root).as_posix()] = path.read_bytes()

    for name, content in entries.items():
        if name in TMLR_IDENTITY_SCAN_EXEMPT_FILES:
            continue
        identity_content = name.encode("utf-8") + b"\n" + content
        if (
            EMAIL_PATTERN.search(identity_content)
            or LOCAL_USER_PATH_PATTERN.search(identity_content)
            or REVISION_PATTERN.search(identity_content)
            or any(_contains_identifier(identity_content, term) for term in forbidden_terms)
        ):
            raise RuntimeError(f"anonymous TMLR package contains identifying data in {name}")
    return entries


def expected_tmlr_supplement_entries(root: Path, anonymous_pdf: bytes) -> dict[str, bytes]:
    """Rebuild the exact current anonymous package inputs for Gate G validation."""

    source_text = _read_required(root, "paper/main.tex").decode("utf-8")
    anonymous_source = anonymize_tmlr_source(source_text)
    identifiers = author_identifiers(source_text) + repository_identifiers(root)
    source_commit, source_tree_clean, _ = _source_tree_state(root, allow_dirty=True)
    _validate_pdf_bytes(
        anonymous_pdf,
        "anonymous TMLR submission",
        anonymous=True,
        forbidden_terms=identifiers,
    )
    return _supplement_entries(
        root,
        anonymous_source,
        identifiers,
        anonymous_pdf,
        source_commit,
        source_tree_clean,
    )


def _source_release_entries(
    root: Path,
    source_files: list[str],
    source_commit: str,
    source_tree_clean: bool,
) -> dict[str, bytes]:
    entries: dict[str, bytes] = {}
    for relative in source_files:
        path = root / relative
        if path.is_file() and path.suffix.lower() != ".pdf":
            entries[relative] = path.read_bytes()
    for relative in ("artifacts/environment/metadata.json", "artifacts/runs/scope-lock.json"):
        path = root / relative
        if path.is_file():
            entries[relative] = path.read_bytes()
    for directory in (root / "results/full-run", root / "results/analysis"):
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.name != "gate-g-review.json":
                entries[path.relative_to(root).as_posix()] = path.read_bytes()
    entries["paper/main.pdf"] = _read_required(root, "paper/main.pdf")
    entries["results/analysis/gate-f-review.json"] = _json_bytes(
        _gate_f_review_for_source(root, source_commit, source_tree_clean)
    )
    return entries


def expected_source_release_entries(root: Path) -> dict[str, bytes]:
    """Rebuild the exact current public source release package inputs."""

    source_commit, source_tree_clean, source_files = _source_tree_state(root, allow_dirty=False)
    return _source_release_entries(root, source_files, source_commit, source_tree_clean)


def build_release_artifacts(
    root: Path,
    output_root: Path,
    *,
    allow_dirty: bool = False,
    stage_only: bool = False,
) -> dict[str, Any]:
    """Build both source ZIPs and stage anonymous TeX for a matching PDF build."""

    root = root.resolve()
    output_root = output_root.resolve()
    if not output_root.is_relative_to(root):
        raise ValueError("artifact output root must remain inside the repository")
    commit, clean, source_files = _source_tree_state(root, allow_dirty)
    paths = artifact_paths(root, commit)
    if not (root / "paper/main.pdf").is_file():
        raise RuntimeError("compile paper/main.tex before building the V1 source release ZIP")

    source_text = _read_required(root, "paper/main.tex").decode("utf-8")
    anonymous_source = anonymize_tmlr_source(source_text)
    identifiers = author_identifiers(source_text) + repository_identifiers(root)
    stage = output_root / "staging" / "tmlr-anonymous"
    (stage / "paper").mkdir(parents=True, exist_ok=True)
    (stage / "paper/main.tex").write_text(anonymous_source, encoding="utf-8", newline="\n")
    for relative in TMLR_PAPER_SUPPORT_FILES:
        destination = stage / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(_read_required(root, relative))
    _validate_tmlr_styles(
        {relative: _read_required(root, relative) for relative in TMLR_STYLE_SHA256}
    )
    for path in sorted((root / "results/analysis/figures").glob("*.png")):
        destination = stage / path.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())

    if stage_only:
        return {
            "source_commit": commit,
            "source_tree_clean": clean,
            "anonymous_source": (stage / "paper/main.tex").as_posix(),
            "stage_only": True,
        }

    anonymous_pdf = output_root / "submission" / paths["tmlr_anonymous_pdf"].name
    if not anonymous_pdf.is_file():
        raise RuntimeError(
            "compile the staged anonymous source and copy its PDF to "
            f"{anonymous_pdf.relative_to(output_root)} before packaging"
        )
    anonymous_pdf_bytes = anonymous_pdf.read_bytes()
    _validate_pdf_bytes(
        anonymous_pdf_bytes,
        "anonymous TMLR submission",
        anonymous=True,
        forbidden_terms=identifiers,
    )
    (stage / "paper/main.pdf").write_bytes(anonymous_pdf_bytes)
    supplement_entries = expected_tmlr_supplement_entries(root, anonymous_pdf_bytes)
    supplement = paths["tmlr_anonymous_supplement_zip"]
    # Honour a caller's staging/output root without weakening Gate G's fixed paths.
    supplement = output_root / "submission" / supplement.name
    supplement_ref = write_package_zip(
        supplement,
        supplement_entries,
        source_commit=None,
        source_tree_clean=clean,
        anonymous=True,
    )

    release_entries = _source_release_entries(root, source_files, commit, clean)
    release = paths["source_release_zip"]
    release = output_root / "release" / release.name
    release_ref = write_package_zip(
        release,
        release_entries,
        source_commit=commit,
        source_tree_clean=clean,
        anonymous=False,
    )
    return {
        "source_commit": commit,
        "source_tree_clean": clean,
        "anonymous_source": (stage / "paper/main.tex").as_posix(),
        "source_release_zip": release_ref,
        "tmlr_anonymous_supplement_zip": supplement_ref,
        "tmlr_anonymous_pdf_expected": (
            output_root / "submission" / paths["tmlr_anonymous_pdf"].name
        ).as_posix(),
    }
