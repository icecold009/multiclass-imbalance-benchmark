# Actual code review and remote backup — 21 November 2026

## Scope and authorization

The author requested inspection of the actual code/repository state and remote preservation of local work, initially selecting GitHub for code and non-sensitive evidence while keeping correspondence private. Automatic approval review subsequently rejected the proposed public archive containing unpublished manuscript PDFs/ZIPs and results. When asked explicitly, the author selected **keep unpublished artifacts private; finish code delivery only**. This final instruction governs delivery: source-code feature branches are backed up, while the manuscript/evidence archive stays private. No merge, venue submission or publication-ready candidate is approved.

The implementation checkout is `RESEARCH PAPER/v1-final-polish`, on `codex/publication-v4-anonymous-evidence`. Before this delivery it was at `9f6f4fa32372a299e1fcb501be71a3e3a4c6fef8`, with six modified T3 files (324 insertions, 58 deletions) and one pre-existing untracked private correspondence file. The other main checkout is detached at `1ce4911...`; the `main` worktree remains at `b8c407330b52f463293ba67471c031de855e7e24`.

GitHub's connector confirmed the repository is public, initially had only `main`, and had no open PRs. Fifteen original local feature branches existed. All were preserved as remote feature refs, including T1 (`4795e31`), T2 (`9f6f4fa`) and the T3 checkpoint (`efaf158`, followed by documentation commits on its branch). No feature branch was merged into `main`.

## Actual code changes reviewed

- `scripts/build_release_artifacts.py`: adds `--anonymous-only` and an isolated default output root without requiring a named PDF for staging.
- `src/release_artifacts.py`: includes the addendum runner/tests, scope lock and environment; rebinds hashes after anonymous transformations; packages exact-calibration evidence; provides a portable supplement README; and preserves the named release paths during an anonymous-only build.
- `scripts/run_publication_addenda.py`: retains the hard-coded frozen cell-means hash and the manifest's reference to those bytes, while accepting a transformed anonymous analysis manifest; writes canonical LF JSON.
- `tests/test_sensitivity_exact.py`: adds rebound-manifest replay verification and handles named versus anonymous manifest identity.
- `tests/test_submission.py`: adds anonymous-only staging, availability wording and nested hash-rebinding checks.
- `docs/release.md`: documents the isolated, explicitly uncommitted preview route.

The six files were committed without silently changing their assertions or representing the preview as a release. Commit `efaf158` is a work-in-progress preservation checkpoint.

## Confirmed unresolved findings

1. **Anonymous owner leak:** packaged `tests/test_submission.py:199` embeds `github.com/icecold009`. `repository_identifiers()` checks full repository URLs, which miss this owner-only spelling. The preview must not be treated as blinded submission material until distributable fixtures/assertions and owner scanning are corrected and verified.
2. **Clean Git checkout fails four targeted integration tests:** three exact-calibration tests require ignored `results/analysis/analysis_manifest.json` or `cell_means.csv`; the supplement-rebinding test requires ignored `artifacts/runs/scope-lock.json`. A separate source-only archive ran all 132 tests with 128 passing and four failing; a subsequent isolated Git clone reproduced all four targeted failures and removed the no-Git ambiguity. The current GitHub workflow checks out source and installs dependencies but does not restore this evidence. Evidence-backed local success therefore does not establish hosted CI success. Repair the test/evidence contract using synthetic unit fixtures plus explicit, hash-verified integration evidence or a reproducible evidence-restoration step; do not hide the failures with unconditional skips.
3. **Candidate mismatch and incomplete release gates:** named upload sets remain V3; the anonymous V4 preview includes T1/T2 but records a dirty source tree and no completed matching Gate F. First-page AI disclosure, author/route/ethics confirmations, independent human review and final artifact approval remain outstanding.
4. **Status records need reconciliation:** the canonical timeline records T1 but not current T2/T3 progress and contains stray `+` characters in its dated T1 paragraph. This delivery record is an audit receipt, not a replacement backlog.

## Verification and limits

The preceding same-session current-tree assessment passed 132 repository tests, 130 extracted-supplement tests, Ruff and `git diff --check`; regenerated all 18 tables and eight figures byte-for-byte; reproduced the exact addendum; inspected all 13 preview PDF pages; and checked package hashes. Source behavior was not changed afterward. The new clean-source/clean-Git checks above expose an additional portability gap rather than invalidating the reported evidence-backed runs.

A signature scan of all named historical blobs reachable from the implementation HEAD found no credential signatures. The separately prepared, private evidence backup scanned selected files and recursively checked nested ZIP members, CRCs and per-file SHA-256 values. Signature scanning is bounded verification, not proof that every possible secret representation is absent. Passing this scan did not authorize public disclosure of unpublished artifacts.

Jev diff review completed normally with `jev-1.14.0`, typed answers and non-empty usage. Coverage was all six changed text files, 30,429 diff characters, no truncation; the pre-existing private correspondence file was explicitly excluded. It returned `correctness_concern` and `resolve_findings_or_human_review` at confidence 0.20, with elevated test-gap signals, without a concrete prose finding. The confirmed issues above remain disclosed. Neither Jev nor a backup checkpoint is publication approval.

## Remote delivery layout

- Code and all original local feature refs: GitHub branches with the original names. The connector creates/validates hosted branches; local Git transports commit objects only.
- Unpublished evidence/artifacts: prepared privately, **not committed or uploaded**. The private backup contains frozen aggregate results, analysis/grouped evidence, publication addendum, historical upload PDFs/ZIPs, the V4 preview, gate receipts, audit reports, otherwise-untracked plans and the non-sensitive Kaggle code package. Its manifest identifies each selected original file by SHA-256. No private cloud destination was selected or used.
- A placeholder remote branch, `codex/research-evidence-backup-20261121`, was created at source checkpoint `efaf158` before the archive upload was blocked. It contains source code only; its existence is not evidence of an uploaded archive.
- Excluded from public delivery by the final user instruction: unpublished artifacts and private correspondence. Raw/processed dataset downloads, credentials, environments, caches, duplicate extraction/replay trees and regenerable TeX intermediates were also excluded. Local copies are preserved, not deleted.

The private backup's README explains evidence restoration for local integration checks. It is not a public release, an anonymous delivery channel or a newly approved paper version. No PR, merge, journal/arXiv submission, message to another person, or final publication was performed by this task. The auto-review rejection was honored, and the proposed public archive was not retried after the author chose to keep it private.
