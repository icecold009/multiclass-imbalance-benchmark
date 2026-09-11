# Gate G: technical release snapshot

Gate G preserves the exact source commit and the local evidence manifests that
support the benchmark's reproducibility boundary. It is a release-candidate
check, not evidence of venue submission or acceptance.

## Scope

The snapshot must be created on a dedicated feature branch with a clean
working tree. It records:

- the current commit and branch;
- a SHA-256 hash of the `git archive` source snapshot and its tracked-file set;
- the passed Gate F clean-checkout review;
- the environment, scope-lock, full-run, Gate D, analysis, and Gate E manifest
  hashes;
- the explicit boundary that no paper, venue, external review, deployment, or
  publication claim is implied.

Generated raw data, environment records, and result tables remain ignored by
Git. Their hashes are recorded in the Gate G payload so the local evidence can
be checked against the preserved commit without committing the generated data.

## Package

The tracked release package consists of:

- `src/submission.py` and `scripts/verify_gate_g.py` for validation;
- `tests/test_submission.py` for snapshot-digest coverage;
- this checklist, the README entry point, and the existing protocol and
  provenance documents;
- the ignored local `results\analysis\gate-g-review.json` evidence payload.

## Acceptance criteria

Gate G passes only when:

- the feature branch is dedicated and the working tree is clean;
- Gate F is passed for the current commit;
- the frozen environment, scope-lock, full-run, Gate D, analysis, and Gate E
  manifests exist with their recorded hashes;
- the source `git archive` matches the tracked-file set and contains no raw or
  generated data;
- the payload records `external_submission: false` unless a separately
  authorized submission workflow is added.

## Verification

After committing the release-candidate source changes and regenerating Gate F
for that exact commit, run:

```powershell
.venv\Scripts\python.exe scripts\verify_gate_f.py
.venv\Scripts\python.exe scripts\verify_gate_g.py
```

The Gate G payload is written to the ignored
`results\analysis\gate-g-review.json`. Preserve that payload together with
the source commit and the environment/results archive. A later source change
requires a new Gate F review and a new Gate G snapshot.

## Non-goals

This package does not submit a manuscript, choose a venue, upload data, merge
to `main`, publish a release, deploy an application, or claim independent
external review. Those actions require their own explicit authorization and
evidence.
