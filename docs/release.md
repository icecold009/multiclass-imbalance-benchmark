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
