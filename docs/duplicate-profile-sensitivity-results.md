# Duplicate-profile sensitivity run record

Run date: 2026-10-14. This document records the completed post hoc sensitivity
under the design frozen in [`duplicate-profile-policy.md`](duplicate-profile-policy.md).
It does not amend the protocol or replace the row-wise V1 benchmark.

## Design and provenance

The six datasets with row-wise profile overlap were CMC, CNAE-9, Glass, Iris,
Wine Quality Red, and Yeast. All 6,000 rows and their recorded targets were
retained. Profile groups use exact raw predictor values with the target
excluded; identical profiles with conflicting labels remain in one group.
Splits use shuffled `StratifiedGroupKFold`, five folds, and seeds 0, 1, and 2.
The locked V1 classifiers, conditions, preprocessing, sampler settings,
random-state routing, training-only fitting, and metrics were reused.

The model-free preflight recorded all 90 planned splits as feasible: every class
appeared in both train and test, and no predictor profile was shared between
the partitions. The runner accounted for all 2,970 model-grid keys. It recorded
2,295 valid results, 540 unsupported combinations, and 135 model failures.
There were no infeasible splits. Unsupported combinations and model failures
remain explicit; neither is treated as a score.

The sensitivity ran on Python 3.13.12 with the 41-package replay/analysis lock
whose SHA-256 is recorded in the run manifest. The frozen analysis manifest
also records Python 3.13.12. The available July Stage 0 environment record says
Python 3.12.10, while the full-run benchmark manifest does not record its
interpreter. The sensitivity environment is known; the original training
interpreter remains unverified.

## Coverage and failures

| Dataset | Expected keys | Valid | Unsupported | Model failures | Complete supported cells |
|---|---:|---:|---:|---:|---:|
| CMC | 495 | 330 | 165 | 0 | 22 / 22 |
| CNAE-9 | 495 | 420 | 75 | 0 | 28 / 28 |
| Glass | 495 | 375 | 75 | 45 | 25 / 28 |
| Iris | 495 | 420 | 75 | 0 | 28 / 28 |
| Wine Quality Red | 495 | 375 | 75 | 45 | 25 / 28 |
| Yeast | 495 | 375 | 75 | 45 | 25 / 28 |
| **Total** | **2,970** | **2,295** | **540** | **135** | **153 / 162** |

All 135 model failures were ADASYN fit failures under the locked settings.
Glass and Wine Quality Red each had 45 failures because the requested ratio
generated no samples. Yeast had 36 such failures and 9 neighbor-support
failures. The 540 unsupported records are non-applicable condition records:
165 for mixed-data CMC and 75 for each numeric dataset. The raw and grouped
failure CSVs preserve the per-seed, per-fold, classifier, condition, stage,
and error details.

## Descriptive comparison with row-wise V1

The comparison uses only cells with all 15 valid seed-fold records in both
designs. There are 153 such dataset/classifier/condition macro-F1 cell pairs;
nine supported ADASYN cells are incomplete in the grouped design. Across the
153 cells, the grouped-minus-row-wise macro-F1 deltas have mean -0.0046,
median -0.0015, range -0.0752 to +0.0400, with 71 positive and 82 negative
deltas. These cells reuse the same six datasets across conditions and
classifiers, so this pooled summary is descriptive and does not define an
independent sample or an inferential estimate.

For the raw baseline, the row-wise-to-grouped macro-F1 means were:

| Dataset | Random forest | XGBoost | Logistic regression |
|---|---:|---:|---:|
| CMC | 0.4778 → 0.4898 (+0.0120) | 0.4874 → 0.4937 (+0.0063) | 0.4804 → 0.4786 (-0.0018) |
| CNAE-9 | 0.9250 → 0.9206 (-0.0044) | 0.9052 → 0.9035 (-0.0017) | 0.9435 → 0.9477 (+0.0042) |
| Glass | 0.7396 → 0.7485 (+0.0089) | 0.6963 → 0.6940 (-0.0023) | 0.5498 → 0.5437 (-0.0061) |
| Iris | 0.9461 → 0.9551 (+0.0090) | 0.9391 → 0.9437 (+0.0046) | 0.9597 → 0.9552 (-0.0045) |
| Wine Quality Red | 0.3478 → 0.2818 (-0.0660) | 0.3554 → 0.2802 (-0.0752) | 0.3011 → 0.3032 (+0.0021) |
| Yeast | 0.5530 → 0.5518 (-0.0012) | 0.5126 → 0.5029 (-0.0097) | 0.5475 → 0.5393 (-0.0082) |

Ranks across all applicable macro-F1 conditions could be compared for nine
dataset/classifier matrices where both designs had complete supported cells.
Across those matrices, 78 condition ranks were available and 27 changed; the
mean absolute rank change was 0.62, with a maximum of six positions. The other
nine matrices were left unranked because the grouped design had incomplete
cells. These are descriptive comparisons of two different split designs, not
tests of whether the original row-wise evaluation is invalid. The larger
raw-baseline decreases for Wine Quality Red and the rank changes warrant
explicit discussion, but do not identify entity identity or establish a
causal effect of duplicated rows.

## Reproduction and output files

From the repository root, create the preflight and then run the fixed model
grid with:

```powershell
python scripts/run_duplicate_profile_sensitivity.py --preflight-only
python scripts/run_duplicate_profile_sensitivity.py --resume
```

The ignored `results/duplicate-profile-sensitivity/` directory contains the
input and environment hashes, profile group codes, all row-to-fold partition
assignments, split feasibility, per-dataset and aggregate results and
failures, cell means, coverage, descriptive design comparisons, rank
comparisons, and `sensitivity_manifest.json` with hashes for every output.
The hashes of the frozen full-run records and all frozen analysis artifacts
were verified before and after the sensitivity; neither frozen directory was
modified. No hypothesis tests were added, no replacement folds were used, and
no raw dataset bytes were added to the release packages.
