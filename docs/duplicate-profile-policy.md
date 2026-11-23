# Duplicate-profile audit and sensitivity design

Status: design frozen on 2026-10-14, before any new sensitivity model scores
were generated. This is a post hoc sensitivity to generalization on unseen
predictor profiles. It does not replace the original row-wise benchmark or
prove that repeated profiles represent the same person, specimen, protein, or
document.

## Exact-profile policy

Keep every row and its recorded target. Do not deduplicate, relabel CMC, change
neighbor settings, or alter the frozen V1 results. An exact predictor profile
means equality across every target-excluded modelling predictor in the
hash-verified local CSV. A matching missing value counts as equal only in the
same predictor column, so missingness pattern is part of the profile. Group
codes are formed from predictors only; labels are never included in the group
identifier. Conflicting target labels remain in one profile group and are
reported as conflicts.

This policy preserves the original sample weighting and handles uncertain
record identity conservatively. The sensitivity asks whether performance
changes when no exact predictor profile is shared between training and test
folds. It does not estimate performance on new people or physical entities
unless the source supplies an identity mapping, which these modelling files
do not.

## Recomputed duplicate and row-wise overlap counts

All 11 eligible CSVs were re-hashed against both the dataset registry and
acquisition manifest before counting. `Exact duplicate excess rows` counts
complete rows identical across predictors and target after the first copy.
`Repeated profile groups` counts predictor profiles with more than one row;
`Profile excess rows` counts repeated rows after the first copy in each group;
`Rows in repeated profiles` counts every row in those groups. Conflicting
groups have more than one distinct target label. Row-wise overlap counts test
row occurrences whose predictor profile also appears in that fold's training
partition, summed across seeds 0, 1, 2 and five folds. Every row is tested once
per seed.

| Dataset | Rows | Unique profiles | Exact duplicate excess rows | Repeated profile groups | Profile excess rows | Rows in repeated profiles | Conflicting groups / rows | Row-wise overlap / test occurrences | Overlap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Balance Scale | 625 | 625 | 0 | 0 | 0 | 0 | 0 / 0 | 0 / 1,875 | 0.00% |
| CMC | 1,473 | 1,358 | 48 | 96 | 115 | 211 | 62 / 140 | 532 / 4,419 | 12.04% |
| CNAE-9 | 1,080 | 1,052 | 28 | 23 | 28 | 51 | 0 / 0 | 128 / 3,240 | 3.95% |
| Dermatology | 366 | 366 | 0 | 0 | 0 | 0 | 0 / 0 | 0 / 1,098 | 0.00% |
| Glass | 214 | 213 | 1 | 1 | 1 | 2 | 0 / 0 | 4 / 642 | 0.62% |
| Iris | 150 | 147 | 3 | 2 | 3 | 5 | 0 / 0 | 15 / 450 | 3.33% |
| Optdigits | 5,620 | 5,620 | 0 | 0 | 0 | 0 | 0 / 0 | 0 / 16,860 | 0.00% |
| Seeds | 210 | 210 | 0 | 0 | 0 | 0 | 0 / 0 | 0 / 630 | 0.00% |
| Wine | 178 | 178 | 0 | 0 | 0 | 0 | 0 / 0 | 0 / 534 | 0.00% |
| Wine Quality Red | 1,599 | 1,359 | 240 | 220 | 240 | 460 | 0 / 0 | 1,102 / 4,797 | 22.97% |
| Yeast | 1,484 | 1,453 | 31 | 31 | 31 | 62 | 0 / 0 | 160 / 4,452 | 3.59% |

The six datasets with any row-wise overlap are the sensitivity scope. CMC has
62 conflicting-label profiles spanning 140 rows. A profile conflict is
preserved; it is not adjudicated as a labeling error. The displayed overlap
percentages are test-row occurrences, not unique entities or people.

## Source identity review

The source descriptions identify record types but do not support mapping exact
profiles to unique entities in the modelling files:

- [CMC](https://archive.ics.uci.edu/dataset/30/contraceptive%26) describes a
  survey subset whose samples are married women. The supplied modelling
  columns have no respondent identifier, so identical responses may belong to
  different women and conflicting labels cannot be resolved by identity.
- [CNAE-9](https://archive.ics.uci.edu/dataset/233/cnae%2B9) describes
  business-description documents represented as word-frequency vectors. The
  modelling file has no document identifier; equal vectors do not establish
  whether the original texts or source businesses were the same.
- [Glass Identification](https://archive.ics.uci.edu/dataset/42/glass%2Bidentification)
  documents an `Id_number` from 1 to 214 and oxide measurements. The locked
  modelling CSV contains the measurements but not that row number or a
  specimen/group identifier. The one repeated profile therefore cannot be
  identified as either a duplicate record or a separate specimen.
- [Iris](https://archive.ics.uci.edu/dataset/53/iris) states that each
  instance is a plant. The three excess rows in repeated profiles have no
  retained plant IDs, so profile equality does not resolve plant identity.
- [Wine Quality](https://archive.ics.uci.edu/dataset/186/wine%2Bquality)
  describes red and white vinho verde samples with physicochemical inputs and
  sensory quality targets. The red modelling file has no sample, brand, or
  batch identifier that would resolve the repeated measurements.
- [Yeast](https://archive.ics.uci.edu/dataset/110/yeast) describes protein
  localization prediction from numeric attributes. The modelling file has no
  protein accession identifier with which to distinguish repeated profiles.

These descriptions motivate keeping all rows and evaluating unseen profiles
as a distinct target. They do not establish that the original row-wise
evaluation is invalid or quantify optimism.

## Frozen sensitivity design

- **Data:** CMC, CNAE-9, Glass, Iris, Wine Quality Red, and Yeast. Use the
  existing hash-verified modelling CSVs and all rows. Do not add or remove
  datasets based on results.
- **Groups:** exact profiles of raw, target-excluded predictors before any
  fitted preprocessing. All rows with one profile, including conflicting
  labels, stay in one partition. Stratification uses the existing row labels;
  labels do not form or modify group IDs.
- **Splits:** `StratifiedGroupKFold`, five folds, shuffled, with seeds `0`,
  `1`, and `2`. No replacement seeds or folds. Before score generation, the
  model-free preflight found at least five distinct profile groups per class
  in each dataset; all 90 train/test folds contained every class and had zero
  shared profile groups. Yeast's rarest class has exactly five profile groups.
- **Model path:** keep the locked classifiers, sampler applicability,
  preprocessing, random states, neighbor settings, and metrics. Fit every
  transformation, sampler, and training-derived weight on the training fold
  only. Never tune from sensitivity outcomes.
- **Failures:** a split missing a class or sharing a profile is marked
  infeasible and is not replaced. Classifier or sampler failures use the
  existing explicit failure accounting. Do not silently drop cells or change
  neighborhood sizes.
- **Outputs:** save input hashes, environment, exact row/profile group and
  split assignments, valid results, failures, and coverage under a separate
  `results/duplicate-profile-sensitivity/` directory. Do not overwrite
  `results/full-run/` or `results/analysis/`.
- **Comparison:** report dataset-level cell means, complete-cell coverage,
  ranks where complete cells permit them, and original-versus-grouped
  differences descriptively. Do not add confirmatory tests or describe the
  design comparison as inferential. If the outcomes materially change the
  manuscript's conclusions, update both manuscript variants with the new
  evidence and limitations.

The model-free audit can be reproduced from the repository root with:

```powershell
python scripts/audit_duplicate_profiles.py
```

It writes its receipt and three audit tables beneath
`results/duplicate-profile-audit/`. The audit checks frozen raw-file hashes
and does not fit classifiers. The hash-verified Dermatology CSV contains eight
missing `Age` values, consistent with the registry's `has_missing=True` flag;
the missingness pattern is included when exact profiles are grouped.
