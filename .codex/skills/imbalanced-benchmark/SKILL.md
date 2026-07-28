---
name: imbalanced-benchmark
description: Implement and review leakage-safe multiclass imbalance experiments in this repository. Use when changing samplers, preprocessing, classifiers, fold loops, class weighting, SMOTENC handling, pilot execution, or experiment artifacts.
---

# Imbalanced benchmark implementation

## Build the comparison correctly

Represent each run with explicit fields for dataset, feature type, classifier,
condition, seed, fold, and configuration version. Keep these categories
distinct:

- raw classifier;
- class-weighted classifier;
- resampling plus classifier;
- RF-only Balanced Random Forest comparator.

Do not collapse different classifier architectures and sampler families into a
single unexplained method name.

## Preserve fold integrity

For every fold:

1. split first;
2. fit imputers, encoders, scalers, and any sampler only on training data;
3. fit the classifier on the transformed/resampled training data;
4. transform the untouched test data with fitted preprocessing only;
5. predict and compute metrics on the untouched test data.

Use `imblearn.pipeline.Pipeline` where samplers are steps. For XGBoost
class-weighted runs, compute training-fold weights and pass `sample_weight` to
the classifier; do not invent a `class_weight` parameter for XGBoost.

## Handle feature types honestly

- Numeric data may use ordinary SMOTE-family samplers.
- Mixed data must preserve categorical columns for SMOTENC before final
  one-hot encoding.
- Ordinary SMOTE on one-hot categorical features is invalid for this study.
- Categorical-only data is excluded until SMOTEN is implemented.
- Record an applicability failure rather than forcing a pipeline to run.

## Lock configurations

Specify and version `sampling_strategy`, neighbor counts, BorderlineSMOTE kind,
cleaning parameters, classifier settings, seeds, and thread counts. Record
pre- and post-sampling row/class counts. Avoid library defaults whose behavior
may change between versions.

## Produce reviewable artifacts

Write long-form fold results, failure records, runtime/RSS measurements, and a
configuration/environment record. A failed sampler is evidence about
applicability, not a blank score. Do not interpret pilot scores as final
scientific results or select datasets from them.
