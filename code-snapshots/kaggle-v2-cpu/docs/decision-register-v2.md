# V2 Decision Register

**Protocol:** `expanded-v2`
**Register version:** `expanded-v2-decisions-1`
**Status:** Locked for implementation; outcome-bearing execution remains gated
by the compute-allocation decision recorded below.

This register makes the methodological choices explicit. It is part of the
hash-locked V2 protocol package and must be reviewed together with
`config/v2.yaml`, `docs/protocol-v2.md`, and `docs/analysis-plan-v2.md`.

| Decision | Locked resolution | Evidence / implementation anchor |
|---|---|---|
| Bayesian likelihood and priors | Five-dimensional exchangeable fold-correlated hierarchical Normal model; `mu ~ Normal(0, 0.5)`, `tau_dataset ~ HalfNormal(0.5)`, `sigma_fold ~ HalfNormal(0.25)`, `rho ~ Beta(2, 2)`; NUTS with 4 chains, 2,000 tune, 2,000 draws, `target_accept=0.90`, seed `20260917`; diagnostics require `r_hat <= 1.01`, bulk/tail ESS >= 400, and zero divergences. | `config/v2.yaml`; `docs/analysis-plan-v2.md`; `src/v2_analysis.py` |
| Primary contrast matrix | Raw versus every applicable non-raw condition; weighting versus each declared sampler; Balanced Random Forest versus each declared RF intervention; and the fixed sampler pairs. A contrast requires both conditions, the same classifier/feature family, five valid outer folds, and at least three complete dataset blocks. | `config/v2.yaml` contrasts; `declared_contrasts()` and `paired_contrast_blocks()` |
| Threshold optimizer | Log-threshold offsets with the lowest encoded class fixed at zero; remaining offsets in `[-2, 2]`; scipy differential evolution, seed `20260916`, `maxiter=40`, `popsize=8`, `polish=false`, `workers=1`; macro-F1 objective; lexicographically smallest tie; ordinary argmax fallback with explicit status on invalid support, non-finite probabilities, or optimizer failure. | `config/v2.yaml`; `src/v2_engineering.py` |
| Native categorical compatibility | XGBoost `3.3.0` with histogram/native categorical mode; LightGBM `4.7.0` with native categorical mode; CatBoost `1.2.10` with native categorical mode and file writing disabled. Mixed synthetic paths use fold-local ordinal sampling and recast categorical columns; no one-hot native path. | `requirements-v2.txt`; `requirements-v2.lock`; Gate 4 tests and smoke artifacts |
| HPO search spaces | Exact YAML spaces are authoritative. Twenty randomized configurations per dataset/outer-fold/classifier/condition, seed `20260915`, scoring `macro_average_precision`, one worker, sampler parameters excluded except the fold-local neighbour rule. Invalid trials are recorded in the trial table; a cell fails explicitly only if all applicable trials fail. | `config/v2.yaml`; `src/v2_tuning.py`; `src/v2_execution.py` |
| Sampler applicability | `random_over` targets `not_majority`; `random_under` targets `not_minority`; synthetic samplers use `k=min(5, minority_support-1)` recomputed on every actual fit split; Borderline-SMOTE bounds `m_neighbors` locally; SMOTENC is mixed-only; ordinary SMOTE/ADASYN/Borderline-SMOTE are numeric-only; SMOTEENN/SMOTETomek use SMOTE or SMOTENC according to feature type. Fewer than two usable examples yields explicit `FAILED / NOT APPLICABLE`; balanced folds use a recorded no-op sampler. | `config/v2.yaml`; `src/v2_engineering.py`; `src/v2_execution.py` |
| Runtime/resource policy | CPU-only, one worker per model, 1,800-second fit timeout, 21,600-second dataset-job timeout, 32,768 MB peak-memory ceiling, one infrastructure retry, zero protocol retries. Runtime, memory, worker count, timeout, and retry fields are recorded per cell. | `config/v2.yaml` execution section; full-run schemas |
| Compute instance | No cloud instance is silently selected. The full matrix requires an explicitly approved compute allocation; the current Windows host is recorded for development and validation only and is not approved as the production benchmark instance. | Readiness blocker; resolve before changing config to `full-run-ready` |
| Optional metrics | Log loss is included as a secondary probability metric using multiclass log loss on untouched outer-test probabilities. Brier score uses the mean over rows and classes of squared probability minus one-hot error. | `config/v2.yaml`; `src/v2_engineering.py` |
| Dataset coverage quotas | Registry eligibility is provenance-only: 40–60 datasets, target approximately 50, at least 20 numeric and 10 mixed, with imbalance-band minima 8/10/15/8. The frozen registry contains 50 eligible rows with 30 numeric, 20 mixed, and band counts 8/19/15/8. No benchmark score is inspected. | `src/v2_registry.py`; `data/dataset_registry_v2.csv` |
| Output schemas | Every expected dataset/fold/classifier/condition key has exactly one `VALID` or `FAILED / NOT APPLICABLE` record. Valid rows contain metrics and provenance; failures contain stage, exception, reason, applicability, resource fields, retry count, and config hash; trial, manifest, posterior, and analysis tables are separate artifacts. | `config/v2.yaml`; `src/v2_execution.py`; `src/v2_analysis.py` |
| Manuscript evidence boundary | Manuscript tables distinguish protocol decisions, observed results, interpretations, and limitations. Smoke/pilot/partial-run output cannot support V2 claims; Bayesian reporting uses posterior probabilities and ROPE language, never confirmatory p-value language. | `docs/protocol-v2.md`; `docs/analysis-plan-v2.md` |

## Readiness state

The methodological register is locked. The benchmark remains blocked until the
PI/implementation team approves and records the compute instance. After that
approval, record the environment, set the configuration readiness state to
`full-run-ready`, rerun the dry-run hash checks, and only then start the
dataset-granular V2 runner.
