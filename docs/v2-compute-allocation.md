# V2 Compute Allocation Brief

This brief records the resource request without selecting or provisioning a
cloud instance. It is an execution prerequisite, not a scientific result.

## Locked workload

| Quantity | Frozen value |
|---|---:|
| Eligible datasets | 50 |
| Outer folds | 5 |
| Classifiers | 5 |
| Conditions in the complete accounting matrix | 11 |
| Expected cell records | 13,750 |
| HPO trials per applicable cell | 20 |
| Inner folds per trial | 3 |
| Additional selected-parameter OOF fits | 3 |
| Final outer-training fit | 1 |
| Model fits per applicable cell | 64 |
| Applicable cells under the frozen numeric/mixed matrix | 2,100 |
| Maximum planned model fits | 134,400 |
| Workers per model | 1 |
| Peak-memory ceiling | 32,768 MB |
| Fit timeout | 1,800 seconds |
| Dataset-job timeout | 21,600 seconds |
| Infrastructure retries | 1 |
| Protocol retries | 0 |

Non-applicable cells remain explicit `FAILED / NOT APPLICABLE` records and are
included in the 13,750-cell denominator. They are never silently removed or
replaced by zero metrics.

## Required approval record

Before changing `config/v2.yaml` to `full-run-ready`, record:

1. provider and instance type;
2. operating system and Python/runtime image;
3. vCPU, RAM, storage, and expected wall-time budget;
4. whether the instance is ephemeral or persistent;
5. data egress and retention policy; and
6. the exact command/commit used to start the dataset-granular runner.

The current Windows host is recorded for development and validation only. No
cloud provider, instance type, or external data transfer is implied by this
file.
