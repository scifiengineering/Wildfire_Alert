# Candidate Filter Sensitivity Report

This side experiment evaluates whether simple candidate eligibility filters can improve Stage 2 precision at fixed recall targets.

The purpose is exploratory. It should not replace the primary matched-volume comparison unless the method section is explicitly updated to include candidate filtering as part of the alerting system.

## Question

Can precision be improved at a chosen recall level by filtering candidate locations before alert selection?

## Method

The experiment uses the existing full-2020 scored candidate files. Stage 1 and Stage 2 scores are not retrained. Instead, candidate rows are filtered by simple eligibility rules, then candidates are ranked by the existing Stage 2 score.

For each filter, the analysis reports the minimum number of top-ranked Stage 2 alerts required to reach recall targets of:

- 0.10
- 0.15
- 0.20
- 0.25

Recall is measured against the base candidate universe:

```text
future candidates within 4 km of current fire
```

The tested filters are:

- distance from current fire: `<= 2 km`, `<= 3 km`, `<= 4 km`;
- Stage 1 probability thresholds: `>= 0.10`, `>= 0.20`, `>= 0.30`;
- Stage 1 top-mask membership;
- wind alignment and forecast wind alignment;
- simple combined rules using distance, Stage 1 probability, and forecast wind alignment.

## Filters Tested and Rationale

The filters are applied after candidate scoring for this sensitivity test. Stage 1 output, Stage 2 model weights, and Stage 2 scores are kept fixed. The experiment only changes which already-scored candidate rows are eligible for alert selection.

| Filter type | Examples | Rationale |
|---|---|---|
| Distance from current fire | `distance <= 2 km`, `distance <= 3 km`, base `distance <= 4 km` | Fire spread is spatially local, so farther candidates may contribute false positives. |
| Stage 1 probability | `stage1_probability >= 0.10`, `>= 0.20`, `>= 0.30` | Very low Stage 1 probability candidates may be weak candidates before Stage 2 ranking. |
| Stage 1 top mask | `stage1_top_mask == 1` | Tests whether restricting to the highest Stage 1 risk pixels improves precision. |
| Wind alignment | `wind_alignment >= 0` | Fire spread may be more plausible in the wind-aligned direction. |
| Forecast wind alignment | `forecast_wind_alignment >= 0` | Tests the same spread-direction idea using forecast wind. |
| Combined rules | `distance <= 3 km and stage1_probability >= 0.10`; `distance <= 3 km and forecast_wind_alignment >= 0` | Tests simple combinations of spatial proximity, Stage 1 risk, and physical spread plausibility. |

These filters were chosen because they are simple, interpretable, based on features already available to Stage 2, and physically plausible. They should be understood as first-pass sensitivity checks, not as an exhaustive search for the best possible filtering policy.

Other filters could also be tested, including:

- additional Stage 1 probability thresholds such as `0.05`, `0.15`, `0.25`, or `0.40`;
- quantile-based Stage 1 filters, such as keeping the top 5%, 10%, or 20% per event-day;
- fire-front distance bands that exclude candidates that are either too close or too far;
- terrain, vegetation, landcover, drought, or fire-danger filters;
- learned candidate-pruning models;
- validation-set optimization of a multi-filter rule subject to a minimum recall constraint.

The main caution is that trying many filters can become test-set tuning if the rule is selected based on 2020 performance. A thesis-safe version would choose any filtering rule using training/validation data, then evaluate it once on the 2020 test set.

## Key Findings

Filtering can improve precision at low and moderate recall targets, especially filters based on minimum Stage 1 probability. The gains are smaller at higher recall because stricter filters begin to remove true positives that are needed to reach the recall target.

### Regenerated-Checkpoint Support Run

| Recall target | Base precision | Best filter | Best precision | Precision gain | Base alerts/day | Filtered alerts/day |
|---:|---:|---|---:|---:|---:|---:|
| 0.10 | 0.2887 | `stage1_probability_ge_0_30` | 0.3305 | +0.0418 | 37.28 | 32.57 |
| 0.15 | 0.2682 | `stage1_probability_ge_0_20` | 0.2852 | +0.0170 | 60.19 | 56.61 |
| 0.20 | 0.2515 | `stage1_probability_ge_0_30` | 0.2568 | +0.0053 | 85.58 | 83.82 |
| 0.25 | 0.2366 | `distance_le_2km` | 0.2366 | +0.0000 | 113.71 | 113.71 |

### Tanisha-Checkpoint Rerun

| Recall target | Base precision | Best filter | Best precision | Precision gain | Base alerts/day | Filtered alerts/day |
|---:|---:|---|---:|---:|---:|---:|
| 0.10 | 0.3118 | `stage1_probability_ge_0_30` | 0.3410 | +0.0292 | 34.52 | 31.56 |
| 0.15 | 0.2798 | `stage1_probability_ge_0_20` | 0.3100 | +0.0302 | 57.71 | 52.08 |
| 0.20 | 0.2649 | `stage1_probability_ge_0_20` | 0.2786 | +0.0137 | 81.27 | 77.27 |
| 0.25 | 0.2486 | `distance_le_3km_and_stage1_probability_ge_0_10` | 0.2559 | +0.0072 | 108.22 | 105.16 |

## Interpretation

The most useful filters were simple Stage 1 probability gates. This suggests that some low-probability candidate locations add false positives without helping much at low-to-moderate recall.

However, the improvement is not uniform:

- At recall `0.10` and `0.15`, filtering gives a meaningful precision improvement.
- At recall `0.20`, filtering still helps, but the gain is smaller.
- At recall `0.25`, filtering gives little or no improvement in the regenerated-checkpoint run and only a small improvement in the Tanisha-checkpoint rerun.

This means filtering is best framed as an operating-point or deployment-policy option, not as evidence that the primary Stage 2 result should be replaced.

## Thesis-Safety Note

If candidate filtering becomes part of the main method, then the thesis must explicitly justify:

- why the filter rule was selected;
- whether the rule was chosen using training/validation data only;
- whether the WSTS baseline gets the same candidate eligibility policy;
- how much recall and event-day catch rate are lost;
- whether improvement comes from Stage 2 learning or from the eligibility filter.

Given limited defense time, this experiment is safest as a secondary sensitivity analysis or future-work direction. It shows that stricter candidate eligibility can improve precision at lower recall targets, but it also changes the alerting policy being evaluated.

## Artifacts

Script:

```text
scripts/33_candidate_filter_sensitivity.py
```

Results:

```text
outputs/stage2/filter_sensitivity/candidate_filter_sensitivity.json
outputs/stage2/filter_sensitivity/candidate_filter_sensitivity_summary.csv
```
