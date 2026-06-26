# Stage 2 Separate Baseline Comparison Report

## Purpose

This report separates the Stage 2 comparison into three independent baseline views for lab-level reporting and future research reuse.

This report reuses the existing regenerated full-2020 artifacts and keeps each baseline in its own section:

- Stage 2 vs WSTS / Stage 1 probability baseline
- Stage 2 vs distance-only baseline
- Stage 2 vs naive 4 km alerting rule

## Recommended Reporting Framing

Use WSTS as the main learned baseline if the goal is to compare against the closest prior-work signal. Keep distance and naive 4 km as separate supporting comparisons:

- WSTS answers: does Stage 2 improve over the WSTS-style Stage 1 probability signal?
- Distance answers: does Stage 2 do more than choose the nearest cells to current fire?
- Naive 4 km answers: does Stage 2 reduce alert fatigue compared with alerting everywhere near the fire front?

These are different benchmark questions, so they should be shown separately rather than forced into one combined baseline table.

## Rolling Alert Experiment: Separate Baselines

The original rolling alert experiment only compared Stage 2 with the naive 4 km rule. The added rolling-baseline comparison below evaluates WSTS and distance in the same rolling 2 km -> 3 km -> 4 km protocol, with each baseline matched to Stage 2's total alert volume.

| Rolling comparison | Stage 2 | Baseline | Stage 2 effect |
|---|---:|---:|---:|
| Alerts/event-day vs WSTS | 75.88 | 75.88 | matched |
| Precision vs WSTS | 0.2621 | 0.2316 | +0.0305 |
| Recall vs WSTS | 0.1881 | 0.1662 | +0.0219 |
| True-positive alerts vs WSTS | 107,046 | 94,597 | +12,449 |
| Mean AP vs WSTS | 0.2409 | 0.1731 | +0.0678 |
| Mean AUC vs WSTS | 0.8200 | 0.6559 | +0.1641 |
| Alerts/event-day vs distance | 75.88 | 75.88 | matched |
| Precision vs distance | 0.2621 | 0.2413 | +0.0208 |
| Recall vs distance | 0.1881 | 0.1731 | +0.0149 |
| True-positive alerts vs distance | 107,046 | 98,538 | +8,508 |
| Mean AP vs distance | 0.2409 | 0.1942 | +0.0467 |
| Mean AUC vs distance | 0.8200 | 0.8118 | +0.0082 |
| Alerts/event-day vs naive 4 km | 75.88 | 2,589.46 | -97.41% |
| Precision vs naive 4 km | 0.1592 | 0.0396 | +0.1197 |

Interpretation: in the rolling alert experiment, Stage 2 beats WSTS and distance at the same alert volume. Against naive 4 km, the comparison is not volume matched; it is an alert-fatigue result showing that Stage 2 produces far fewer alerts with higher precision.

## 1. Stage 2 vs WSTS

### Matched-Volume Alert Quality

This is the cleanest primary baseline comparison. WSTS is used as an alerter with one global Stage 1 probability threshold selected to match Stage 2's alert volume.

| Metric | Stage 2 | WSTS-as-alerter | Stage 2 gain |
|---|---:|---:|---:|
| Alerts/event-day | 75.88 | 75.89 | volume matched |
| Precision | 0.2621 | 0.2316 | +0.0305 |
| Recall | 0.1881 | 0.1662 | +0.0219 |
| True-positive alerts | 107,046 | 94,602 | +12,444 |
| Event-day catch rate | 0.5418 | 0.3948 | +0.1470 |

Interpretation: at essentially the same alert volume, Stage 2 catches more positive alert occurrences than WSTS and improves precision, recall, and event-day catch rate.

### Ranking Quality

| Metric | Stage 2 | WSTS / Stage 1 | Stage 2 gain |
|---|---:|---:|---:|
| Average Precision | 0.2511 | 0.1844 | +0.0666 |
| ROC AUC | 0.8186 | 0.7089 | +0.1098 |
| Within-event AUC | 0.8600 | 0.6961 | +0.1639 |

Interpretation: Stage 2 ranks candidate alert locations better than WSTS / Stage 1 alone across AP, global AUC, and within-event AUC.

## 2. Stage 2 vs Distance

This comparison asks whether the model adds value beyond a simple nearest-to-current-fire rule.

### 2020 Full-Regenerated Ranking Quality

| Metric | Stage 2 | Distance-only | Stage 2 gain |
|---|---:|---:|---:|
| Average Precision | 0.2511 | 0.2040 | +0.0471 |
| ROC AUC | 0.8186 | 0.7930 | +0.0256 |
| Within-event AUC | 0.8600 | 0.8507 | +0.0093 |

Interpretation: Stage 2 improves over distance-only ranking on all three 2020 ranking metrics, though the within-event AUC gain is smaller than the WSTS gain.

### 2019 Fold-Level Distance Comparison

| Held-out fold | Stage 2 AP | Distance AP | AP gain |
|---|---:|---:|---:|
| Fold 0 | 0.2093 | 0.1541 | +0.0551 |
| Fold 1 | 0.2700 | 0.2082 | +0.0618 |
| Fold 2 | 0.2870 | 0.1790 | +0.1080 |

Interpretation: Stage 2 improves AP over distance-only ranking in all regenerated 2019 folds.

## 3. Stage 2 vs Naive 4 km

This comparison is an operational alert-fatigue comparison. The naive baseline alerts broadly within 4 km, so the key question is whether Stage 2 reduces unnecessary alerts while preserving better precision.

### 2020 Rolling Alert Volume

| Metric | Stage 2 | Naive 4 km | Stage 2 effect |
|---|---:|---:|---:|
| Alerts/event-day | 75.88 | 2,589.46 | -97.41% |
| Precision | 0.1592 | 0.0396 | +0.1197 |

Interpretation: Stage 2 sends far fewer alerts than naive 4 km alerting and has much higher precision.

### 2019 Rolling Alert Volume

| Metric | Stage 2 | Naive 4 km | Stage 2 effect |
|---|---:|---:|---:|
| Alerts/event-day | 39.91 | 1,079.99 | -96.88% |
| Precision | 0.1188 | 0.0271 | +0.0917 |

Interpretation: the alert-fatigue result is consistent in the regenerated 2019 rolling evaluation.

## Baseline Choice Guide

| Research question | Use this comparison |
|---|---|
| Stage 2 improves over the closest learned WSTS signal | Stage 2 vs WSTS |
| Stage 2 is not merely a distance heuristic | Stage 2 vs distance |
| Stage 2 reduces alert overload for operations | Stage 2 vs naive 4 km |

## Source Artifacts

- WSTS matched-volume summary: `outputs/stage2/matched_volume_imagenet_noamp_full2020_retrained_stage2/stage2_2020_external_ensemble_imagenet_noamp_full2020_retrained_stage2_matched_volume_summary.json`
- WSTS/distance rolling matched-volume summary: `outputs/stage2/rolling_baseline_comparisons_2020_imagenet_noamp_full2020_retrained_stage2/stage2_2020_external_ensemble_imagenet_noamp_full2020_retrained_stage2_rolling_baseline_summary.json`
- 2020 ranking summary: `outputs/stage2/evaluation_2020_imagenet_noamp_full2020_retrained_stage2/stage2_2020_external_ensemble_imagenet_noamp_full2020_retrained_stage2_evaluation.json`
- 2020 rolling alert summary: `outputs/stage2/rolling_alerts_2020_imagenet_noamp_full2020_retrained_stage2/stage2_2020_external_ensemble_imagenet_noamp_full2020_retrained_stage2_rolling_summary.json`
- 2019 aggregate table summary: `outputs/stage2/regenerated_checkpoint_tables/tables_3_2_3_3_3_4_summary.json`
