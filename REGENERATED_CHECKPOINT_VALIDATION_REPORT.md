# Supportive Regenerated-Checkpoint Validation Report

> Superseded note, 2026-06-15: this report was produced from an incomplete 50-event 2020 download caused by the Google Drive folder download limit. The full 2020 universe has since been restored to 201 events and 2,184 active-current-fire samples. Use `FULL_2020_REGENERATED_CHECKPOINT_VALIDATION_REPORT.md` for the corrected validation result. This file is preserved as a record of the incomplete-subset audit.

This report is a supportive reproducibility and integrity check of the thesis framework using regenerated ImageNet-initialized Stage 1 checkpoints. It is not a competing experiment and should not be framed as "the regenerated result versus the thesis result." The purpose is to check whether the same comparison framework produces the same direction of evidence, and to clearly flag any places where regenerated run differs from the reported thesis comparison.

This is not an exact artifact reproduction of the reported thesis result because the original checkpoint LFS objects were not available locally. The run below uses regenerated checkpoints and the currently available 2020 data/artifacts.

## How to Read This Report

The numbers below should be read as a sensitivity check under regenerated artifacts, not as an alternative thesis result. the reported thesis result remains the primary result for the thesis claim because it uses the frozen pipeline and artifacts.

The most important difference is alert volume. the saved thesis 2020 run produces about 89.53 Stage 2 alerts/event-day. The regenerated-checkpoint run produces about 25.23 Stage 2 alerts/event-day using the newly calibrated regenerated-pipeline Stage 2 threshold, `0.8117221967059977`. That difference is expected because this run uses a smaller available 2020 data universe and regenerated Stage 1 checkpoints, which change the Stage 2 score distribution.

| Input / volume check | saved thesis 2020 run | The regenerated-checkpoint run |
|---|---:|---:|
| 2020 events | 201 | 50 |
| 2020 samples | 2,184 | 342 |
| Stage 2 candidate rows | 5,212,367 | 373,223 |
| Naive 4 km alerts/event-day | 2,589.46 | 1,156.02 |
| Stage 2 threshold | 0.7777810642 | 0.8117221967 |
| Stage 2 alerts/event-day | 89.53 | 25.23 |

Because the alert budgets differ so much, the regenerated-checkpoint run should not be used as an exact numerical reproduction of the thesis point 1. It is still valuable as an integrity check: if the same framework gives the same direction, that supports the claim; if it gives a different direction, that difference must be highlighted rather than hidden. In this run, the matched-volume WSTS comparison is different from the reported thesis direction on precision, recall, and true-positive alerts, while the alert-fatigue and ranking-quality comparisons point in the same direction as the thesis.

## Audit Finding at a Glance

| Thesis comparison | the reported thesis direction | Regenerated-checkpoint direction | Integrity/audit reading |
|---|---|---|---|
| Stage 2 vs WSTS-as-alerter at matched volume | Stage 2 better on precision, recall, true-positive alerts, and event-day catch rate | Stage 2 better only on event-day catch rate; WSTS better on precision, recall, and true-positive alerts | Different result direction for precision/recall/TP. This does not validate the full point 1 claim without the original artifacts. |
| Stage 2 vs naive 4 km flooding | Stage 2 far fewer alerts and higher precision | Stage 2 far fewer alerts and higher precision | Same direction. Supports the alert-fatigue claim. |
| Stage 2 vs WSTS/Stage 1 and distance ranking | Stage 2 better ranking quality | Stage 2 better AP/AUC than WSTS/Stage 1 and distance | Same direction. Supports the ranking-quality claim. |

## Script and Artifact Audit

The audit checked the public repo scripts that are shared between the original thesis documented 2020 workflow and this run:

- `scripts/20_evaluate_rolling_alerts.py`
- `scripts/21_predict_stage1_year.py`
- `scripts/22_ensemble_stage1_predictions.py`
- `scripts/23_build_stage2_candidates_year.py`
- `scripts/25_score_stage2_ensemble.py`

There were no local code diffs in those shared scripts. The documented `--exclude-empty-current-fire` flag was also tested on available 2020 data and produced the same candidate count, so it does not explain the lower true-positive count.

After the original `26_compare_matched_alert_volume.py` and `27_summarize_matched_alert_volume.py` were shared into `Specifications/`, the comparison checked them with the local implementation and ran the original comparison script on the scored candidates. The original script could not run standalone because the helper module `wildfire_alert.evaluation.matched_alerts` is not present in the repo, so this run used a compatibility implementation matching the API and metric definitions imported by the original script.

the original script on the scored candidates produced the same matched-volume precision, recall, and true-positive discrepancy:

| Metric | Stage 2 | WSTS-as-alerter |
|---|---:|---:|
| Alert count | 18,545 | 18,566 |
| Alerts/event-day | 25.23 | 25.26 |
| Precision | 0.2203 | 0.2457 |
| Recall | 0.1423 | 0.1589 |
| True-positive alerts | 4,085 | 4,561 |
| Event-day catch rate | 0.6047 | 0.3419 |

This confirms that the lower true-positive count is not caused by the local matched-volume script. One metric-definition difference was found: the first local summary computed event-day catch rate over all rolling decisions, while the original script defines it over positive rolling decisions only. Using the original thesis definition, Stage 2's event-day catch-rate advantage in this run is `0.6047` vs `0.3419`.

The main remaining differences are artifact/data differences:

- This run uses regenerated Stage 1 checkpoints, not the original Stage 1 checkpoints.
- The available 2020 data/artifact universe is smaller: 50 events and 342 samples rather than 201 events and 2,184 samples.
- The run corrected the earlier hybrid run by retraining Stage 2 from regenerated 2019 Stage 1 out-of-fold predictions. The corrected Stage 2 models are in `outputs/stage2/models_3fold_r4km_imagenet_noamp/`.

Therefore, the lower true-positive count is not currently explained by a known shared-script mismatch or by the earlier Stage 2 shortcut. After retraining Stage 2 consistently, the discrepancy remains. The remaining likely causes are regenerated Stage 1 checkpoints and incomplete/different 2020 data coverage compared with the original thesis full 201-event run.

## Experiment 1: Matched-Volume WSTS-as-Alerter

Claim tested: at the same alert budget, Stage 2 should choose better alert locations than WSTS/Stage 1 used directly as an alerter.

| Metric | Original Stage 2 | Original WSTS | Regenerated Stage 2 | Regenerated WSTS |
|---|---:|---:|---:|---:|
| Alerts/event-day | 89.53 | 89.52 | 25.23 | 25.26 |
| Precision | 0.2590 | 0.2166 | 0.2203 | 0.2457 |
| Recall | 0.2193 | 0.1834 | 0.1423 | 0.1589 |
| True-positive alerts | 124,790 | 104,361 | 4,085 | 4,561 |
| Event-day catch rate | 0.7084 | 0.4908 | 0.6047 | 0.3419 |

Audit conclusion: different result direction for the main matched-volume claim. This run confirms the matched-volume setup and shows Stage 2 catches more event-days than WSTS, but it does not reproduce the original thesis higher precision or higher true-positive alert count under the smaller regenerated-checkpoint alert budget. This should be reported openly as an artifact-sensitive discrepancy, not hidden and not framed as a rival thesis result.

Why the alert volume is lower: the runs use different frozen Stage 2 thresholds because mine was recalibrated from regenerated 2019 OOF Stage 2 scores: original run uses `0.7777810642240797`, while the regenerated pipeline uses `0.8117221967059977` and a smaller 2020 candidate universe. the saved thesis 2020 rolling run has 201 events, 2,184 samples, and a naive 4 km volume of 2,589.46 alerts/event-day. Available 2020 run has 50 events, 342 samples, and a naive 4 km volume of 1,156.02 alerts/event-day. Therefore, the regenerated-pipeline Stage 2 threshold produces about 89.53 alerts/event-day in the saved thesis run but about 25.23 alerts/event-day in mine.

## Experiment 2: Alert Fatigue vs Naive 4 km Flooding

Claim tested: Stage 2 should issue far fewer alerts than the naive 4 km rule while improving alert precision.

| Metric | Original Stage 2 | Original Naive 4 km | Regenerated Stage 2 | Regenerated Naive 4 km |
|---|---:|---:|---:|---:|
| Alerts/event-day | 89.53 | 2,589.46 | 25.23 | 1,156.02 |
| Precision | 0.1747 | 0.0396 | 0.1215 | 0.0296 |
| Alert reduction vs naive | 97.01% | - | 98.11% | - |

Supportive validation conclusion: agrees with the thesis direction. The regenerated-checkpoint run confirms the alert-fatigue claim: Stage 2 produces dramatically fewer alerts than naive 4 km flooding and has much higher precision.

## Experiment 3: Ranking Quality

Claim tested: Stage 2 should rank candidate locations better than WSTS/Stage 1 and distance-only baselines.

| Metric | Original Stage 2 | Original WSTS/Stage 1 | Original Distance | Regenerated Stage 2 | Regenerated WSTS/Stage 1 | Regenerated Distance |
|---|---:|---:|---:|---:|---:|---:|
| Average Precision | 0.2459 | secondary baseline lower in reported folds | secondary baseline lower in reported folds | 0.1998 | 0.1549 | 0.1367 |
| ROC AUC | 0.8220 | - | - | 0.8179 | 0.7606 | 0.7658 |
| Within-event AUC | - | - | - | 0.8455 | 0.6768 | 0.8280 |
| AP gain over WSTS | - | - | - | +0.0449 | - | - |
| AP gain over distance | - | - | - | +0.0631 | - | - |

Supportive validation conclusion: agrees with the thesis direction. The regenerated-checkpoint run confirms that Stage 2 ranks candidates better than WSTS/Stage 1 and distance-only baselines by AP and AUC.

## Overall Validator Summary

| Thesis point | Reproducibility / integrity result |
|---|---|
| 1. Stage 2 beats WSTS-as-alerter at matched alert volume | Not reproduced in full. Stage 2 improves event-day catch rate in this run, but WSTS has better precision, recall, and true-positive count. This must be highlighted as a discrepancy. |
| 2. Stage 2 reduces alert fatigue vs naive 4 km flooding | Reproduced directionally. Stage 2 strongly reduces alert volume and improves precision. |
| 3. Stage 2 improves ranking quality vs WSTS/Stage 1 and distance | Reproduced directionally. Stage 2 has better AP/AUC than WSTS/Stage 1 and distance. |

The independent regenerated-checkpoint run supports the thesis operational alert-fatigue claim and ranking-quality claim. It does not independently reproduce the full matched-volume WSTS-as-alerter claim. That difference is the main audit finding: the framework itself is runnable and fair, but the strongest point 1 result depends on the exact checkpoint/data artifacts used in the reported thesis run.

## Artifacts Created

- `scripts/30_run_regenerated_stage2_pipeline.sh`
- `outputs/predictions/stage1_2019_3fold_imagenet_noamp_oof/`
- `outputs/stage2/candidates_3fold_r4km_fair_imagenet_noamp/`
- `outputs/stage2/models_3fold_r4km_imagenet_noamp/`
- `outputs/stage2/calibration_3fold_r4km_imagenet_noamp/stage2_oof_threshold.json`
- `outputs/stage2/scored_2020_r4km_fair_imagenet_noamp_retrained_stage2/`
- `outputs/stage2/matched_volume_imagenet_noamp_retrained_stage2/`
- `outputs/stage2/matched_volume_imagenet_noamp_retrained_stage2_reference_script/`
- `outputs/stage2/rolling_alerts_2020_imagenet_noamp_retrained_stage2/`
- `outputs/stage2/evaluation_2020_imagenet_noamp_retrained_stage2/`
