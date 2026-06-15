# Checkpoint Comparison and Dual Catch-Rate Report

This report compares three related 2020 matched-volume results:

1. Tanisha's reported 2020 result.
2. A local rerun using Tanisha's Stage 1 checkpoint files from `Specifications/`.
3. My support experiment using locally regenerated Stage 1 checkpoints, without Tanisha's checkpoint weights.

The purpose is not to compete with Tanisha's result. The purpose is to show whether the framework conclusion is stable when we run the full 2020 universe, and to report both event-day catch-rate definitions Tanisha asked for.

## Result Types

| Result | What it uses | Status |
|---|---|---|
| Tanisha reported 2020 | Her saved checkpoint and pipeline artifacts | Primary thesis result provided by Tanisha |
| Tanisha-checkpoint rerun | `Specifications/stage1_fold_0/1/2_best.pt`, local Stage 2 retrain/recalibration | Closest local rerun using her Stage 1 weights |
| Regenerated-checkpoint support run | My locally trained ImageNet-initialized Stage 1 checkpoints, local Stage 2 retrain/recalibration | Independent support experiment without her checkpoints |

The Tanisha-checkpoint rerun is not guaranteed bit-for-bit identical to Tanisha's reported result because I retrained/recalibrated Stage 2 locally from her Stage 1 checkpoint outputs. Still, it is much closer than the regenerated-checkpoint support run because it uses her Stage 1 checkpoint weights.

## Checkpoint Provenance

| Fold | Tanisha checkpoint | My regenerated checkpoint |
|---|---|---|
| 0 | `Specifications/stage1_fold_0_best.pt`<br>`sha256=e78d59e5...` | `outputs/checkpoints/stage1_2019_3fold_imagenet_noamp_20260615/stage1_fold_0_best.pt`<br>`sha256=67f083a3...` |
| 1 | `Specifications/stage1_fold_1_best.pt`<br>`sha256=b6e963bf...` | `outputs/checkpoints/stage1_2019_3fold_imagenet_noamp_20260615/stage1_fold_1_best.pt`<br>`sha256=8aa779b3...` |
| 2 | `Specifications/stage1_fold_2_best.pt`<br>`sha256=21cf2083...` | `outputs/checkpoints/stage1_2019_3fold_imagenet_noamp_20260615/stage1_fold_2_best.pt`<br>`sha256=8ab97bca...` |

## Data Universe

Both local full-2020 runs now use the corrected 2020 universe:

| Check | Value |
|---|---:|
| 2020 event folders | 201 |
| 2020 HDF5 events | 201 |
| Raw temporal samples | 3,287 |
| Samples after `--exclude-empty-current-fire` | 2,184 |
| Rolling decision rows | 5,382 |
| Positive rolling decision rows | 4,493 |

Candidate rows differ slightly because the Stage 1 probabilities affect the candidate feature rows:

| Run | Candidate rows | Candidate positives |
|---|---:|---:|
| Tanisha reported | 5,212,367 | not reported in message |
| Tanisha-checkpoint rerun | 5,212,363 | 255,761 |
| Regenerated-checkpoint support run | 5,212,827 | 255,105 |

## Primary Matched-Volume Comparison

All rows use one global WSTS/Stage 1 probability threshold matched to that run's Stage 2 alert volume. No per-event-day top-K thresholding is used.

| Run | Method | Alerts/event-day | Precision | Recall | True-positive alerts | Catch rate, positive event-days | Catch rate, all event-days |
|---|---|---:|---:|---:|---:|---:|---:|
| Tanisha reported | Stage 2 | 89.53 | 0.2590 | 0.2193 | 124,790 | 0.7084 | not available |
| Tanisha reported | WSTS-as-alerter | 89.52 | 0.2166 | 0.1834 | 104,361 | 0.4908 | not available |
| Tanisha-checkpoint rerun | Stage 2 | 114.29 | 0.2521 | 0.2725 | 155,098 | 0.7334 | 0.6122 |
| Tanisha-checkpoint rerun | WSTS-as-alerter | 114.30 | 0.2062 | 0.2229 | 126,837 | 0.5304 | 0.4428 |
| Regenerated-checkpoint support | Stage 2 | 75.88 | 0.2621 | 0.1881 | 107,046 | 0.6490 | 0.5418 |
| Regenerated-checkpoint support | WSTS-as-alerter | 75.89 | 0.2316 | 0.1662 | 94,602 | 0.4730 | 0.3948 |

## Gains Over WSTS

| Run | TP alert gain | Relative TP gain | Precision gain | Recall gain | Positive-day catch-rate gain | All-day catch-rate gain |
|---|---:|---:|---:|---:|---:|---:|
| Tanisha reported | 20,429 | 19.58% | +0.0424 | +0.0359 | +0.2176 | not available |
| Tanisha-checkpoint rerun | 28,261 | 22.28% | +0.0460 | +0.0497 | +0.2030 | +0.1695 |
| Regenerated-checkpoint support | 12,444 | 13.15% | +0.0305 | +0.0219 | +0.1761 | +0.1470 |

All three comparisons support the same core conclusion: at matched alert volume, Stage 2 produces higher-quality alerts than WSTS-as-alerter.

## Alert Fatigue vs Naive 4 km

| Run | Method | Alerts/event-day | Precision | Alert reduction vs naive |
|---|---|---:|---:|---:|
| Tanisha reported | Stage 2 | 89.53 | 0.1747 | 97.01% |
| Tanisha reported | Naive 4 km | 2,589.46 | 0.0396 | - |
| Tanisha-checkpoint rerun | Stage 2 | 114.29 | 0.1704 | 96.35% |
| Tanisha-checkpoint rerun | Naive 4 km | 2,589.46 | 0.0396 | - |
| Regenerated-checkpoint support | Stage 2 | 75.88 | 0.1592 | 97.41% |
| Regenerated-checkpoint support | Naive 4 km | 2,589.46 | 0.0396 | - |

Both local runs support Tanisha's alert-fatigue claim: Stage 2 sharply reduces alert volume while improving precision relative to naive 4 km flooding.

## Ranking Metrics

| Run | Stage 2 AP | WSTS AP | Distance AP | Stage 2 AUC | WSTS AUC | Distance AUC |
|---|---:|---:|---:|---:|---:|---:|
| Tanisha-checkpoint rerun | 0.2615 | 0.1866 | 0.2063 | 0.8250 | 0.7090 | 0.7935 |
| Regenerated-checkpoint support | 0.2511 | 0.1844 | 0.2040 | 0.8186 | 0.7089 | 0.7930 |

Both local runs support the ranking-quality claim: Stage 2 ranks candidates better than WSTS/Stage 1 and distance-only baselines.

## Catch-Rate Definitions

Tanisha's original event-day catch rate:

`positive-event-day catch rate = positive rolling decision rows with at least one true-positive alert / rolling decision rows with at least one positive candidate`

Additional metric requested by Tanisha:

`all-event-day catch rate = all rolling decision rows with at least one true-positive alert / all rolling decision rows`

The positive-event-day definition answers: when there was something to catch, how often did the method catch at least one positive alert?

The all-event-day definition answers: across the entire rolling alert schedule, how often did the method issue at least one true-positive alert?

Both definitions show Stage 2 ahead of WSTS in both local full-2020 runs.

## Commands and Artifacts

Main Tanisha-checkpoint runner:

```bash
scripts/31_run_tanisha_checkpoint_full2020_pipeline.sh \
  > outputs/logs/tanisha_ckpt_full2020_pipeline_20260615.log 2>&1
```

Key Tanisha-checkpoint artifacts:

- `outputs/calibration/stage1_2019_3fold_tanisha_ckpt/`
- `outputs/predictions/stage1_2019_3fold_tanisha_ckpt_oof/`
- `outputs/stage2/candidates_3fold_r4km_fair_tanisha_ckpt/`
- `outputs/stage2/models_3fold_r4km_tanisha_ckpt/`
- `outputs/stage2/calibration_3fold_r4km_tanisha_ckpt/stage2_oof_threshold.json`
- `outputs/predictions/stage1_2020_tanisha_ckpt_full2020_ensemble/`
- `outputs/stage2/candidates_2020_r4km_fair_tanisha_ckpt_full2020/`
- `outputs/stage2/scored_2020_r4km_fair_tanisha_ckpt_full2020/`
- `outputs/stage2/matched_volume_tanisha_ckpt_full2020/`
- `outputs/stage2/matched_volume_tanisha_ckpt_full2020_dual_catch/`
- `outputs/stage2/rolling_alerts_2020_tanisha_ckpt_full2020/`
- `outputs/stage2/evaluation_2020_tanisha_ckpt_full2020/`

Key regenerated-checkpoint dual catch artifact:

- `outputs/stage2/matched_volume_imagenet_noamp_full2020_retrained_stage2_dual_catch/stage2_2020_regenerated_ckpt_full2020_dual_catch_rate.json`

## Bottom Line

The earlier 50-event discrepancy was caused by incomplete data, not by Tanisha's method. On the corrected full 2020 universe:

- Tanisha's reported result shows Stage 2 beating WSTS-as-alerter.
- The local rerun using Tanisha's Stage 1 checkpoints also shows Stage 2 beating WSTS-as-alerter.
- The independent support experiment using my regenerated checkpoints also shows Stage 2 beating WSTS-as-alerter.
- Both catch-rate definitions show the same direction: Stage 2 catches more event-days than WSTS.

This strengthens the integrity story: the conclusion is not dependent on a single reporting definition or only on my regenerated checkpoint run.
