# Experiments Summary

This document summarizes the reproduction and validation work done around Tanisha's matched-volume wildfire alerting experiment. It focuses on the important differences we found, why the numbers changed between runs, and how the corrected experiments support the same thesis argument.

## Core Thesis Argument

Tanisha's alerting argument is not that Stage 2 produces the exact same number under every checkpoint or machine. The argument is:

1. At matched alert volume, Stage 2 is a better alerting system than WSTS/Stage 1 used directly as an alerter.
2. Stage 2 sharply reduces alert fatigue compared with naive 4 km flooding.
3. Stage 2 ranks candidate alert locations better than WSTS/Stage 1 and distance-only baselines.

The fairest comparison for point 1 uses one global WSTS threshold matched to Stage 2's total/mean alert volume. We did not use per-event-day top-K thresholding.

## Main Reproduction Lessons

### 1. The first 2020 run was incomplete

The first reproduced 2020 run used only 50 events because the Google Drive folder download stopped at the folder-download limit. That produced a misleading 50-event subset:

| Check | Incomplete subset | Correct full 2020 |
|---|---:|---:|
| 2020 events | 50 | 201 |
| Active-current-fire samples | 342 | 2,184 |
| Candidate rows | 373,223 | about 5.21M |

The 50-event result should not be used to judge Tanisha's result. It was an incomplete-data artifact.

### 2. After restoring the full data, the conclusion changed

Once the full 201-event 2020 universe was restored and converted to HDF5, the regenerated-checkpoint support run changed direction and supported Tanisha's matched-volume claim.

| Run | Stage 2 TP alerts | WSTS TP alerts | Result |
|---|---:|---:|---|
| Incomplete 50-event subset | 4,085 | 4,561 | Did not support point 1 |
| Full 2020 regenerated-checkpoint run | 107,046 | 94,602 | Supports point 1 |

This is the most important audit finding: the earlier discrepancy was driven by incomplete data, not by evidence that Tanisha's method was invalid.

### 3. Checkpoint differences change absolute values

We used two local checkpoint tracks:

| Track | What it used | Purpose |
|---|---|---|
| Tanisha-checkpoint rerun | `Specifications/stage1_fold_0/1/2_best.pt` | Closest local rerun using her Stage 1 weights |
| Regenerated-checkpoint support run | Locally retrained ImageNet-initialized Stage 1 checkpoints | Independent support experiment without her checkpoint weights |

The checkpoint files are not identical. This changes Stage 1 probability maps, Stage 2 candidate features, Stage 2 model scores, and the frozen Stage 2 alert threshold. Therefore, exact alert volume and true-positive counts differ across runs.

That difference is expected and does not undermine the thesis if the comparison direction remains stable.

## Primary Matched-Volume Results

All rows below compare Stage 2 against WSTS-as-alerter at matched alert volume using one global WSTS threshold.

| Run | Method | Alerts/day | Precision | Recall | TP alerts | Positive-day catch | All-day catch |
|---|---|---:|---:|---:|---:|---:|---:|
| Tanisha reported | Stage 2 | 89.53 | 0.2590 | 0.2193 | 124,790 | 0.7084 | not available |
| Tanisha reported | WSTS | 89.52 | 0.2166 | 0.1834 | 104,361 | 0.4908 | not available |
| Tanisha-checkpoint rerun | Stage 2 | 114.29 | 0.2521 | 0.2725 | 155,098 | 0.7334 | 0.6122 |
| Tanisha-checkpoint rerun | WSTS | 114.30 | 0.2062 | 0.2229 | 126,837 | 0.5304 | 0.4428 |
| Regenerated-checkpoint support | Stage 2 | 75.88 | 0.2621 | 0.1881 | 107,046 | 0.6490 | 0.5418 |
| Regenerated-checkpoint support | WSTS | 75.89 | 0.2316 | 0.1662 | 94,602 | 0.4730 | 0.3948 |

All full-2020 matched-volume comparisons support the same argument: Stage 2 produces higher-quality alerts than WSTS-as-alerter at the same alert budget.

## Gains Over WSTS

| Run | TP alert gain | Relative TP gain | Precision gain | Recall gain | Positive-day catch gain | All-day catch gain |
|---|---:|---:|---:|---:|---:|---:|
| Tanisha reported | 20,429 | 19.58% | +0.0424 | +0.0359 | +0.2176 | not available |
| Tanisha-checkpoint rerun | 28,261 | 22.28% | +0.0460 | +0.0497 | +0.2030 | +0.1695 |
| Regenerated-checkpoint support | 12,444 | 13.15% | +0.0305 | +0.0219 | +0.1761 | +0.1470 |

These are not identical numeric results, but they support the same claim. The value of the extra runs is that the conclusion survives both her checkpoint files and independently regenerated checkpoints.

## Alert Fatigue Comparison

Stage 2 also supports the alert-fatigue argument against naive 4 km flooding.

| Run | Method | Alerts/day | Precision | Alert reduction vs naive |
|---|---|---:|---:|---:|
| Tanisha reported | Stage 2 | 89.53 | 0.1747 | 97.01% |
| Tanisha reported | Naive 4 km | 2,589.46 | 0.0396 | - |
| Tanisha-checkpoint rerun | Stage 2 | 114.29 | 0.1704 | 96.35% |
| Tanisha-checkpoint rerun | Naive 4 km | 2,589.46 | 0.0396 | - |
| Regenerated-checkpoint support | Stage 2 | 75.88 | 0.1592 | 97.41% |
| Regenerated-checkpoint support | Naive 4 km | 2,589.46 | 0.0396 | - |

This supports the practical alerting story: Stage 2 gives far fewer alerts than naive flooding while maintaining much better precision.

## Ranking Quality

Both local full-2020 runs also support the ranking-quality claim.

| Run | Stage 2 AP | WSTS AP | Distance AP | Stage 2 AUC | WSTS AUC | Distance AUC |
|---|---:|---:|---:|---:|---:|---:|
| Tanisha-checkpoint rerun | 0.2615 | 0.1866 | 0.2063 | 0.8250 | 0.7090 | 0.7935 |
| Regenerated-checkpoint support | 0.2511 | 0.1844 | 0.2040 | 0.8186 | 0.7089 | 0.7930 |

Stage 2 ranks candidate locations better than both WSTS/Stage 1 and distance-only baselines.

## Catch-Rate Definitions

Tanisha's requested clarification is useful because the two catch-rate definitions answer different questions.

`positive-day catch rate` asks: when there was at least one positive candidate to catch, how often did the method catch at least one?

`all-day catch rate` asks: across the full rolling alert schedule, how often did the method issue at least one true-positive alert?

Both definitions support Stage 2:

| Run | Method | Positive-day catch | All-day catch |
|---|---|---:|---:|
| Tanisha-checkpoint rerun | Stage 2 | 0.7334 | 0.6122 |
| Tanisha-checkpoint rerun | WSTS | 0.5304 | 0.4428 |
| Regenerated-checkpoint support | Stage 2 | 0.6490 | 0.5418 |
| Regenerated-checkpoint support | WSTS | 0.4730 | 0.3948 |

This means the conclusion is not dependent on only one catch-rate definition.

## Anything Not Supporting Her Argument?

The only result that did not support her primary matched-volume claim was the early 50-event subset run. That run showed WSTS ahead of Stage 2 on precision, recall, and true-positive alerts. However, that run is not a valid full-2020 reproduction because the 2020 data download was incomplete.

After correcting the data universe to 201 events and 2,184 samples:

- Tanisha's reported result supports the argument.
- The local rerun using Tanisha's Stage 1 checkpoints supports the argument.
- The independent regenerated-checkpoint support run supports the argument.
- Both catch-rate definitions support the argument.
- Alert fatigue and ranking metrics support the argument.

So the meaningful final audit finding is supportive. The non-supportive result is preserved as a data-completeness lesson, not as evidence against the thesis method.

## Reports and Artifacts

- `FULL_2020_REGENERATED_CHECKPOINT_VALIDATION_REPORT.md`
- `CHECKPOINT_COMPARISON_AND_DUAL_CATCH_REPORT.md`
- `REGENERATED_CHECKPOINT_VALIDATION_REPORT.md` superseded 50-event subset report
- `REPRODUCTION_COMMANDS_AND_VALIDATION.md` superseded 50-event command report
- `outputs/stage2/matched_volume_tanisha_ckpt_full2020/`
- `outputs/stage2/matched_volume_tanisha_ckpt_full2020_dual_catch/`
- `outputs/stage2/matched_volume_imagenet_noamp_full2020_retrained_stage2/`
- `outputs/stage2/matched_volume_imagenet_noamp_full2020_retrained_stage2_dual_catch/`
