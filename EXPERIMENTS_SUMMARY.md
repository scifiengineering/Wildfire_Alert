# Experiments Summary

This document summarizes the reproduction and validation work done around Tanisha's matched-volume wildfire alerting experiment. It focuses on the important differences we found, why the numbers changed between runs, and how the corrected experiments support the same thesis argument.

## Core Thesis Argument

Tanisha's alerting argument is not that Stage 2 produces the exact same number under every checkpoint or machine. The argument is:

1. At matched alert volume, Stage 2 is a better alerting system than WSTS/Stage 1 used directly as an alerter.
2. Stage 2 sharply reduces alert fatigue compared with naive 4 km flooding.
3. Stage 2 ranks candidate alert locations better than WSTS/Stage 1 and distance-only baselines.

The fairest comparison for point 1 uses one global WSTS threshold matched to Stage 2's total/mean alert volume. We did not use per-event-day top-K thresholding.

## Main Reproduction Lessons

### 1. Checkpoint differences change absolute values

We used two local checkpoint tracks:

| Track | What it used | Purpose |
|---|---|---|
| Tanisha-checkpoint rerun | `Specifications/stage1_fold_0/1/2_best.pt` | Closest local rerun using her Stage 1 weights |
| Regenerated-checkpoint support run | Locally retrained ImageNet-initialized Stage 1 checkpoints | Independent support experiment without her checkpoint weights |

The checkpoint files are not identical. This changes Stage 1 probability maps, Stage 2 candidate features, Stage 2 model scores, and the frozen Stage 2 alert threshold. Therefore, exact alert volume and true-positive counts differ across runs.

That difference is expected and does not undermine the thesis if the comparison direction remains stable.

### 2. The full-2020 runs support the same framework direction

All full-2020 runs use the same 201-event / 2,184-sample 2020 evaluation universe. The absolute alert volumes differ because each run has its own frozen checkpoint and threshold artifacts, but the Stage 2 versus WSTS comparison direction remains stable.

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

## Follow-up Feature Ablations

Tanisha also requested a smaller follow-up around explainability, coordinate conversion, and whether the added Stage 2 features matter. The SHAP and coordinate-conversion files were already present in the repository:

- SHAP generation and artifacts: `scripts/15_compute_stage2_shap.py`, `outputs/stage2/shap/`
- Pixel-to-latitude/longitude conversion: `scripts/17_generate_pipeline_visualizations.py`
- Existing coordinate table: `outputs/figures/final_pipeline/05_stage2_high_alert_locations.csv`

I trained two-fold LightGBM variants on the 2019 `r2km_fair` candidate tables. The accepted full-feature models are the reference. The no-extra-feature baseline kept only Stage 1 probability/threshold-mask features plus current-fire distance/geometry features.

| Variant | Mean AP | Mean within-event AUC | Mean AP drop vs full | Mean AUC drop vs full | Discriminative check |
|---|---:|---:|---:|---:|---:|
| No added features | 0.1444 | 0.7509 | 65.1% | 6.0% | 0/2 folds passed |
| Drop weather/drought features | 0.3031 | 0.8133 | 27.3% | -1.8% | 2/2 folds passed |
| Drop fire geometry features | 0.3404 | 0.7391 | 17.5% | 7.5% | 0/2 folds passed |
| Drop terrain/vegetation features | 0.3776 | 0.8332 | 8.4% | -4.3% | 2/2 folds passed |
| Drop wind features | 0.4130 | 0.7987 | -0.3% | -0.0% | 2/2 folds passed |
| Drop Stage 1 threshold features | 0.4847 | 0.8221 | -18.3% | -2.9% | 2/2 folds passed |

The clearest thesis-facing finding is that the model without added features performs much worse: AP falls by about 65% and both no-extra-feature folds fail the discriminative check. Among individual feature groups, weather/drought features cause the largest AP loss when removed, while fire-geometry features cause the largest within-event AUC loss and fail the discriminative check.

The follow-up figures should be additions, not replacements for Tanisha's existing `outputs/figures/final_pipeline/` set. I rejected the initial top-5 SHAP scatter plot generated during this follow-up because it was not presentation-worthy: the x-axis labels were crowded, annotations collided with the title, repeated event labels dominated the plot, and the visual did not communicate a clean claim.

The follow-up figure folder contains only the three new figures for Tanisha's request, under `outputs/tanisha_followup_experiments/figures/`. The detailed follow-up report `outputs/tanisha_followup_experiments/TANISHA_FOLLOWUP_RESULTS.md` includes a figure guide explaining what each diagram shows and why it matters:

- `01_top5_added_feature_shap_heatmap.png`: highest-scoring sampled alert across five unique fire events, with grouped added-feature SHAP contributions.
- `02_top5_alert_locations_lat_lon_table.png`: standalone coordinate table for the highest-risk Stage 2 alert locations.
- `03_feature_group_ablation_impact.png`: percent performance drop by feature group for AP and within-event AUC.

For the thesis/report, these should be considered supplemental follow-up figures beside Tanisha's existing `outputs/figures/final_pipeline/` set, not replacements.

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

In the full-2020 thesis-facing results summarized here, I did not find a result that reverses Tanisha's main argument:

- Tanisha's reported result supports the argument.
- The local rerun using Tanisha's Stage 1 checkpoints supports the argument.
- The independent regenerated-checkpoint support run supports the argument.
- Both catch-rate definitions support the argument.
- Alert fatigue and ranking metrics support the argument.

So the meaningful final audit finding is supportive. The remaining differences are differences in absolute values caused by non-identical checkpoint, score, and threshold artifacts.

## Reports and Artifacts

- `FULL_2020_REGENERATED_CHECKPOINT_VALIDATION_REPORT.md`
- `CHECKPOINT_COMPARISON_AND_DUAL_CATCH_REPORT.md`
- `outputs/stage2/matched_volume_tanisha_ckpt_full2020/`
- `outputs/stage2/matched_volume_tanisha_ckpt_full2020_dual_catch/`
- `outputs/stage2/matched_volume_imagenet_noamp_full2020_retrained_stage2/`
- `outputs/stage2/matched_volume_imagenet_noamp_full2020_retrained_stage2_dual_catch/`
