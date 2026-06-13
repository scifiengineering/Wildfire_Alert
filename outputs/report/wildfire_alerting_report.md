# Wildfire Next-Day Discriminative Alerting Report

## Executive Summary

This project implements a two-stage post-ignition wildfire alerting pipeline on the
WildfireSpreadTS (WSTS) dataset. The system asks a practical operational question:

> Given an already active wildfire, which nearby candidate locations are most likely to burn tomorrow, and why?

The implemented prototype uses 2019 WSTS data only as a focused proof-of-concept scope.
Within that reduced scope, the pipeline is working end to end:

1. Stage 1 produces next-day fire probability maps using a ResNet-18 U-Net.
2. Stage 1 probabilities are calibrated to usable operating thresholds.
3. Stage 2 extracts candidate alert locations near current fire fronts.
4. Stage 2 trains a LightGBM discriminative alert model.
5. Stage 2 is evaluated against distance-only and Stage-1-only baselines.
6. TreeSHAP explanations are generated globally and locally.

The strongest evidence for the discriminative-alerting objective is the held-out Stage 2
evaluation. The fair Stage 2 model beats both distance-only and Stage-1-only baselines:

| Evaluation direction | Model AP | Distance AP | Stage 1 AP | Model within-event AUC | Distance within-event AUC | Stage 1 within-event AUC |
|---|---:|---:|---:|---:|---:|---:|
| Fold 0 -> Fold 1 | 0.3940 | 0.1933 | 0.1527 | 0.7992 | 0.7949 | 0.6217 |
| Fold 1 -> Fold 0 | 0.4301 | 0.1451 | 0.1635 | 0.7980 | 0.7558 | 0.5645 |

In simple words, the model ranks real next-day burn candidates substantially better than
simple rules such as "choose the closest cells" or "use the Stage 1 probability directly."
The within-event AUC of approximately 0.80 means that, within a fire event-day, the model
ranks a burned candidate above an unburned candidate about 80% of the time.

## Problem Objective

The project objective is not general pre-ignition fire risk mapping. It is a post-ignition
alerting task:

1. A fire is already detected.
2. The system observes recent fire, weather, vegetation, topography, and drought variables.
3. The system predicts which nearby cells are more likely to burn tomorrow.
4. The system explains why one candidate location is riskier than another nearby candidate.

The key expected output is therefore discriminative:

> Two nearby locations at similar distance from a fire should be allowed to receive different
> alert scores if wind, terrain, drought, fuel, or learned spread probability differ.

## Related Work: What It Helped With and What Was Missing

The main related work is the WSTS dataset and its baseline modeling setup. It helped this
project in four ways:

1. It provided a clean daily time-series wildfire spread dataset.
2. It defined the next-day active-fire-map prediction task.
3. It emphasized severe class imbalance and recommended Average Precision.
4. It exposed useful physical channels such as wind, NDVI, slope, aspect, drought, and active fire.

However, the related work mostly focuses on spatial fire spread prediction as a segmentation
task. That is useful, but incomplete for alerting. In an operational alerting setting,
decision-makers need more than a probability map:

1. They need candidate-level alert scores near an active fire.
2. They need to know whether the model is doing more than ranking by distance.
3. They need explanations for why one nearby location is more dangerous than another.
4. They need interpretable evidence from wind, terrain, drought, vegetation, and model outputs.

This project fills that missing layer by adding a second-stage discriminative alert model
and SHAP explanations on top of Stage 1 spread prediction.

## Challenges and How They Connect to the Objectives

| Challenge | Why it matters for the objective | Project response |
|---|---|---|
| Extreme class imbalance | Very few pixels burn tomorrow, so accuracy is misleading. | Used AP, Dice/F1, candidate positive-rate checks, and balanced LightGBM training. |
| Explainability | Alerts must be justifiable to a professor or decision-maker. | Added global and local TreeSHAP explanations. |
| Related-work gap | Existing WSTS baselines predict maps but do not explain per-location alerts. | Added Stage 2 discriminative candidate ranking and SHAP outputs. |

The objectives and challenges are therefore directly correlated. The goal required a model
that can rank rare burn locations while still explaining why one nearby location receives a
higher alert score than another. The implementation therefore focuses on better candidate
ranking, class-imbalance-aware learning, and interpretable evidence for the alert decision.

## Contributions

The project contributions are:

1. A reproducible two-stage wildfire alerting prototype on WSTS. Stage 1 converts raster
   observations into next-day burn probability signals, and Stage 2 converts nearby candidate
   locations into ranked alert scores.
2. A stable Stage 1 U-Net training pipeline with event-split support, mixed precision,
   finite-value guards, and checkpoint safety. These safeguards helped prevent unstable
   training runs from corrupting the best checkpoints and made validation metrics reliable.
3. Candidate-level feature engineering for discriminative alerting. Each candidate row
   combines spatial context, Stage 1 probability, wind direction/speed/alignment, terrain,
   drought, humidity, temperature, vegetation, and fire-context features. This gives the
   classifier more evidence than a simple "closest to fire" rule.
4. Class-imbalance-aware Stage 2 LightGBM training. The model uses balanced class weighting
   so rare burned candidates influence the training loss more strongly than the large number
   of non-burn candidates.
5. Baseline-driven validation against distance-only and Stage-1-only ranking. This verifies
   that the full model improves candidate prioritization rather than merely reproducing a
   simple heuristic.
6. Within-event discrimination evaluation. This directly measures whether the model ranks
   burned cells above unburned cells inside the same fire event-day, which matches the central
   discriminative-alert objective.
7. TreeSHAP global and local explanations, including paired similar-distance examples. These
   explanations show how features such as wind, drought, terrain, vegetation, and Stage 1
   probability push individual alert scores upward or downward.

## Dataset and Experimental Scope

The original plan targeted full WSTS evaluation. The implemented experiment uses 2019 as
the current proof-of-concept scope.

Key scope decisions:

- The current prototype focuses on 2019 WSTS events.
- Event-level train/validation splits were used within 2019.
- Two stable event-split Stage 1 folds were trained.
- Stage 2 was evaluated in two directions: fold0 -> fold1 and fold1 -> fold0.

This is not yet a full official WSTS leave-year-out benchmark. It is a focused research
prototype demonstrating the discriminative-alerting concept.

Held-out/test fire IDs used in the current Stage 2 evaluation:

| Fold | Fire IDs |
|---|---|
| Fold 0 validation/test set | fire_22710141, fire_22938749, fire_23036871, fire_23037424, fire_23159637, fire_23159647, fire_23159789, fire_23159838, fire_23159862, fire_23160488, fire_23300839, fire_23300886, fire_23301037, fire_23301386, fire_23410619 |
| Fold 1 validation/test set | fire_22863125, fire_22938746, fire_23036565, fire_23037424, fire_23159637, fire_23159647, fire_23159838, fire_23159842, fire_23159855, fire_23159860, fire_23300665, fire_23300699, fire_23301046, fire_23301386, fire_23572745 |

## Stage 1: Spread Prediction and Calibration

Stage 1 uses a ResNet-18 U-Net to produce per-pixel next-day burn probabilities. The initial
qualitative inspection showed that the model worked mainly as a ranking signal. Therefore,
we selected practical operating thresholds and top-risk budgets from validation data.

| Fold | AP | Best threshold | Dice | IoU | Precision | Recall | Best top-risk budget |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fold 0 | 0.0872 | 7.84e-05 | 0.2311 | 0.1307 | 0.2385 | 0.2242 | 0.05% |
| Fold 1 | 0.0927 | 8.35e-05 | 0.2093 | 0.1169 | 0.1780 | 0.2540 | 0.05% |

Interpretation:

- Stage 1 works better as a ranking signal than as a directly calibrated probability map.
- The usable threshold is around 8e-05, not 0.5.
- A top 0.05% risk budget was a stable operating point.
- Stage 1 alone was weaker than Stage 2 for candidate-level discrimination.

Important qualitative figures:

- `outputs/figures/stage1_qualitative_calibrated_ranked/fold0_00_fire_22938749_2019-06-18.png`
- `outputs/figures/stage1_qualitative_calibrated_ranked/fold1_00_fire_22938746_2019-06-08.png`

## Stage 2: Candidate Extraction

The Stage 2 model operates on candidate locations rather than every raster pixel.

Candidate generation initially included target pixels for diagnostics, but this was corrected.
The fair candidate set does not use tomorrow's burn mask. It uses:

1. A 2 km ring around current active fire pixels.
2. Stage 1 top-risk pixels.

This matches an operational alerting scenario where alerts are produced from current
observations and model risk signals.

Fair candidate statistics:

| Fold | Candidate rows | Positive rows | Positive rate |
|---|---:|---:|---:|
| Fold 0 | 51,330 | 3,269 | 6.37% |
| Fold 1 | 35,136 | 1,930 | 5.49% |

These rates are inside the desired 5-20% range for Stage 2 candidate learning.

## Stage 2: Model Results

Stage 2 uses LightGBM with balanced class weighting. It was trained in two directions:

1. Train on fold 0 candidates, validate on fold 1.
2. Train on fold 1 candidates, validate on fold 0.

| Direction | AP | ROC AUC | Within-event AUC | Best F1 | Best threshold |
|---|---:|---:|---:|---:|---:|
| Fold 0 -> Fold 1 | 0.3940 | 0.8008 | 0.7992 | 0.3922 | 0.3417 |
| Fold 1 -> Fold 0 | 0.4301 | 0.8232 | 0.7980 | 0.4001 | 0.6468 |

The most important metric for the thesis claim is within-event AUC. A value near 0.80 means
that within the same fire event-day, the model usually ranks burned candidate cells above
unburned candidate cells.

## Baseline Comparison

The model was compared against:

1. Distance-only ranking.
2. Stage-1-probability-only ranking.
3. Forecast-wind-alignment-only ranking.

| Direction | Model AP | Distance AP | Stage 1 AP | AP gain over distance | AP gain over Stage 1 |
|---|---:|---:|---:|---:|---:|
| Fold 0 -> Fold 1 | 0.3940 | 0.1933 | 0.1527 | 0.2008 | 0.2414 |
| Fold 1 -> Fold 0 | 0.4301 | 0.1451 | 0.1635 | 0.2850 | 0.2665 |

| Direction | Model within-event AUC | Distance within-event AUC | Stage 1 within-event AUC | Gain over distance | Gain over Stage 1 |
|---|---:|---:|---:|---:|---:|
| Fold 0 -> Fold 1 | 0.7992 | 0.7949 | 0.6217 | 0.0043 | 0.1775 |
| Fold 1 -> Fold 0 | 0.7980 | 0.7558 | 0.5645 | 0.0422 | 0.2335 |

Interpretation:

- The model clearly beats distance-only and Stage-1-only on AP.
- The model strongly beats Stage 1 on within-event discrimination.
- The model beats distance on within-event AUC, but the fold0 -> fold1 gain is small.
- This means distance remains a strong driver, but the full model adds useful information.

## SHAP Explainability

TreeSHAP was computed for both fair Stage 2 models. The SHAP outputs show that distance is
important, but not the only factor.

Top global SHAP features:

| Model | Top features |
|---|---|
| Fold 0 -> Fold 1 | top_threshold, distance_to_current_fire_px, elevation, pdsi, stage1_probability, forecast_temperature, forecast_wind_direction, minimum_temperature |
| Fold 1 -> Fold 0 | distance_to_current_fire_px, pdsi, stage1_probability, forecast_wind_speed, forecast_wind_direction, has_current_fire, minimum_temperature, specific_humidity |

Important SHAP artifacts:

- `outputs/stage2/shap/stage2_gbm_r2km_fair_fold0_to_fold1/global_beeswarm.png`
- `outputs/stage2/shap/stage2_gbm_r2km_fair_fold1_to_fold0/global_beeswarm.png`
- `outputs/stage2/shap/stage2_gbm_r2km_fair_fold0_to_fold1/paired_similar_distance_examples.json`
- `outputs/stage2/shap/stage2_gbm_r2km_fair_fold1_to_fold0/paired_similar_distance_examples.json`

The paired examples are especially important because they compare nearby candidate cells
with similar distance-to-fire but different model scores. These are the examples that support
the claim that the model is discriminative, not merely distance-based.

## Expected Output vs Achieved Output

Expected:

- A pipeline that can produce per-location alerts around an active wildfire.
- Alerts should be ranked better than simple distance-to-fire.
- Alerts should be explainable using physical and learned features.

Achieved:

- Stage 1 probability maps are generated and calibrated.
- Stage 2 candidate alerts are generated from current-fire proximity and Stage 1 top-risk signals.
- Stage 2 beats distance-only and Stage-1-only baselines on AP.
- Stage 2 reaches approximately 0.80 within-event AUC.
- SHAP explanations identify distance, Stage 1 probability, drought, wind, humidity,
  elevation, temperature, and vegetation features as contributors.

Therefore, the project achieved a working research prototype of discriminative wildfire
alerting. It is not yet a full official WSTS benchmark, but it demonstrates the central
thesis idea.

## Limitations

1. The prototype uses only 2019 data due time and compute limits.
2. Stage 1 performance is below the original full-data WSTS baseline target, likely because
   the experiment is reduced-data and event-split.
3. Distance remains a strong predictor; the model improves beyond it, but not equally strongly
   in both directions.
4. Stage 2 probabilities are not yet isotonic-calibrated.
5. Path-integrated features from the prompt were not fully implemented yet.
6. Official leave-year-out multi-year evaluation remains future work.
7. SHAP explanations describe the trained model's behavior, not causal wildfire physics.

## Recommended Next Steps

1. Generate alert-map visualizations showing Stage 2 scores around selected fires.
2. Add isotonic calibration for Stage 2 probabilities.
3. Add path-integrated features such as mean NDVI, elevation gain, and Stage 1 probability
   along the fire-to-candidate line.
4. Expand from 2019-only to multiple years as future project scope.
5. Run official WSTS year-based folds for stronger comparison with related work.
6. Convert this report into thesis sections: methodology, results, discussion, limitations.

## Reproducibility Artifacts

Key scripts:

- `scripts/03_train_stage1.py`
- `scripts/10_calibrate_stage1_threshold.py`
- `scripts/11_predict_stage1_event_split.py`
- `scripts/12_build_stage2_candidates.py`
- `scripts/13_train_stage2_gbm.py`
- `scripts/14_evaluate_stage2_model.py`
- `scripts/15_compute_stage2_shap.py`

Key outputs:

- `outputs/calibration/`
- `outputs/figures/stage1_qualitative_calibrated_ranked/`
- `outputs/predictions/stage1_2019_event/`
- `outputs/stage2/candidates_r2km_fair/`
- `outputs/stage2/models/`
- `outputs/stage2/evaluation/`
- `outputs/stage2/shap/`
