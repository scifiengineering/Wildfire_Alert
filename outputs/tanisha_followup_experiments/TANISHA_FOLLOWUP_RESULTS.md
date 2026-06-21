# Tanisha Follow-up Experiment Results

Generated: 2026-06-21

## Figure Guide

All new follow-up figures are in `outputs/tanisha_followup_experiments/figures/`. This folder now contains only the three new figures that correspond to Tanisha's requested follow-up experiments. The existing final-pipeline figures remain in `outputs/figures/final_pipeline/` and are not duplicated here.

| Figure | What it shows | Why it matters |
|---|---|---|
| `01_top5_added_feature_shap_heatmap.png` | Highest-scoring sampled alert across five unique fire events, with grouped added-feature SHAP contributions. | Direct answer to the SHAP request: it shows how added feature groups contribute at top alert locations across five unique fire events. |
| `02_top5_alert_locations_lat_lon_table.png` | Top Stage 2 raster row/column alert locations converted to latitude/longitude. | Direct answer to the coordinate-conversion request. |
| `03_feature_group_ablation_impact.png` | Percent AP and within-event AUC drop after removing added feature groups. | Direct answer to the no-added-feature and ablation request. |

Related figure tables:

- Added-feature SHAP table: `outputs/tanisha_followup_experiments/figures/01_top5_added_feature_group_shap.csv`
- Standalone top-5 coordinate table: `outputs/tanisha_followup_experiments/figures/02_top5_alert_locations_lat_lon_table.csv`
- Top alert lat/lon table: `outputs/tanisha_followup_experiments/tables/top_alert_locations_lat_lon.csv`
- Note: `.venv` does not currently include the `shap` package, so SHAP was not recomputed; the new view uses existing tracked SHAP outputs under `outputs/stage2/shap/`.

## Table Definitions

### Figure Source Tables

`01_top5_added_feature_group_shap.csv` is the source table for Figure 1.

| Column | Definition |
|---|---|
| `event_id`, `target_date` | Fire event and prediction date for the selected top alert. |
| `row`, `col` | Raster pixel location produced by the model. |
| `latitude`, `longitude` | Pixel location converted to geographic coordinates. |
| `model_score` | Stage 2 LightGBM risk score for that alert location. |
| `burned_tomorrow` | Whether that candidate pixel burned on the next day. |
| `Fire geometry` | Sum of SHAP contributions from current-fire and distance/geometry features. |
| `Wind + alignment` | Sum of SHAP contributions from wind and spread-alignment features. |
| `Weather + drought` | Sum of SHAP contributions from weather, forecast weather, ERC, humidity, and PDSI features. |
| `Terrain + veg` | Sum of SHAP contributions from slope, aspect, elevation, vegetation, and landcover features. |

`02_top5_alert_locations_lat_lon_table.csv` is the source table for Figure 2.

| Column | Definition |
|---|---|
| `Rank` | Rank by Stage 2 risk score. |
| `Fire event`, `Date` | Event and target date for the alert. |
| `Latitude`, `Longitude` | Converted geographic coordinates for the raster alert pixel. |
| `Risk` | Stage 2 LightGBM risk score. |
| `Burned?` | Whether that alert location burned on the next day. |

### Ablation Result Tables

`Mean Ablation Results` averages the two validation directions. `Fold-Level Results` shows each direction separately.

| Column | Definition |
|---|---|
| `variant` | Feature set used for that LightGBM run. `no_extra_features` keeps only basic Stage 1/current-fire/distance-style features; `drop_*` removes one added feature group from the full feature set. |
| `direction` | Training/validation direction. `fold0_to_fold1` means trained on fold 0 candidates and evaluated on fold 1 candidates; `fold1_to_fold0` is the reverse. |
| `average_precision` | Area under the precision-recall curve for candidate ranking. Higher is better. |
| `within_event_auc` | Mean ROC AUC computed inside event-day candidate groups. Higher is better for ranking candidates within the same fire context. |
| `ap_drop_pct_of_full` | Percent AP loss relative to the accepted full-feature model for the same fold direction. Positive means performance dropped when the feature group was removed. |
| `within_event_auc_drop_pct_of_full` | Percent within-event AUC loss relative to the accepted full-feature model for the same fold direction. Positive means performance dropped. |
| `passes_discriminative_check` | Whether the model met the existing discriminative check: within-event AUC at least 0.65 and better than distance-only baseline. |

## Mean Ablation Results

| variant                          | average_precision | within_event_auc | ap_drop_pct_of_full | within_event_auc_drop_pct_of_full | passes_discriminative_check |
| -------------------------------- | ----------------- | ---------------- | ------------------- | --------------------------------- | --------------------------- |
| no_extra_features                | 0.1444            | 0.7509           | 65.0727             | 5.9701                            | 0.0000                      |
| drop_weather_drought_features    | 0.3031            | 0.8133           | 27.3042             | -1.8360                           | 1.0000                      |
| drop_fire_geometry_features      | 0.3404            | 0.7391           | 17.4597             | 7.4519                            | 0.0000                      |
| drop_terrain_vegetation_features | 0.3776            | 0.8332           | 8.3944              | -4.3319                           | 1.0000                      |
| drop_wind_features               | 0.4130            | 0.7987           | -0.3466             | -0.0097                           | 1.0000                      |
| drop_stage1_threshold_features   | 0.4847            | 0.8221           | -18.2682            | -2.9404                           | 1.0000                      |

## Fold-Level Results

| variant                          | direction      | average_precision | within_event_auc | ap_drop_pct_of_full | within_event_auc_drop_pct_of_full | passes_discriminative_check |
| -------------------------------- | -------------- | ----------------- | ---------------- | ------------------- | --------------------------------- | --------------------------- |
| drop_fire_geometry_features      | fold0_to_fold1 | 0.3188            | 0.7342           | 19.0772             | 8.1383                            | False                       |
| drop_fire_geometry_features      | fold1_to_fold0 | 0.3620            | 0.7440           | 15.8422             | 6.7655                            | False                       |
| drop_stage1_threshold_features   | fold0_to_fold1 | 0.5232            | 0.8287           | -32.7925            | -3.6835                           | True                        |
| drop_stage1_threshold_features   | fold1_to_fold0 | 0.4462            | 0.8155           | -3.7438             | -2.1972                           | True                        |
| drop_terrain_vegetation_features | fold0_to_fold1 | 0.3583            | 0.8442           | 9.0721              | -5.6255                           | True                        |
| drop_terrain_vegetation_features | fold1_to_fold0 | 0.3969            | 0.8222           | 7.7167              | -3.0383                           | True                        |
| drop_weather_drought_features    | fold0_to_fold1 | 0.2090            | 0.8202           | 46.9561             | -2.6241                           | True                        |
| drop_weather_drought_features    | fold1_to_fold0 | 0.3972            | 0.8063           | 7.6523              | -1.0480                           | True                        |
| drop_wind_features               | fold0_to_fold1 | 0.4052            | 0.8130           | -2.8418             | -1.7170                           | True                        |
| drop_wind_features               | fold1_to_fold0 | 0.4208            | 0.7844           | 2.1486              | 1.6977                            | True                        |
| no_extra_features                | fold0_to_fold1 | 0.1266            | 0.7663           | 67.8727             | 4.1246                            | False                       |
| no_extra_features                | fold1_to_fold0 | 0.1623            | 0.7356           | 62.2727             | 7.8157                            | False                       |

## Interpretation

- Removing added features drops mean AP by 65.1% and mean within-event AUC by 6.0% versus the accepted full-feature models.
- Among single group removals, `drop_weather_drought_features` has the largest mean AP drop (27.3% of full-model AP).
- For within-event ranking, `drop_fire_geometry_features` has the largest mean AUC drop (7.5% of full-model AUC).
- Negative drops mean the ablated model scored higher than the original full-feature reference on that metric/fold; treat those as evidence of feature redundancy or split variance, not as a claim that the removed group is harmful.
