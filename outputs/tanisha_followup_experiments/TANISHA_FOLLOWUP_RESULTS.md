# Tanisha Follow-up Experiment Results

Generated: 2026-06-23

## Figure Guide

All new follow-up figures are in `outputs/tanisha_followup_experiments/figures/`. This folder now contains only the three new figures that correspond to Tanisha's requested follow-up experiments. The existing final-pipeline figures remain in `outputs/figures/final_pipeline/` and are not duplicated here.

| Figure | What it shows | Why it matters |
|---|---|---|
| `01_top5_added_feature_shap_heatmap.png` | Highest-scoring regenerated-checkpoint sampled alert across five unique fire events, with grouped added-feature SHAP contributions. | Direct answer to the SHAP request: it shows how added feature groups contribute at top alert locations across five unique fire events. |
| `02_top5_alert_locations_lat_lon_table.png` | Top Stage 2 raster row/column alert locations converted to latitude/longitude. | Direct answer to the coordinate-conversion request. |
| `03_feature_group_ablation_impact.png` | 2020 regenerated-checkpoint percent AP and within-event AUC drop after removing added feature groups. | Direct answer to the no-added-feature and ablation request. |

Related figure tables:

- Added-feature SHAP table: `outputs/tanisha_followup_experiments/figures/01_top5_added_feature_group_shap.csv`
- Standalone top-5 coordinate table: `outputs/tanisha_followup_experiments/figures/02_top5_alert_locations_lat_lon_table.csv`
- Top alert lat/lon table: `outputs/tanisha_followup_experiments/tables/top_alert_locations_lat_lon.csv`
- Provenance: the active follow-up figures and 2020 ablation table use the regenerated ImageNet/no-AMP checkpoint artifacts, not `*_tanisha_ckpt*` artifacts.

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

### 2020 Ablation Result Table

`outputs/tanisha_followup_experiments/2020_ablation/2020_ablation_metrics.csv` is the source table for the 2020 regenerated-checkpoint ablation.

| Column | Definition |
|---|---|
| `variant` | Feature set used for that LightGBM run. `no_extra_features` keeps only basic Stage 1/current-fire/distance-style features; `drop_*` removes one added feature group from the full feature set. |
| `average_precision` | Area under the precision-recall curve for candidate ranking. Higher is better. |
| `within_event_auc` | Mean ROC AUC computed inside event-day candidate groups. Higher is better for ranking candidates within the same fire context. |
| `ap_drop_percent_vs_full` | Percent AP loss relative to the regenerated full-feature model. Positive means performance dropped when the feature group was removed. |
| `within_event_auc_drop_percent_vs_full` | Percent within-event AUC loss relative to the regenerated full-feature model. Positive means performance dropped. |

## 2020 Regenerated-Checkpoint Ablation Results

| variant | average_precision | within_event_auc | ap_drop_percent_vs_full | within_event_auc_drop_percent_vs_full |
|---|---:|---:|---:|---:|
| drop_fire_geometry_features | 0.1629 | 0.6752 | 35.1393 | 21.4829 |
| drop_weather_drought_features | 0.2491 | 0.8625 | 0.8038 | -0.2906 |
| drop_terrain_vegetation_features | 0.2492 | 0.8605 | 0.7342 | -0.0587 |
| full_features | 0.2511 | 0.8600 | 0.0000 | 0.0000 |
| no_extra_features | 0.2620 | 0.8590 | -4.3518 | 0.1141 |
| drop_wind_features | 0.2695 | 0.8571 | -7.3242 | 0.3367 |

## Interpretation

- The full-feature row matches the regenerated-checkpoint 2020 discrimination summary: AP `0.2511` and within-event AUC `0.8600`.
- Removing fire-geometry features causes the largest drop: AP drops by `35.1%` and within-event AUC drops by `21.5%`.
- Weather/drought and terrain/vegetation removals cause small AP drops and nearly unchanged within-event AUC.
- The no-extra and no-wind variants score slightly higher on AP than the full model in this regenerated 2020 run; treat negative drops as split/test-set variance or feature redundancy, not as a claim that those features are harmful.
