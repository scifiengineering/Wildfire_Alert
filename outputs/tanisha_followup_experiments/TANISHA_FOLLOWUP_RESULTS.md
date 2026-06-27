# Follow-up Experiment Results

Generated: 2026-06-23

## Figure Guide

All new follow-up figures are in this follow-up output folder's `figures/` directory. This folder now contains only the three new figures that correspond to the requested follow-up experiments. The existing final-pipeline figures remain in `outputs/figures/final_pipeline/` and are not duplicated here.

| Figure | What it shows | Why it matters |
|---|---|---|
| `01_top5_added_feature_shap_heatmap.png` | Highest-scoring regenerated-checkpoint sampled alert across five unique fire events, with grouped added-feature SHAP contributions. | Direct answer to the SHAP request: it shows how added feature groups contribute at top alert locations across five unique fire events. |
| `02_top5_alert_locations_lat_lon_table.png` | Top Stage 2 raster row/column alert locations converted to latitude/longitude. | Direct answer to the coordinate-conversion request. |
| `03_feature_group_ablation_impact.png` | 2020 regenerated-checkpoint percent AP and within-event AUC drop after removing the specific distance, bearing, alignment, and Stage 1 calibration features. | Direct answer to the corrected ablation request. |

Related figure tables:

- Added-feature SHAP table: `figures/01_top5_added_feature_group_shap.csv`
- Standalone top-5 coordinate table: `figures/02_top5_alert_locations_lat_lon_table.csv`
- Top alert lat/lon table: `tables/top_alert_locations_lat_lon.csv`
- Provenance: the active follow-up figures and 2020 ablation table use the regenerated ImageNet/no-AMP checkpoint artifacts, not the original-checkpoint artifact family.

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

`2020_ablation/2020_ablation_metrics.csv` is the source table for the 2020 regenerated-checkpoint ablation.

| Column | Definition |
|---|---|
| `row` | Row number matching the corrected ablation table. |
| `variant` | Feature set used for that LightGBM run. `drop_*` removes the named feature or feature family from the full feature set; `minimal_geometry_baseline` keeps only `stage1_probability` and `distance_to_current_fire_km`. |
| `feature_configuration` | Human-readable feature configuration for the row. |
| `average_precision` | Area under the precision-recall curve for candidate ranking. Higher is better. |
| `within_event_auc` | Mean ROC AUC computed inside event-day candidate groups. Higher is better for ranking candidates within the same fire context. |
| `ap_drop_percent_vs_full` | Percent AP loss relative to the regenerated full-feature model. Positive means performance dropped when the feature group was removed. |
| `within_event_auc_drop_percent_vs_full` | Percent within-event AUC loss relative to the regenerated full-feature model. Positive means performance dropped. |

## 2020 Regenerated-Checkpoint Ablation Results

| row | variant | average_precision | within_event_auc | ap_drop_percent_vs_full | within_event_auc_drop_percent_vs_full |
|---:|---|---:|---:|---:|---:|
| 1 | full_features | 0.2511 | 0.8600 | 0.0000 | 0.0000 |
| 2 | drop_distance_features | 0.1901 | 0.6875 | 24.2821 | 20.0550 |
| 3 | drop_bearing_feature | 0.2507 | 0.8601 | 0.1355 | -0.0127 |
| 4 | drop_wind_alignment_features | 0.2517 | 0.8580 | -0.2296 | 0.2262 |
| 5 | drop_slope_alignment_feature | 0.2513 | 0.8606 | -0.1025 | -0.0680 |
| 6 | drop_stage1_calibration_group | 0.2564 | 0.8588 | -2.1202 | 0.1331 |
| 7 | drop_all_geometry_alignment_features | 0.1879 | 0.6885 | 25.1528 | 19.9344 |
| 8 | minimal_geometry_baseline | 0.2533 | 0.8579 | -0.8663 | 0.2458 |

## Interpretation

- The full-feature row matches the regenerated-checkpoint 2020 discrimination summary: AP `0.2511` and within-event AUC `0.8600`.
- Distance features carry the clearest added signal: removing `distance_to_current_fire_px` and `distance_to_current_fire_km` drops AP by `24.3%` and within-event AUC by `20.1%`.
- Removing all geometry/alignment features gives a similar degradation: AP drops by `25.2%` and within-event AUC by `19.9%`.
- Bearing, wind alignment, slope alignment, and Stage 1 calibration removals are close to the full model in this regenerated 2020 run; small negative drops should be treated as split/test-set variance or feature redundancy, not as evidence that the features are harmful.
- The minimal two-feature baseline is competitive on AP but slightly lower on within-event AUC, so the cleanest contribution claim is that distance/geometry features drive most of the measurable gain in within-event discrimination.
