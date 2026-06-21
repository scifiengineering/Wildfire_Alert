# Tanisha Follow-up Figures

This folder contains only the new figures for Tanisha's follow-up request. It does not duplicate or replace the existing final-pipeline figures in `outputs/figures/final_pipeline/`.

## Figure 1: Added-Feature SHAP Contributions

- File: `01_top5_added_feature_shap_heatmap.png`
- Source table: `01_top5_added_feature_group_shap.csv`
- Purpose: shows the highest-scoring sampled alert across five unique fire events and summarizes how added feature groups contribute to each alert score.
- Use for: Tanisha's request to use the SHAP file to explain top alert locations across five unique fire events.

## Figure 2: Raster Pixels Converted to Latitude/Longitude

- File: `02_top5_alert_locations_lat_lon_table.png`
- Source table: `02_top5_alert_locations_lat_lon_table.csv`
- Purpose: converts the top Stage 2 raster row/column alert locations into latitude/longitude coordinates.
- Use for: Tanisha's request to show what model output locations look like after coordinate conversion.

## Figure 3: Added-Feature Ablation Impact

- File: `03_feature_group_ablation_impact.png`
- Purpose: shows the percent performance drop after removing added feature groups, using average precision and within-event AUC.
- Use for: Tanisha's request to test whether LightGBM performance goes down without added features and which feature groups contribute most.

The earlier scatter plot attempt was removed because it was less clear than the heatmap.
