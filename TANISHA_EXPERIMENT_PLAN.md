# Tanisha Follow-up Experiment Plan

Date: 2026-06-21

## Inputs Found

- SHAP code/artifacts:
  - `scripts/15_compute_stage2_shap.py`
  - `outputs/stage2/shap/stage2_gbm_r2km_fair_fold0_to_fold1/`
  - `outputs/stage2/shap/stage2_gbm_r2km_fair_fold1_to_fold0/`
- Pixel-to-lat/lon code/artifact:
  - `scripts/17_generate_pipeline_visualizations.py`
  - `outputs/figures/final_pipeline/05_stage2_high_alert_locations.csv`
  - local HDF5 metadata root: `wsts_hdf5/`
- Stage 2 LightGBM candidates/models:
  - `outputs/stage2/candidates_r2km_fair/stage2_fold0_validation_candidates.csv`
  - `outputs/stage2/candidates_r2km_fair/stage2_fold1_validation_candidates.csv`
  - `outputs/stage2/models/stage2_gbm_r2km_fair_fold0_to_fold1.joblib`
  - `outputs/stage2/models/stage2_gbm_r2km_fair_fold1_to_fold0.joblib`

Remote check note: the SHAP and lat/lon files Tanisha referenced are visible locally and tracked in the current branch, so no remote file recovery is needed for those. Remote heads were checked on `origin` and `upstream` for awareness.

## Execution Plan

1. Re-run/confirm SHAP outputs for the accepted Stage 2 models.
   - Use `scripts/15_compute_stage2_shap.py`.
   - Produce local/global SHAP CSVs and plots in a new experiment output folder.
   - Add a top alert fire-event contribution figure/table from `sample_shap_values.csv`, focusing on grouped added-feature SHAP contributions.

2. Re-run/confirm the raster-pixel to latitude/longitude table.
   - Use `scripts/17_generate_pipeline_visualizations.py` with the existing HDF5 metadata root.
   - Preserve the generated CSV/table under the experiment output folder.

3. Train LightGBM without additional engineered features.
   - Keep only a compact baseline feature set:
     - `stage1_probability`
     - `distance_to_current_fire_px`
     - `distance_to_current_fire_km`
     - `has_current_fire`
     - `current_fire_at_candidate`
   - Train fold0->fold1 and fold1->fold0 variants.
   - Evaluate each against the matching validation fold.
   - Compare AP and within-event AUC against the full-feature accepted models.

4. Run ablations over extra feature groups.
   - Candidate extra groups:
     - forecast wind: `forecast_wind_*`
     - weather/drought: temperature, humidity, precipitation, ERC, BI, FM100, FM1000, PDSI
     - terrain/landcover: elevation, slope, aspect, population, vegetation, landcover where present
   - Train/evaluate one model per group removed.
   - Report percent contribution as performance drop relative to the full-feature model:
     - `100 * (full_metric - ablated_metric) / full_metric`

5. Write a compact summary report.
   - Include command log, produced artifacts, metric tables, and interpretation.

## Status

- [x] Located SHAP code and existing SHAP artifacts.
- [x] Located pixel-to-lat/lon conversion code and existing top-alert coordinate CSV.
- [x] Confirmed local `wsts_hdf5/` metadata exists for coordinate conversion.
- [x] Generate follow-up SHAP top-5 contribution view.
- [x] Generate/copy coordinate table into the follow-up experiment folder.
- [x] Train/evaluate no-additional-feature LightGBM models.
- [x] Train/evaluate ablation models.
- [x] Summarize findings.

## Results Written

- Summary report: `outputs/tanisha_followup_experiments/TANISHA_FOLLOWUP_RESULTS.md`
- Mean ablation metrics: `outputs/tanisha_followup_experiments/tables/ablation_metrics_mean.csv`
- Fold-level ablation metrics: `outputs/tanisha_followup_experiments/tables/ablation_metrics_by_fold.csv`
- Presentation candidate figures:
  - `outputs/tanisha_followup_experiments/figures/01_top5_added_feature_shap_heatmap.png`
  - `outputs/tanisha_followup_experiments/figures/02_top5_alert_locations_lat_lon_table.png`
  - `outputs/tanisha_followup_experiments/figures/03_feature_group_ablation_impact.png`
- Added-feature SHAP table: `outputs/tanisha_followup_experiments/figures/01_top5_added_feature_group_shap.csv`
- Coordinate table: `outputs/tanisha_followup_experiments/tables/top_alert_locations_lat_lon.csv`

Environment note: `.venv` has LightGBM and the geo dependencies but not `shap`, so SHAP was not recomputed from scratch. The follow-up SHAP view uses the existing tracked SHAP artifacts already present in `outputs/stage2/shap/`.

Figure-quality note: the first scatter attempt was rejected and removed because it was visually weaker than Tanisha's existing `outputs/figures/final_pipeline/` figures.
