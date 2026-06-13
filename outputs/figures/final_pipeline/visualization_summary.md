# Final Pipeline Visualization Index

These figures are ordered in the same order as the pipeline.

## Generated Figures

1. `outputs\figures\final_pipeline\01_pipeline_overview.png`
2. `outputs\figures\final_pipeline\02_stage1_qualitative.png`
3. `outputs\figures\final_pipeline\03_stage2_baseline_comparison.png`
4. `outputs\figures\final_pipeline\04_feature_engineering_shap_importance.png`
5. `outputs\figures\final_pipeline\05_stage2_candidate_alert_map.png`
6. `outputs\figures\final_pipeline\06_similar_distance_discrimination.png`

## Interpretation

- Stage 1 turns five days of raster context into a next-day probability map.
- Stage 2 turns candidate locations into alert scores.
- Feature engineering helps because the model sees distance, wind, terrain, drought, vegetation, and Stage 1 risk together.
- The SHAP figures show which engineered features influenced the alert scores.

Top SHAP features in the selected model: distance_to_current_fire_px, pdsi, stage1_probability, forecast_wind_speed, forecast_wind_direction, has_current_fire, minimum_temperature, specific_humidity.
