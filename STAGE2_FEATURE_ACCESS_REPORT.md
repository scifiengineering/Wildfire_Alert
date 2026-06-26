# Stage 2 Feature Access Report

This report documents which candidate-location features are available to the Stage 2 model.

## Summary

Stage 2 is a candidate-level LightGBM classifier. It receives engineered numeric features for each candidate location and predicts the probability that the candidate location will burn on the next target day.

The model does not use the following metadata columns as predictive features:

- `fold`
- `split`
- `event_id`
- `target_date`
- `row`
- `col`
- `label`

The `label` column is used only as the training/evaluation target, not as an input feature.

## Confirmed Feature Set

The saved Stage 2 model bundles confirm that the model uses 33 numeric input features:

| Group | Feature |
|---|---|
| Stage 1 / WSTS signal | `stage1_probability` |
| Stage 1 / WSTS signal | `stage1_threshold_mask` |
| Stage 1 / WSTS signal | `stage1_top_mask` |
| Stage 1 / WSTS signal | `threshold` |
| Stage 1 / WSTS signal | `top_fraction` |
| Stage 1 / WSTS signal | `top_threshold` |
| Current-fire context | `has_current_fire` |
| Current-fire context | `current_fire_at_candidate` |
| Current-fire context | `distance_to_current_fire_px` |
| Current-fire context | `distance_to_current_fire_km` |
| Direction / spread geometry | `bearing_from_fire_deg` |
| Direction / spread geometry | `wind_alignment` |
| Direction / spread geometry | `forecast_wind_alignment` |
| Direction / spread geometry | `slope_alignment` |
| Vegetation | `ndvi` |
| Vegetation | `evi2` |
| Weather / fire danger | `total_precipitation` |
| Weather / fire danger | `wind_speed` |
| Weather / fire danger | `wind_direction` |
| Weather / fire danger | `minimum_temperature` |
| Weather / fire danger | `maximum_temperature` |
| Weather / fire danger | `energy_release_component` |
| Weather / fire danger | `specific_humidity` |
| Terrain | `slope` |
| Terrain | `aspect` |
| Terrain | `elevation` |
| Drought / landcover | `pdsi` |
| Drought / landcover | `landcover_class` |
| Forecast weather | `forecast_total_precipitation` |
| Forecast weather | `forecast_wind_speed` |
| Forecast weather | `forecast_wind_direction` |
| Forecast weather | `forecast_temperature` |
| Forecast weather | `forecast_specific_humidity` |

## What This Means

Stage 2 is not only reusing the WSTS / Stage 1 probability. It combines the Stage 1 risk signal with local context around the active fire:

- how close the candidate is to the current fire;
- whether the candidate lies in a direction consistent with wind and terrain;
- vegetation and landcover conditions;
- weather, forecast weather, drought, and fire-danger variables;
- terrain variables such as slope, aspect, and elevation.

This is why Stage 2 can be interpreted as a discriminative alert-ranking layer: it starts from candidate locations near current fire or high Stage 1 risk and then re-ranks them using local physical and environmental context.

## Code Evidence

Feature rows are created in:

```text
src/wildfire_alert/stage2/candidates.py
```

The Stage 2 trainer excludes metadata columns and uses the remaining numeric columns as features:

```text
scripts/13_train_stage2_gbm.py
```

The relevant excluded columns are:

```text
fold, split, event_id, target_date, row, col, label
```

The saved model bundles also store the exact feature list under `feature_columns`, and both the regenerated-checkpoint run and the original-checkpoint rerun use the same 33-feature list.

## Integrity Note

The model does not have access to the answer at prediction time. The future burn label is used to train and evaluate the model, but it is explicitly excluded from the predictive feature columns.

