# 2020 External Evaluation

The 2020 evaluation is a frozen temporal test. Models, features, calibration values,
and alert rules come only from 2019. Do not tune them after inspecting 2020 labels.

## Frozen Values

- Stage 1 ensemble threshold: `0.7314453125`
- Stage 1 top-risk fraction: `0.001`
- Stage 2 threshold: `0.7777810642240797`
- Fair candidate radius: `4 km`
- Rolling radii: `2 km -> 3 km -> 4 km`
- Fixed naive baseline: locations within `4 km` of the initial Day-5 fire map,
  evaluated against the final Day-8 target

## Commands

Convert 2020 only:

```powershell
.\.venv\Scripts\python.exe scripts\02_convert_geotiff_to_hdf5.py `
  --data-dir wsts_data --target-dir wsts_hdf5 `
  --years 2020 --compression lzf
```

Run `scripts/21_predict_stage1_year.py` once per Stage 1 fold, pairing each checkpoint
with its corresponding calibration JSON. Save the outputs as:

```text
outputs/predictions/stage1_2020_fold_0/
outputs/predictions/stage1_2020_fold_1/
outputs/predictions/stage1_2020_fold_2/
```

Ensemble Stage 1:

```powershell
.\.venv\Scripts\python.exe scripts\22_ensemble_stage1_predictions.py `
  --prediction-dirs outputs\predictions\stage1_2020_fold_0 `
    outputs\predictions\stage1_2020_fold_1 `
    outputs\predictions\stage1_2020_fold_2 `
  --threshold 0.7314453125 --top-fraction 0.001 `
  --output-dir outputs\predictions\stage1_2020_ensemble
```

Build fair candidates and score the Stage 2 ensemble:

```powershell
.\.venv\Scripts\python.exe scripts\23_build_stage2_candidates_year.py `
  --predictions-dir outputs\predictions\stage1_2020_ensemble `
  --data-root wsts_hdf5 --backend hdf5 --year 2020 `
  --exclude-empty-current-fire --ring-radius-km 4 `
  --output-csv outputs\stage2\candidates_2020_r4km_fair\stage2_2020_candidates.csv

.\.venv\Scripts\python.exe scripts\25_score_stage2_ensemble.py `
  --models outputs\stage2\models_3fold_r4km\stage2_gbm_3fold_r4km_holdout0.joblib `
    outputs\stage2\models_3fold_r4km\stage2_gbm_3fold_r4km_holdout1.joblib `
    outputs\stage2\models_3fold_r4km\stage2_gbm_3fold_r4km_holdout2.joblib `
  --candidate-csv outputs\stage2\candidates_2020_r4km_fair\stage2_2020_candidates.csv `
  --output-csv outputs\stage2\scored_2020_r4km_fair\stage2_2020_scored_candidates.csv
```

Evaluate:

```powershell
.\.venv\Scripts\python.exe scripts\20_evaluate_rolling_alerts.py `
  --scored-candidate-csv `
    outputs\stage2\scored_2020_r4km_fair\stage2_2020_scored_candidates.csv `
  --model-name stage2_2020_external_ensemble `
  --threshold 0.7777810642240797 `
  --radii-km 2 3 4 --naive-radius-km 4 `
  --target-artifacts-dir outputs\predictions\stage1_2020_ensemble `
  --output-dir outputs\stage2\rolling_alerts_2020_external
```

## Completed Results

- 201 HDF5 events and 2,184 eligible event-days
- 5,212,367 fair Stage 2 candidates
- 1,794 rolling windows and 5,382 rolling event-days
- Rolling mean AP: `0.2459`
- Rolling mean AUC: `0.8220`
- Rolling alerts: `89.53` per event-day at precision `0.1747`
- Daily naive 4 km alerts: `2589.46` at precision `0.0396`
- Rolling alert reduction against daily naive alerting: `97.01%`
- Final Day-8 model alerts: `88.58` at precision `0.1641`
- Fixed Day-5-to-Day-8 naive alerts: `2615.44` at precision `0.0459`
- Final-day alert reduction against the fixed +3-day baseline: `96.41%`
