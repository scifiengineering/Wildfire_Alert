# WildFire — Project Log

Maintained by Claude Code. Each session appends a new entry.
For architecture and task spec, see `PROJECT_PROMPT.md`.

---

## Project Overview

**Goal:** Two-stage ML pipeline for next-day wildfire spread prediction.
- **Stage 1:** ResNet-18 U-Net producing per-pixel burn probability maps (AUC-PR target: 0.28–0.32).
- **Stage 2:** LightGBM discriminative alert model with TreeSHAP explanations. Headline metric: Within-Event Discrimination AUC.

**Dataset:** WildfireSpreadTS (WSTS) — 607 events, 2018–2021, 23 channels at 375m resolution. Path: `wsts_data/`.

**Evaluation:** 4 folds, one per test year (not 12-fold — see decision below).

---

## Key Decisions

| Decision | Rationale |
|---|---|
| 4 folds (test=each year) instead of 12-fold CV | 12-fold is 4×3 permutations of (test, val) — redundant for a master's thesis. 4 folds covers inter-year variance and is directly comparable to published baselines. |
| Train/val/test are strict year splits | Temporal leakage is severe with random splits. Enforced in `splits.py`, tested in `test_splits.py`. |
| Circular encoding for wind/aspect (23→26 channels) | Avoids angle discontinuity at 0°/360°. 3 directional channels → 6 sin/cos channels. |
| Augmentation direction-corrected | Rotations/flips also transform wind/aspect channels physically. Implemented in `transforms.py`. |
| AUC-PR (Average Precision) as Stage 1 headline metric | Matches Gerard et al. 2023 WSTS paper baseline. Class imbalance makes accuracy/IoU misleading. |
| Focal (γ=2) + Dice loss, 1:1 weight | BCE fails on severe imbalance. Focal down-weights easy negatives; Dice optimizes overlap directly. |
| crop_size=256 with random crop (train) / center crop (val) | WSTS tiles vary in spatial size; fixed crop makes batching efficient without resizing physical rasters. |

---

## Phase Status

| Phase | Scope | Status |
|---|---|---|
| 0 | Environment, data loading, splits, synthetic data | ✅ Complete |
| 1 | Stage 1 U-Net — training loop, checkpointing, AUC-PR | 🔄 In Progress |
| 2 | Stage 2 candidate extraction + feature engineering | ⏳ Not started |
| 3 | LightGBM + SHAP + isotonic calibration | ⏳ Not started |
| 4 | Evaluation metrics (IoU, AUC-PR, Discrimination AUC) | ⏳ Not started |
| 5 | Figures + auto-generated markdown report | ⏳ Not started |

---

## Session Log

---

### Session 5 - Stage 2 candidate extraction and GBM training

**Current focus:**
- Train the Stage 2 discriminative alert classifier using the 2019 event-split candidate tables.
- Preferred candidate set: `outputs/stage2/candidates_r2km/`, because it gives balanced candidate labels:
  - Fold 0: 55,738 candidates, 13.77% positives
  - Fold 1: 37,800 candidates, 12.15% positives

**Completed so far in this phase:**
- Installed Stage 2 ML dependencies in `.venv`: LightGBM, scikit-learn, pandas, joblib.
- Confirmed the original 5 km candidate ring is too negative-heavy; 2 km is better for the time-constrained 2019-only thesis prototype.
- Important correction: target pixels must not be included by default when generating validation/test candidates, because tomorrow's burn map is not available at alert time. Target inclusion is now an explicit diagnostic flag only.

**Next immediate step:**
- Fair Stage 2 candidates were rebuilt in `outputs/stage2/candidates_r2km_fair/` using ring+top-risk only.
- `scripts/13_train_stage2_gbm.py` trained two fair LightGBM sanity-check runs:
  - fold0 -> fold1: AP `0.3940`, ROC AUC `0.8008`, within-event AUC `0.7992`, best F1 `0.3922`.
  - fold1 -> fold0: AP `0.4301`, ROC AUC `0.8232`, within-event AUC `0.7980`, best F1 `0.4001`.
- Model artifacts are in `outputs/stage2/models/`.
- Top features include distance to current fire, Stage 1 probability/top threshold, forecast wind alignment/speed, bearing, drought/elevation/weather features.

**Next immediate step:**
- `scripts/14_evaluate_stage2_model.py` was added to test saved Stage 2 models against simple baselines.
- Held-out fair candidate evaluations:
  - fold0 -> fold1:
    - Model AP `0.3940` vs distance AP `0.1933` vs Stage 1 AP `0.1527`.
    - Model within-event AUC `0.7992` vs distance `0.7949` vs Stage 1 `0.6217`.
  - fold1 -> fold0:
    - Model AP `0.4301` vs distance AP `0.1451` vs Stage 1 AP `0.1635`.
    - Model within-event AUC `0.7980` vs distance `0.7558` vs Stage 1 `0.5645`.
- Both runs pass the current discriminative sanity check: model within-event AUC >= `0.65` and above distance baseline.
- Note: the fold0 -> fold1 within-event gain over distance is small (`+0.0043`), so SHAP/reporting should be honest that distance remains a strong driver.

**Next immediate step:**
- SHAP explanation phase completed for both fair Stage 2 models.
- Installed `shap` in `.venv` and updated `pyproject.toml` to require `shap>=0.52`.
- Added `scripts/15_compute_stage2_shap.py`.
- Canonical SHAP outputs:
  - `outputs/stage2/shap/stage2_gbm_r2km_fair_fold0_to_fold1/`
  - `outputs/stage2/shap/stage2_gbm_r2km_fair_fold1_to_fold0/`
- Each SHAP folder contains:
  - `global_shap_importance.csv`
  - `sample_shap_values.csv`
  - `global_beeswarm.png`
  - `local/top_*.png`
  - `paired_similar_distance_examples.json`
- SHAP interpretation:
  - Distance remains the strongest global driver, as expected.
  - Non-distance drivers also contribute: Stage 1 probability, drought (`pdsi`), forecast wind speed/direction/alignment, humidity, elevation, energy release component, NDVI/slope.
  - This supports the discriminative-alert claim, but reporting should be honest that distance is still dominant.

**Next immediate step:**
- Generated the comprehensive report at `outputs/report/wildfire_alerting_report.md`.
- The report covers objectives, challenges, related-work gap, contributions, Stage 1 calibration, Stage 2 candidate extraction, baseline comparisons, SHAP explanations, limitations, and next steps.
- It explicitly explains how challenges map to objectives and how each contribution addresses a challenge.

**Next immediate step:**
- Final pipeline visualizations were generated with `scripts/17_generate_pipeline_visualizations.py`.
- Outputs are in `outputs/figures/final_pipeline/`:
  - `01_pipeline_overview.png`
  - `02_stage1_qualitative.png`
  - `03_stage2_baseline_comparison.png`
  - `04_feature_engineering_shap_importance.png`
  - `05_stage2_candidate_alert_map.png`
  - `05_stage2_high_alert_locations.csv`
  - `06_similar_distance_discrimination.png`
  - `manifest.json`
  - `visualization_summary.md`
- Temporal clarification for reporting:
  - The prediction/alert horizon is one day ahead.
  - Stage 1 uses five previous daily WSTS rasters to predict the next-day fire map.
  - Stage 2 uses latest-day engineered candidate features plus Stage 1 risk to rank one-day-ahead alert candidates.
- 2 km candidate ring rationale:
  - The 5 km ring created too many easy negatives far from the active fire.
  - The 2 km ring keeps the candidate task closer to realistic short-term spread and forces the model to rank plausible nearby locations more precisely.
- Figure 5 now converts raster row/column alert pixels into GPS coordinates using the HDF5 CRS/transform metadata.
- The top alert table excludes pixels already burning today, so it represents new-location next-day spread alerts.
- A concise secondary thesis report was saved at `SECONDARY_THESIS_REPORT.md`, with final-pipeline visuals embedded and contributions summarized at the end.
- The secondary thesis report was rendered to `SECONDARY_THESIS_REPORT.pdf` using `scripts/18_render_secondary_report_pdf.py`.

**Major methodology revision after committee-risk review:**
- The earlier fold0/fold1 Stage 2 evaluation had event leakage because the two held-out sets were independent seeded splits and shared fire IDs.
- Added leakage-free event k-fold support with `make_event_kfold_split()` / `make_event_kfolds()` in `src/wildfire_alert/data/event_split.py`.
- Generated the real 2019 3-fold manifest at `data/splits/2019_event_3folds.json`; it reports `has_validation_overlap: false`.
- Stage 1 and Stage 2 scripts now support `--split-mode event_kfold --event-folds 3`.
- `WSTSDataset` and `Stage1TorchDataset` now support `--target-offset-days`, so the same loader can support one-day rolling targets and a direct +3-day baseline target.
- `scripts/13_train_stage2_gbm.py` now accepts multiple `--train-csv` paths so Stage 2 can train on two folds and test on the third.
- Added `scripts/19_write_event_kfolds.py` for split manifests and leakage audits.
- Added `scripts/20_evaluate_rolling_alerts.py` for 3-day rolling alerts, naive 4 km baseline comparison, alert reduction, dropped previous alerts, AP, and AUC.
- New intended result framing:
  - Minimum 3 folds with no shared validation fire IDs.
  - Baseline: naive 4 km alerting for +3-day spread.
  - Proposed model: rolling 2 km -> 3 km -> 4 km discriminative alerts.
  - Report both location discrimination and alert-fatigue reduction.

**No-leakage 3-fold rerun results:**
- Stage 1 was retrained for all three 2019 event k-folds using `--split-mode event_kfold --event-folds 3 --target-offset-days 1 --exclude-empty-current-fire`.
- Stage 1 best validation AP:
  - fold 0: `0.064376`
  - fold 1: `0.115760`
  - fold 2: `0.136186`
- Active-fire-only Stage 1 calibration:
  - fold 0 threshold `0.90625`, top fraction `0.001`
  - fold 1 threshold `0.788086`, top fraction `0.001`
  - fold 2 threshold `0.5`, top fraction `0.001`
- Out-of-fold Stage 1 predictions saved under `outputs/predictions/stage1_2019_3fold/`.
- Fair 4 km Stage 2 candidates saved under `outputs/stage2/candidates_3fold_r4km_fair/`; no target candidates injected.
- Candidate positive rates:
  - fold 0: `2.97%`
  - fold 1: `3.59%`
  - fold 2: `2.54%`
- Stage 2 trained with two folds as train and one fold as held-out validation:
  - holdout 0: AP `0.2097`, within-event AUC `0.8631`
  - holdout 1: AP `0.2913`, within-event AUC `0.8935`
  - holdout 2: AP `0.2822`, within-event AUC `0.8724`
- Baseline comparison:
  - Stage 2 beats distance-only and Stage-1-only AP in all three folds.
  - Distance-only still slightly beats Stage 2 within-event AUC in holdout folds 0 and 2, so do not claim Stage 2 dominates distance on every metric.
  - Stronger defensible claim: Stage 2 improves AP/alert precision under an alert budget and greatly reduces alert volume.
- Rolling 3-day alert evaluation (`2 km -> 3 km -> 4 km`) against naive 4 km alerting:
  - total event-days evaluated: `1419`
  - total rolling windows: `473`
  - mean model alerts/event-day: `37.13`
  - mean naive 4 km alerts/event-day: `1099.20`
  - mean model precision: `0.1266`
  - mean naive precision: `0.0269`
  - mean alert reduction vs naive: `97.12%`
  - mean rolling AP: `0.2644`
  - mean rolling AUC: `0.8414`
- Quality gates after code changes: focused pytest `11 passed`, ruff passed, mypy passed.

**2020 frozen external evaluation:**
- Added `scripts/21_predict_stage1_year.py` for all-event inference in a requested year.
- Added `scripts/22_ensemble_stage1_predictions.py` to average matching Stage 1 rasters.
- Added `scripts/23_build_stage2_candidates_year.py` for streamed full-year fair candidates.
- Added `scripts/24_calibrate_stage2_oof_threshold.py` to freeze a pooled 2019 OOF threshold.
- Added `scripts/25_score_stage2_ensemble.py` for chunked three-model Stage 2 scoring.
- Updated `scripts/20_evaluate_rolling_alerts.py` to accept pre-scored candidates and evaluate
  the exact fixed Day-5-to-Day-8 naive 4 km baseline.
- Frozen values from 2019 only:
  - Stage 1 ensemble threshold `0.7314453125`
  - Stage 1 top fraction `0.001`
  - Stage 2 pooled OOF threshold `0.7777810642240797`
- Converted all 201 raw 2020 events to LZF HDF5 under ignored `wsts_hdf5/2020/`.
- Ran all three frozen Stage 1 checkpoints on 2,184 eligible 2020 event-days and ensembled them.
- Built 5,212,367 fair 4 km candidates with no target injection.
- 2020 external evaluation covered 1,794 rolling windows / 5,382 event-days:
  - rolling AP `0.2459`
  - rolling AUC `0.8220`
  - rolling model alerts `89.53` with precision `0.1747`
  - daily naive alerts `2589.46` with precision `0.0396`
  - rolling alert reduction `97.01%`
  - final Day-8 model alerts `88.58` with precision `0.1641`
  - fixed Day-5-to-Day-8 naive alerts `2615.44` with precision `0.0459`
  - final-day alert reduction against fixed baseline `96.41%`
- Full command sequence and interpretation are in `docs/2020_EXTERNAL_EVALUATION.md`.
- A detailed thesis-facing explanation of the complete methodological evolution is in
  `METHODOLOGICAL_EVOLUTION_AND_2020_TEST_REPORT.md`. It covers the move from the
  superseded overlapping two-split prototype to three leakage-free event folds, the
  rolling Day 6/7/8 temporal protocol, and the frozen 2020 external test.
- Treat the older two-fold methodology in `SECONDARY_THESIS_REPORT.md` as historical;
  the three-fold plus 2020 methodology is authoritative.
- Quality gates for the 2020 implementation:
  - full pytest: `30 passed`
  - strict mypy on `src`: passed
  - Ruff on all new/modified 2020 pipeline files: passed
  - full-repo Ruff still reports a pre-existing import-order issue in
    `scripts/18_render_secondary_report_pdf.py`

---

### Session 1 — Phase 0 (prior session)

**What was built:**
- `src/wildfire_alert/data/wsts_loader.py` — inventory-first lazy loader, backends: npz/geotiff/hdf5
- `src/wildfire_alert/data/splits.py` — 12 leave-year-out folds (Fold dataclass)
- `src/wildfire_alert/data/channels.py` — 23 WSTS channel metadata
- `src/wildfire_alert/data/preprocessing.py` — NaN handling, z-score norm, circular direction encoding (23→26 ch)
- `src/wildfire_alert/data/synthetic.py` — deterministic test fixtures
- `src/wildfire_alert/data/transforms.py` — physically consistent spatial augmentation
- `src/wildfire_alert/data/audit.py` — dataset inventory without full load
- `src/wildfire_alert/stage1/unet.py` — SpreadUNet (ResNet-18, single/multi-day)
- `src/wildfire_alert/stage1/losses.py` — BinaryFocalLoss, BinaryDiceLoss, CombinedFocalDiceLoss
- `src/wildfire_alert/utils/seeds.py` — seed_everything()
- `configs/data.yaml`, `configs/stage1_unet.yaml`, `configs/stage2_gbm.yaml`, `configs/eval.yaml`
- Scripts: `00_audit_data.py`, `00_create_synthetic_data.py`, `00_visualize_batch.py`, `00_write_splits.py`
- Tests: `test_splits.py`, `test_transforms.py`, `test_wsts_loader.py`, `test_preprocessing.py`

**Outputs:**
- `outputs/phase0_audit.json`, `outputs/phase0_batch.png`
- `data/splits/official_year_folds.json`

**Quality gates:** pytest 16 passed, ruff ✅, mypy --strict ✅, black ✅

---

### Session 2 — Phase 1 partial (prior session)

**What was built:**
- `src/wildfire_alert/stage1/dataset.py` — Stage1TorchDataset; random crop (train) + center crop (val); exclude-empty-fire filter
- `src/wildfire_alert/evaluation/metrics.py` — BinarySegmentationMetrics: AP, Dice, IoU, precision, recall, positive_rate
- `scripts/03_train_stage1.py` — fold-aware training loop with per-epoch validation, best/latest checkpointing, JSON history

**Smoke test result (fold 0, 2 batches, 1 epoch):**
```
fold: 0, train years: 2020+2021, val year: 2018, test year: 2019
train_loss: 1.2333, val_loss: 2.0801, val_AP: 0.000131
```
Low AP expected — untrained model, 2 batches only. Infrastructure verified correct.

**Artifacts:**
- `outputs/checkpoints/stage1_smoke/history_fold_0.json`
- `outputs/checkpoints/stage1_smoke/stage1_fold_0_best.pt` / `_latest.pt`

**Quality gates:** pytest 19 passed, ruff ✅, mypy --strict ✅, black ✅

---

### Session 3 — Phase 1 production training

**What was built:**
- `scripts/03_train_stage1.py` — upgraded with mixed precision (torch.autocast + GradScaler), CosineAnnealingWarmRestarts LR scheduler, W&B logging (offline fallback), `--max-train-batches`/`--max-val-batches` default to `None` (full dataset)
- `src/wildfire_alert/stage1/predict.py` — inference module; loads checkpoint, runs full-dataset inference, saves per-event probability rasters as `.npz` to `outputs/predictions/`

**Next action:** Run real Stage 1 training on fold 0 (`--epochs 30 --batch-size 2 --encoder-weights imagenet`). Target val AUC-PR ≥ 0.28.

**Quality gates:** pytest ✅, ruff ✅, mypy --strict ✅, black ✅
