# Supportive Reproduction Commands and Validation Report

> Superseded note, 2026-06-15: this command report documents the earlier incomplete 50-event 2020 subset run. The full 2020 universe has since been restored to 201 events and 2,184 active-current-fire samples. Use `FULL_2020_REGENERATED_CHECKPOINT_VALIDATION_REPORT.md` for the corrected full-2020 results, while keeping this file as provenance for the debugging path.

This document records how I reproduced Tanisha's Stage 2 alerting framework using regenerated Stage 1 checkpoints, which scripts were used, which train/validation/test commands were run, and what conclusions I reached against the three thesis claims.

This is a supportive reproducibility and integrity check, not a competing experiment. The goal is not to prove that my regenerated-checkpoint run is better than Tanisha's run. The goal is to check whether the same comparison framework produces the same direction of evidence, and to clearly flag places where my regenerated run differs from Tanisha's reported comparison.

This was not an exact artifact reproduction of Tanisha's reported checkpoint run. The checkpoint paths were present in the repo, but the original Git LFS checkpoint objects were not retrievable locally, so I regenerated ImageNet-initialized Stage 1 checkpoints and then ran the same Stage 2 alerting framework.

## Objective

Tanisha's framework makes three alerting-system claims:

1. At matched alert volume, Stage 2 should be better than WSTS/Stage 1 used directly as an alerter.
2. Stage 2 should reduce alert fatigue compared with naive 4 km flooding.
3. Stage 2 should rank candidate locations better than WSTS/Stage 1 and distance-only baselines.

The key fairness rule for claim 1 is one global WSTS threshold matched to Stage 2's mean alert volume. I did not use per-event-day top-K thresholding.

## How to Read This Run

The regenerated-checkpoint results should be read as a sensitivity check under different artifacts, not as an alternative thesis result. Tanisha's reported result remains the primary thesis result because it uses her frozen pipeline and artifacts.

The main comparability issue is alert volume. Tanisha's saved 2020 run produces about 89.53 Stage 2 alerts/event-day. My regenerated-checkpoint run produces about 25.23 Stage 2 alerts/event-day using the newly calibrated regenerated-pipeline Stage 2 threshold, `0.8117221967059977`.

| Input / volume check | Tanisha saved 2020 run | My regenerated-checkpoint run |
|---|---:|---:|
| 2020 events | 201 | 50 |
| 2020 samples | 2,184 | 342 |
| Stage 2 candidate rows | 5,212,367 | 373,223 |
| Naive 4 km alerts/event-day | 2,589.46 | 1,156.02 |
| Stage 2 threshold | 0.7777810642 | 0.8117221967 |
| Stage 2 alerts/event-day | 89.53 | 25.23 |

This means the regenerated-checkpoint run should not be described as "beating" or "competing with" Tanisha's result. It checks whether the same framework points in the same thesis direction under regenerated artifacts. Where it does not, that difference is part of the audit and should be reported plainly. In this run, the alert-fatigue and ranking-quality claims point in the same direction as Tanisha's reported thesis results, while the matched-volume WSTS comparison does not reproduce the same precision/true-positive advantage.

## Audit Finding at a Glance

| Thesis comparison | Tanisha's reported direction | My regenerated-checkpoint direction | Integrity/audit reading |
|---|---|---|---|
| Stage 2 vs WSTS-as-alerter at matched volume | Stage 2 better on precision, recall, true-positive alerts, and event-day catch rate | Stage 2 better only on event-day catch rate; WSTS better on precision, recall, and true-positive alerts | Different result direction for precision/recall/TP. This does not validate the full point 1 claim without Tanisha's exact artifacts. |
| Stage 2 vs naive 4 km flooding | Stage 2 far fewer alerts and higher precision | Stage 2 far fewer alerts and higher precision | Same direction. Supports the alert-fatigue claim. |
| Stage 2 vs WSTS/Stage 1 and distance ranking | Stage 2 better ranking quality | Stage 2 better AP/AUC than WSTS/Stage 1 and distance | Same direction. Supports the ranking-quality claim. |

## Script and Artifact Audit Before Reporting

I compared the public repo scripts used by Tanisha's documented 2020 workflow with the scripts used in my run:

- `scripts/20_evaluate_rolling_alerts.py`
- `scripts/21_predict_stage1_year.py`
- `scripts/22_ensemble_stage1_predictions.py`
- `scripts/23_build_stage2_candidates_year.py`
- `scripts/25_score_stage2_ensemble.py`

There were no local diffs in those shared scripts. I also tested the documented `--exclude-empty-current-fire` flag for candidate generation on my available 2020 data; it produced the same candidate count, so that flag does not explain the lower true-positive count.

The exact matched-volume scripts Tanisha named, `scripts/26_compare_matched_alert_volume.py` and `scripts/27_summarize_matched_alert_volume.py`, were later provided in `Specifications/`. I compared them against my local implementation and ran Tanisha's comparison script on my corrected scored candidates from the retrained Stage 2 pipeline. Her script could not run standalone because the helper module `wildfire_alert.evaluation.matched_alerts` is not present in the repo, so I used a compatibility implementation matching the imported API and the metric definitions in her script.

Tanisha's script on my scored candidates produced the same precision, recall, and true-positive discrepancy:

| Metric | Stage 2 | WSTS-as-alerter |
|---|---:|---:|
| Alert count | 18,545 | 18,566 |
| Alerts/event-day | 25.23 | 25.26 |
| Precision | 0.2203 | 0.2457 |
| Recall | 0.1423 | 0.1589 |
| True-positive alerts | 4,085 | 4,561 |
| Event-day catch rate | 0.6047 | 0.3419 |

This means the lower true-positive count is not caused by my local matched-volume script. One metric-definition correction was found: my first local summary computed event-day catch rate over all rolling decisions, while Tanisha's script defines it over positive rolling decisions only. Using Tanisha's definition, Stage 2's event-day catch-rate advantage in my run is `0.6047` vs `0.3419`.

Before Tanisha's scripts were available, I implemented the described matched-volume logic locally as:

- `scripts/27_compare_matched_alert_volume.py`
- `scripts/28_summarize_matched_alert_volume.py`

Those local scripts use one global WSTS threshold matched to Stage 2's total/mean alert volume. They do not use per-event-day top-K.

The main remaining differences are artifact/data differences rather than visible shared-script differences:

- My run uses regenerated Stage 1 checkpoints, not Tanisha's exact Stage 1 checkpoints.
- My 2020 data/artifact universe is smaller: 50 events and 342 samples rather than 201 events and 2,184 samples.
- I corrected the earlier hybrid run by retraining Stage 2 from regenerated 2019 Stage 1 out-of-fold predictions. The corrected Stage 2 models are in `outputs/stage2/models_3fold_r4km_imagenet_noamp/`.

So the lower true-positive count is not explained by a known shared-script mismatch or by the earlier Stage 2 shortcut. After retraining Stage 2 consistently, the discrepancy remains. The remaining likely causes are regenerated Stage 1 checkpoints and incomplete/different 2020 data coverage compared with Tanisha's full 201-event run.

## Environment

- Repo: `/home/Sifiso/Projects/Wildfire_Alert`
- Machine used for this run: `user-MS-7D75`
- GPU reported by `nvidia-smi`: `NVIDIA GeForce RTX 5090`
- Python entry point: `.venv/bin/python`
- 2020 data available locally: `50` HDF5 event files
- Primary regenerated checkpoint directory: `outputs/checkpoints/stage1_2019_3fold_imagenet_noamp_20260615/`

## Scripts Used

| Step | Script | Purpose |
|---|---|---|
| Stage 1 training | `scripts/03_train_stage1.py` | Train one 2019 event-disjoint fold and save best/latest checkpoints. |
| Stage 1 threshold calibration | `scripts/10_calibrate_stage1_threshold.py` | Calibrate each Stage 1 checkpoint on its 2019 validation fold. |
| Stage 1 2020 prediction | `scripts/21_predict_stage1_year.py` | Generate 2020 prediction rasters for each fold checkpoint. |
| Stage 1 ensemble | `scripts/22_ensemble_stage1_predictions.py` | Average the three fold prediction rasters. |
| Stage 2 candidate generation | `scripts/23_build_stage2_candidates_year.py` | Build fair 2020 Stage 2 candidate rows from the Stage 1 ensemble. |
| Stage 2 training | `scripts/13_train_stage2_gbm.py` | Retrain three LightGBM Stage 2 models from regenerated 2019 Stage 1 candidate features. |
| Stage 2 threshold calibration | `scripts/24_calibrate_stage2_oof_threshold.py` | Freeze one Stage 2 threshold from regenerated 2019 out-of-fold scores. |
| Stage 2 ensemble scoring | `scripts/25_score_stage2_ensemble.py` | Score 2020 candidates with the three retrained Stage 2 GBM models. |
| Matched-volume WSTS comparison | `scripts/27_compare_matched_alert_volume.py` | Compare Stage 2 vs WSTS using one global WSTS threshold matched to Stage 2 alert volume. |
| Matched-volume report | `scripts/28_summarize_matched_alert_volume.py` | Render the matched-volume summary as Markdown. |
| Rolling alert fatigue | `scripts/20_evaluate_rolling_alerts.py` | Compare Stage 2 alert volume and precision against naive 4 km flooding. |
| Ranking validation | inline Python using `sklearn.metrics` | Compare Stage 2 AP/AUC against WSTS/Stage 1 and distance baselines on the same scored candidates. |
| Full corrected run wrapper | `scripts/30_run_regenerated_stage2_pipeline.sh` | Rebuild 2019 Stage 2 candidates, retrain Stage 2, calibrate threshold, score 2020, and evaluate. |

## Stage 1 Training Commands

The successful checkpoints used for the validation were trained with ImageNet encoder initialization and AMP disabled. AMP was disabled because the AMP runs on this machine produced non-finite/GradScaler failures for later folds. These commands are the effective commands used for the successful regenerated checkpoint set.

```bash
.venv/bin/python scripts/03_train_stage1.py \
  --data-root wsts_hdf5 \
  --backend hdf5 \
  --split-mode event_kfold \
  --years 2019 \
  --event-folds 3 \
  --fold 0 \
  --epochs 30 \
  --batch-size 2 \
  --encoder-weights imagenet \
  --no-amp \
  --output-dir outputs/checkpoints/stage1_2019_3fold_imagenet_noamp_20260615 \
  --no-wandb \
  > outputs/logs/retrain_stage1_imagenet_noamp_clean_fold0_20260615.log 2>&1
```

```bash
.venv/bin/python scripts/03_train_stage1.py \
  --data-root wsts_hdf5 \
  --backend hdf5 \
  --split-mode event_kfold \
  --years 2019 \
  --event-folds 3 \
  --fold 1 \
  --epochs 30 \
  --batch-size 2 \
  --encoder-weights imagenet \
  --no-amp \
  --output-dir outputs/checkpoints/stage1_2019_3fold_imagenet_noamp_20260615 \
  --no-wandb \
  > outputs/logs/retrain_stage1_imagenet_noamp_clean_fold1_20260615.log 2>&1
```

```bash
.venv/bin/python scripts/03_train_stage1.py \
  --data-root wsts_hdf5 \
  --backend hdf5 \
  --split-mode event_kfold \
  --years 2019 \
  --event-folds 3 \
  --fold 2 \
  --epochs 30 \
  --batch-size 2 \
  --encoder-weights imagenet \
  --no-amp \
  --output-dir outputs/checkpoints/stage1_2019_3fold_imagenet_noamp_20260615 \
  --no-wandb \
  > outputs/logs/retrain_stage1_imagenet_noamp_clean_fold2_20260615.log 2>&1
```

Training outputs:

- `outputs/checkpoints/stage1_2019_3fold_imagenet_noamp_20260615/stage1_fold_0_best.pt`
- `outputs/checkpoints/stage1_2019_3fold_imagenet_noamp_20260615/stage1_fold_1_best.pt`
- `outputs/checkpoints/stage1_2019_3fold_imagenet_noamp_20260615/stage1_fold_2_best.pt`
- `outputs/checkpoints/stage1_2019_3fold_imagenet_noamp_20260615/history_fold_0.json`
- `outputs/checkpoints/stage1_2019_3fold_imagenet_noamp_20260615/history_fold_1.json`
- `outputs/checkpoints/stage1_2019_3fold_imagenet_noamp_20260615/history_fold_2.json`

Best validation AP from the regenerated Stage 1 histories:

| Fold | Best validation AP |
|---|---:|
| 0 | 0.0673 |
| 1 | 0.1204 |
| 2 | 0.1463 |

## Stage 1 Validation / Calibration Commands

Each regenerated checkpoint was calibrated on its corresponding 2019 held-out event fold.

```bash
.venv/bin/python scripts/10_calibrate_stage1_threshold.py \
  --checkpoint outputs/checkpoints/stage1_2019_3fold_imagenet_noamp_20260615/stage1_fold_0_best.pt \
  --data-root wsts_hdf5 \
  --backend hdf5 \
  --split-mode event_kfold \
  --years 2019 \
  --event-folds 3 \
  --seed 42 \
  --fold 0 \
  --batch-size 2 \
  --output-dir outputs/calibration/stage1_2019_3fold_imagenet_noamp_20260615
```

```bash
.venv/bin/python scripts/10_calibrate_stage1_threshold.py \
  --checkpoint outputs/checkpoints/stage1_2019_3fold_imagenet_noamp_20260615/stage1_fold_1_best.pt \
  --data-root wsts_hdf5 \
  --backend hdf5 \
  --split-mode event_kfold \
  --years 2019 \
  --event-folds 3 \
  --seed 42 \
  --fold 1 \
  --batch-size 2 \
  --output-dir outputs/calibration/stage1_2019_3fold_imagenet_noamp_20260615
```

```bash
.venv/bin/python scripts/10_calibrate_stage1_threshold.py \
  --checkpoint outputs/checkpoints/stage1_2019_3fold_imagenet_noamp_20260615/stage1_fold_2_best.pt \
  --data-root wsts_hdf5 \
  --backend hdf5 \
  --split-mode event_kfold \
  --years 2019 \
  --event-folds 3 \
  --seed 42 \
  --fold 2 \
  --batch-size 2 \
  --output-dir outputs/calibration/stage1_2019_3fold_imagenet_noamp_20260615
```

Calibration outputs:

| Fold | Validation samples | Validation events | Best threshold | Best top fraction |
|---|---:|---:|---:|---:|
| 0 | 201 | 25 | 0.83252 | 0.001 |
| 1 | 229 | 25 | 0.677734 | 0.001 |
| 2 | 187 | 24 | 0.75293 | 0.001 |

## 2020 Test Prediction Commands

Each fold checkpoint was then frozen and applied to the available 2020 HDF5 data.

```bash
.venv/bin/python scripts/21_predict_stage1_year.py \
  --checkpoint outputs/checkpoints/stage1_2019_3fold_imagenet_noamp_20260615/stage1_fold_0_best.pt \
  --calibration-json outputs/calibration/stage1_2019_3fold_imagenet_noamp_20260615/stage1_fold0_thresholds.json \
  --data-root wsts_hdf5 \
  --backend hdf5 \
  --year 2020 \
  --batch-size 2 \
  --output-dir outputs/predictions/stage1_2020_imagenet_noamp_fold_0
```

```bash
.venv/bin/python scripts/21_predict_stage1_year.py \
  --checkpoint outputs/checkpoints/stage1_2019_3fold_imagenet_noamp_20260615/stage1_fold_1_best.pt \
  --calibration-json outputs/calibration/stage1_2019_3fold_imagenet_noamp_20260615/stage1_fold1_thresholds.json \
  --data-root wsts_hdf5 \
  --backend hdf5 \
  --year 2020 \
  --batch-size 2 \
  --output-dir outputs/predictions/stage1_2020_imagenet_noamp_fold_1
```

```bash
.venv/bin/python scripts/21_predict_stage1_year.py \
  --checkpoint outputs/checkpoints/stage1_2019_3fold_imagenet_noamp_20260615/stage1_fold_2_best.pt \
  --calibration-json outputs/calibration/stage1_2019_3fold_imagenet_noamp_20260615/stage1_fold2_thresholds.json \
  --data-root wsts_hdf5 \
  --backend hdf5 \
  --year 2020 \
  --batch-size 2 \
  --output-dir outputs/predictions/stage1_2020_imagenet_noamp_fold_2
```

- `outputs/predictions/stage1_2020_imagenet_noamp_fold_0/manifest.json`
- `outputs/predictions/stage1_2020_imagenet_noamp_fold_1/manifest.json`
- `outputs/predictions/stage1_2020_imagenet_noamp_fold_2/manifest.json`

Each fold produced 342 prediction records for the available 2020 data.

## Stage 1 Ensemble Command

```bash
.venv/bin/python scripts/22_ensemble_stage1_predictions.py \
  --prediction-dirs \
    outputs/predictions/stage1_2020_imagenet_noamp_fold_0 \
    outputs/predictions/stage1_2020_imagenet_noamp_fold_1 \
    outputs/predictions/stage1_2020_imagenet_noamp_fold_2 \
  --threshold 0.7314453125 \
  --top-fraction 0.001 \
  --output-dir outputs/predictions/stage1_2020_imagenet_noamp_ensemble
```

Output:

- `outputs/predictions/stage1_2020_imagenet_noamp_ensemble/manifest.json`
- Ensemble sample count: 342

## Stage 2 Candidate Generation Command

```bash
.venv/bin/python scripts/23_build_stage2_candidates_year.py \
  --predictions-dir outputs/predictions/stage1_2020_imagenet_noamp_ensemble \
  --data-root wsts_hdf5 \
  --backend hdf5 \
  --year 2020 \
  --ring-radius-km 4 \
  --output-csv outputs/stage2/candidates_2020_r4km_fair_imagenet_noamp/stage2_2020_candidates.csv
```

Output summary:

- Event count: 50
- Sample count: 342
- Candidate count: 373,223
- Positive candidate count: 14,371
- Output CSV: `outputs/stage2/candidates_2020_r4km_fair_imagenet_noamp/stage2_2020_candidates.csv`

## Corrected Stage 2 Retraining and Scoring Commands

The earlier hybrid run reused the repo's existing Stage 2 GBM models. That was not an uncompromised regenerated-checkpoint reproduction. The corrected run retrained Stage 2 from regenerated 2019 Stage 1 out-of-fold predictions and candidates.

I captured the corrected sequence in:

```bash
bash scripts/30_run_regenerated_stage2_pipeline.sh \
  > outputs/logs/retrained_stage2_from_regenerated_stage1_20260615.log 2>&1
```

The wrapper performs the following operations without sampling or reducing model settings:

1. Generate 2019 validation-fold Stage 1 predictions from the regenerated Stage 1 checkpoints.
2. Build 2019 Stage 2 candidates from those regenerated predictions.
3. Train three new Stage 2 LightGBM holdout models.
4. Calibrate one Stage 2 threshold from regenerated 2019 OOF scores.
5. Score the available 2020 candidates with the retrained Stage 2 ensemble.
6. Run matched-volume WSTS comparison, rolling alert-fatigue comparison, and ranking-quality evaluation.

Stage 2 retraining produced:

- `outputs/stage2/models_3fold_r4km_imagenet_noamp/stage2_gbm_3fold_r4km_imagenet_noamp_holdout0.joblib`
- `outputs/stage2/models_3fold_r4km_imagenet_noamp/stage2_gbm_3fold_r4km_imagenet_noamp_holdout1.joblib`
- `outputs/stage2/models_3fold_r4km_imagenet_noamp/stage2_gbm_3fold_r4km_imagenet_noamp_holdout2.joblib`

The regenerated 2019 OOF calibration selected:

```text
Stage 2 threshold = 0.8117221967059977
OOF AP = 0.23671475682787785
OOF rows = 662,963
```

The 2020 scoring command inside the wrapper was:

```bash
.venv/bin/python scripts/25_score_stage2_ensemble.py \
  --models \
    outputs/stage2/models_3fold_r4km_imagenet_noamp/stage2_gbm_3fold_r4km_imagenet_noamp_holdout0.joblib \
    outputs/stage2/models_3fold_r4km_imagenet_noamp/stage2_gbm_3fold_r4km_imagenet_noamp_holdout1.joblib \
    outputs/stage2/models_3fold_r4km_imagenet_noamp/stage2_gbm_3fold_r4km_imagenet_noamp_holdout2.joblib \
  --candidate-csv outputs/stage2/candidates_2020_r4km_fair_imagenet_noamp/stage2_2020_candidates.csv \
  --output-csv outputs/stage2/scored_2020_r4km_fair_imagenet_noamp_retrained_stage2/stage2_2020_scored_candidates.csv
```

Output summary:

- Candidate count: 373,223
- Ensemble size: 3
- Mean score: 0.1834
- Maximum score: 0.9604
- Output CSV: `outputs/stage2/scored_2020_r4km_fair_imagenet_noamp_retrained_stage2/stage2_2020_scored_candidates.csv`

## Experiment 1: Matched-Volume WSTS-as-Alerter

This tests whether Stage 2 beats WSTS/Stage 1 when both operate under the same alert budget. The WSTS threshold was selected globally to match Stage 2's mean alert volume.

```bash
.venv/bin/python scripts/27_compare_matched_alert_volume.py \
  --scored-candidate-csv outputs/stage2/scored_2020_r4km_fair_imagenet_noamp_retrained_stage2/stage2_2020_scored_candidates.csv \
  --model-name stage2_2020_external_ensemble_imagenet_noamp_retrained_stage2 \
  --stage2-threshold 0.8117221967059977 \
  --radii-km 2 3 4 \
  --output-dir outputs/stage2/matched_volume_imagenet_noamp_retrained_stage2
```

```bash
.venv/bin/python scripts/28_summarize_matched_alert_volume.py \
  --summary-json outputs/stage2/matched_volume_imagenet_noamp_retrained_stage2/stage2_2020_external_ensemble_imagenet_noamp_retrained_stage2_matched_volume_summary.json \
  --primary-name stage2_2020_external_ensemble_imagenet_noamp_retrained_stage2 \
  --output-md outputs/stage2/matched_volume_imagenet_noamp_retrained_stage2/MATCHED_VOLUME_ALERT_QUALITY_REPORT_IMAGENET_NOAMP_RETRAINED_STAGE2.md
```

Result:

| Metric | Stage 2 | WSTS-as-alerter |
|---|---:|---:|
| Alerts/event-day | 25.23 | 25.26 |
| Precision | 0.2203 | 0.2457 |
| Recall | 0.1423 | 0.1589 |
| True-positive alerts | 4,085 | 4,561 |
| Event-day catch rate | 0.6047 | 0.3419 |

Audit conclusion: different result direction for the main matched-volume claim. Stage 2 catches more event-days, but WSTS has higher precision and more true-positive alert occurrences in this regenerated-checkpoint run. This should not be framed as a competing result against Tanisha, but it must be highlighted as a reproducibility discrepancy. It shows that Tanisha's strongest matched-volume WSTS claim depends on the exact frozen checkpoints and full 2020 artifact set.

Why the alert volume is much lower than Tanisha's reported 89.53 alerts/event-day: this run used a regenerated-pipeline Stage 2 threshold calibrated from regenerated 2019 OOF scores, `0.8117221967059977`, while Tanisha's saved 2020 run used `0.7777810642240797`. The threshold operates on a different regenerated-checkpoint score distribution and a smaller 2020 data universe. Tanisha's saved 2020 candidate summary references 201 events, 2,184 samples, and 5,212,367 candidates. My regenerated-checkpoint run used 50 events, 342 samples, and 373,223 generated candidates. The naive 4 km baseline is also smaller in my run, 1,156.02 alerts/event-day versus Tanisha's 2,589.46. So the low `25.23` alerts/event-day is not because WSTS was matched incorrectly; WSTS was matched to the Stage 2 volume that the regenerated-pipeline threshold produced in my run.

## Experiment 2: Alert Fatigue vs Naive 4 km

This tests whether Stage 2 sends fewer alerts than naive 4 km flooding while improving precision.

```bash
.venv/bin/python scripts/20_evaluate_rolling_alerts.py \
  --scored-candidate-csv outputs/stage2/scored_2020_r4km_fair_imagenet_noamp_retrained_stage2/stage2_2020_scored_candidates.csv \
  --model-name stage2_2020_external_ensemble_imagenet_noamp_retrained_stage2 \
  --threshold 0.8117221967059977 \
  --radii-km 2 3 4 \
  --naive-radius-km 4 \
  --output-dir outputs/stage2/rolling_alerts_2020_imagenet_noamp_retrained_stage2
```

Result:

| Metric | Stage 2 | Naive 4 km |
|---|---:|---:|
| Alerts/event-day | 25.23 | 1,156.02 |
| Precision | 0.1215 | 0.0296 |
| Alert reduction vs naive | 98.11% | - |

Supportive validation conclusion: agrees with Tanisha's thesis direction. The regenerated-checkpoint run confirms that Stage 2 dramatically reduces alert volume and improves precision versus naive 4 km flooding.

## Experiment 3: Ranking Quality

This tests whether Stage 2 ranks candidate locations better than WSTS/Stage 1 and distance-only baselines on the same candidate table.

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

csv = Path("outputs/stage2/scored_2020_r4km_fair_imagenet_noamp_retrained_stage2/stage2_2020_scored_candidates.csv")
out = Path("outputs/stage2/evaluation_2020_imagenet_noamp_retrained_stage2")
out.mkdir(parents=True, exist_ok=True)

cols = [
    "event_id",
    "target_date",
    "label",
    "model_score",
    "stage1_probability",
    "distance_to_current_fire_px",
]
df = pd.read_csv(csv, usecols=lambda c: c in cols)
labels = df["label"].astype(int).to_numpy()

def safe_auc(y, s):
    return float(roc_auc_score(y, s)) if np.unique(y).size > 1 else 0.0

def within_event_auc(scores):
    tmp = df[["event_id", "target_date", "label"]].copy()
    tmp["score"] = scores
    vals = []
    for _, group in tmp.groupby(["event_id", "target_date"], sort=False):
        y = group["label"].astype(int).to_numpy()
        if np.unique(y).size > 1:
            vals.append(float(roc_auc_score(y, group["score"].to_numpy())))
    return float(np.mean(vals)) if vals else 0.0

scores = {
    "stage2": df["model_score"].fillna(0.0).to_numpy(),
    "wsts_stage1": df["stage1_probability"].fillna(0.0).to_numpy(),
    "distance": -df["distance_to_current_fire_px"].fillna(1e9).to_numpy(),
}

report = {
    "candidate_csv": str(csv),
    "rows": int(len(df)),
    "positive_count": int(labels.sum()),
    "positive_rate": float(labels.mean()),
    "metrics": {},
}
for name, score in scores.items():
    report["metrics"][name] = {
        "average_precision": float(average_precision_score(labels, score)),
        "roc_auc": safe_auc(labels, score),
        "within_event_auc": within_event_auc(score),
    }

report["summary"] = {
    "stage2_ap": report["metrics"]["stage2"]["average_precision"],
    "wsts_stage1_ap": report["metrics"]["wsts_stage1"]["average_precision"],
    "distance_ap": report["metrics"]["distance"]["average_precision"],
    "stage2_auc": report["metrics"]["stage2"]["roc_auc"],
    "wsts_stage1_auc": report["metrics"]["wsts_stage1"]["roc_auc"],
    "distance_auc": report["metrics"]["distance"]["roc_auc"],
    "stage2_within_event_auc": report["metrics"]["stage2"]["within_event_auc"],
    "wsts_stage1_within_event_auc": report["metrics"]["wsts_stage1"]["within_event_auc"],
    "distance_within_event_auc": report["metrics"]["distance"]["within_event_auc"],
    "ap_gain_over_wsts": report["metrics"]["stage2"]["average_precision"] - report["metrics"]["wsts_stage1"]["average_precision"],
    "ap_gain_over_distance": report["metrics"]["stage2"]["average_precision"] - report["metrics"]["distance"]["average_precision"],
}

path = out / "stage2_2020_external_ensemble_imagenet_noamp_retrained_stage2_evaluation.json"
path.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report["summary"], indent=2))
PY
```

Result:

| Metric | Stage 2 | WSTS/Stage 1 | Distance |
|---|---:|---:|---:|
| Average Precision | 0.1998 | 0.1549 | 0.1367 |
| ROC AUC | 0.8179 | 0.7606 | 0.7658 |
| Within-event AUC | 0.8455 | 0.6768 | 0.8280 |

Supportive validation conclusion: agrees with Tanisha's thesis direction. Stage 2 ranks candidates better than both WSTS/Stage 1 and distance-only baselines in this regenerated-checkpoint run.

## Overall Comparison to Tanisha's Thesis Claims

| Thesis point | Tanisha's reported result | My regenerated-checkpoint result | Reproducibility / integrity conclusion |
|---|---|---|---|
| 1. Stage 2 beats WSTS-as-alerter at matched volume | Stage 2 wins precision, recall, true-positive alerts, and event-day catch rate. | Stage 2 wins event-day catch rate only; WSTS wins precision, recall, and true-positive alerts under my smaller alert budget. | Not reproduced in full. This discrepancy must be highlighted. |
| 2. Stage 2 reduces alert fatigue vs naive 4 km flooding | Stage 2 reduces alerts by about 97% and improves precision. | Stage 2 reduces alerts by 98.11% and improves precision. | Reproduced directionally. |
| 3. Stage 2 improves ranking quality vs WSTS/Stage 1 and distance | Stage 2 AP wins across reported validation folds; 2020 AP/AUC remain stable. | Stage 2 AP/AUC beat WSTS/Stage 1 and distance on the regenerated-checkpoint 2020 run. | Reproduced directionally. |

## Important Caveats

- This run used regenerated Stage 1 checkpoints, not Tanisha's exact original checkpoint artifacts.
- The available 2020 HDF5 set used here contains 50 events and 342 eligible samples.
- Tanisha's saved 2020 summaries reference a larger run with 201 events and 2,184 samples.
- Therefore this validates the framework and two of the three thesis conclusions under regenerated artifacts, but it should not be presented as an exact reproduction of Tanisha's reported 2020 numbers.

## Primary Output Artifacts

- `outputs/checkpoints/stage1_2019_3fold_imagenet_noamp_20260615/`
- `outputs/calibration/stage1_2019_3fold_imagenet_noamp_20260615/`
- `outputs/predictions/stage1_2020_imagenet_noamp_ensemble/`
- `outputs/stage2/candidates_2020_r4km_fair_imagenet_noamp/`
- `outputs/stage2/scored_2020_r4km_fair_imagenet_noamp_retrained_stage2/`
- `outputs/stage2/matched_volume_imagenet_noamp_retrained_stage2/`
- `outputs/stage2/rolling_alerts_2020_imagenet_noamp_retrained_stage2/`
- `outputs/stage2/evaluation_2020_imagenet_noamp_retrained_stage2/`
