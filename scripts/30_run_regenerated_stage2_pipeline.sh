#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"

STAGE1_CKPT_DIR="outputs/checkpoints/stage1_2019_3fold_imagenet_noamp_20260615"
STAGE1_CAL_DIR="outputs/calibration/stage1_2019_3fold_imagenet_noamp_20260615"
STAGE1_2019_PRED_DIR="outputs/predictions/stage1_2019_3fold_imagenet_noamp_oof"
STAGE2_2019_CAND_DIR="outputs/stage2/candidates_3fold_r4km_fair_imagenet_noamp"
STAGE2_MODEL_DIR="outputs/stage2/models_3fold_r4km_imagenet_noamp"
STAGE2_CAL_JSON="outputs/stage2/calibration_3fold_r4km_imagenet_noamp/stage2_oof_threshold.json"
STAGE2_2020_SCORED_DIR="outputs/stage2/scored_2020_r4km_fair_imagenet_noamp_retrained_stage2"
MATCHED_DIR="outputs/stage2/matched_volume_imagenet_noamp_retrained_stage2"
ROLLING_DIR="outputs/stage2/rolling_alerts_2020_imagenet_noamp_retrained_stage2"

mkdir -p outputs/logs

for fold in 0 1 2; do
  "$PYTHON" scripts/11_predict_stage1_event_split.py \
    --checkpoint "$STAGE1_CKPT_DIR/stage1_fold_${fold}_best.pt" \
    --calibration-json "$STAGE1_CAL_DIR/stage1_fold${fold}_thresholds.json" \
    --data-root wsts_hdf5 \
    --backend hdf5 \
    --years 2019 \
    --split-mode event_kfold \
    --event-folds 3 \
    --seed 42 \
    --fold "$fold" \
    --split validation \
    --batch-size 2 \
    --exclude-empty-current-fire \
    --output-dir "$STAGE1_2019_PRED_DIR"

  "$PYTHON" scripts/12_build_stage2_candidates.py \
    --predictions-dir "$STAGE1_2019_PRED_DIR/fold_${fold}/validation" \
    --data-root wsts_hdf5 \
    --backend hdf5 \
    --years 2019 \
    --split-mode event_kfold \
    --event-folds 3 \
    --seed 42 \
    --fold "$fold" \
    --split validation \
    --exclude-empty-current-fire \
    --ring-radius-km 4 \
    --output-dir "$STAGE2_2019_CAND_DIR"
done

"$PYTHON" scripts/13_train_stage2_gbm.py \
  --train-csv \
    "$STAGE2_2019_CAND_DIR/stage2_fold1_validation_candidates.csv" \
    "$STAGE2_2019_CAND_DIR/stage2_fold2_validation_candidates.csv" \
  --val-csv "$STAGE2_2019_CAND_DIR/stage2_fold0_validation_candidates.csv" \
  --output-dir "$STAGE2_MODEL_DIR" \
  --model-name stage2_gbm_3fold_r4km_imagenet_noamp_holdout0

"$PYTHON" scripts/13_train_stage2_gbm.py \
  --train-csv \
    "$STAGE2_2019_CAND_DIR/stage2_fold0_validation_candidates.csv" \
    "$STAGE2_2019_CAND_DIR/stage2_fold2_validation_candidates.csv" \
  --val-csv "$STAGE2_2019_CAND_DIR/stage2_fold1_validation_candidates.csv" \
  --output-dir "$STAGE2_MODEL_DIR" \
  --model-name stage2_gbm_3fold_r4km_imagenet_noamp_holdout1

"$PYTHON" scripts/13_train_stage2_gbm.py \
  --train-csv \
    "$STAGE2_2019_CAND_DIR/stage2_fold0_validation_candidates.csv" \
    "$STAGE2_2019_CAND_DIR/stage2_fold1_validation_candidates.csv" \
  --val-csv "$STAGE2_2019_CAND_DIR/stage2_fold2_validation_candidates.csv" \
  --output-dir "$STAGE2_MODEL_DIR" \
  --model-name stage2_gbm_3fold_r4km_imagenet_noamp_holdout2

"$PYTHON" scripts/24_calibrate_stage2_oof_threshold.py \
  --models \
    "$STAGE2_MODEL_DIR/stage2_gbm_3fold_r4km_imagenet_noamp_holdout0.joblib" \
    "$STAGE2_MODEL_DIR/stage2_gbm_3fold_r4km_imagenet_noamp_holdout1.joblib" \
    "$STAGE2_MODEL_DIR/stage2_gbm_3fold_r4km_imagenet_noamp_holdout2.joblib" \
  --candidate-csvs \
    "$STAGE2_2019_CAND_DIR/stage2_fold0_validation_candidates.csv" \
    "$STAGE2_2019_CAND_DIR/stage2_fold1_validation_candidates.csv" \
    "$STAGE2_2019_CAND_DIR/stage2_fold2_validation_candidates.csv" \
  --output-json "$STAGE2_CAL_JSON"

THRESHOLD="$("$PYTHON" - <<'PY'
import json
from pathlib import Path
report = json.loads(Path("outputs/stage2/calibration_3fold_r4km_imagenet_noamp/stage2_oof_threshold.json").read_text())
print(report["frozen_threshold"])
PY
)"

mkdir -p "$STAGE2_2020_SCORED_DIR"
"$PYTHON" scripts/25_score_stage2_ensemble.py \
  --models \
    "$STAGE2_MODEL_DIR/stage2_gbm_3fold_r4km_imagenet_noamp_holdout0.joblib" \
    "$STAGE2_MODEL_DIR/stage2_gbm_3fold_r4km_imagenet_noamp_holdout1.joblib" \
    "$STAGE2_MODEL_DIR/stage2_gbm_3fold_r4km_imagenet_noamp_holdout2.joblib" \
  --candidate-csv outputs/stage2/candidates_2020_r4km_fair_imagenet_noamp/stage2_2020_candidates.csv \
  --output-csv "$STAGE2_2020_SCORED_DIR/stage2_2020_scored_candidates.csv"

"$PYTHON" scripts/27_compare_matched_alert_volume.py \
  --scored-candidate-csv "$STAGE2_2020_SCORED_DIR/stage2_2020_scored_candidates.csv" \
  --model-name stage2_2020_external_ensemble_imagenet_noamp_retrained_stage2 \
  --stage2-threshold "$THRESHOLD" \
  --radii-km 2 3 4 \
  --output-dir "$MATCHED_DIR"

"$PYTHON" scripts/20_evaluate_rolling_alerts.py \
  --scored-candidate-csv "$STAGE2_2020_SCORED_DIR/stage2_2020_scored_candidates.csv" \
  --model-name stage2_2020_external_ensemble_imagenet_noamp_retrained_stage2 \
  --threshold "$THRESHOLD" \
  --radii-km 2 3 4 \
  --naive-radius-km 4 \
  --output-dir "$ROLLING_DIR"

"$PYTHON" - <<'PY'
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

csv = Path("outputs/stage2/scored_2020_r4km_fair_imagenet_noamp_retrained_stage2/stage2_2020_scored_candidates.csv")
out = Path("outputs/stage2/evaluation_2020_imagenet_noamp_retrained_stage2")
out.mkdir(parents=True, exist_ok=True)
cols = ["event_id", "target_date", "label", "model_score", "stage1_probability", "distance_to_current_fire_px"]
df = pd.read_csv(csv, usecols=lambda column: column in cols)
labels = df["label"].astype(int).to_numpy()

def safe_auc(y, scores):
    return float(roc_auc_score(y, scores)) if np.unique(y).size > 1 else 0.0

def within_event_auc(scores):
    tmp = df[["event_id", "target_date", "label"]].copy()
    tmp["score"] = scores
    values = []
    for _, group in tmp.groupby(["event_id", "target_date"], sort=False):
        y = group["label"].astype(int).to_numpy()
        if np.unique(y).size > 1:
            values.append(float(roc_auc_score(y, group["score"].to_numpy())))
    return float(np.mean(values)) if values else 0.0

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

echo "$THRESHOLD" > outputs/stage2/calibration_3fold_r4km_imagenet_noamp/frozen_threshold.txt
