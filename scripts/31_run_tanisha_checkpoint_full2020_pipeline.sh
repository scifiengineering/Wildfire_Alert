#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"

STAGE1_CKPT_DIR="Specifications"
STAGE1_CAL_DIR="outputs/calibration/stage1_2019_3fold_tanisha_ckpt"
STAGE1_2019_PRED_DIR="outputs/predictions/stage1_2019_3fold_tanisha_ckpt_oof"
STAGE2_2019_CAND_DIR="outputs/stage2/candidates_3fold_r4km_fair_tanisha_ckpt"
STAGE2_MODEL_DIR="outputs/stage2/models_3fold_r4km_tanisha_ckpt"
STAGE2_CAL_JSON="outputs/stage2/calibration_3fold_r4km_tanisha_ckpt/stage2_oof_threshold.json"
STAGE1_2020_PREFIX="outputs/predictions/stage1_2020_tanisha_ckpt_full2020"
STAGE1_2020_ENSEMBLE="${STAGE1_2020_PREFIX}_ensemble"
STAGE2_2020_CAND_DIR="outputs/stage2/candidates_2020_r4km_fair_tanisha_ckpt_full2020"
STAGE2_2020_SCORED_DIR="outputs/stage2/scored_2020_r4km_fair_tanisha_ckpt_full2020"
MATCHED_DIR="outputs/stage2/matched_volume_tanisha_ckpt_full2020"
MATCHED_DUAL_DIR="outputs/stage2/matched_volume_tanisha_ckpt_full2020_dual_catch"
ROLLING_DIR="outputs/stage2/rolling_alerts_2020_tanisha_ckpt_full2020"
EVAL_DIR="outputs/stage2/evaluation_2020_tanisha_ckpt_full2020"

mkdir -p outputs/logs "$STAGE1_CAL_DIR" "$(dirname "$STAGE2_CAL_JSON")"

for fold in 0 1 2; do
  "$PYTHON" scripts/10_calibrate_stage1_threshold.py \
    --checkpoint "$STAGE1_CKPT_DIR/stage1_fold_${fold}_best.pt" \
    --data-root wsts_hdf5 \
    --backend hdf5 \
    --split-mode event_kfold \
    --years 2019 \
    --event-folds 3 \
    --seed 42 \
    --fold "$fold" \
    --batch-size 2 \
    --num-workers 2 \
    --exclude-empty-current-fire \
    --output-dir "$STAGE1_CAL_DIR"

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
    --num-workers 2 \
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
  --model-name stage2_gbm_3fold_r4km_tanisha_ckpt_holdout0

"$PYTHON" scripts/13_train_stage2_gbm.py \
  --train-csv \
    "$STAGE2_2019_CAND_DIR/stage2_fold0_validation_candidates.csv" \
    "$STAGE2_2019_CAND_DIR/stage2_fold2_validation_candidates.csv" \
  --val-csv "$STAGE2_2019_CAND_DIR/stage2_fold1_validation_candidates.csv" \
  --output-dir "$STAGE2_MODEL_DIR" \
  --model-name stage2_gbm_3fold_r4km_tanisha_ckpt_holdout1

"$PYTHON" scripts/13_train_stage2_gbm.py \
  --train-csv \
    "$STAGE2_2019_CAND_DIR/stage2_fold0_validation_candidates.csv" \
    "$STAGE2_2019_CAND_DIR/stage2_fold1_validation_candidates.csv" \
  --val-csv "$STAGE2_2019_CAND_DIR/stage2_fold2_validation_candidates.csv" \
  --output-dir "$STAGE2_MODEL_DIR" \
  --model-name stage2_gbm_3fold_r4km_tanisha_ckpt_holdout2

"$PYTHON" scripts/24_calibrate_stage2_oof_threshold.py \
  --models \
    "$STAGE2_MODEL_DIR/stage2_gbm_3fold_r4km_tanisha_ckpt_holdout0.joblib" \
    "$STAGE2_MODEL_DIR/stage2_gbm_3fold_r4km_tanisha_ckpt_holdout1.joblib" \
    "$STAGE2_MODEL_DIR/stage2_gbm_3fold_r4km_tanisha_ckpt_holdout2.joblib" \
  --candidate-csvs \
    "$STAGE2_2019_CAND_DIR/stage2_fold0_validation_candidates.csv" \
    "$STAGE2_2019_CAND_DIR/stage2_fold1_validation_candidates.csv" \
    "$STAGE2_2019_CAND_DIR/stage2_fold2_validation_candidates.csv" \
  --output-json "$STAGE2_CAL_JSON"

THRESHOLD="$("$PYTHON" - <<PY
import json
from pathlib import Path
print(json.loads(Path("$STAGE2_CAL_JSON").read_text())["frozen_threshold"])
PY
)"

for fold in 0 1 2; do
  "$PYTHON" scripts/21_predict_stage1_year.py \
    --checkpoint "$STAGE1_CKPT_DIR/stage1_fold_${fold}_best.pt" \
    --calibration-json "$STAGE1_CAL_DIR/stage1_fold${fold}_thresholds.json" \
    --data-root wsts_hdf5 \
    --backend hdf5 \
    --year 2020 \
    --batch-size 2 \
    --num-workers 2 \
    --exclude-empty-current-fire \
    --output-dir "${STAGE1_2020_PREFIX}_fold_${fold}"
done

ENSEMBLE_THRESHOLD="$("$PYTHON" - <<PY
import json
from pathlib import Path
values = []
for fold in range(3):
    report = json.loads(Path("$STAGE1_CAL_DIR") .joinpath(f"stage1_fold{fold}_thresholds.json").read_text())
    values.append(float(report["best_threshold_by_dice"]["threshold"]))
print(sum(values) / len(values))
PY
)"

"$PYTHON" scripts/22_ensemble_stage1_predictions.py \
  --prediction-dirs \
    "${STAGE1_2020_PREFIX}_fold_0" \
    "${STAGE1_2020_PREFIX}_fold_1" \
    "${STAGE1_2020_PREFIX}_fold_2" \
  --threshold "$ENSEMBLE_THRESHOLD" \
  --top-fraction 0.001 \
  --output-dir "$STAGE1_2020_ENSEMBLE"

"$PYTHON" scripts/23_build_stage2_candidates_year.py \
  --predictions-dir "$STAGE1_2020_ENSEMBLE" \
  --data-root wsts_hdf5 \
  --backend hdf5 \
  --year 2020 \
  --exclude-empty-current-fire \
  --ring-radius-km 4 \
  --output-csv "$STAGE2_2020_CAND_DIR/stage2_2020_candidates.csv"

"$PYTHON" scripts/25_score_stage2_ensemble.py \
  --models \
    "$STAGE2_MODEL_DIR/stage2_gbm_3fold_r4km_tanisha_ckpt_holdout0.joblib" \
    "$STAGE2_MODEL_DIR/stage2_gbm_3fold_r4km_tanisha_ckpt_holdout1.joblib" \
    "$STAGE2_MODEL_DIR/stage2_gbm_3fold_r4km_tanisha_ckpt_holdout2.joblib" \
  --candidate-csv "$STAGE2_2020_CAND_DIR/stage2_2020_candidates.csv" \
  --output-csv "$STAGE2_2020_SCORED_DIR/stage2_2020_scored_candidates.csv"

"$PYTHON" scripts/27_compare_matched_alert_volume.py \
  --scored-candidate-csv "$STAGE2_2020_SCORED_DIR/stage2_2020_scored_candidates.csv" \
  --model-name stage2_2020_tanisha_ckpt_full2020 \
  --stage2-threshold "$THRESHOLD" \
  --radii-km 2 3 4 \
  --output-dir "$MATCHED_DIR"

"$PYTHON" scripts/20_evaluate_rolling_alerts.py \
  --scored-candidate-csv "$STAGE2_2020_SCORED_DIR/stage2_2020_scored_candidates.csv" \
  --model-name stage2_2020_tanisha_ckpt_full2020 \
  --threshold "$THRESHOLD" \
  --radii-km 2 3 4 \
  --naive-radius-km 4 \
  --output-dir "$ROLLING_DIR"

"$PYTHON" - <<PY
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

scored = Path("$STAGE2_2020_SCORED_DIR/stage2_2020_scored_candidates.csv")
matched_summary = Path("$MATCHED_DIR/stage2_2020_tanisha_ckpt_full2020_matched_volume_summary.json")
matched_rows = Path("$MATCHED_DIR/stage2_2020_tanisha_ckpt_full2020_matched_volume_event_days.csv")
dual_dir = Path("$MATCHED_DUAL_DIR")
eval_dir = Path("$EVAL_DIR")
dual_dir.mkdir(parents=True, exist_ok=True)
eval_dir.mkdir(parents=True, exist_ok=True)

summary = json.loads(matched_summary.read_text())
rows = pd.read_csv(matched_rows)
positive_rows = rows[rows["positive_count"].astype(int) > 0]

dual = {
    "source_summary": str(matched_summary),
    "event_day_count_all": int(len(rows)),
    "event_day_count_positive": int(len(positive_rows)),
    "stage2": {
        "catch_rate_all_event_days": float((rows["stage2_true_positive_alert_count"].astype(int) > 0).mean()),
        "catch_rate_positive_event_days": float((positive_rows["stage2_true_positive_alert_count"].astype(int) > 0).mean()),
    },
    "wsts_as_alerter": {
        "catch_rate_all_event_days": float((rows["wsts_true_positive_alert_count"].astype(int) > 0).mean()),
        "catch_rate_positive_event_days": float((positive_rows["wsts_true_positive_alert_count"].astype(int) > 0).mean()),
    },
}
dual["stage2"]["alerts_per_event_day"] = summary["stage2_alerts_per_event_day"]
dual["stage2"]["precision"] = summary["stage2_precision"]
dual["stage2"]["recall"] = summary["stage2_recall"]
dual["stage2"]["true_positive_alert_count"] = summary["stage2_true_positive_alert_count"]
dual["wsts_as_alerter"]["alerts_per_event_day"] = summary["wsts_alerts_per_event_day"]
dual["wsts_as_alerter"]["precision"] = summary["wsts_precision"]
dual["wsts_as_alerter"]["recall"] = summary["wsts_recall"]
dual["wsts_as_alerter"]["true_positive_alert_count"] = summary["wsts_true_positive_alert_count"]
dual["comparison"] = {
    "true_positive_alert_gain": summary["true_positive_alert_gain"],
    "relative_true_positive_alert_gain": summary["relative_true_positive_alert_gain"],
    "catch_rate_all_event_days_gain": dual["stage2"]["catch_rate_all_event_days"] - dual["wsts_as_alerter"]["catch_rate_all_event_days"],
    "catch_rate_positive_event_days_gain": dual["stage2"]["catch_rate_positive_event_days"] - dual["wsts_as_alerter"]["catch_rate_positive_event_days"],
}
(dual_dir / "stage2_2020_tanisha_ckpt_full2020_dual_catch_rate.json").write_text(json.dumps(dual, indent=2) + "\n")

cols = ["event_id", "target_date", "label", "model_score", "stage1_probability", "distance_to_current_fire_px"]
df = pd.read_csv(scored, usecols=lambda column: column in cols)
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
    "candidate_csv": str(scored),
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
(eval_dir / "stage2_2020_tanisha_ckpt_full2020_evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"threshold": "$THRESHOLD", "dual_catch": dual, "ranking": report["summary"]}, indent=2))
PY

echo "$THRESHOLD" > "$(dirname "$STAGE2_CAL_JSON")/frozen_threshold.txt"
