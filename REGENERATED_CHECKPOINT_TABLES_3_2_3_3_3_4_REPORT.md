# Regenerated-Checkpoint Results for Tables 3.2, 3.3, and 3.4

## Scope

This report uses Sifiso's regenerated checkpoint support run only. It does not mix in the original checkpoints or the original-checkpoint rerun.

- Stage 1 checkpoint family: `outputs/checkpoints/stage1_2019_3fold_imagenet_noamp_20260615/stage1_fold_{0,1,2}_best.pt`
- Stage 2 model family: `outputs/stage2/models_3fold_r4km_imagenet_noamp/`
- 2019 rows use the regenerated three-fold Stage 2 models and their fold-specific best-F1 thresholds.
- 2020 rows use the regenerated 2020 Stage 2 scored candidates and frozen 2020 operating threshold from the regenerated run.

## Table 3.2: Alert Volume vs. Naive 4 km Rule

| Split | Mean alerts (model) | Mean alerts (naive 4 km) | Alert reduction | Precision (model vs naive) |
|---|---:|---:|---:|---:|
| 2019 rolling | 39.9 | 1080.0 | 96.9% | 0.119 vs 0.027 |
| 2020 rolling | 75.9 | 2589.5 | 97.4% | 0.159 vs 0.040 |

The regenerated run keeps the main alert-fatigue finding intact: the model sends far fewer alerts than the naive 4 km rule while maintaining substantially higher precision.

## Optional High-Confidence Operating Points

These rows answer the professor's request for a compact view of the most confident alerts. The alert budget is produced by ranking candidate locations by confidence and keeping only the top-scoring rows. A smaller alerts/day budget therefore means a stricter confidence cutoff.

| Alerts/day | Stage 2 min confidence | Stage 2 precision | Stage 2 recall | Stage 2 F1 | WSTS min score | WSTS precision | WSTS recall | WSTS F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.9425 | 0.6963 | 0.0298 | 0.0572 | 0.8719 | 0.5663 | 0.0242 | 0.0465 |
| 10 | 0.9173 | 0.6480 | 0.0555 | 0.1022 | 0.8538 | 0.5251 | 0.0450 | 0.0828 |
| 15 | 0.8958 | 0.5892 | 0.0757 | 0.1341 | 0.8372 | 0.4924 | 0.0632 | 0.1121 |
| 20 | 0.8841 | 0.5281 | 0.0904 | 0.1544 | 0.8215 | 0.4601 | 0.0788 | 0.1345 |
| 25 | 0.8757 | 0.4880 | 0.1044 | 0.1721 | 0.8063 | 0.4362 | 0.0934 | 0.1538 |

| Alerts/day | Stage 2 min confidence | Stage 2 precision | Stage 2 recall | Stage 2 F1 | WSTS min score | WSTS precision | WSTS recall | WSTS F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 0.8455 | 0.3866 | 0.1655 | 0.2318 | 0.7358 | 0.3512 | 0.1503 | 0.2106 |
| 75 | 0.8185 | 0.3399 | 0.2183 | 0.2659 | 0.6574 | 0.2973 | 0.1909 | 0.2325 |
| 90 | 0.8014 | 0.3203 | 0.2468 | 0.2787 | 0.6079 | 0.2762 | 0.2128 | 0.2404 |
| 114 | 0.7676 | 0.2965 | 0.2893 | 0.2928 | 0.5423 | 0.2515 | 0.2455 | 0.2484 |
| 150 | 0.7122 | 0.2676 | 0.3436 | 0.3009 | 0.4797 | 0.2243 | 0.2881 | 0.2522 |
| 250 | 0.5869 | 0.2201 | 0.4712 | 0.3001 | 0.3529 | 0.1735 | 0.3713 | 0.2365 |

At 10 alerts/day, Stage 2 keeps candidates scoring at least 0.917 and reaches 0.648 precision with 0.055 recall. At 5 alerts/day, it keeps candidates scoring at least 0.943 and reaches 0.696 precision with 0.030 recall. This is useful as an operational high-confidence framing, but the recall cost should be stated clearly.

## Table 3.3: Ranking Quality

| Split | AP | AUC |
|---|---:|---:|
| 2019 Fold 0 | 0.2093 | 0.8623 |
| 2019 Fold 1 | 0.2700 | 0.8861 |
| 2019 Fold 2 | 0.2870 | 0.8726 |
| 2019 rolling | 0.2678 | 0.8414 |
| 2020 rolling | 0.2409 | 0.8200 |

## Table 3.4: Location Discrimination

| Held-out fold | Stage 2 AP | Distance AP | Stage-1 AP | Gain over distance |
|---|---:|---:|---:|---:|
| Fold 0 | 0.2093 | 0.1541 | 0.1036 | +0.0551 |
| Fold 1 | 0.2700 | 0.2082 | 0.1506 | +0.0618 |
| Fold 2 | 0.2870 | 0.1790 | 0.2127 | +0.1080 |

For 2019, Stage 2 improves AP over both distance and Stage-1 ranking in all three folds. Within-event AUC is more mixed: Fold 1 is above distance, while Folds 0 and 2 are slightly below distance. That means the regenerated run supports the AP-based discrimination claim cleanly, but the report should avoid overclaiming that every ranking metric beats distance on every fold.

## 2020 Discrimination Check

| Metric | Stage 2 | WSTS/Stage 1 | Distance |
|---|---:|---:|---:|
| AP | 0.2511 | 0.1844 | 0.2040 |
| Global AUC | 0.8186 | 0.7089 | 0.7930 |
| Within-event AUC | 0.8600 | 0.6961 | 0.8507 |

The 2020 regenerated-checkpoint run supports the thesis direction: Stage 2 is above WSTS/Stage 1 and distance on AP, global AUC, and within-event AUC.

## Commands Used

```bash
for fold in 0 1 2; do
  threshold=$(python - "$fold" <<'PY'
import json, sys
from pathlib import Path
fold = sys.argv[1]
path = Path(f"outputs/stage2/models_3fold_r4km_imagenet_noamp/stage2_gbm_3fold_r4km_imagenet_noamp_holdout{fold}_metrics.json")
print(json.loads(path.read_text())["best_threshold_by_f1"]["threshold"])
PY
)
  .venv/bin/python scripts/20_evaluate_rolling_alerts.py \
    --model outputs/stage2/models_3fold_r4km_imagenet_noamp/stage2_gbm_3fold_r4km_imagenet_noamp_holdout${fold}.joblib \
    --candidate-csv outputs/stage2/candidates_3fold_r4km_fair_imagenet_noamp/stage2_fold${fold}_validation_candidates.csv \
    --model-name stage2_gbm_3fold_r4km_imagenet_noamp_holdout${fold} \
    --threshold "$threshold" \
    --radii-km 2 3 4 \
    --naive-radius-km 4 \
    --output-dir outputs/stage2/rolling_alerts_3fold_r4km_imagenet_noamp

  .venv/bin/python scripts/14_evaluate_stage2_model.py \
    --model outputs/stage2/models_3fold_r4km_imagenet_noamp/stage2_gbm_3fold_r4km_imagenet_noamp_holdout${fold}.joblib \
    --candidate-csv outputs/stage2/candidates_3fold_r4km_fair_imagenet_noamp/stage2_fold${fold}_validation_candidates.csv \
    --model-name stage2_gbm_3fold_r4km_imagenet_noamp_holdout${fold} \
    --output-dir outputs/stage2/evaluation_3fold_r4km_imagenet_noamp
done

.venv/bin/python scripts/34_summarize_regenerated_checkpoint_tables.py
```

## Output Artifacts

- Aggregate JSON: `outputs/stage2/regenerated_checkpoint_tables/tables_3_2_3_3_3_4_summary.json`
- Report: `REGENERATED_CHECKPOINT_TABLES_3_2_3_3_3_4_REPORT.md`
- 2019 rolling summaries: `outputs/stage2/rolling_alerts_3fold_r4km_imagenet_noamp/*_rolling_summary.json`
- 2019 discrimination summaries: `outputs/stage2/evaluation_3fold_r4km_imagenet_noamp/*_evaluation.json`
