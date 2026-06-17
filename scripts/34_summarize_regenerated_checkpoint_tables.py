"""Summarize regenerated-checkpoint results for thesis tables 3.2-3.4."""

# ruff: noqa: E501

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path("outputs/stage2")
ROLLING_2019_DIR = ROOT / "rolling_alerts_3fold_r4km_imagenet_noamp"
EVAL_2019_DIR = ROOT / "evaluation_3fold_r4km_imagenet_noamp"
MODEL_2019_DIR = ROOT / "models_3fold_r4km_imagenet_noamp"
ROLLING_2020_SUMMARY = (
    ROOT
    / "rolling_alerts_2020_imagenet_noamp_full2020_retrained_stage2"
    / "stage2_2020_external_ensemble_imagenet_noamp_full2020_retrained_stage2_rolling_summary.json"
)
EVAL_2020_SUMMARY = (
    ROOT
    / "evaluation_2020_imagenet_noamp_full2020_retrained_stage2"
    / "stage2_2020_external_ensemble_imagenet_noamp_full2020_retrained_stage2_evaluation.json"
)
PRECISION_DIAGNOSTIC = ROOT / "precision_diagnostics" / "precision_budget_diagnostics.json"
OUTPUT_DIR = ROOT / "regenerated_checkpoint_tables"
SUMMARY_PATH = OUTPUT_DIR / "tables_3_2_3_3_3_4_summary.json"
REPORT_PATH = Path("REGENERATED_CHECKPOINT_TABLES_3_2_3_3_3_4_REPORT.md")


def main() -> None:
    """Create the aggregate JSON and Markdown report."""

    summary = build_summary()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(render_report(summary), encoding="utf-8")
    print(f"wrote {SUMMARY_PATH}")
    print(f"wrote {REPORT_PATH}")


def build_summary() -> dict[str, Any]:
    """Collect the regenerated-checkpoint table values."""

    rolling_2019 = aggregate_2019_rolling()
    rolling_2020 = read_json(ROLLING_2020_SUMMARY)
    eval_2020 = read_json(EVAL_2020_SUMMARY)["summary"]
    fold_rows = [fold_metrics(fold) for fold in range(3)]
    precision_budget = read_json(PRECISION_DIAGNOSTIC)["regenerated"]["budgets"]

    return {
        "scope": "regenerated-checkpoint support run only",
        "stage1_checkpoint_family": (
            "outputs/checkpoints/stage1_2019_3fold_imagenet_noamp_20260615/"
            "stage1_fold_{0,1,2}_best.pt"
        ),
        "stage2_model_family": "outputs/stage2/models_3fold_r4km_imagenet_noamp/",
        "table_3_2_2019_rolling": rolling_2019,
        "table_3_2_2020_rolling": {
            "event_day_count": rolling_2020["event_day_count"],
            "window_count": rolling_2020["window_count"],
            "mean_model_alert_count": rolling_2020["mean_model_alert_count"],
            "mean_naive_alert_count": rolling_2020["mean_naive_alert_count"],
            "mean_model_precision": rolling_2020["mean_model_precision"],
            "mean_naive_precision": rolling_2020["mean_naive_precision"],
            "mean_alert_reduction_vs_naive": rolling_2020["mean_alert_reduction_vs_naive"],
            "mean_ap": rolling_2020["mean_ap"],
            "mean_auc": rolling_2020["mean_auc"],
            "threshold": rolling_2020["threshold"],
        },
        "table_3_3_fold_metrics": fold_rows,
        "table_3_3_2019_rolling": {
            "ap": rolling_2019["mean_ap"],
            "auc": rolling_2019["mean_auc"],
        },
        "table_3_3_2020_rolling": {
            "ap": rolling_2020["mean_ap"],
            "auc": rolling_2020["mean_auc"],
        },
        "table_3_4_fold_metrics": fold_rows,
        "table_3_4_2020_discrimination": eval_2020,
        "precision_budget_rows": precision_budget,
    }


def aggregate_2019_rolling() -> dict[str, float | int]:
    """Pool the three 2019 rolling event-day CSVs."""

    frames = []
    for fold in range(3):
        path = (
            ROLLING_2019_DIR
            / f"stage2_gbm_3fold_r4km_imagenet_noamp_holdout{fold}_rolling_event_days.csv"
        )
        frame = pd.read_csv(path)
        frame["fold"] = fold
        frames.append(frame)
    rows = pd.concat(frames, ignore_index=True)
    summary: dict[str, float | int] = {
        "event_day_count": int(len(rows)),
        "window_count": int(rows["window_id"].nunique()),
    }
    for column in [
        "model_alert_count",
        "naive_alert_count",
        "model_precision",
        "naive_precision",
        "alert_reduction_vs_naive",
        "ap",
        "auc",
    ]:
        summary[f"mean_{column}"] = float(pd.to_numeric(rows[column], errors="coerce").mean())
    return summary


def fold_metrics(fold: int) -> dict[str, float | int]:
    """Read the regenerated fold evaluation and training metric artifacts."""

    metrics = read_json(
        MODEL_2019_DIR / f"stage2_gbm_3fold_r4km_imagenet_noamp_holdout{fold}_metrics.json"
    )
    evaluation = read_json(
        EVAL_2019_DIR / f"stage2_gbm_3fold_r4km_imagenet_noamp_holdout{fold}_evaluation.json"
    )["summary"]
    threshold = metrics["best_threshold_by_f1"]
    return {
        "fold": fold,
        "stage2_ap": evaluation["model_average_precision"],
        "distance_ap": evaluation["distance_baseline_average_precision"],
        "stage1_ap": evaluation["stage1_baseline_average_precision"],
        "gain_over_distance": evaluation["ap_gain_over_distance"],
        "stage2_within_event_auc": metrics["validation_within_event_auc"],
        "stage2_roc_auc": metrics["validation_roc_auc"],
        "distance_within_event_auc": evaluation["distance_baseline_within_event_auc"],
        "stage1_within_event_auc": evaluation["stage1_baseline_within_event_auc"],
        "within_event_auc_gain_over_distance": evaluation[
            "within_event_auc_gain_over_distance"
        ],
        "threshold": threshold["threshold"],
        "threshold_precision": threshold["precision"],
        "threshold_recall": threshold["recall"],
        "threshold_f1": threshold["f1"],
    }


def render_report(summary: dict[str, Any]) -> str:
    """Render a thesis-facing Markdown report."""

    t32_2019 = summary["table_3_2_2019_rolling"]
    t32_2020 = summary["table_3_2_2020_rolling"]
    fold_rows = summary["table_3_3_fold_metrics"]
    t33_2019 = summary["table_3_3_2019_rolling"]
    t33_2020 = summary["table_3_3_2020_rolling"]
    t34_2020 = summary["table_3_4_2020_discrimination"]
    budget_rows = {row["alerts_per_day"]: row for row in summary["precision_budget_rows"]}

    high_confidence_days = [5, 10, 15, 20, 25]
    broader_days = [50, 75, 90, 114, 150, 250]

    lines = [
        "# Regenerated-Checkpoint Results for Tables 3.2, 3.3, and 3.4",
        "",
        "## Scope",
        "",
        "This report uses Sifiso's regenerated checkpoint support run only. It does not mix in Tanisha's original checkpoints or the Tanisha-checkpoint rerun.",
        "",
        f"- Stage 1 checkpoint family: `{summary['stage1_checkpoint_family']}`",
        f"- Stage 2 model family: `{summary['stage2_model_family']}`",
        "- 2019 rows use the regenerated three-fold Stage 2 models and their fold-specific best-F1 thresholds.",
        "- 2020 rows use the regenerated 2020 Stage 2 scored candidates and frozen 2020 operating threshold from the regenerated run.",
        "",
        "## Table 3.2: Alert Volume vs. Naive 4 km Rule",
        "",
        "| Split | Mean alerts (model) | Mean alerts (naive 4 km) | Alert reduction | Precision (model vs naive) |",
        "|---|---:|---:|---:|---:|",
        t32_row("2019 rolling", t32_2019),
        t32_row("2020 rolling", t32_2020),
        "",
        "The regenerated run keeps the main alert-fatigue finding intact: the model sends far fewer alerts than the naive 4 km rule while maintaining substantially higher precision.",
        "",
        "## Optional High-Confidence Operating Points",
        "",
        "These rows answer the professor's request for a compact view of the most confident alerts. They are not a replacement for the rolling operating point above; they show what precision, recall, and F1 look like if the alert budget is tightened.",
        "",
        "| Alerts/day | Stage 2 precision | Stage 2 recall | Stage 2 F1 | WSTS precision | WSTS recall | WSTS F1 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(budget_row(budget_rows[day]) for day in high_confidence_days)
    lines.extend(
        [
            "",
            "| Alerts/day | Stage 2 precision | Stage 2 recall | Stage 2 F1 | WSTS precision | WSTS recall | WSTS F1 |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    lines.extend(budget_row(budget_rows[day]) for day in broader_days)
    lines.extend(
        [
            "",
            "At 10 alerts/day, Stage 2 reaches 0.648 precision with 0.055 recall; at 5 alerts/day, it reaches 0.696 precision with 0.030 recall. This is useful as an operational high-confidence framing, but the recall cost should be stated clearly.",
            "",
            "## Table 3.3: Ranking Quality",
            "",
            "| Split | AP | AUC |",
            "|---|---:|---:|",
        ]
    )
    for row in fold_rows:
        lines.append(
            f"| 2019 Fold {row['fold']} | {row['stage2_ap']:.4f} | {row['stage2_within_event_auc']:.4f} |"
        )
    lines.extend(
        [
            f"| 2019 rolling | {t33_2019['ap']:.4f} | {t33_2019['auc']:.4f} |",
            f"| 2020 rolling | {t33_2020['ap']:.4f} | {t33_2020['auc']:.4f} |",
            "",
            "## Table 3.4: Location Discrimination",
            "",
            "| Held-out fold | Stage 2 AP | Distance AP | Stage-1 AP | Gain over distance |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in fold_rows:
        lines.append(
            f"| Fold {row['fold']} | {row['stage2_ap']:.4f} | {row['distance_ap']:.4f} | {row['stage1_ap']:.4f} | {row['gain_over_distance']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "For 2019, Stage 2 improves AP over both distance and Stage-1 ranking in all three folds. Within-event AUC is more mixed: Fold 1 is above distance, while Folds 0 and 2 are slightly below distance. That means the regenerated run supports the AP-based discrimination claim cleanly, but the report should avoid overclaiming that every ranking metric beats distance on every fold.",
            "",
            "## 2020 Discrimination Check",
            "",
            "| Metric | Stage 2 | WSTS/Stage 1 | Distance |",
            "|---|---:|---:|---:|",
            f"| AP | {t34_2020['stage2_ap']:.4f} | {t34_2020['wsts_stage1_ap']:.4f} | {t34_2020['distance_ap']:.4f} |",
            f"| Global AUC | {t34_2020['stage2_auc']:.4f} | {t34_2020['wsts_stage1_auc']:.4f} | {t34_2020['distance_auc']:.4f} |",
            f"| Within-event AUC | {t34_2020['stage2_within_event_auc']:.4f} | {t34_2020['wsts_stage1_within_event_auc']:.4f} | {t34_2020['distance_within_event_auc']:.4f} |",
            "",
            "The 2020 regenerated-checkpoint run supports Tanisha's direction: Stage 2 is above WSTS/Stage 1 and distance on AP, global AUC, and within-event AUC.",
            "",
            "## Commands Used",
            "",
            "```bash",
            "for fold in 0 1 2; do",
            "  threshold=$(python - \"$fold\" <<'PY'",
            "import json, sys",
            "from pathlib import Path",
            "fold = sys.argv[1]",
            "path = Path(f\"outputs/stage2/models_3fold_r4km_imagenet_noamp/stage2_gbm_3fold_r4km_imagenet_noamp_holdout{fold}_metrics.json\")",
            "print(json.loads(path.read_text())[\"best_threshold_by_f1\"][\"threshold\"])",
            "PY",
            ")",
            "  .venv/bin/python scripts/20_evaluate_rolling_alerts.py \\",
            "    --model outputs/stage2/models_3fold_r4km_imagenet_noamp/stage2_gbm_3fold_r4km_imagenet_noamp_holdout${fold}.joblib \\",
            "    --candidate-csv outputs/stage2/candidates_3fold_r4km_fair_imagenet_noamp/stage2_fold${fold}_validation_candidates.csv \\",
            "    --model-name stage2_gbm_3fold_r4km_imagenet_noamp_holdout${fold} \\",
            "    --threshold \"$threshold\" \\",
            "    --radii-km 2 3 4 \\",
            "    --naive-radius-km 4 \\",
            "    --output-dir outputs/stage2/rolling_alerts_3fold_r4km_imagenet_noamp",
            "",
            "  .venv/bin/python scripts/14_evaluate_stage2_model.py \\",
            "    --model outputs/stage2/models_3fold_r4km_imagenet_noamp/stage2_gbm_3fold_r4km_imagenet_noamp_holdout${fold}.joblib \\",
            "    --candidate-csv outputs/stage2/candidates_3fold_r4km_fair_imagenet_noamp/stage2_fold${fold}_validation_candidates.csv \\",
            "    --model-name stage2_gbm_3fold_r4km_imagenet_noamp_holdout${fold} \\",
            "    --output-dir outputs/stage2/evaluation_3fold_r4km_imagenet_noamp",
            "done",
            "",
            ".venv/bin/python scripts/34_summarize_regenerated_checkpoint_tables.py",
            "```",
            "",
            "## Output Artifacts",
            "",
            f"- Aggregate JSON: `{SUMMARY_PATH}`",
            f"- Report: `{REPORT_PATH}`",
            "- 2019 rolling summaries: `outputs/stage2/rolling_alerts_3fold_r4km_imagenet_noamp/*_rolling_summary.json`",
            "- 2019 discrimination summaries: `outputs/stage2/evaluation_3fold_r4km_imagenet_noamp/*_evaluation.json`",
            "",
        ]
    )
    return "\n".join(lines)


def t32_row(label: str, row: dict[str, float | int]) -> str:
    """Render one Table 3.2 row."""

    return (
        f"| {label} | {row['mean_model_alert_count']:.1f} | "
        f"{row['mean_naive_alert_count']:.1f} | "
        f"{100 * row['mean_alert_reduction_vs_naive']:.1f}% | "
        f"{row['mean_model_precision']:.3f} vs {row['mean_naive_precision']:.3f} |"
    )


def budget_row(row: dict[str, Any]) -> str:
    """Render one precision-budget row."""

    stage2_f1 = f1(row["stage2_precision"], row["stage2_recall"])
    wsts_f1 = f1(row["wsts_precision"], row["wsts_recall"])
    return (
        f"| {int(row['alerts_per_day'])} | {row['stage2_precision']:.4f} | "
        f"{row['stage2_recall']:.4f} | {stage2_f1:.4f} | "
        f"{row['wsts_precision']:.4f} | {row['wsts_recall']:.4f} | {wsts_f1:.4f} |"
    )


def f1(precision: float, recall: float) -> float:
    """Compute F1 from precision and recall."""

    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def read_json(path: Path) -> Any:
    """Read JSON from disk."""

    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
