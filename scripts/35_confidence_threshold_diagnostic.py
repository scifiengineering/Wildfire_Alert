"""Evaluate alert performance at fixed confidence thresholds."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_THRESHOLDS = (
    0.95,
    0.94,
    0.93,
    0.92,
    0.91,
    0.90,
    0.875,
    0.85,
    0.825,
    0.80,
    0.775,
    0.75,
    0.70,
    0.65,
    0.60,
    0.55,
    0.50,
)
REQUIRED_COLUMNS = (
    "event_id",
    "target_date",
    "label",
    "model_score",
    "stage1_probability",
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Measure alerts/day, precision, recall, and F1 at fixed score thresholds."
    )
    parser.add_argument(
        "--scored-candidate-csv",
        type=Path,
        required=True,
        help="2020 scored candidate CSV with model_score and stage1_probability columns.",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=list(DEFAULT_THRESHOLDS),
        help="Confidence thresholds to evaluate.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Path where the confidence-threshold JSON should be written.",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        required=True,
        help="Path where the Markdown report should be written.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the confidence-threshold diagnostic."""

    args = parse_args()
    report = evaluate(args.scored_candidate_csv, args.thresholds)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.output_report.write_text(render_report(report), encoding="utf-8")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_report}")


def evaluate(scored_candidate_csv: Path, thresholds: list[float]) -> dict[str, Any]:
    """Evaluate Stage 2 and WSTS at fixed thresholds."""

    frame = pd.read_csv(scored_candidate_csv, usecols=list(REQUIRED_COLUMNS))
    labels = frame["label"].astype(int)
    total_rows = int(len(frame))
    total_positives = int(labels.sum())
    event_days = int(frame[["event_id", "target_date"]].drop_duplicates().shape[0])
    base_rate = total_positives / total_rows if total_rows else 0.0

    rows = []
    for threshold in sorted(set(thresholds), reverse=True):
        rows.append(
            {
                "threshold": threshold,
                "stage2": threshold_metrics(
                    labels=labels,
                    scores=frame["model_score"],
                    threshold=threshold,
                    event_days=event_days,
                    total_positives=total_positives,
                    base_rate=base_rate,
                ),
                "wsts": threshold_metrics(
                    labels=labels,
                    scores=frame["stage1_probability"],
                    threshold=threshold,
                    event_days=event_days,
                    total_positives=total_positives,
                    base_rate=base_rate,
                ),
            }
        )

    return {
        "scored_candidate_csv": str(scored_candidate_csv),
        "rows": total_rows,
        "positives": total_positives,
        "candidate_positive_rate": base_rate,
        "event_days": event_days,
        "thresholds": rows,
    }


def threshold_metrics(
    *,
    labels: pd.Series,
    scores: pd.Series,
    threshold: float,
    event_days: int,
    total_positives: int,
    base_rate: float,
) -> dict[str, float | int]:
    """Calculate alert metrics for a score threshold."""

    selected = scores >= threshold
    alerts = int(selected.sum())
    true_positives = int(labels[selected].sum())
    precision = true_positives / alerts if alerts else 0.0
    recall = true_positives / total_positives if total_positives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    lift = precision / base_rate if base_rate else 0.0
    alerts_per_day = alerts / event_days if event_days else 0.0
    return {
        "alerts": alerts,
        "alerts_per_day": alerts_per_day,
        "true_positive_alerts": true_positives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "lift_over_base": lift,
    }


def render_report(report: dict[str, Any]) -> str:
    """Render the confidence-threshold report."""

    high_thresholds = [row for row in report["thresholds"] if row["threshold"] >= 0.85]
    broader_thresholds = [row for row in report["thresholds"] if row["threshold"] < 0.85]
    lines = [
        "# Confidence-Threshold Operating Points",
        "",
        "This report uses fixed confidence thresholds as the operating point. For each threshold, candidate locations with scores greater than or equal to that threshold are alerted, then alerts/day, precision, recall, F1, and true-positive alerts are measured.",
        "",
        f"- Candidate rows: {report['rows']:,}",
        f"- Positive candidate rows: {report['positives']:,}",
        f"- Candidate positive rate: {report['candidate_positive_rate']:.4f}",
        f"- Event-days: {report['event_days']:,}",
        "",
        "## Stage 2 High-Confidence Region",
        "",
        "| Confidence threshold | Alerts/day | Precision | Recall | F1 | True-positive alerts | Lift over base |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(stage2_row(row) for row in high_thresholds)
    lines.extend(
        [
            "",
            "## Stage 2 Broader Threshold Range",
            "",
            "| Confidence threshold | Alerts/day | Precision | Recall | F1 | True-positive alerts | Lift over base |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    lines.extend(stage2_row(row) for row in broader_thresholds)
    lines.extend(
        [
            "",
            "## WSTS Score Threshold Context",
            "",
            "This table applies the same numeric thresholds to the WSTS/Stage-1 probability score. It is useful context, but it should not be treated as a calibrated confidence equality between models.",
            "",
            "| Score threshold | Alerts/day | Precision | Recall | F1 | True-positive alerts | Lift over base |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    lines.extend(wsts_row(row) for row in report["thresholds"])
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This confidence-threshold view answers the professor's possible alternative framing directly. Instead of choosing 5, 10, or 25 alerts/day first, we choose a confidence threshold first and observe how many alerts/day the system emits.",
            "",
            "The pattern is consistent with the alert-budget view: stricter confidence thresholds produce fewer alerts/day and higher precision, while relaxed thresholds capture more true positives and increase recall at the cost of precision.",
            "",
        ]
    )
    return "\n".join(lines)


def stage2_row(row: dict[str, Any]) -> str:
    """Render one Stage 2 threshold row."""

    metrics = row["stage2"]
    return metric_row(row["threshold"], metrics)


def wsts_row(row: dict[str, Any]) -> str:
    """Render one WSTS threshold row."""

    metrics = row["wsts"]
    return metric_row(row["threshold"], metrics)


def metric_row(threshold: float, metrics: dict[str, float | int]) -> str:
    """Render one metrics row."""

    return (
        f"| {threshold:.3f} | {metrics['alerts_per_day']:.2f} | "
        f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | "
        f"{metrics['f1']:.4f} | {int(metrics['true_positive_alerts']):,} | "
        f"{metrics['lift_over_base']:.1f}x |"
    )


if __name__ == "__main__":
    main()
