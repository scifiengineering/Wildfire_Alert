"""Run precision-by-alert-budget diagnostics for full-2020 Stage 2 outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


DEFAULT_BUDGETS = (5, 10, 25, 50, 75, 90, 114, 150, 250)
REQUIRED_COLUMNS = (
    "event_id",
    "target_date",
    "label",
    "model_score",
    "stage1_probability",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure how Stage 2 and WSTS precision change as the global "
            "alert budget changes."
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        nargs=2,
        metavar=("NAME", "SCORED_CANDIDATE_CSV"),
        required=True,
        help="Named scored-candidate CSV to evaluate. Repeat for multiple runs.",
    )
    parser.add_argument(
        "--alerts-per-day",
        type=float,
        nargs="+",
        default=list(DEFAULT_BUDGETS),
        help="Global alert budgets to evaluate, expressed as alerts per event-day.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Path where the diagnostic JSON should be written.",
    )
    return parser.parse_args()


def top_k_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    k: int,
    total_positives: int,
    base_rate: float,
) -> dict[str, float | int | None]:
    if k <= 0:
        raise ValueError("k must be positive")

    k = min(k, len(labels))
    top_indices = np.argpartition(scores, -k)[-k:]
    true_positives = int(labels[top_indices].sum())
    precision = true_positives / k
    recall = true_positives / total_positives if total_positives else 0.0
    lift = precision / base_rate if base_rate else None

    return {
        "alerts": k,
        "precision": precision,
        "recall": recall,
        "true_positive_alerts": true_positives,
        "lift_over_base": lift,
    }


def evaluate_run(
    name: str,
    scored_candidate_csv: Path,
    alerts_per_day: list[float],
) -> dict[str, object]:
    if not scored_candidate_csv.exists():
        raise FileNotFoundError(scored_candidate_csv)

    df = pd.read_csv(scored_candidate_csv, usecols=list(REQUIRED_COLUMNS))
    labels = df["label"].to_numpy(dtype=np.int8)
    stage2_scores = df["model_score"].to_numpy(dtype=np.float64)
    wsts_scores = df["stage1_probability"].to_numpy(dtype=np.float64)

    total_rows = int(len(df))
    total_positives = int(labels.sum())
    event_days = int(df[["event_id", "target_date"]].drop_duplicates().shape[0])
    base_rate = total_positives / total_rows if total_rows else 0.0

    budgets: list[dict[str, object]] = []
    for budget in alerts_per_day:
        alerts = int(round(budget * event_days))
        stage2 = top_k_metrics(
            labels,
            stage2_scores,
            k=alerts,
            total_positives=total_positives,
            base_rate=base_rate,
        )
        wsts = top_k_metrics(
            labels,
            wsts_scores,
            k=alerts,
            total_positives=total_positives,
            base_rate=base_rate,
        )
        budgets.append(
            {
                "alerts_per_day": budget,
                "alerts": alerts,
                "stage2_precision": stage2["precision"],
                "stage2_recall": stage2["recall"],
                "stage2_tp": stage2["true_positive_alerts"],
                "stage2_lift_over_base": stage2["lift_over_base"],
                "wsts_precision": wsts["precision"],
                "wsts_recall": wsts["recall"],
                "wsts_tp": wsts["true_positive_alerts"],
                "wsts_lift_over_base": wsts["lift_over_base"],
            }
        )

    return {
        "name": name,
        "scored_candidate_csv": str(scored_candidate_csv),
        "rows": total_rows,
        "positives": total_positives,
        "candidate_positive_rate": base_rate,
        "event_days": event_days,
        "stage2_ap": float(average_precision_score(labels, stage2_scores)),
        "wsts_ap": float(average_precision_score(labels, wsts_scores)),
        "stage2_auc": float(roc_auc_score(labels, stage2_scores)),
        "wsts_auc": float(roc_auc_score(labels, wsts_scores)),
        "budgets": budgets,
    }


def main() -> None:
    args = parse_args()
    results = {
        name: evaluate_run(name, Path(csv_path), args.alerts_per_day)
        for name, csv_path in args.run
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
