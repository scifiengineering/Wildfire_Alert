"""Evaluate candidate eligibility filters at fixed recall targets."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = (
    "event_id",
    "target_date",
    "label",
    "model_score",
    "stage1_probability",
    "stage1_top_mask",
    "current_fire_at_candidate",
    "distance_to_current_fire_km",
    "wind_alignment",
    "forecast_wind_alignment",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply simple candidate eligibility filters and report precision at "
            "fixed recall targets."
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
        "--recall-targets",
        type=float,
        nargs="+",
        default=[0.10, 0.15, 0.20, 0.25],
        help="Recall targets for precision-at-recall reporting.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Path where detailed JSON results should be written.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Path where a flat summary CSV should be written.",
    )
    return parser.parse_args()


def filter_specs() -> dict[str, Callable[[pd.DataFrame], pd.Series]]:
    return {
        "base_future_distance_le_4km": lambda df: base_mask(df),
        "distance_le_3km": lambda df: base_mask(df) & (df["distance_to_current_fire_km"] <= 3.0),
        "distance_le_2km": lambda df: base_mask(df) & (df["distance_to_current_fire_km"] <= 2.0),
        "stage1_probability_ge_0_10": lambda df: base_mask(df)
        & (df["stage1_probability"] >= 0.10),
        "stage1_probability_ge_0_20": lambda df: base_mask(df)
        & (df["stage1_probability"] >= 0.20),
        "stage1_probability_ge_0_30": lambda df: base_mask(df)
        & (df["stage1_probability"] >= 0.30),
        "stage1_top_mask": lambda df: base_mask(df) & (df["stage1_top_mask"] > 0),
        "wind_alignment_ge_0": lambda df: base_mask(df) & (df["wind_alignment"] >= 0.0),
        "forecast_wind_alignment_ge_0": lambda df: base_mask(df)
        & (df["forecast_wind_alignment"] >= 0.0),
        "distance_le_3km_and_forecast_wind_ge_0": lambda df: base_mask(df)
        & (df["distance_to_current_fire_km"] <= 3.0)
        & (df["forecast_wind_alignment"] >= 0.0),
        "distance_le_3km_and_stage1_probability_ge_0_10": lambda df: base_mask(df)
        & (df["distance_to_current_fire_km"] <= 3.0)
        & (df["stage1_probability"] >= 0.10),
    }


def base_mask(df: pd.DataFrame) -> pd.Series:
    return (df["current_fire_at_candidate"] == 0) & (df["distance_to_current_fire_km"] <= 4.0)


def evaluate_run(
    name: str,
    path: Path,
    recall_targets: list[float],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path, usecols=list(REQUIRED_COLUMNS))
    total_positives = int(df.loc[base_mask(df), "label"].sum())
    records: list[dict[str, object]] = []
    detailed_filters: list[dict[str, object]] = []

    for filter_name, filter_fn in filter_specs().items():
        eligible = df[filter_fn(df)].copy()
        base_event_days = int(df.loc[base_mask(df), ["event_id", "target_date"]].drop_duplicates().shape[0])
        metrics = evaluate_filter(
            eligible,
            recall_targets=recall_targets,
            base_positive_count=total_positives,
            base_event_days=base_event_days,
        )
        detailed_filters.append(
            {
                "filter": filter_name,
                "eligible_candidates": int(len(eligible)),
                "eligible_positives": int(eligible["label"].sum()),
                "eligible_positive_rate": safe_div(float(eligible["label"].sum()), len(eligible)),
                "max_recall_against_base": safe_div(
                    float(eligible["label"].sum()), total_positives
                ),
                "targets": metrics,
            }
        )
        for metric in metrics:
            records.append(
                {
                    "run": name,
                    "filter": filter_name,
                    "eligible_candidates": int(len(eligible)),
                    "eligible_positives": int(eligible["label"].sum()),
                    "eligible_positive_rate": safe_div(
                        float(eligible["label"].sum()), len(eligible)
                    ),
                    "max_recall_against_base": safe_div(
                        float(eligible["label"].sum()), total_positives
                    ),
                    **metric,
                }
            )

    detail = {
        "run": name,
        "scored_candidate_csv": str(path),
        "base_candidates": int(base_mask(df).sum()),
        "base_positives": total_positives,
        "base_positive_rate": safe_div(float(total_positives), int(base_mask(df).sum())),
        "base_event_days": base_event_days,
        "filters": detailed_filters,
    }
    return detail, records


def evaluate_filter(
    eligible: pd.DataFrame,
    *,
    recall_targets: list[float],
    base_positive_count: int,
    base_event_days: int,
) -> list[dict[str, object]]:
    if eligible.empty or base_positive_count == 0:
        return [
            empty_target_record(recall_target, "empty_filter")
            for recall_target in recall_targets
        ]

    ranked = eligible.sort_values("model_score", ascending=False).reset_index(drop=True)
    labels = ranked["label"].to_numpy(dtype=np.int64)
    cumulative_tp = np.cumsum(labels)
    records: list[dict[str, object]] = []
    for recall_target in recall_targets:
        target_tp = int(np.ceil(recall_target * base_positive_count))
        if target_tp <= 0:
            records.append(empty_target_record(recall_target, "zero_target"))
            continue
        if int(cumulative_tp[-1]) < target_tp:
            records.append(empty_target_record(recall_target, "target_unreachable"))
            continue
        index = int(np.searchsorted(cumulative_tp, target_tp, side="left"))
        alert_count = index + 1
        true_positive_alerts = int(cumulative_tp[index])
        precision = true_positive_alerts / alert_count
        recall = true_positive_alerts / base_positive_count
        eligible_event_day_count = ranked[["event_id", "target_date"]].drop_duplicates().shape[0]
        records.append(
            {
                "recall_target": recall_target,
                "status": "ok",
                "score_threshold": float(ranked.loc[index, "model_score"]),
                "alerts": int(alert_count),
                "alerts_per_base_event_day": safe_div(alert_count, base_event_days),
                "alerts_per_eligible_event_day": safe_div(alert_count, eligible_event_day_count),
                "true_positive_alerts": true_positive_alerts,
                "precision": precision,
                "recall_against_base": recall,
                "base_event_days": int(base_event_days),
                "eligible_event_days": int(eligible_event_day_count),
            }
        )
    return records


def empty_target_record(recall_target: float, status: str) -> dict[str, object]:
    return {
        "recall_target": recall_target,
        "status": status,
        "score_threshold": None,
        "alerts": 0,
        "alerts_per_base_event_day": 0.0,
        "alerts_per_eligible_event_day": 0.0,
        "true_positive_alerts": 0,
        "precision": None,
        "recall_against_base": 0.0,
        "base_event_days": 0,
        "eligible_event_days": 0,
    }


def safe_div(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def main() -> None:
    args = parse_args()
    details: dict[str, object] = {}
    records: list[dict[str, object]] = []
    for name, csv_path in args.run:
        detail, rows = evaluate_run(name, Path(csv_path), args.recall_targets)
        details[name] = detail
        records.extend(rows)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
