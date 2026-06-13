"""Evaluate rolling 3-day discriminative alerts and alert-fatigue reduction."""

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def main() -> None:
    """Score candidate rows and evaluate a rolling alert protocol."""

    parser = argparse.ArgumentParser(description="Evaluate rolling Stage 2 alerts.")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--candidate-csv", type=Path)
    parser.add_argument(
        "--scored-candidate-csv",
        type=Path,
        help="Candidate CSV that already contains an ensembled model_score column.",
    )
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--radii-km", type=float, nargs="+", default=[2.0, 3.0, 4.0])
    parser.add_argument("--naive-radius-km", type=float, default=4.0)
    parser.add_argument(
        "--target-artifacts-dir",
        type=Path,
        help="Stage 1 artifact directory used for the fixed Day-5 to Day-8 baseline.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stage2/rolling_alerts"))
    args = parser.parse_args()

    if not args.radii_km:
        raise ValueError("--radii-km must contain at least one value.")
    candidates, source = load_scored_candidates(args)

    event_day_rows = evaluate_rolling_alerts(
        candidates,
        threshold=args.threshold,
        radii_km=tuple(args.radii_km),
        naive_radius_km=args.naive_radius_km,
        target_artifacts_dir=args.target_artifacts_dir,
    )
    summary = summarize(event_day_rows)
    summary.update(
        {
            **source,
            "model_name": args.model_name,
            "threshold": args.threshold,
            "radii_km": args.radii_km,
            "naive_radius_km": args.naive_radius_km,
            "fixed_baseline": (
                "initial-current-fire 4 km alerts evaluated on final target day"
                if args.target_artifacts_dir is not None
                else None
            ),
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / f"{args.model_name}_rolling_event_days.csv"
    summary_path = args.output_dir / f"{args.model_name}_rolling_summary.json"
    pd.DataFrame(event_day_rows).to_csv(rows_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"saved rolling event-day rows to {rows_path}")
    print(f"saved rolling summary to {summary_path}")


def evaluate_rolling_alerts(
    candidates: pd.DataFrame,
    threshold: float,
    radii_km: tuple[float, ...],
    naive_radius_km: float,
    target_artifacts_dir: Path | None = None,
) -> list[dict[str, int | float | str]]:
    """Evaluate rolling windows within each fire event."""

    rows: list[dict[str, int | float | str]] = []
    future = candidates[candidates["current_fire_at_candidate"] == 0].copy()
    for event_id, event in future.groupby("event_id", sort=True):
        dates = sorted(str(date) for date in event["target_date"].unique())
        if len(dates) < len(radii_km):
            continue
        for start_index in range(0, len(dates) - len(radii_km) + 1):
            previous_alerts: set[tuple[int, int]] = set()
            previous_2km_alerts: set[tuple[int, int]] = set()
            window_dates = dates[start_index : start_index + len(radii_km)]
            window_id = f"{event_id}_{window_dates[0]}_{window_dates[-1]}"
            initial_group = event[event["target_date"].astype(str) == window_dates[0]]
            fixed_naive_candidates = initial_group[
                initial_group["distance_to_current_fire_km"] <= naive_radius_km
            ]
            for step, (target_date, radius_km) in enumerate(
                zip(window_dates, radii_km, strict=True)
            ):
                group = event[event["target_date"].astype(str) == target_date]
                naive_candidates = group[group["distance_to_current_fire_km"] <= naive_radius_km]
                model_candidates = group[group["distance_to_current_fire_km"] <= radius_km]
                alerts = model_candidates[model_candidates["model_score"] >= threshold]
                alert_locations = location_set(alerts)
                current_2km_alerts = location_set(
                    alerts[alerts["distance_to_current_fire_km"] <= 2.0]
                )

                rows.append(
                    event_day_metrics(
                        event_id=str(event_id),
                        target_date=str(target_date),
                        window_id=window_id,
                        step=step,
                        radius_km=radius_km,
                        naive_radius_km=naive_radius_km,
                        group=group,
                        model_candidates=model_candidates,
                        naive_candidates=naive_candidates,
                        alerts=alerts,
                        previous_alerts=previous_alerts,
                        previous_2km_alerts=previous_2km_alerts,
                        fixed_naive_candidates=fixed_naive_candidates,
                        final_target=(
                            load_target_artifact(
                                target_artifacts_dir,
                                str(event_id),
                                window_dates[-1],
                            )
                            if target_artifacts_dir is not None and step == len(radii_km) - 1
                            else None
                        ),
                    )
                )
                previous_alerts = alert_locations
                previous_2km_alerts = current_2km_alerts
    return rows


def event_day_metrics(
    event_id: str,
    target_date: str,
    window_id: str,
    step: int,
    radius_km: float,
    naive_radius_km: float,
    group: pd.DataFrame,
    model_candidates: pd.DataFrame,
    naive_candidates: pd.DataFrame,
    alerts: pd.DataFrame,
    previous_alerts: set[tuple[int, int]],
    previous_2km_alerts: set[tuple[int, int]],
    fixed_naive_candidates: pd.DataFrame,
    final_target: np.ndarray[Any, np.dtype[np.uint8]] | None,
) -> dict[str, int | float | str]:
    """Compute one rolling event-day metric row."""

    alert_locations = location_set(alerts)
    true_alerts = int(alerts["label"].sum())
    naive_true_alerts = int(naive_candidates["label"].sum())
    dropped_previous_alerts = len(previous_alerts - alert_locations)
    dropped_previous_2km_alerts = len(previous_2km_alerts - alert_locations)
    fixed_true_alerts = (
        target_positive_count(fixed_naive_candidates, final_target)
        if final_target is not None
        else None
    )
    fixed_alert_count = len(fixed_naive_candidates) if final_target is not None else None
    final_model_alert_count = len(alerts) if final_target is not None else None
    final_model_true_alerts = true_alerts if final_target is not None else None

    return {
        "event_id": event_id,
        "target_date": target_date,
        "window_id": window_id,
        "rolling_step": step,
        "radius_km": radius_km,
        "naive_radius_km": naive_radius_km,
        "candidate_count": int(len(group)),
        "model_candidate_count": int(len(model_candidates)),
        "naive_candidate_count": int(len(naive_candidates)),
        "model_alert_count": int(len(alerts)),
        "naive_alert_count": int(len(naive_candidates)),
        "model_true_alert_count": true_alerts,
        "naive_true_alert_count": naive_true_alerts,
        "model_precision": safe_div(true_alerts, len(alerts)),
        "naive_precision": safe_div(naive_true_alerts, len(naive_candidates)),
        "alert_reduction_vs_naive": 1.0 - safe_div(len(alerts), len(naive_candidates)),
        "dropped_previous_alerts": dropped_previous_alerts,
        "dropped_previous_2km_alerts": dropped_previous_2km_alerts,
        "ap": safe_ap(model_candidates),
        "auc": safe_auc(model_candidates),
        "fixed_3day_naive_alert_count": fixed_alert_count,
        "fixed_3day_naive_true_alert_count": fixed_true_alerts,
        "fixed_3day_naive_precision": (
            safe_div(fixed_true_alerts, fixed_alert_count)
            if fixed_true_alerts is not None and fixed_alert_count is not None
            else float("nan")
        ),
        "final_day_model_alert_count": final_model_alert_count,
        "final_day_model_true_alert_count": final_model_true_alerts,
        "final_day_model_precision": (
            safe_div(final_model_true_alerts, final_model_alert_count)
            if final_model_true_alerts is not None and final_model_alert_count is not None
            else float("nan")
        ),
        "final_day_alert_reduction_vs_fixed_naive": (
            1.0 - safe_div(len(alerts), fixed_alert_count)
            if fixed_alert_count is not None
            else float("nan")
        ),
    }


def summarize(rows: list[dict[str, int | float | str]]) -> dict[str, float | int]:
    """Summarize rolling event-day metrics."""

    if not rows:
        return {"event_day_count": 0}
    frame = pd.DataFrame(rows)
    numeric_columns = [
        "model_alert_count",
        "naive_alert_count",
        "model_precision",
        "naive_precision",
        "alert_reduction_vs_naive",
        "dropped_previous_alerts",
        "dropped_previous_2km_alerts",
        "ap",
        "auc",
        "fixed_3day_naive_alert_count",
        "fixed_3day_naive_precision",
        "final_day_model_alert_count",
        "final_day_model_precision",
        "final_day_alert_reduction_vs_fixed_naive",
    ]
    summary: dict[str, float | int] = {
        "event_day_count": int(len(frame)),
        "window_count": int(frame["window_id"].nunique()),
    }
    for column in numeric_columns:
        values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        summary[f"mean_{column}"] = float(values.mean())
    return summary


def location_set(frame: pd.DataFrame) -> set[tuple[int, int]]:
    """Return row/column locations for candidate rows."""

    return set(zip(frame["row"].astype(int), frame["col"].astype(int), strict=True))


def safe_div(numerator: int | float, denominator: int | float) -> float:
    """Divide with zero guard."""

    return float(numerator / denominator) if denominator else 0.0


def safe_ap(frame: pd.DataFrame) -> float:
    """Average precision for an event-day, or NaN when undefined."""

    if frame.empty or frame["label"].nunique() < 2:
        return float("nan")
    return float(average_precision_score(frame["label"].astype(int), frame["model_score"]))


def safe_auc(frame: pd.DataFrame) -> float:
    """ROC AUC for an event-day, or NaN when undefined."""

    if frame.empty or frame["label"].nunique() < 2:
        return float("nan")
    return float(roc_auc_score(frame["label"].astype(int), frame["model_score"]))


def load_scored_candidates(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, str | None]]:
    """Load pre-scored candidates or score them with one legacy model."""

    if args.scored_candidate_csv is not None:
        if args.model is not None or args.candidate_csv is not None:
            raise ValueError(
                "Use either --scored-candidate-csv or --model with --candidate-csv, not both."
            )
        candidates = pd.read_csv(args.scored_candidate_csv)
        if "model_score" not in candidates:
            raise ValueError("--scored-candidate-csv must contain a model_score column.")
        return candidates, {
            "model": None,
            "candidate_csv": str(args.scored_candidate_csv),
        }
    if args.model is None or args.candidate_csv is None:
        raise ValueError("Provide --scored-candidate-csv or both --model and --candidate-csv.")

    bundle = joblib.load(args.model)
    feature_columns = list(bundle["feature_columns"])
    candidates = pd.read_csv(args.candidate_csv)
    features = candidates[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    candidates = candidates.copy()
    candidates["model_score"] = bundle["model"].predict_proba(features)[:, 1]
    return candidates, {
        "model": str(args.model),
        "candidate_csv": str(args.candidate_csv),
    }


def load_target_artifact(directory: Path, event_id: str, target_date: str) -> np.ndarray:
    """Load the full target raster for a fixed three-day baseline."""

    path = directory / event_id / f"{target_date}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing target artifact for fixed baseline: {path}")
    with np.load(path, allow_pickle=False) as artifact:
        return np.asarray(artifact["target"], dtype=np.uint8)


def target_positive_count(candidates: pd.DataFrame, target: np.ndarray) -> int:
    """Count candidate locations that are positive in a supplied target raster."""

    rows = candidates["row"].to_numpy(dtype=int)
    cols = candidates["col"].to_numpy(dtype=int)
    return int((target[rows, cols] > 0).sum())


if __name__ == "__main__":
    main()
