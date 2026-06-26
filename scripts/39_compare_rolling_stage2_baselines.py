"""Compare rolling Stage 2 alerts with WSTS and distance baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

NON_FEATURE_COLUMNS = {
    "fold",
    "split",
    "event_id",
    "target_date",
    "row",
    "col",
    "label",
}


def main() -> None:
    """Run matched-volume rolling comparisons against WSTS and distance."""

    parser = argparse.ArgumentParser(
        description=(
            "Compare Stage 2 rolling alerts with WSTS and distance-only baselines. "
            "Each baseline receives one global threshold calibrated to match the "
            "Stage 2 alert volume over the same rolling alert universe."
        )
    )
    parser.add_argument("--scored-candidate-csv", type=Path)
    parser.add_argument("--candidate-csv", type=Path)
    parser.add_argument("--models", type=Path, nargs="+")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--stage2-threshold", type=float, required=True)
    parser.add_argument("--radii-km", type=float, nargs="+", default=[2.0, 3.0, 4.0])
    parser.add_argument("--stage2-score-column", default="model_score")
    parser.add_argument("--wsts-score-column", default="stage1_probability")
    parser.add_argument("--distance-column", default="distance_to_current_fire_km")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/stage2/rolling_baseline_comparisons"),
    )
    args = parser.parse_args()

    candidates, source = load_candidates(args)
    rows = compare_rolling_baselines(
        candidates=candidates,
        radii_km=tuple(args.radii_km),
        stage2_threshold=args.stage2_threshold,
        stage2_score_column=args.stage2_score_column,
        wsts_score_column=args.wsts_score_column,
        distance_column=args.distance_column,
    )
    summary = summarize(rows)
    summary.update(
        {
            **source,
            "model_name": args.model_name,
            "stage2_threshold": args.stage2_threshold,
            "radii_km": args.radii_km,
            "stage2_score_column": args.stage2_score_column,
            "wsts_score_column": args.wsts_score_column,
            "distance_score": f"-{args.distance_column}",
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / f"{args.model_name}_rolling_baseline_event_days.csv"
    summary_path = args.output_dir / f"{args.model_name}_rolling_baseline_summary.json"
    pd.DataFrame(rows).to_csv(rows_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"saved rolling baseline event-day rows to {rows_path}")
    print(f"saved rolling baseline summary to {summary_path}")


def load_candidates(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load pre-scored candidates or score candidates with one or more models."""

    if args.scored_candidate_csv is not None:
        if args.candidate_csv is not None or args.models is not None:
            raise ValueError("Use either --scored-candidate-csv or --candidate-csv with --models.")
        reject_lfs_pointer(args.scored_candidate_csv)
        return pd.read_csv(args.scored_candidate_csv), {
            "scored_candidate_csv": str(args.scored_candidate_csv),
            "candidate_csv": None,
            "models": None,
        }

    if args.candidate_csv is None or not args.models:
        raise ValueError("Provide --scored-candidate-csv or both --candidate-csv and --models.")
    reject_lfs_pointer(args.candidate_csv)
    candidates = pd.read_csv(args.candidate_csv)
    scored = add_ensemble_scores(candidates, args.models, args.stage2_score_column)
    return scored, {
        "scored_candidate_csv": None,
        "candidate_csv": str(args.candidate_csv),
        "models": [str(path) for path in args.models],
    }


def reject_lfs_pointer(path: Path) -> None:
    """Fail early when a CSV path is only a Git LFS pointer."""

    with path.open("rb") as handle:
        head = handle.read(128)
    if head.startswith(b"version https://git-lfs.github.com/spec"):
        raise RuntimeError(
            f"{path} is a Git LFS pointer, not a materialized CSV. "
            "Regenerate the candidate/scored-candidate CSV or pull LFS data before running."
        )


def add_ensemble_scores(
    candidates: pd.DataFrame,
    model_paths: list[Path],
    score_column: str,
) -> pd.DataFrame:
    """Add the mean predicted probability from one or more saved Stage 2 models."""

    bundles = [joblib.load(path) for path in model_paths]
    feature_columns = list(bundles[0]["feature_columns"])
    for path, bundle in zip(model_paths, bundles, strict=True):
        if list(bundle["feature_columns"]) != feature_columns:
            raise ValueError(f"Feature columns differ in {path}.")
    features = candidates[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    model_scores = [bundle["model"].predict_proba(features)[:, 1] for bundle in bundles]
    scored = candidates.copy()
    scored[score_column] = np.mean(np.stack(model_scores), axis=0)
    return scored


def compare_rolling_baselines(
    candidates: pd.DataFrame,
    radii_km: tuple[float, ...],
    stage2_threshold: float,
    stage2_score_column: str,
    wsts_score_column: str,
    distance_column: str,
) -> list[dict[str, int | float | str]]:
    """Create rolling event-day rows with matched WSTS and distance baselines."""

    required = {
        "event_id",
        "target_date",
        "label",
        "current_fire_at_candidate",
        distance_column,
        stage2_score_column,
        wsts_score_column,
    }
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"Candidate table is missing required columns: {missing}")

    future = candidates[candidates["current_fire_at_candidate"] == 0].copy()
    future["_distance_score"] = -future[distance_column].astype(float)
    frames = collect_event_day_frames(
        future=future,
        radii_km=radii_km,
        stage2_threshold=stage2_threshold,
        stage2_score_column=stage2_score_column,
        distance_column=distance_column,
    )
    target_alert_count = sum(
        len(item["stage2_alerts"])
        for item in frames
        if isinstance(item["stage2_alerts"], pd.DataFrame)
    )
    wsts_selection = select_global_topk(frames, wsts_score_column, target_alert_count)
    distance_selection = select_global_topk(frames, "_distance_score", target_alert_count)

    rows: list[dict[str, int | float | str]] = []
    for frame_index, item in enumerate(frames):
        alert_universe = checked_frame(item["alert_universe"], "alert_universe")
        stage2_alerts = checked_frame(item["stage2_alerts"], "stage2_alerts")
        wsts_alerts = alert_universe.iloc[wsts_selection["positions"].get(frame_index, [])]
        distance_alerts = alert_universe.iloc[
            distance_selection["positions"].get(frame_index, [])
        ]
        rows.append(
            event_day_row(
                event_id=str(item["event_id"]),
                target_date=str(item["target_date"]),
                window_id=str(item["window_id"]),
                step=int(item["rolling_step"]),
                radius_km=float(item["radius_km"]),
                alert_universe=alert_universe,
                stage2_alerts=stage2_alerts,
                wsts_alerts=wsts_alerts,
                distance_alerts=distance_alerts,
                stage2_score_column=stage2_score_column,
                wsts_score_column=wsts_score_column,
                distance_score_column="_distance_score",
                wsts_threshold=wsts_selection["min_selected_score"],
                distance_threshold=distance_selection["min_selected_score"],
            )
        )
    return rows


def collect_event_day_frames(
    future: pd.DataFrame,
    radii_km: tuple[float, ...],
    stage2_threshold: float,
    stage2_score_column: str,
    distance_column: str,
) -> list[dict[str, object]]:
    """Collect rolling alert universes and Stage 2 alert sets."""

    frames: list[dict[str, object]] = []
    for event_id, event in future.groupby("event_id", sort=True):
        dates = sorted(str(date) for date in event["target_date"].unique())
        if len(dates) < len(radii_km):
            continue
        for start_index in range(0, len(dates) - len(radii_km) + 1):
            window_dates = dates[start_index : start_index + len(radii_km)]
            window_id = f"{event_id}_{window_dates[0]}_{window_dates[-1]}"
            for step, (target_date, radius_km) in enumerate(
                zip(window_dates, radii_km, strict=True)
            ):
                day = event[event["target_date"].astype(str) == target_date]
                alert_universe = day[day[distance_column] <= radius_km].copy()
                stage2_alerts = alert_universe[
                    alert_universe[stage2_score_column] >= stage2_threshold
                ]
                frames.append(
                    {
                        "event_id": str(event_id),
                        "target_date": str(target_date),
                        "window_id": window_id,
                        "rolling_step": step,
                        "radius_km": radius_km,
                        "alert_universe": alert_universe,
                        "stage2_alerts": stage2_alerts,
                    }
                )
    return frames


def select_global_topk(
    event_day_frames: list[dict[str, object]],
    score_column: str,
    target_alert_count: int,
) -> dict[str, Any]:
    """Select exactly the top-k rolling alert occurrences for one baseline score."""

    scored_positions: list[tuple[float, int, int]] = []
    for frame_index, item in enumerate(event_day_frames):
        alert_universe = checked_frame(item["alert_universe"], "alert_universe")
        scores = pd.to_numeric(alert_universe[score_column], errors="coerce").to_numpy()
        for row_position, score in enumerate(scores):
            if np.isfinite(score):
                scored_positions.append((float(score), frame_index, row_position))

    if target_alert_count <= 0 or not scored_positions:
        return {"positions": {}, "min_selected_score": float("inf")}

    scored_positions.sort(key=lambda item: (-item[0], item[1], item[2]))
    selected = scored_positions[: min(target_alert_count, len(scored_positions))]
    positions: dict[int, list[int]] = {}
    for score, frame_index, row_position in selected:
        positions.setdefault(frame_index, []).append(row_position)
    return {
        "positions": positions,
        "min_selected_score": float(selected[-1][0]),
    }


def checked_frame(value: object, name: str) -> pd.DataFrame:
    """Type-check a stored frame before metric computation."""

    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"{name} must be a DataFrame.")
    return value


def event_day_row(
    event_id: str,
    target_date: str,
    window_id: str,
    step: int,
    radius_km: float,
    alert_universe: pd.DataFrame,
    stage2_alerts: pd.DataFrame,
    wsts_alerts: pd.DataFrame,
    distance_alerts: pd.DataFrame,
    stage2_score_column: str,
    wsts_score_column: str,
    distance_score_column: str,
    wsts_threshold: float,
    distance_threshold: float,
) -> dict[str, int | float | str]:
    """Summarize one rolling event-day for Stage 2 and both baselines."""

    positives = int(alert_universe["label"].sum())
    stage2_tp = int(stage2_alerts["label"].sum())
    wsts_tp = int(wsts_alerts["label"].sum())
    distance_tp = int(distance_alerts["label"].sum())
    return {
        "event_id": event_id,
        "target_date": target_date,
        "window_id": window_id,
        "rolling_step": step,
        "radius_km": radius_km,
        "candidate_count": int(len(alert_universe)),
        "positive_count": positives,
        "stage2_alert_count": int(len(stage2_alerts)),
        "stage2_true_positive_alert_count": stage2_tp,
        "stage2_precision": safe_div(stage2_tp, len(stage2_alerts)),
        "stage2_recall": safe_div(stage2_tp, positives),
        "wsts_alert_count": int(len(wsts_alerts)),
        "wsts_true_positive_alert_count": wsts_tp,
        "wsts_precision": safe_div(wsts_tp, len(wsts_alerts)),
        "wsts_recall": safe_div(wsts_tp, positives),
        "distance_alert_count": int(len(distance_alerts)),
        "distance_true_positive_alert_count": distance_tp,
        "distance_precision": safe_div(distance_tp, len(distance_alerts)),
        "distance_recall": safe_div(distance_tp, positives),
        "wsts_threshold": wsts_threshold,
        "distance_score_threshold": distance_threshold,
        "stage2_ap": safe_average_precision(alert_universe, stage2_score_column),
        "wsts_ap": safe_average_precision(alert_universe, wsts_score_column),
        "distance_ap": safe_average_precision(alert_universe, distance_score_column),
        "stage2_auc": safe_auc(alert_universe, stage2_score_column),
        "wsts_auc": safe_auc(alert_universe, wsts_score_column),
        "distance_auc": safe_auc(alert_universe, distance_score_column),
    }


def summarize(rows: list[dict[str, int | float | str]]) -> dict[str, int | float]:
    """Aggregate rolling event-day rows."""

    if not rows:
        return {"event_day_count": 0}
    frame = pd.DataFrame(rows)
    summary: dict[str, int | float] = {
        "event_day_count": int(len(frame)),
        "window_count": int(frame["window_id"].nunique()),
        "candidate_count": int(frame["candidate_count"].sum()),
        "positive_count": int(frame["positive_count"].sum()),
    }
    for name in ["stage2", "wsts", "distance"]:
        alerts = int(frame[f"{name}_alert_count"].sum())
        tp = int(frame[f"{name}_true_positive_alert_count"].sum())
        summary[f"{name}_alert_count"] = alerts
        summary[f"{name}_alerts_per_event_day"] = safe_div(alerts, len(frame))
        summary[f"{name}_true_positive_alert_count"] = tp
        summary[f"{name}_precision"] = safe_div(tp, alerts)
        summary[f"{name}_recall"] = safe_div(tp, summary["positive_count"])
        summary[f"{name}_event_day_catch_rate"] = float(
            (frame[f"{name}_true_positive_alert_count"].astype(int) > 0).mean()
        )
        summary[f"{name}_mean_ap"] = nanmean(frame[f"{name}_ap"])
        summary[f"{name}_mean_auc"] = nanmean(frame[f"{name}_auc"])

    for baseline in ["wsts", "distance"]:
        summary[f"true_positive_alert_gain_over_{baseline}"] = (
            int(summary["stage2_true_positive_alert_count"])
            - int(summary[f"{baseline}_true_positive_alert_count"])
        )
        summary[f"precision_gain_over_{baseline}"] = (
            float(summary["stage2_precision"]) - float(summary[f"{baseline}_precision"])
        )
        summary[f"recall_gain_over_{baseline}"] = (
            float(summary["stage2_recall"]) - float(summary[f"{baseline}_recall"])
        )
        summary[f"mean_ap_gain_over_{baseline}"] = (
            float(summary["stage2_mean_ap"]) - float(summary[f"{baseline}_mean_ap"])
        )
        summary[f"mean_auc_gain_over_{baseline}"] = (
            float(summary["stage2_mean_auc"]) - float(summary[f"{baseline}_mean_auc"])
        )
    summary["wsts_threshold"] = float(frame["wsts_threshold"].iloc[0])
    summary["distance_score_threshold"] = float(frame["distance_score_threshold"].iloc[0])
    return summary


def safe_div(numerator: int | float, denominator: int | float) -> float:
    """Divide with zero guard."""

    return float(numerator / denominator) if denominator else 0.0


def safe_average_precision(frame: pd.DataFrame, score_column: str) -> float:
    """Average precision for an event-day, or NaN when undefined."""

    if frame.empty or frame["label"].nunique() < 2:
        return float("nan")
    return float(average_precision_score(frame["label"].astype(int), frame[score_column]))


def safe_auc(frame: pd.DataFrame, score_column: str) -> float:
    """ROC AUC for an event-day, or NaN when undefined."""

    if frame.empty or frame["label"].nunique() < 2:
        return float("nan")
    return float(roc_auc_score(frame["label"].astype(int), frame[score_column]))


def nanmean(values: pd.Series) -> float:
    """Mean that tolerates NaN and infinities."""

    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return float(numeric.mean())


if __name__ == "__main__":
    main()
