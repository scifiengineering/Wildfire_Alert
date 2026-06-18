"""Compare Stage 2 alerts with WSTS alerts at matched mean event-day volume."""

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
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Stage 2 against a WSTS-as-alerter baseline with the WSTS alert "
            "count matched to Stage 2 by one global threshold."
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
    parser.add_argument(
        "--wsts-threshold",
        type=float,
        help=(
            "Optional fixed WSTS threshold. By default the script selects one global "
            "threshold whose total alert count is closest to the Stage 2 total."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stage2/matched_volume"))
    args = parser.parse_args()

    candidates, source = load_candidates(args)
    rows = compare_rolling_event_days(
        candidates=candidates,
        stage2_threshold=args.stage2_threshold,
        radii_km=tuple(args.radii_km),
        stage2_score_column=args.stage2_score_column,
        wsts_score_column=args.wsts_score_column,
        wsts_threshold=args.wsts_threshold,
    )
    summary = summarize(rows)
    summary.update(
        {
            **source,
            "model_name": args.model_name,
            "stage2_threshold": args.stage2_threshold,
            "wsts_threshold": rows[0]["wsts_threshold"] if rows else args.wsts_threshold,
            "wsts_threshold_source": (
                "provided" if args.wsts_threshold is not None else "calibrated"
            ),
            "radii_km": args.radii_km,
            "stage2_score_column": args.stage2_score_column,
            "wsts_score_column": args.wsts_score_column,
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / f"{args.model_name}_matched_volume_event_days.csv"
    summary_path = args.output_dir / f"{args.model_name}_matched_volume_summary.json"
    pd.DataFrame(rows).to_csv(rows_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"saved matched-volume event-day rows to {rows_path}")
    print(f"saved matched-volume summary to {summary_path}")


def load_candidates(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    if args.scored_candidate_csv is not None:
        if args.candidate_csv is not None or args.models is not None:
            raise ValueError("Use either --scored-candidate-csv or --candidate-csv with --models.")
        reject_lfs_pointer(args.scored_candidate_csv)
        candidates = pd.read_csv(args.scored_candidate_csv)
        return candidates, {
            "scored_candidate_csv": str(args.scored_candidate_csv),
            "candidate_csv": None,
            "models": None,
        }

    if args.candidate_csv is None or not args.models:
        raise ValueError("Provide --scored-candidate-csv or both --candidate-csv and --models.")
    reject_lfs_pointer(args.candidate_csv)
    candidates = pd.read_csv(args.candidate_csv)
    candidates = add_ensemble_scores(candidates, args.models, args.stage2_score_column)
    return candidates, {
        "scored_candidate_csv": None,
        "candidate_csv": str(args.candidate_csv),
        "models": [str(path) for path in args.models],
    }


def reject_lfs_pointer(path: Path) -> None:
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


def compare_rolling_event_days(
    candidates: pd.DataFrame,
    stage2_threshold: float,
    radii_km: tuple[float, ...],
    stage2_score_column: str,
    wsts_score_column: str,
    wsts_threshold: float | None = None,
) -> list[dict[str, int | float | str]]:
    required = {
        "event_id",
        "target_date",
        "label",
        "distance_to_current_fire_km",
        stage2_score_column,
        wsts_score_column,
    }
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"Candidate table is missing required columns: {missing}")

    future = candidates.copy()
    if "current_fire_at_candidate" in future.columns:
        future = future[future["current_fire_at_candidate"] == 0].copy()

    event_day_frames = collect_event_day_frames(
        future=future,
        radii_km=radii_km,
        stage2_threshold=stage2_threshold,
        stage2_score_column=stage2_score_column,
    )
    if wsts_threshold is None:
        wsts_threshold = calibrate_global_wsts_threshold(
            event_day_frames,
            wsts_score_column=wsts_score_column,
        )

    rows: list[dict[str, int | float | str]] = []
    for item in event_day_frames:
        alert_universe = item["alert_universe"]
        if not isinstance(alert_universe, pd.DataFrame):
            raise TypeError("alert_universe must be a DataFrame.")
        stage2_alerts = item["stage2_alerts"]
        if not isinstance(stage2_alerts, pd.DataFrame):
            raise TypeError("stage2_alerts must be a DataFrame.")
        wsts_alerts = alert_universe[alert_universe[wsts_score_column] >= wsts_threshold]
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
                stage2_score_column=stage2_score_column,
                wsts_score_column=wsts_score_column,
                wsts_threshold=wsts_threshold,
            )
        )
    return rows


def collect_event_day_frames(
    future: pd.DataFrame,
    radii_km: tuple[float, ...],
    stage2_threshold: float,
    stage2_score_column: str,
) -> list[dict[str, object]]:
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
                alert_universe = day[day["distance_to_current_fire_km"] <= radius_km].copy()
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


def calibrate_global_wsts_threshold(
    event_day_frames: list[dict[str, object]],
    wsts_score_column: str,
) -> float:
    target_alert_count = sum(
        len(item["stage2_alerts"])
        for item in event_day_frames
        if isinstance(item["stage2_alerts"], pd.DataFrame)
    )
    scores = [
        np.asarray(item["alert_universe"][wsts_score_column], dtype=float)
        for item in event_day_frames
        if isinstance(item["alert_universe"], pd.DataFrame) and len(item["alert_universe"]) > 0
    ]
    if not scores:
        return float("inf")
    all_scores = np.concatenate(scores)
    if target_alert_count <= 0:
        return float(np.nextafter(np.nanmax(all_scores), np.inf))
    if target_alert_count >= all_scores.size:
        return float(np.nanmin(all_scores))

    sorted_scores = np.sort(all_scores[~np.isnan(all_scores)])[::-1]
    if sorted_scores.size == 0:
        return float("inf")
    index = min(target_alert_count - 1, sorted_scores.size - 1)
    return float(sorted_scores[index])


def event_day_row(
    event_id: str,
    target_date: str,
    window_id: str,
    step: int,
    radius_km: float,
    alert_universe: pd.DataFrame,
    stage2_alerts: pd.DataFrame,
    wsts_alerts: pd.DataFrame,
    stage2_score_column: str,
    wsts_score_column: str,
    wsts_threshold: float,
) -> dict[str, int | float | str]:
    positives = int(alert_universe["label"].sum())
    stage2_tp = int(stage2_alerts["label"].sum())
    wsts_tp = int(wsts_alerts["label"].sum())
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
        "wsts_threshold": wsts_threshold,
        "stage2_ap": safe_average_precision(alert_universe, stage2_score_column),
        "wsts_ap": safe_average_precision(alert_universe, wsts_score_column),
        "stage2_auc": safe_auc(alert_universe, stage2_score_column),
        "wsts_auc": safe_auc(alert_universe, wsts_score_column),
    }


def summarize(rows: list[dict[str, int | float | str]]) -> dict[str, int | float]:
    if not rows:
        return {"event_day_count": 0}
    frame = pd.DataFrame(rows)
    stage2_alerts = int(frame["stage2_alert_count"].sum())
    wsts_alerts = int(frame["wsts_alert_count"].sum())
    stage2_tp = int(frame["stage2_true_positive_alert_count"].sum())
    wsts_tp = int(frame["wsts_true_positive_alert_count"].sum())
    positives = int(frame["positive_count"].sum())
    stage2_catch = frame["stage2_true_positive_alert_count"].astype(int) > 0
    wsts_catch = frame["wsts_true_positive_alert_count"].astype(int) > 0
    return {
        "event_day_count": int(len(frame)),
        "window_count": int(frame["window_id"].nunique()),
        "candidate_count": int(frame["candidate_count"].sum()),
        "positive_count": positives,
        "stage2_alert_count": stage2_alerts,
        "stage2_alerts_per_event_day": safe_div(stage2_alerts, len(frame)),
        "stage2_true_positive_alert_count": stage2_tp,
        "stage2_precision": safe_div(stage2_tp, stage2_alerts),
        "stage2_recall": safe_div(stage2_tp, positives),
        "stage2_event_day_catch_rate": float(stage2_catch.mean()),
        "wsts_alert_count": wsts_alerts,
        "wsts_alerts_per_event_day": safe_div(wsts_alerts, len(frame)),
        "wsts_true_positive_alert_count": wsts_tp,
        "wsts_precision": safe_div(wsts_tp, wsts_alerts),
        "wsts_recall": safe_div(wsts_tp, positives),
        "wsts_event_day_catch_rate": float(wsts_catch.mean()),
        "true_positive_alert_gain": stage2_tp - wsts_tp,
        "relative_true_positive_alert_gain": safe_div(stage2_tp - wsts_tp, wsts_tp),
        "stage2_mean_ap": nanmean(frame["stage2_ap"]),
        "wsts_mean_ap": nanmean(frame["wsts_ap"]),
        "stage2_mean_auc": nanmean(frame["stage2_auc"]),
        "wsts_mean_auc": nanmean(frame["wsts_auc"]),
    }


def safe_div(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def safe_average_precision(frame: pd.DataFrame, score_column: str) -> float:
    if frame.empty or frame["label"].nunique() < 2:
        return float("nan")
    return float(average_precision_score(frame["label"].astype(int), frame[score_column]))


def safe_auc(frame: pd.DataFrame, score_column: str) -> float:
    if frame.empty or frame["label"].nunique() < 2:
        return float("nan")
    return float(roc_auc_score(frame["label"].astype(int), frame[score_column]))


def nanmean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return float(numeric.mean())


if __name__ == "__main__":
    main()
