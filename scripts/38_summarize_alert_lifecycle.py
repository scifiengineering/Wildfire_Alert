"""Summarize new, continued, removed, and active Stage 2 alerts by event-day."""

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

KEY_COLUMNS = ["event_id", "target_date", "row", "col"]
REQUIRED_COLUMNS = [
    *KEY_COLUMNS,
    "model_score",
    "current_fire_at_candidate",
    "distance_to_current_fire_km",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Post-process scored Stage 2 candidates into alert lifecycle counts. "
            "A selected alert is a candidate with model_score >= threshold, not already "
            "burning/current-fire, and optionally within --max-radius-km."
        )
    )
    parser.add_argument("--scored-candidate-csv", type=Path, required=True)
    parser.add_argument("--threshold", type=float)
    parser.add_argument(
        "--threshold-json",
        type=Path,
        help="JSON file containing frozen_threshold or best_operating_point.threshold.",
    )
    parser.add_argument(
        "--max-radius-km",
        type=float,
        help="Optional candidate-region radius to apply before selecting/continuing alerts.",
    )
    parser.add_argument(
        "--rolling-radii-km",
        type=float,
        nargs="+",
        help=(
            "Evaluate lifecycle within overlapping rolling windows using these step radii. "
            "Use this when comparing against rolling-alert table summaries."
        ),
    )
    parser.add_argument(
        "--carry-over-gaps",
        action="store_true",
        help="Carry active alerts across missing calendar dates. By default, gaps reset state.",
    )
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/stage2/alert_lifecycle"),
    )
    args = parser.parse_args()

    threshold = resolve_threshold(args.threshold, args.threshold_json)
    candidates = load_candidates(args.scored_candidate_csv)
    if args.rolling_radii_km is not None:
        if not args.rolling_radii_km:
            raise ValueError("--rolling-radii-km must contain at least one value.")
        rows = summarize_rolling_alert_lifecycle(
            candidates,
            threshold=threshold,
            radii_km=tuple(args.rolling_radii_km),
        )
        protocol = "rolling_windows"
    else:
        rows = summarize_alert_lifecycle(
            candidates,
            threshold=threshold,
            max_radius_km=args.max_radius_km,
            carry_over_gaps=args.carry_over_gaps,
        )
        protocol = "unique_event_days"
    summary = summarize(rows)
    summary.update(source_universe_summary(candidates))
    summary.update(
        {
            "protocol": protocol,
            "model_name": args.model_name,
            "scored_candidate_csv": str(args.scored_candidate_csv),
            "threshold": threshold,
            "threshold_json": str(args.threshold_json) if args.threshold_json else None,
            "max_radius_km": args.max_radius_km,
            "rolling_radii_km": args.rolling_radii_km,
            "carry_over_gaps": args.carry_over_gaps,
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / f"{args.model_name}_alert_lifecycle_event_days.csv"
    summary_path = args.output_dir / f"{args.model_name}_alert_lifecycle_summary.json"
    pd.DataFrame(rows).to_csv(rows_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"saved alert lifecycle event-day rows to {rows_path}")
    print(f"saved alert lifecycle summary to {summary_path}")


def summarize_alert_lifecycle(
    candidates: pd.DataFrame,
    threshold: float,
    max_radius_km: float | None = None,
    carry_over_gaps: bool = False,
) -> list[dict[str, int | float | str]]:
    """Compute alert lifecycle counts for each available event-day."""

    frame = candidates.copy()
    frame["target_date"] = frame["target_date"].astype(str)
    frame["_target_date_dt"] = pd.to_datetime(frame["target_date"], errors="raise").dt.date
    frame["_coord"] = list(zip(frame["row"].astype(int), frame["col"].astype(int), strict=True))
    frame["_eligible_region"] = frame["current_fire_at_candidate"].astype(int).eq(0)
    if max_radius_km is not None:
        frame["_eligible_region"] &= frame["distance_to_current_fire_km"].astype(float).le(
            max_radius_km
        )
    frame["_selected"] = frame["_eligible_region"] & frame["model_score"].astype(float).ge(
        threshold
    )

    rows: list[dict[str, int | float | str]] = []
    for event_id, event in frame.groupby("event_id", sort=True):
        previous_active: set[tuple[int, int]] = set()
        previous_date: date | None = None
        event_days = event.sort_values("_target_date_dt").groupby(
            ["_target_date_dt", "target_date"], sort=True
        )
        for (target_date_dt, target_date), day in event_days:
            date_gap_days = (
                (target_date_dt - previous_date).days if previous_date is not None else 0
            )
            if previous_date is None or (date_gap_days != 1 and not carry_over_gaps):
                previous_active = set()

            selected_today = set(day.loc[day["_selected"], "_coord"])
            continued = previous_active & selected_today
            new = selected_today - previous_active
            removed = previous_active - selected_today
            removal_reasons = count_removal_reasons(
                day=day,
                removed=removed,
                threshold=threshold,
                max_radius_km=max_radius_km,
            )

            rows.append(
                {
                    "event_id": str(event_id),
                    "target_date": str(target_date),
                    "date_gap_days": int(date_gap_days),
                    "candidate_count": int(len(day)),
                    "eligible_candidate_count": int(day["_eligible_region"].sum()),
                    "selected_alert_count": int(len(selected_today)),
                    "previous_active_alert_count": int(len(previous_active)),
                    "new_alert_count": int(len(new)),
                    "continued_alert_count": int(len(continued)),
                    "removed_alert_count": int(len(removed)),
                    "active_alert_count": int(len(new) + len(continued)),
                    **removal_reasons,
                }
            )
            previous_active = selected_today
            previous_date = target_date_dt
    return rows


def summarize_rolling_alert_lifecycle(
    candidates: pd.DataFrame,
    threshold: float,
    radii_km: tuple[float, ...],
) -> list[dict[str, int | float | str]]:
    """Compute lifecycle counts on the same overlapping windows as rolling alerts."""

    frame = candidates.copy()
    frame["target_date"] = frame["target_date"].astype(str)
    frame["_coord"] = list(zip(frame["row"].astype(int), frame["col"].astype(int), strict=True))
    future = frame[frame["current_fire_at_candidate"].astype(int).eq(0)].copy()

    rows: list[dict[str, int | float | str]] = []
    for event_id, event in future.groupby("event_id", sort=True):
        dates = sorted(str(date) for date in event["target_date"].unique())
        if len(dates) < len(radii_km):
            continue
        for start_index in range(0, len(dates) - len(radii_km) + 1):
            previous_active: set[tuple[int, int]] = set()
            window_dates = dates[start_index : start_index + len(radii_km)]
            window_id = f"{event_id}_{window_dates[0]}_{window_dates[-1]}"
            for step, (target_date, radius_km) in enumerate(
                zip(window_dates, radii_km, strict=True)
            ):
                day = event[event["target_date"] == target_date].copy()
                day["_eligible_region"] = day["distance_to_current_fire_km"].astype(float).le(
                    radius_km
                )
                day["_selected"] = day["_eligible_region"] & day["model_score"].astype(
                    float
                ).ge(threshold)

                selected_today = set(day.loc[day["_selected"], "_coord"])
                continued = previous_active & selected_today
                new = selected_today - previous_active
                removed = previous_active - selected_today
                removal_reasons = count_removal_reasons(
                    day=day,
                    removed=removed,
                    threshold=threshold,
                    max_radius_km=radius_km,
                )

                rows.append(
                    {
                        "event_id": str(event_id),
                        "target_date": str(target_date),
                        "window_id": window_id,
                        "rolling_step": int(step),
                        "radius_km": float(radius_km),
                        "candidate_count": int(len(day)),
                        "eligible_candidate_count": int(day["_eligible_region"].sum()),
                        "selected_alert_count": int(len(selected_today)),
                        "previous_active_alert_count": int(len(previous_active)),
                        "new_alert_count": int(len(new)),
                        "continued_alert_count": int(len(continued)),
                        "removed_alert_count": int(len(removed)),
                        "active_alert_count": int(len(new) + len(continued)),
                        **removal_reasons,
                    }
                )
                previous_active = selected_today
    return rows


def count_removal_reasons(
    day: pd.DataFrame,
    removed: set[tuple[int, int]],
    threshold: float,
    max_radius_km: float | None,
) -> dict[str, int]:
    """Classify removed previous alerts using today's candidate row when present."""

    by_coord = {
        coord: row
        for coord, row in zip(day["_coord"], day.to_dict(orient="records"), strict=True)
    }
    burned = 0
    no_longer_candidate = 0
    below_threshold = 0
    for coord in removed:
        row = by_coord.get(coord)
        if row is None:
            no_longer_candidate += 1
        elif int(row["current_fire_at_candidate"]) == 1:
            burned += 1
        elif (
            max_radius_km is not None
            and float(row["distance_to_current_fire_km"]) > max_radius_km
        ):
            no_longer_candidate += 1
        elif float(row["model_score"]) < threshold:
            below_threshold += 1
        else:
            no_longer_candidate += 1
    return {
        "removed_burned_count": burned,
        "removed_no_longer_candidate_count": no_longer_candidate,
        "removed_below_threshold_count": below_threshold,
    }


def summarize(rows: list[dict[str, int | float | str]]) -> dict[str, int | float]:
    """Aggregate lifecycle counts across event-days."""

    if not rows:
        return {"event_day_count": 0}
    frame = pd.DataFrame(rows)
    total_new = int(frame["new_alert_count"].sum())
    total_continued = int(frame["continued_alert_count"].sum())
    total_removed = int(frame["removed_alert_count"].sum())
    total_active = int(frame["active_alert_count"].sum())
    return {
        "event_day_count": int(len(frame)),
        "event_count": int(frame["event_id"].nunique()),
        **(
            {"window_count": int(frame["window_id"].nunique())}
            if "window_id" in frame.columns
            else {}
        ),
        "total_new_alerts": total_new,
        "total_continued_alerts": total_continued,
        "total_removed_alerts": total_removed,
        "total_active_alerts": total_active,
        "mean_new_alerts_per_event_day": float(frame["new_alert_count"].mean()),
        "mean_continued_alerts_per_event_day": float(frame["continued_alert_count"].mean()),
        "mean_removed_alerts_per_event_day": float(frame["removed_alert_count"].mean()),
        "mean_active_alerts_per_event_day": float(frame["active_alert_count"].mean()),
        "mean_selected_alerts_per_event_day": float(frame["selected_alert_count"].mean()),
        "total_removed_burned": int(frame["removed_burned_count"].sum()),
        "total_removed_no_longer_candidate": int(
            frame["removed_no_longer_candidate_count"].sum()
        ),
        "total_removed_below_threshold": int(frame["removed_below_threshold_count"].sum()),
    }


def source_universe_summary(candidates: pd.DataFrame) -> dict[str, int]:
    """Describe the input candidate universe before lifecycle windowing."""

    event_days = candidates[["event_id", "target_date"]].drop_duplicates()
    return {
        "source_event_count": int(candidates["event_id"].nunique()),
        "source_event_day_count": int(len(event_days)),
    }


def load_candidates(path: Path) -> pd.DataFrame:
    """Load only columns needed for alert lifecycle post-processing."""

    header = pd.read_csv(path, nrows=0)
    missing = sorted(set(REQUIRED_COLUMNS) - set(header.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    columns = REQUIRED_COLUMNS
    if "label" in header.columns:
        columns = [*columns, "label"]
    frame = pd.read_csv(path, usecols=columns)
    if frame.empty:
        raise ValueError(f"{path} contains no candidate rows.")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    required_without_distance = [
        column for column in REQUIRED_COLUMNS if column != "distance_to_current_fire_km"
    ]
    return frame.dropna(subset=required_without_distance)


def resolve_threshold(threshold: float | None, threshold_json: Path | None) -> float:
    """Resolve a numeric Stage 2 alert threshold from CLI args."""

    if (threshold is None) == (threshold_json is None):
        raise ValueError("Provide exactly one of --threshold or --threshold-json.")
    if threshold is not None:
        return float(threshold)
    assert threshold_json is not None
    data: dict[str, Any] = json.loads(threshold_json.read_text(encoding="utf-8"))
    if "frozen_threshold" in data:
        return float(data["frozen_threshold"])
    operating_point = data.get("best_operating_point", {})
    if isinstance(operating_point, dict) and "threshold" in operating_point:
        return float(operating_point["threshold"])
    raise ValueError(f"No frozen_threshold found in {threshold_json}.")


if __name__ == "__main__":
    main()
