"""Write a Markdown report from matched-volume comparison summaries."""

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize matched-volume comparisons.")
    parser.add_argument("--summary-json", type=Path, nargs="+", required=True)
    parser.add_argument("--primary-name", default="stage2_2020_external_ensemble")
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("MATCHED_VOLUME_ALERT_QUALITY_REPORT.md"),
    )
    args = parser.parse_args()

    summaries = [load_summary(path) for path in args.summary_json]
    primary = next(
        (summary for summary in summaries if summary.get("model_name") == args.primary_name),
        summaries[0],
    )
    report = render_report(primary, summaries)
    args.output_md.write_text(report, encoding="utf-8")
    print(f"saved matched-volume report to {args.output_md}")


def load_summary(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["_path"] = str(path)
    return summary


def render_report(primary: dict[str, Any], summaries: list[dict[str, Any]]) -> str:
    lines = [
        "# Matched-Volume Alert Quality Report",
        "",
        "This report compares Stage 2 alerts with WSTS-as-alerter alerts using one "
        "global WSTS threshold selected to match Stage 2's mean alert volume.",
        "",
        "## Primary Result",
        "",
        "| Metric | Stage 2 | WSTS-as-alerter |",
        "|---|---:|---:|",
        metric_row(
            "Alerts/event-day",
            primary,
            "stage2_alerts_per_event_day",
            "wsts_alerts_per_event_day",
            "{:.2f}",
        ),
        metric_row("Precision", primary, "stage2_precision", "wsts_precision", "{:.4f}"),
        metric_row("Recall", primary, "stage2_recall", "wsts_recall", "{:.4f}"),
        metric_row(
            "True-positive alerts",
            primary,
            "stage2_true_positive_alert_count",
            "wsts_true_positive_alert_count",
            "{:,.0f}",
        ),
        metric_row(
            "Event-day catch rate",
            primary,
            "stage2_event_day_catch_rate",
            "wsts_event_day_catch_rate",
            "{:.4f}",
        ),
        "",
        primary_gain_sentence(primary),
        "",
        "## Secondary Ranking Metrics",
        "",
        "| Metric | Stage 2 | WSTS-as-alerter |",
        "|---|---:|---:|",
        metric_row("Mean AP", primary, "stage2_mean_ap", "wsts_mean_ap", "{:.4f}"),
        metric_row("Mean AUC", primary, "stage2_mean_auc", "wsts_mean_auc", "{:.4f}"),
        "",
        "## Runs",
        "",
        "| Run | Event-days | Stage 2 precision | WSTS precision | TP gain |",
        "|---|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            "| {name} | {event_days:,} | {stage2_precision:.4f} | "
            "{wsts_precision:.4f} | {tp_gain:,} |".format(
                name=summary.get("model_name", summary.get("_path", "run")),
                event_days=int(summary.get("event_day_count", 0)),
                stage2_precision=float(summary.get("stage2_precision", 0.0)),
                wsts_precision=float(summary.get("wsts_precision", 0.0)),
                tp_gain=int(summary.get("true_positive_alert_gain", 0)),
            )
        )
    lines.extend(
        [
            "",
            "## Inputs",
            "",
            f"- Primary summary: `{primary.get('_path', '')}`",
            f"- Stage 2 score column: `{primary.get('stage2_score_column', 'model_score')}`",
            f"- WSTS score column: `{primary.get('wsts_score_column', 'stage1_probability')}`",
            f"- Stage 2 threshold: `{primary.get('stage2_threshold', '')}`",
            f"- WSTS threshold: `{primary.get('wsts_threshold', '')}`",
            f"- WSTS threshold source: `{primary.get('wsts_threshold_source', '')}`",
            f"- Radii km: `{primary.get('radii_km', '')}`",
            "",
        ]
    )
    return "\n".join(lines)


def metric_row(
    label: str,
    summary: dict[str, Any],
    stage2_key: str,
    wsts_key: str,
    fmt: str,
) -> str:
    return (
        f"| {label} | {fmt.format(float(summary.get(stage2_key, 0.0)))} | "
        f"{fmt.format(float(summary.get(wsts_key, 0.0)))} |"
    )


def primary_gain_sentence(summary: dict[str, Any]) -> str:
    gain = int(summary.get("true_positive_alert_gain", 0))
    relative_gain = float(summary.get("relative_true_positive_alert_gain", 0.0)) * 100.0
    return (
        f"Stage 2 catches {gain:,} more positive alert occurrences "
        f"({relative_gain:.2f}% relative gain) at globally matched mean alert volume."
    )


if __name__ == "__main__":
    main()
