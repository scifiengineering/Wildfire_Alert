"""Evaluate Stage 1 probability maps with one global alert threshold.

This script treats each saved Stage 1 ``probability`` raster as the alert score
for one event-day and reports precision under one deployable fixed rule: choose
one global probability threshold to average about the requested number of
alerts/event-day. Daily alert counts are allowed to vary. No AP is computed.
"""

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ResultRow:
    method_name: str
    split_protocol: str
    global_probability_threshold: float
    event_days_evaluated: int
    target_mean_alerts_per_event_day: float
    achieved_mean_alerts_per_event_day: float
    total_alerts: int
    true_positive_alerts: int
    false_positive_alerts: int
    precision: float


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Stage-1 alert precision with one global probability threshold."
    )
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=Path("outputs/predictions/stage1_2019_3fold"),
        help="Root containing fold_*/validation/*.npz Stage-1 prediction artifacts.",
    )
    parser.add_argument("--data-root", type=Path, default=Path("wsts_hdf5/2019"))
    parser.add_argument("--year", type=int, default=2019)
    parser.add_argument("--target-mean-alerts-per-event-day", type=float, default=90.0)
    parser.add_argument(
        "--split-protocol",
        default=(
            "2019 3-fold event-level out-of-fold validation; "
            "existing Stage-1 probability artifacts"
        ),
    )
    parser.add_argument(
        "--method-name",
        default="Stage-1 WSTS-style probability baseline",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/tanisha_2019_stage1_baseline"),
    )
    args = parser.parse_args()

    prediction_paths = discover_prediction_paths(args.prediction_root)
    if not prediction_paths:
        raise FileNotFoundError(f"No prediction artifacts found under {args.prediction_root}")

    row = evaluate_global_threshold(
        prediction_paths,
        method_name=args.method_name,
        split_protocol=args.split_protocol,
        target_mean_alerts_per_event_day=args.target_mean_alerts_per_event_day,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "stage1_2019_alert_budget_results.csv", [row])
    write_json(args.output_dir / "stage1_2019_alert_budget_results.json", [row])
    write_markdown(args.output_dir / "stage1_2019_alert_budget_results.md", [row])
    write_reproducibility_json(
        args.output_dir / "stage1_2019_alert_budget_reproducibility.json",
        args=args,
        prediction_count=len(prediction_paths),
        result=row,
    )
    print(json.dumps(asdict(row), indent=2))


def discover_prediction_paths(prediction_root: Path) -> list[Path]:
    """Return one sorted list of validation prediction artifacts."""

    manifest_paths = sorted(prediction_root.glob("fold_*/validation/manifest.json"))
    paths: list[Path] = []
    if manifest_paths:
        for manifest_path in manifest_paths:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for record in manifest.get("records", []):
                path = Path(str(record["path"]).replace("\\", "/"))
                if not path.is_absolute():
                    path = Path.cwd() / path
                paths.append(path)
        return sorted(paths)
    return sorted(prediction_root.glob("fold_*/validation/*/*.npz"))


def evaluate_global_threshold(
    prediction_paths: list[Path],
    method_name: str,
    split_protocol: str,
    target_mean_alerts_per_event_day: float,
) -> ResultRow:
    """Evaluate one global probability threshold for the requested average budget."""

    probabilities: list[NDArray[np.float32]] = []
    targets: list[NDArray[np.uint8]] = []
    for path in prediction_paths:
        probability, target = load_prediction(path)
        probabilities.append(probability.reshape(-1))
        targets.append(target.reshape(-1))

    all_probabilities = np.concatenate(probabilities)
    all_targets = np.concatenate(targets)
    requested_alerts = min(
        int(round(target_mean_alerts_per_event_day * len(prediction_paths))),
        int(all_probabilities.size),
    )
    if requested_alerts <= 0:
        raise ValueError("Requested alert count must be positive.")

    threshold = float(np.partition(all_probabilities, -requested_alerts)[-requested_alerts])
    selected = all_probabilities >= threshold
    total_alerts = int(selected.sum())
    true_positive_alerts = int(all_targets[selected].sum())
    false_positive_alerts = total_alerts - true_positive_alerts
    achieved_mean_alerts_per_event_day = total_alerts / len(prediction_paths)
    precision = true_positive_alerts / total_alerts if total_alerts else 0.0

    return ResultRow(
        method_name=method_name,
        split_protocol=split_protocol,
        global_probability_threshold=threshold,
        event_days_evaluated=len(prediction_paths),
        target_mean_alerts_per_event_day=float(target_mean_alerts_per_event_day),
        achieved_mean_alerts_per_event_day=achieved_mean_alerts_per_event_day,
        total_alerts=total_alerts,
        true_positive_alerts=true_positive_alerts,
        false_positive_alerts=false_positive_alerts,
        precision=precision,
    )


def load_prediction(path: Path) -> tuple[NDArray[np.float32], NDArray[np.uint8]]:
    with np.load(path, allow_pickle=False) as artifact:
        probability = np.nan_to_num(
            np.asarray(artifact["probability"], dtype=np.float32),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )
        target = (np.asarray(artifact["target"], dtype=np.uint8) > 0).astype(np.uint8)
    if probability.shape != target.shape:
        raise ValueError(f"Probability and target shape mismatch in {path}")
    return probability, target


def write_csv(path: Path, rows: list[ResultRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def write_json(path: Path, rows: list[ResultRow]) -> None:
    path.write_text(json.dumps([asdict(row) for row in rows], indent=2) + "\n", encoding="utf-8")


def write_markdown(path: Path, rows: list[ResultRow]) -> None:
    fields = list(asdict(rows[0]).keys())
    lines = [
        "# Stage-1 2019 Global-Threshold Alert-Budget Results",
        "",
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        values = asdict(row)
        formatted = [format_markdown_value(values[field]) for field in fields]
        lines.append("| " + " | ".join(formatted) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_reproducibility_json(
    path: Path,
    args: argparse.Namespace,
    prediction_count: int,
    result: ResultRow,
) -> None:
    payload = {
        "script": "scripts/26_evaluate_stage1_alert_budget.py",
        "command": (
            ".venv/bin/python scripts/26_evaluate_stage1_alert_budget.py "
            f"--prediction-root {args.prediction_root} "
            f"--data-root {args.data_root} "
            f"--year {args.year} "
            f"--target-mean-alerts-per-event-day {args.target_mean_alerts_per_event_day} "
            f"--output-dir {args.output_dir}"
        ),
        "prediction_root": str(args.prediction_root),
        "data_root": str(args.data_root),
        "year": args.year,
        "prediction_artifacts_evaluated": prediction_count,
        "result": asdict(result),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def format_markdown_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


if __name__ == "__main__":
    main()
