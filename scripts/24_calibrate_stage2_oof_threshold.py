"""Freeze one Stage 2 alert threshold from leakage-free 2019 OOF scores."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate a global Stage 2 OOF threshold.")
    parser.add_argument("--models", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate-csvs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if len(args.models) != len(args.candidate_csvs):
        raise ValueError("--models and --candidate-csvs must have the same length.")

    scored_frames: list[pd.DataFrame] = []
    for model_path, csv_path in zip(args.models, args.candidate_csvs, strict=True):
        bundle = joblib.load(model_path)
        candidates = pd.read_csv(csv_path)
        feature_columns = list(bundle["feature_columns"])
        features = candidates[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        scored_frames.append(
            pd.DataFrame(
                {
                    "label": candidates["label"].astype(int),
                    "score": bundle["model"].predict_proba(features)[:, 1],
                }
            )
        )
    scored = pd.concat(scored_frames, ignore_index=True)
    sweep = threshold_sweep(scored["label"].to_numpy(), scored["score"].to_numpy())
    best = max(sweep, key=lambda row: row["f1"])
    report = {
        "calibration_data": "2019 leakage-free out-of-fold candidate predictions",
        "models": [str(path) for path in args.models],
        "candidate_csvs": [str(path) for path in args.candidate_csvs],
        "row_count": len(scored),
        "positive_rate": float(scored["label"].mean()),
        "average_precision": float(average_precision_score(scored["label"], scored["score"])),
        "selection_metric": "pooled_oof_f1",
        "frozen_threshold": best["threshold"],
        "best_operating_point": best,
        "threshold_sweep": sweep,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    concise_report = {
        key: value for key, value in report.items() if key != "threshold_sweep"
    }
    print(json.dumps(concise_report, indent=2))


def threshold_sweep(labels: np.ndarray, scores: np.ndarray) -> list[dict[str, float]]:
    thresholds = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 201)))
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        predictions = scores >= threshold
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            predictions,
            average="binary",
            zero_division=0,
        )
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "alert_rate": float(predictions.mean()),
            }
        )
    return rows


if __name__ == "__main__":
    main()
