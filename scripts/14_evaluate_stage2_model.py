"""Evaluate Stage 2 models against simple alerting baselines."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def main() -> None:
    """Evaluate a saved Stage 2 model on a candidate table."""

    parser = argparse.ArgumentParser(description="Evaluate Stage 2 discriminative alerts.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stage2/evaluation"))
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    model = bundle["model"]
    feature_columns = list(bundle["feature_columns"])
    candidates = pd.read_csv(args.candidate_csv)
    labels = candidates["label"].astype(int).to_numpy()
    features = candidates[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    model_scores = model.predict_proba(features)[:, 1]
    distance_scores = -candidates["distance_to_current_fire_px"].fillna(1e6).to_numpy()
    stage1_scores = candidates["stage1_probability"].fillna(0.0).to_numpy()
    forecast_wind_scores = candidates["forecast_wind_alignment"].fillna(0.0).to_numpy()

    report = {
        "model": str(args.model),
        "candidate_csv": str(args.candidate_csv),
        "model_name": args.model_name,
        "rows": int(len(candidates)),
        "positive_count": int(labels.sum()),
        "positive_rate": float(labels.mean()),
        "model_metrics": score_report(candidates, labels, model_scores),
        "distance_baseline_metrics": score_report(candidates, labels, distance_scores),
        "stage1_probability_baseline_metrics": score_report(candidates, labels, stage1_scores),
        "forecast_wind_alignment_baseline_metrics": score_report(
            candidates, labels, forecast_wind_scores
        ),
    }
    report["summary"] = build_summary(report)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.model_name}_evaluation.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"saved evaluation report to {output_path}")


def score_report(
    candidates: pd.DataFrame,
    labels: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float | dict[str, float]]:
    """Return global and within-event metrics for a score vector."""

    threshold_sweep = threshold_metrics(labels, scores)
    best_f1 = max(threshold_sweep, key=lambda row: row["f1"])
    return {
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": safe_auc(labels, scores),
        "within_event_auc": within_event_auc(candidates, scores),
        "best_threshold_by_f1": best_f1,
    }


def threshold_metrics(labels: np.ndarray, scores: np.ndarray) -> list[dict[str, float]]:
    """Evaluate precision/recall/F1 over score quantile thresholds."""

    thresholds = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 101)))
    records: list[dict[str, float]] = []
    for threshold in thresholds:
        predictions = scores >= threshold
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            predictions,
            average="binary",
            zero_division=0,
        )
        records.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "alert_rate": float(predictions.mean()),
            }
        )
    return records


def within_event_auc(candidates: pd.DataFrame, scores: np.ndarray) -> float:
    """Mean candidate-ranking ROC AUC inside event-day groups."""

    scored = candidates[["event_id", "target_date", "label"]].copy()
    scored["score"] = scores
    aucs: list[float] = []
    for _, group in scored.groupby(["event_id", "target_date"], sort=False):
        labels = group["label"].to_numpy()
        if np.unique(labels).size < 2:
            continue
        aucs.append(float(roc_auc_score(labels, group["score"].to_numpy())))
    return float(np.mean(aucs)) if aucs else 0.0


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Return ROC AUC, or 0 if the labels contain only one class."""

    if np.unique(labels).size < 2:
        return 0.0
    return float(roc_auc_score(labels, scores))


def build_summary(report: dict[str, object]) -> dict[str, float | bool]:
    """Build a compact pass/fail-oriented summary."""

    model_metrics = report["model_metrics"]
    distance_metrics = report["distance_baseline_metrics"]
    stage1_metrics = report["stage1_probability_baseline_metrics"]
    if not isinstance(model_metrics, dict):
        raise TypeError("model_metrics must be a dict.")
    if not isinstance(distance_metrics, dict):
        raise TypeError("distance_baseline_metrics must be a dict.")
    if not isinstance(stage1_metrics, dict):
        raise TypeError("stage1_probability_baseline_metrics must be a dict.")

    model_ap = float(model_metrics["average_precision"])
    model_within_auc = float(model_metrics["within_event_auc"])
    distance_ap = float(distance_metrics["average_precision"])
    distance_within_auc = float(distance_metrics["within_event_auc"])
    stage1_ap = float(stage1_metrics["average_precision"])
    stage1_within_auc = float(stage1_metrics["within_event_auc"])
    return {
        "model_average_precision": model_ap,
        "distance_baseline_average_precision": distance_ap,
        "stage1_baseline_average_precision": stage1_ap,
        "model_within_event_auc": model_within_auc,
        "distance_baseline_within_event_auc": distance_within_auc,
        "stage1_baseline_within_event_auc": stage1_within_auc,
        "ap_gain_over_distance": model_ap - distance_ap,
        "within_event_auc_gain_over_distance": model_within_auc - distance_within_auc,
        "ap_gain_over_stage1": model_ap - stage1_ap,
        "within_event_auc_gain_over_stage1": model_within_auc - stage1_within_auc,
        "passes_discriminative_check": bool(
            model_within_auc >= 0.65 and model_within_auc > distance_within_auc
        ),
    }


if __name__ == "__main__":
    main()
