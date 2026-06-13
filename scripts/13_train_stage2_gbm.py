"""Train the Stage 2 LightGBM discriminative alert classifier."""

import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

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
    """Train and evaluate a Stage 2 LightGBM model."""

    parser = argparse.ArgumentParser(description="Train Stage 2 LightGBM alert model.")
    parser.add_argument("--train-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stage2/models"))
    parser.add_argument("--model-name", type=str, default="stage2_gbm_fold0_to_fold1")
    parser.add_argument("--n-estimators", type=int, default=1000)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_frames = [pd.read_csv(path) for path in args.train_csv]
    train = pd.concat(train_frames, ignore_index=True)
    validation = pd.read_csv(args.val_csv)
    feature_columns = feature_column_names(train)
    x_train = train[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_train = train["label"].astype(int)
    x_validation = validation[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_validation = validation["label"].astype(int)

    model = lgb.LGBMClassifier(
        objective="binary",
        class_weight="balanced",
        n_estimators=args.n_estimators,
        num_leaves=args.num_leaves,
        learning_rate=args.learning_rate,
        random_state=args.seed,
        n_jobs=-1,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_validation, y_validation)],
        eval_metric="average_precision",
        callbacks=[
            lgb.early_stopping(args.early_stopping_rounds),
            lgb.log_evaluation(period=25),
        ],
    )

    validation_scores = model.predict_proba(x_validation)[:, 1]
    threshold_sweep = threshold_metrics(y_validation.to_numpy(), validation_scores)
    best_threshold = max(threshold_sweep, key=lambda row: row["f1"])
    metrics = {
        "train_csv": [str(path) for path in args.train_csv],
        "val_csv": str(args.val_csv),
        "feature_count": len(feature_columns),
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "train_positive_rate": float(y_train.mean()),
        "validation_positive_rate": float(y_validation.mean()),
        "best_iteration": int(model.best_iteration_ or args.n_estimators),
        "validation_average_precision": float(
            average_precision_score(y_validation, validation_scores)
        ),
        "validation_roc_auc": safe_auc(y_validation.to_numpy(), validation_scores),
        "validation_within_event_auc": within_event_auc(validation, validation_scores),
        "best_threshold_by_f1": best_threshold,
        "threshold_sweep": threshold_sweep,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / f"{args.model_name}.joblib"
    metrics_path = args.output_dir / f"{args.model_name}_metrics.json"
    importance_path = args.output_dir / f"{args.model_name}_feature_importance.csv"
    joblib.dump({"model": model, "feature_columns": feature_columns}, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    write_feature_importance(model, feature_columns, importance_path)

    print(json.dumps(metrics, indent=2))
    print(f"saved model to {model_path}")
    print(f"saved metrics to {metrics_path}")
    print(f"saved feature importance to {importance_path}")


def feature_column_names(frame: pd.DataFrame) -> list[str]:
    """Return numeric feature columns from a candidate table."""

    return [
        column
        for column in frame.columns
        if column not in NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(frame[column])
    ]


def threshold_metrics(labels: np.ndarray, scores: np.ndarray) -> list[dict[str, float]]:
    """Evaluate precision/recall/F1 over score quantile thresholds."""

    quantiles = np.linspace(0.0, 1.0, 101)
    thresholds = np.unique(np.quantile(scores, quantiles))
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


def within_event_auc(frame: pd.DataFrame, scores: np.ndarray) -> float:
    """Mean ROC AUC over event-day candidate sets with both classes present."""

    scored = frame[["event_id", "target_date", "label"]].copy()
    scored["score"] = scores
    aucs: list[float] = []
    for _, group in scored.groupby(["event_id", "target_date"], sort=False):
        labels = group["label"].to_numpy()
        if np.unique(labels).size < 2:
            continue
        aucs.append(float(roc_auc_score(labels, group["score"].to_numpy())))
    return float(np.mean(aucs)) if aucs else 0.0


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Return ROC AUC, or 0 when only one class is present."""

    if np.unique(labels).size < 2:
        return 0.0
    return float(roc_auc_score(labels, scores))


def write_feature_importance(
    model: lgb.LGBMClassifier,
    feature_columns: list[str],
    path: Path,
) -> None:
    """Write LightGBM split/gain importances."""

    booster = model.booster_
    frame = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance_split": booster.feature_importance(importance_type="split"),
            "importance_gain": booster.feature_importance(importance_type="gain"),
        }
    ).sort_values("importance_gain", ascending=False)
    frame.to_csv(path, index=False)


if __name__ == "__main__":
    main()
