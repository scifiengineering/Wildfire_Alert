"""Compute SHAP explanations for Stage 2 LightGBM alert models."""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

matplotlib.use("Agg")


METADATA_COLUMNS = ["event_id", "target_date", "row", "col", "label"]


def main() -> None:
    """Compute global and local SHAP explanations for one Stage 2 model."""

    parser = argparse.ArgumentParser(description="Compute Stage 2 SHAP explanations.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--max-rows", type=int, default=5000)
    parser.add_argument("--local-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stage2/shap"))
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    bundle = joblib.load(args.model)
    model = bundle["model"]
    feature_columns = list(bundle["feature_columns"])
    candidates = pd.read_csv(args.candidate_csv)
    features = candidates[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scores = model.predict_proba(features)[:, 1]
    sampled = sample_for_shap(candidates, features, scores, args.max_rows, rng)
    sampled_features = sampled["features"]
    sampled_candidates = sampled["candidates"]
    sampled_scores = sampled["scores"]

    explainer = shap.TreeExplainer(model)
    shap_values = positive_class_shap_values(explainer.shap_values(sampled_features))
    expected_value = positive_class_expected_value(explainer.expected_value)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.output_dir / args.model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    importance = global_importance(shap_values, feature_columns)
    importance_path = model_dir / "global_shap_importance.csv"
    importance.to_csv(importance_path, index=False)

    rows_path = model_dir / "sample_shap_values.csv"
    write_sample_shap_rows(
        path=rows_path,
        candidates=sampled_candidates,
        scores=sampled_scores,
        shap_values=shap_values,
        feature_columns=feature_columns,
    )

    beeswarm_path = model_dir / "global_beeswarm.png"
    save_beeswarm(shap_values, sampled_features, feature_columns, beeswarm_path)

    local_dir = model_dir / "local"
    local_dir.mkdir(exist_ok=True)
    local_records = save_local_explanations(
        local_dir=local_dir,
        candidates=sampled_candidates,
        features=sampled_features,
        scores=sampled_scores,
        shap_values=shap_values,
        feature_columns=feature_columns,
        expected_value=expected_value,
        count=args.local_count,
    )
    paired_records = save_paired_examples(
        output_dir=model_dir,
        candidates=sampled_candidates,
        scores=sampled_scores,
        shap_values=shap_values,
        feature_columns=feature_columns,
    )

    summary = {
        "model": str(args.model),
        "candidate_csv": str(args.candidate_csv),
        "model_name": args.model_name,
        "candidate_rows": int(len(candidates)),
        "shap_rows": int(len(sampled_candidates)),
        "feature_count": len(feature_columns),
        "expected_value_log_odds": expected_value,
        "top_global_features": importance.head(10).to_dict(orient="records"),
        "local_explanations": local_records,
        "paired_examples": paired_records,
        "outputs": {
            "global_importance_csv": str(importance_path),
            "sample_shap_values_csv": str(rows_path),
            "global_beeswarm_png": str(beeswarm_path),
            "local_dir": str(local_dir),
        },
    }
    summary_path = model_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def sample_for_shap(
    candidates: pd.DataFrame,
    features: pd.DataFrame,
    scores: np.ndarray,
    max_rows: int,
    rng: np.random.Generator,
) -> dict[str, pd.DataFrame | np.ndarray]:
    """Sample rows while always keeping positives and highest-risk candidates."""

    if max_rows <= 0 or len(candidates) <= max_rows:
        return {"candidates": candidates, "features": features, "scores": scores}

    positive_indices = np.flatnonzero(candidates["label"].to_numpy(dtype=int) == 1)
    top_count = min(max_rows // 3, len(candidates))
    top_indices = np.argsort(-scores)[:top_count]
    required = np.unique(np.concatenate([positive_indices, top_indices]))

    if required.size >= max_rows:
        selected = rng.choice(required, size=max_rows, replace=False)
    else:
        remaining = np.setdiff1d(np.arange(len(candidates)), required, assume_unique=False)
        random_count = max_rows - required.size
        random_indices = rng.choice(remaining, size=random_count, replace=False)
        selected = np.concatenate([required, random_indices])
    selected = np.sort(selected)
    return {
        "candidates": candidates.iloc[selected].reset_index(drop=True),
        "features": features.iloc[selected].reset_index(drop=True),
        "scores": scores[selected],
    }


def positive_class_shap_values(values: object) -> np.ndarray:
    """Return SHAP values for the positive class as a 2D array."""

    if isinstance(values, list):
        return np.asarray(values[-1], dtype=np.float64)
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 3:
        return array[:, :, -1]
    if array.ndim != 2:
        raise ValueError(f"Unexpected SHAP value shape: {array.shape}")
    return array


def positive_class_expected_value(value: object) -> float:
    """Return expected value for the positive class."""

    if isinstance(value, list):
        return float(value[-1])
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        return float(array)
    return float(array.reshape(-1)[-1])


def global_importance(shap_values: np.ndarray, feature_columns: list[str]) -> pd.DataFrame:
    """Return mean absolute SHAP importance."""

    return pd.DataFrame(
        {
            "feature": feature_columns,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            "mean_shap": shap_values.mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)


def write_sample_shap_rows(
    path: Path,
    candidates: pd.DataFrame,
    scores: np.ndarray,
    shap_values: np.ndarray,
    feature_columns: list[str],
) -> None:
    """Write metadata, model score, and SHAP values for sampled rows."""

    output = candidates[METADATA_COLUMNS].copy()
    output["model_score"] = scores
    for index, feature in enumerate(feature_columns):
        output[f"shap_{feature}"] = shap_values[:, index]
    output.to_csv(path, index=False)


def save_beeswarm(
    shap_values: np.ndarray,
    features: pd.DataFrame,
    feature_columns: list[str],
    path: Path,
) -> None:
    """Save a global SHAP beeswarm plot."""

    explanation = shap.Explanation(
        values=shap_values,
        data=features.to_numpy(),
        feature_names=feature_columns,
    )
    plt.figure()
    shap.plots.beeswarm(explanation, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def save_local_explanations(
    local_dir: Path,
    candidates: pd.DataFrame,
    features: pd.DataFrame,
    scores: np.ndarray,
    shap_values: np.ndarray,
    feature_columns: list[str],
    expected_value: float,
    count: int,
) -> list[dict[str, object]]:
    """Save local bar explanations for high-risk candidates."""

    selected = np.argsort(-scores)[:count]
    records: list[dict[str, object]] = []
    for rank, row_index in enumerate(selected):
        row = candidates.iloc[row_index]
        explanation = shap.Explanation(
            values=shap_values[row_index],
            base_values=expected_value,
            data=features.iloc[row_index].to_numpy(),
            feature_names=feature_columns,
        )
        output_path = local_dir / f"top_{rank:02d}_{row['event_id']}_{row['target_date']}.png"
        plt.figure()
        shap.plots.bar(explanation, max_display=12, show=False)
        plt.tight_layout()
        plt.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close()
        records.append(
            {
                "rank": int(rank),
                "event_id": str(row["event_id"]),
                "target_date": str(row["target_date"]),
                "row": int(row["row"]),
                "col": int(row["col"]),
                "label": int(row["label"]),
                "model_score": float(scores[row_index]),
                "plot": str(output_path),
            }
        )
    return records


def save_paired_examples(
    output_dir: Path,
    candidates: pd.DataFrame,
    scores: np.ndarray,
    shap_values: np.ndarray,
    feature_columns: list[str],
) -> list[dict[str, object]]:
    """Save paired examples with similar distance but different model risk."""

    frame = candidates.copy()
    frame["model_score"] = scores
    frame["_sample_index"] = np.arange(len(frame))
    pairs: list[dict[str, object]] = []
    for _, group in frame.groupby(["event_id", "target_date"], sort=False):
        if group["label"].nunique() < 2 or len(group) < 10:
            continue
        positives = group[group["label"] == 1].sort_values("model_score", ascending=False)
        negatives = group[group["label"] == 0].sort_values("model_score", ascending=True)
        for _, positive in positives.head(5).iterrows():
            distance = float(positive["distance_to_current_fire_px"])
            nearby_negatives = negatives[
                (negatives["distance_to_current_fire_px"] - distance).abs() <= 2.0
            ]
            if nearby_negatives.empty:
                continue
            negative = nearby_negatives.iloc[0]
            pairs.append(
                {
                    "event_id": str(positive["event_id"]),
                    "target_date": str(positive["target_date"]),
                    "positive_row": int(positive["row"]),
                    "positive_col": int(positive["col"]),
                    "positive_score": float(positive["model_score"]),
                    "negative_row": int(negative["row"]),
                    "negative_col": int(negative["col"]),
                    "negative_score": float(negative["model_score"]),
                    "distance_px": distance,
                    "score_gap": float(positive["model_score"] - negative["model_score"]),
                    "positive_top_shap": top_shap_features(
                        shap_values[int(positive["_sample_index"])], feature_columns
                    ),
                    "negative_top_shap": top_shap_features(
                        shap_values[int(negative["_sample_index"])], feature_columns
                    ),
                }
            )
            break
        if len(pairs) >= 8:
            break

    pair_path = output_dir / "paired_similar_distance_examples.json"
    pair_path.write_text(json.dumps(pairs, indent=2) + "\n", encoding="utf-8")
    return pairs


def top_shap_features(
    values: np.ndarray,
    feature_columns: list[str],
    count: int = 5,
) -> list[dict[str, float | str]]:
    """Return strongest local SHAP contributions."""

    indices = np.argsort(-np.abs(values))[:count]
    return [
        {"feature": feature_columns[index], "shap_value": float(values[index])} for index in indices
    ]


if __name__ == "__main__":
    main()
