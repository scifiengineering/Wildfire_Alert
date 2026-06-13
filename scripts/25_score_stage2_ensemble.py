"""Average probability scores from multiple frozen Stage 2 models."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Score candidates with a Stage 2 ensemble.")
    parser.add_argument("--models", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    if len(args.models) < 2:
        raise ValueError("Provide at least two --models.")

    bundles = [joblib.load(path) for path in args.models]
    expected_features: list[str] | None = None
    for model_path, bundle in zip(args.models, bundles, strict=True):
        feature_columns = list(bundle["feature_columns"])
        if expected_features is None:
            expected_features = feature_columns
        elif feature_columns != expected_features:
            raise ValueError(f"Feature columns differ in {model_path}.")
    assert expected_features is not None

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    candidate_count = 0
    score_sum = 0.0
    maximum_score = 0.0
    first_chunk = True
    for candidates in pd.read_csv(args.candidate_csv, chunksize=100_000):
        features = (
            candidates[expected_features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        )
        model_scores = [
            bundle["model"].predict_proba(features)[:, 1] for bundle in bundles
        ]
        scored = candidates.copy()
        for index, scores in enumerate(model_scores):
            scored[f"model_score_{index}"] = scores
        scored["model_score"] = np.mean(np.stack(model_scores), axis=0)
        scored.to_csv(
            args.output_csv,
            index=False,
            mode="w" if first_chunk else "a",
            header=first_chunk,
        )
        first_chunk = False
        candidate_count += len(scored)
        score_sum += float(scored["model_score"].sum())
        maximum_score = max(maximum_score, float(scored["model_score"].max()))

    summary = {
        "models": [str(path) for path in args.models],
        "candidate_csv": str(args.candidate_csv),
        "output_csv": str(args.output_csv),
        "candidate_count": candidate_count,
        "ensemble_size": len(args.models),
        "mean_score": score_sum / candidate_count,
        "maximum_score": maximum_score,
    }
    args.output_csv.with_name(f"{args.output_csv.stem}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
