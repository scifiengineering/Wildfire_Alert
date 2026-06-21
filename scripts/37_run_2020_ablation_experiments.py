"""Run 2020 test-set Stage 2 feature ablations for Tanisha follow-up."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
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
BASELINE_FEATURES = [
    "stage1_probability",
    "stage1_threshold_mask",
    "stage1_top_mask",
    "threshold",
    "top_fraction",
    "top_threshold",
    "has_current_fire",
    "current_fire_at_candidate",
    "distance_to_current_fire_px",
    "distance_to_current_fire_km",
    "bearing_from_fire_deg",
]
FEATURE_GROUPS = {
    "fire_geometry_features": [
        "has_current_fire",
        "current_fire_at_candidate",
        "distance_to_current_fire_px",
        "distance_to_current_fire_km",
        "bearing_from_fire_deg",
    ],
    "wind_features": [
        "wind_alignment",
        "forecast_wind_alignment",
        "slope_alignment",
        "wind_speed",
        "wind_direction",
        "forecast_wind_speed",
        "forecast_wind_direction",
    ],
    "weather_drought_features": [
        "total_precipitation",
        "minimum_temperature",
        "maximum_temperature",
        "energy_release_component",
        "specific_humidity",
        "pdsi",
        "forecast_total_precipitation",
        "forecast_temperature",
        "forecast_specific_humidity",
    ],
    "terrain_vegetation_features": [
        "ndvi",
        "evi2",
        "slope",
        "aspect",
        "elevation",
        "landcover_class",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fold-csvs",
        type=Path,
        nargs="+",
        default=[
            Path(
                "outputs/stage2/candidates_3fold_r4km_fair_tanisha_ckpt/"
                "stage2_fold0_validation_candidates.csv"
            ),
            Path(
                "outputs/stage2/candidates_3fold_r4km_fair_tanisha_ckpt/"
                "stage2_fold1_validation_candidates.csv"
            ),
            Path(
                "outputs/stage2/candidates_3fold_r4km_fair_tanisha_ckpt/"
                "stage2_fold2_validation_candidates.csv"
            ),
        ],
    )
    parser.add_argument(
        "--full-models",
        type=Path,
        nargs="+",
        default=[
            Path(
                "outputs/stage2/models_3fold_r4km_tanisha_ckpt/"
                "stage2_gbm_3fold_r4km_tanisha_ckpt_holdout0.joblib"
            ),
            Path(
                "outputs/stage2/models_3fold_r4km_tanisha_ckpt/"
                "stage2_gbm_3fold_r4km_tanisha_ckpt_holdout1.joblib"
            ),
            Path(
                "outputs/stage2/models_3fold_r4km_tanisha_ckpt/"
                "stage2_gbm_3fold_r4km_tanisha_ckpt_holdout2.joblib"
            ),
        ],
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=Path(
            "outputs/stage2/candidates_2020_r4km_fair_tanisha_ckpt_full2020/"
            "stage2_2020_candidates.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/tanisha_followup_experiments/2020_ablation"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_frames = [pd.read_csv(path) for path in args.fold_csvs]
    full_feature_columns = feature_column_names(fold_frames[0])
    variants = build_variants(full_feature_columns)

    records = []
    full_scores = score_ensemble(args.full_models, args.test_csv)
    test_labels, test_groups = read_test_labels(args.test_csv)
    full_metrics = compute_metrics(test_labels, test_groups, full_scores)
    records.append({"variant": "full_features", **full_metrics})

    models_dir = args.output_dir / "models"
    models_dir.mkdir(exist_ok=True)
    for variant_name, feature_columns in variants.items():
        model_paths = train_variant_models(
            variant_name=variant_name,
            fold_frames=fold_frames,
            feature_columns=feature_columns,
            output_dir=models_dir,
            seed=args.seed,
        )
        variant_scores = score_ensemble(model_paths, args.test_csv)
        records.append(
            {"variant": variant_name, **compute_metrics(test_labels, test_groups, variant_scores)}
        )

    table = pd.DataFrame(records)
    full_ap = float(table.loc[table["variant"] == "full_features", "average_precision"].iloc[0])
    full_auc = float(table.loc[table["variant"] == "full_features", "within_event_auc"].iloc[0])
    table["ap_drop_vs_full"] = full_ap - table["average_precision"]
    table["within_event_auc_drop_vs_full"] = full_auc - table["within_event_auc"]
    table["ap_drop_percent_vs_full"] = percentage_drop(full_ap, table["average_precision"])
    table["within_event_auc_drop_percent_vs_full"] = percentage_drop(
        full_auc, table["within_event_auc"]
    )
    table = table.sort_values("ap_drop_vs_full", ascending=False)
    table.to_csv(args.output_dir / "2020_ablation_metrics.csv", index=False)
    (args.output_dir / "2020_ablation_metrics.json").write_text(
        json.dumps(table.to_dict(orient="records"), indent=2) + "\n",
        encoding="utf-8",
    )
    print(table.to_string(index=False))


def feature_column_names(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column not in NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(frame[column])
    ]


def build_variants(full_feature_columns: list[str]) -> dict[str, list[str]]:
    variants = {
        "no_extra_features": [
            feature for feature in BASELINE_FEATURES if feature in full_feature_columns
        ]
    }
    for group_name, group_features in FEATURE_GROUPS.items():
        variants[f"drop_{group_name}"] = [
            feature for feature in full_feature_columns if feature not in group_features
        ]
    return variants


def train_variant_models(
    variant_name: str,
    fold_frames: list[pd.DataFrame],
    feature_columns: list[str],
    output_dir: Path,
    seed: int,
) -> list[Path]:
    model_paths = []
    for holdout_index, validation in enumerate(fold_frames):
        train = pd.concat(
            [frame for index, frame in enumerate(fold_frames) if index != holdout_index],
            ignore_index=True,
        )
        x_train = train[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y_train = train["label"].astype(int)
        x_validation = validation[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y_validation = validation["label"].astype(int)

        model = lgb.LGBMClassifier(
            objective="binary",
            class_weight="balanced",
            n_estimators=1000,
            num_leaves=63,
            learning_rate=0.03,
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_validation, y_validation)],
            eval_metric="average_precision",
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(period=0)],
        )
        model_path = output_dir / f"stage2_2020_{variant_name}_holdout{holdout_index}.joblib"
        joblib.dump({"model": model, "feature_columns": feature_columns}, model_path)
        model_paths.append(model_path)
    return model_paths


def score_ensemble(model_paths: list[Path], test_csv: Path) -> np.ndarray:
    bundles = [joblib.load(path) for path in model_paths]
    feature_columns = list(bundles[0]["feature_columns"])
    scores: list[np.ndarray] = []
    for chunk in pd.read_csv(test_csv, chunksize=200_000):
        features = chunk[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        chunk_scores = [
            bundle["model"].predict_proba(features)[:, 1] for bundle in bundles
        ]
        scores.append(np.mean(np.stack(chunk_scores), axis=0))
    return np.concatenate(scores)


def read_test_labels(test_csv: Path) -> tuple[np.ndarray, pd.DataFrame]:
    groups = []
    labels = []
    for chunk in pd.read_csv(
        test_csv,
        usecols=["event_id", "target_date", "label"],
        chunksize=200_000,
    ):
        groups.append(chunk[["event_id", "target_date"]].copy())
        labels.append(chunk["label"].astype(int).to_numpy())
    return np.concatenate(labels), pd.concat(groups, ignore_index=True)


def compute_metrics(
    labels: np.ndarray,
    groups: pd.DataFrame,
    scores: np.ndarray,
) -> dict[str, float]:
    return {
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": safe_auc(labels, scores),
        "within_event_auc": within_event_auc(labels, groups, scores),
    }


def within_event_auc(labels: np.ndarray, groups: pd.DataFrame, scores: np.ndarray) -> float:
    scored = groups.copy()
    scored["label"] = labels
    scored["score"] = scores
    aucs = []
    for _, group in scored.groupby(["event_id", "target_date"], sort=False):
        group_labels = group["label"].to_numpy()
        if np.unique(group_labels).size < 2:
            continue
        aucs.append(float(roc_auc_score(group_labels, group["score"].to_numpy())))
    return float(np.mean(aucs)) if aucs else 0.0


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if np.unique(labels).size < 2:
        return 0.0
    return float(roc_auc_score(labels, scores))


def percentage_drop(full_value: float, variant_values: pd.Series) -> pd.Series:
    if full_value == 0:
        return pd.Series(np.zeros(len(variant_values)), index=variant_values.index)
    return (full_value - variant_values) / full_value * 100.0


if __name__ == "__main__":
    main()
