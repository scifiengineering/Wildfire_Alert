"""Generate ordered visualizations for the wildfire alerting pipeline."""

import argparse
import json
from pathlib import Path

import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio.warp


def main() -> None:
    """Generate Stage 1 and Stage 2 presentation figures."""

    parser = argparse.ArgumentParser(description="Generate final pipeline visualizations.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("outputs/stage2/models/stage2_gbm_r2km_fair_fold1_to_fold0.joblib"),
    )
    parser.add_argument(
        "--candidate-csv",
        type=Path,
        default=Path("outputs/stage2/candidates_r2km_fair/stage2_fold0_validation_candidates.csv"),
    )
    parser.add_argument(
        "--evaluation-json",
        type=Path,
        default=Path(
            "outputs/stage2/evaluation/stage2_gbm_r2km_fair_fold1_to_fold0_evaluation.json"
        ),
    )
    parser.add_argument(
        "--shap-summary",
        type=Path,
        default=Path("outputs/stage2/shap/stage2_gbm_r2km_fair_fold1_to_fold0/summary.json"),
    )
    parser.add_argument(
        "--shap-importance",
        type=Path,
        default=Path(
            "outputs/stage2/shap/stage2_gbm_r2km_fair_fold1_to_fold0/global_shap_importance.csv"
        ),
    )
    parser.add_argument(
        "--paired-json",
        type=Path,
        default=Path(
            "outputs/stage2/shap/stage2_gbm_r2km_fair_fold1_to_fold0/"
            "paired_similar_distance_examples.json"
        ),
    )
    parser.add_argument(
        "--stage1-figure",
        type=Path,
        default=Path(
            "outputs/figures/stage1_qualitative_calibrated_ranked/"
            "fold1_01_fire_23159637_2019-08-17.png"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/figures/final_pipeline"),
    )
    parser.add_argument(
        "--hdf5-root",
        type=Path,
        default=Path("wsts_hdf5"),
        help="HDF5 dataset root used to recover CRS/transform metadata.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates, scores = load_candidates_with_scores(args.candidate_csv, args.model)
    evaluation = read_json(args.evaluation_json)
    shap_summary = read_json(args.shap_summary)
    shap_importance = pd.read_csv(args.shap_importance)
    paired = read_json(args.paired_json)

    outputs = [
        draw_pipeline_diagram(args.output_dir / "01_pipeline_overview.png"),
        copy_stage1_summary(args.stage1_figure, args.output_dir / "02_stage1_qualitative.png"),
        draw_baseline_comparison(
            evaluation,
            args.output_dir / "03_stage2_baseline_comparison.png",
        ),
        draw_shap_importance(
            shap_importance,
            args.output_dir / "04_feature_engineering_shap_importance.png",
        ),
        draw_alert_map(
            candidates,
            scores,
            args.hdf5_root,
            args.output_dir / "05_stage2_candidate_alert_map.png",
        ),
        draw_paired_discrimination(
            paired,
            args.output_dir / "06_similar_distance_discrimination.png",
        ),
    ]
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"figures": [str(path) for path in outputs]}, indent=2))

    summary_path = args.output_dir / "visualization_summary.md"
    summary_path.write_text(build_summary(outputs, shap_summary), encoding="utf-8")
    print(f"saved {len(outputs)} visualizations to {args.output_dir}")
    print(f"saved summary to {summary_path}")


def load_candidates_with_scores(
    candidate_csv: Path, model_path: Path
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load candidate rows and score them with the saved Stage 2 model."""

    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_columns = list(bundle["feature_columns"])
    candidates = pd.read_csv(candidate_csv)
    features = candidates[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scores = model.predict_proba(features)[:, 1]
    candidates = candidates.copy()
    candidates["model_score"] = scores
    return candidates, scores


def read_json(path: Path) -> dict[str, object] | list[object]:
    """Read a JSON artifact."""

    return json.loads(path.read_text(encoding="utf-8"))


def draw_pipeline_diagram(path: Path) -> Path:
    """Draw an ordered Stage 1 -> Stage 2 pipeline diagram."""

    fig, ax = plt.subplots(figsize=(15, 4))
    ax.axis("off")
    boxes = [
        ("5-day WSTS rasters\nweather, terrain, vegetation,\ndrought, active fire", 0.08),
        ("Stage 1 U-Net\nnext-day probability map", 0.29),
        ("Candidate extraction\n2 km current-fire ring\n+ Stage 1 top-risk cells", 0.50),
        ("Stage 2 LightGBM\ncandidate risk scores", 0.70),
        ("SHAP explanations\nwhy this location?", 0.90),
    ]
    for text, x_pos in boxes:
        ax.text(
            x_pos,
            0.55,
            text,
            ha="center",
            va="center",
            fontsize=11,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "#eef5ff", "edgecolor": "#2f5f9f"},
            transform=ax.transAxes,
        )
    for x_start, x_end in zip([0.17, 0.38, 0.59, 0.80], [0.23, 0.44, 0.64, 0.85], strict=True):
        ax.annotate(
            "",
            xy=(x_end, 0.55),
            xytext=(x_start, 0.55),
            arrowprops={"arrowstyle": "->", "lw": 2, "color": "#444"},
            xycoords=ax.transAxes,
        )
    ax.set_title("Wildfire Discriminative Alerting Pipeline", fontsize=15, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def copy_stage1_summary(source: Path, target: Path) -> Path:
    """Copy the selected Stage 1 qualitative figure into the final sequence."""

    import shutil

    shutil.copyfile(source, target)
    return target


def draw_baseline_comparison(evaluation: dict[str, object] | list[object], path: Path) -> Path:
    """Draw AP and within-event AUC comparison against baselines."""

    if not isinstance(evaluation, dict):
        raise TypeError("evaluation must be a dict")
    summary = evaluation["summary"]
    if not isinstance(summary, dict):
        raise TypeError("evaluation summary must be a dict")

    labels = ["Stage 2 model", "Distance only", "Stage 1 only"]
    ap_values = [
        float(summary["model_average_precision"]),
        float(summary["distance_baseline_average_precision"]),
        float(summary["stage1_baseline_average_precision"]),
    ]
    auc_values = [
        float(summary["model_within_event_auc"]),
        float(summary["distance_baseline_within_event_auc"]),
        float(summary["stage1_baseline_within_event_auc"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = ["#1f77b4", "#999999", "#ff7f0e"]
    axes[0].bar(labels, ap_values, color=colors)
    axes[0].set_title("Candidate Ranking AP")
    axes[0].set_ylim(0, max(ap_values) * 1.25)
    axes[0].set_ylabel("Average Precision")
    axes[1].bar(labels, auc_values, color=colors)
    axes[1].set_title("Within-Event Discrimination AUC")
    axes[1].set_ylim(0.45, 0.9)
    for axis in axes:
        axis.tick_params(axis="x", rotation=25)
        for container in axis.containers:
            axis.bar_label(container, fmt="%.3f")
    fig.suptitle("Feature-Engineered Stage 2 Model vs Simple Baselines", weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def draw_shap_importance(importance: pd.DataFrame, path: Path) -> Path:
    """Draw global SHAP importance for engineered features."""

    top = importance.sort_values("mean_abs_shap", ascending=False).head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["mean_abs_shap"], color="#2ca02c")
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_title("How Feature Engineering Helps Discriminative Alerts", weight="bold")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def draw_alert_map(
    candidates: pd.DataFrame, scores: np.ndarray, hdf5_root: Path, path: Path
) -> Path:
    """Draw a candidate-level alert map for the event-day with highest positive score."""

    frame = candidates.copy()
    frame["model_score"] = scores
    positive_rows = frame[frame["label"] == 1]
    if positive_rows.empty:
        sample = frame.sort_values("model_score", ascending=False).iloc[0]
    else:
        sample = positive_rows.sort_values("model_score", ascending=False).iloc[0]
    group = frame[
        (frame["event_id"] == sample["event_id"]) & (frame["target_date"] == sample["target_date"])
    ].copy()

    future_candidates = group[group["current_fire_at_candidate"] == 0]
    alert_source = future_candidates if not future_candidates.empty else group
    top_alert_cutoff = float(alert_source["model_score"].quantile(0.9))
    high_alerts = alert_source[alert_source["model_score"] >= top_alert_cutoff]
    positives = future_candidates[future_candidates["label"] == 1]
    current_fire = group[group["current_fire_at_candidate"] == 1]
    top_locations = top_alert_location_table(alert_source, hdf5_root, top_n=8)
    top_locations.to_csv(path.with_name("05_stage2_high_alert_locations.csv"), index=False)

    fig = plt.figure(figsize=(22, 7))
    grid = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.35], wspace=0.35)
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[0, 2]),
    ]
    table_axis = fig.add_subplot(grid[0, 3])
    for axis in axes[1:]:
        axis.sharex(axes[0])
        axis.sharey(axes[0])
    distance_plot = axes[0].scatter(
        group["col"],
        group["row"],
        c=group["distance_to_current_fire_km"],
        s=14,
        cmap="viridis",
        alpha=0.9,
    )
    axes[0].scatter(
        current_fire["col"],
        current_fire["row"],
        marker="x",
        c="#1f77b4",
        s=26,
        linewidths=1.5,
        label="current fire",
    )
    axes[0].set_title("1. Candidate Search Area")
    axes[0].text(
        0.5,
        -0.17,
        "Every dot is a location Stage 2 is allowed to rank.\n"
        "Color shows distance from today's active fire.",
        transform=axes[0].transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )
    score_plot = axes[1].scatter(
        group["col"],
        group["row"],
        c=group["model_score"],
        s=14,
        cmap="magma",
        vmin=0,
        vmax=1,
    )
    axes[1].scatter(
        high_alerts["col"],
        high_alerts["row"],
        facecolors="none",
        edgecolors="#00e5ff",
        s=42,
        linewidths=0.8,
        label="top 10% new-location alerts",
    )
    axes[1].set_title("2. Model Alert Score")
    axes[1].text(
        0.5,
        -0.17,
        "Brighter points are locations the Stage 2 model\n"
        "believes are more likely to burn tomorrow.",
        transform=axes[1].transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )
    axes[2].scatter(group["col"], group["row"], c="#d0d0d0", s=10, label="candidate")
    axes[2].scatter(
        high_alerts["col"],
        high_alerts["row"],
        facecolors="none",
        edgecolors="#00e5ff",
        s=42,
        linewidths=0.8,
        label="top 10% new-location alerts",
    )
    axes[2].scatter(
        positives["col"],
        positives["row"],
        c="red",
        s=24,
        label="burned tomorrow",
        zorder=3,
    )
    axes[2].set_title("3. Next-Day Ground Truth")
    axes[2].text(
        0.5,
        -0.17,
        "Red points are the candidates that actually burned.\n"
        "Current-fire pixels are excluded to focus on spread.",
        transform=axes[2].transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )
    for axis in axes:
        axis.invert_yaxis()
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("raster column (pixel x-position)")
        axis.set_ylabel("raster row (pixel y-position)")
        axis.grid(alpha=0.2, linewidth=0.5)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[1].legend(loc="upper right", fontsize=8)
    axes[2].legend(loc="upper right", fontsize=8)
    draw_high_alert_table(table_axis, top_locations)
    fig.colorbar(
        distance_plot,
        ax=axes[0],
        fraction=0.046,
        pad=0.04,
        label="distance to current fire (km)",
    )
    fig.colorbar(score_plot, ax=axes[1], fraction=0.046, pad=0.04, label="Stage 2 risk score")
    summary = (
        f"{len(group):,} candidate pixels | {len(positives):,} new burned pixels | "
        f"new-location alert cutoff >= {top_alert_cutoff:.2f}"
    )
    fig.suptitle(
        (
            f"Figure 5. How Stage 2 Turns Candidate Pixels into Next-Day Fire Alerts\n"
            f"{sample['event_id']} -> {sample['target_date']} | {summary}"
        ),
        weight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        0.01,
        "Rows and columns are raster/image coordinates, not latitude-longitude. "
        "They show where each candidate pixel sits inside the WSTS event patch.",
        ha="center",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f5f5f5", "edgecolor": "#cccccc"},
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def top_alert_location_table(group: pd.DataFrame, hdf5_root: Path, top_n: int) -> pd.DataFrame:
    """Return top alert rows with latitude/longitude coordinates."""

    event_id = str(group["event_id"].iloc[0])
    target_date = str(group["target_date"].iloc[0])
    year = target_date[:4]
    hdf5_path = hdf5_root / year / f"{event_id}.hdf5"
    if not hdf5_path.exists():
        raise FileNotFoundError(f"missing HDF5 event file for coordinate conversion: {hdf5_path}")

    with h5py.File(hdf5_path, "r") as handle:
        dataset = handle["data"]
        crs = str(dataset.attrs["crs"])
        transform = np.asarray(dataset.attrs["transform"], dtype=float)

    top = group.sort_values("model_score", ascending=False).head(top_n).copy()
    lon, lat = rows_cols_to_lon_lat(
        top["row"].to_numpy(dtype=float),
        top["col"].to_numpy(dtype=float),
        transform,
        crs,
    )
    table = pd.DataFrame(
        {
            "rank": np.arange(1, len(top) + 1),
            "event_id": event_id,
            "target_date": target_date,
            "row": top["row"].to_numpy(dtype=int),
            "col": top["col"].to_numpy(dtype=int),
            "latitude": lat,
            "longitude": lon,
            "risk_score": top["model_score"].to_numpy(dtype=float),
            "distance_km": top["distance_to_current_fire_km"].to_numpy(dtype=float),
            "burned_tomorrow": top["label"].to_numpy(dtype=int),
        }
    )
    return table


def rows_cols_to_lon_lat(
    rows: np.ndarray, cols: np.ndarray, transform: np.ndarray, crs: str
) -> tuple[list[float], list[float]]:
    """Convert raster row/column pixel centers to longitude/latitude."""

    x_coords = transform[0] * (cols + 0.5) + transform[1] * (rows + 0.5) + transform[2]
    y_coords = transform[3] * (cols + 0.5) + transform[4] * (rows + 0.5) + transform[5]
    lon, lat = rasterio.warp.transform(crs, "EPSG:4326", x_coords.tolist(), y_coords.tolist())
    return lon, lat


def draw_high_alert_table(axis: plt.Axes, top_locations: pd.DataFrame) -> None:
    """Draw a latitude/longitude table of top Stage 2 alert locations."""

    axis.axis("off")
    display = top_locations.copy()
    display["latitude"] = display["latitude"].map(lambda value: f"{value:.5f}")
    display["longitude"] = display["longitude"].map(lambda value: f"{value:.5f}")
    display["risk_score"] = display["risk_score"].map(lambda value: f"{value:.3f}")
    display["burned_tomorrow"] = display["burned_tomorrow"].map({1: "yes", 0: "no"})
    display = display[["rank", "latitude", "longitude", "risk_score", "burned_tomorrow"]]
    display.columns = ["Rank", "Latitude", "Longitude", "Risk", "Burned?"]

    table = axis.table(
        cellText=display.values,
        colLabels=display.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.05, 1.6)
    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#dceeff")
            cell.set_text_props(weight="bold")
        elif display.iloc[row - 1]["Burned?"] == "yes":
            cell.set_facecolor("#ffe0e0")
    axis.set_title("4. Top Alert Locations\nGPS Coordinates", weight="bold")
    axis.text(
        0.5,
        0.03,
        "These are the highest-risk new-location alerts converted\n"
        "from raster pixels to latitude/longitude.",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
    )


def draw_paired_discrimination(paired: dict[str, object] | list[object], path: Path) -> Path:
    """Draw a paired similar-distance example showing feature-level explanation."""

    if not isinstance(paired, list) or not paired:
        raise ValueError("paired examples must be a non-empty list")
    pair = max(paired, key=lambda item: float(item["score_gap"]) if isinstance(item, dict) else 0)
    if not isinstance(pair, dict):
        raise TypeError("paired example must be a dict")

    pos_features = pair["positive_top_shap"]
    neg_features = pair["negative_top_shap"]
    if not isinstance(pos_features, list) or not isinstance(neg_features, list):
        raise TypeError("paired SHAP features must be lists")
    pos = pd.DataFrame(pos_features)
    neg = pd.DataFrame(neg_features)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].bar(
        ["burned", "not burned"],
        [pair["positive_score"], pair["negative_score"]],
        color=["red", "gray"],
    )
    axes[0].set_title("Similar Distance, Different Risk")
    axes[0].set_ylabel("Stage 2 risk score")
    axes[0].text(
        0.5,
        max(float(pair["positive_score"]), float(pair["negative_score"])) * 0.85,
        (
            f"distance ~= {float(pair['distance_px']):.1f} px\n"
            f"score gap = {float(pair['score_gap']):.3f}"
        ),
        ha="center",
    )
    axes[1].barh(pos["feature"], pos["shap_value"], color="#d62728")
    axes[1].set_title("Burned Candidate: Top SHAP")
    axes[2].barh(neg["feature"], neg["shap_value"], color="#777777")
    axes[2].set_title("Non-Burned Candidate: Top SHAP")
    for axis in axes[1:]:
        axis.axvline(0, color="black", linewidth=1)
        axis.set_xlabel("SHAP value")
    fig.suptitle(
        f"Paired Discriminative Example: {pair['event_id']} -> {pair['target_date']}",
        weight="bold",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def build_summary(paths: list[Path], shap_summary: dict[str, object] | list[object]) -> str:
    """Build a small markdown index for the generated visualization sequence."""

    if isinstance(shap_summary, dict):
        top_features = [
            str(row["feature"])
            for row in shap_summary.get("top_global_features", [])[:8]
            if isinstance(row, dict)
        ]
    else:
        top_features = []
    lines = [
        "# Final Pipeline Visualization Index",
        "",
        "These figures are ordered in the same order as the pipeline.",
        "",
        "## Generated Figures",
        "",
    ]
    lines.extend(f"{index}. `{path}`" for index, path in enumerate(paths, start=1))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Stage 1 turns five days of raster context into a next-day probability map.",
            "- Stage 2 turns candidate locations into alert scores.",
            "- Feature engineering helps because the model sees distance, wind, terrain, "
            "drought, vegetation, and Stage 1 risk together.",
            "- The SHAP figures show which engineered features influenced the alert scores.",
            "",
            f"Top SHAP features in the selected model: {', '.join(top_features)}.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
