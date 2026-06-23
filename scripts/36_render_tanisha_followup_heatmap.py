"""Render Tanisha follow-up 2020 test-set SHAP heatmap."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import joblib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio.warp

FEATURE_COLUMNS = [
    "Fire geometry",
    "Wind + alignment",
    "Weather + drought",
    "Terrain + veg",
]
METADATA_COLUMNS = ["event_id", "target_date", "row", "col", "label", "model_score"]
FEATURE_GROUPS = {
    "Fire geometry": [
        "has_current_fire",
        "current_fire_at_candidate",
        "distance_to_current_fire_px",
        "distance_to_current_fire_km",
        "bearing_from_fire_deg",
    ],
    "Wind + alignment": [
        "wind_alignment",
        "forecast_wind_alignment",
        "slope_alignment",
        "wind_speed",
        "wind_direction",
        "forecast_wind_speed",
        "forecast_wind_direction",
    ],
    "Weather + drought": [
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
    "Terrain + veg": [
        "ndvi",
        "evi2",
        "slope",
        "aspect",
        "elevation",
        "landcover_class",
    ],
}
MODEL_SOURCE = "stage2_gbm_3fold_r4km_imagenet_noamp_ensemble"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scored-candidate-csv",
        type=Path,
        default=Path(
            "outputs/stage2/scored_2020_r4km_fair_imagenet_noamp_full2020_retrained_stage2/"
            "stage2_2020_scored_candidates.csv"
        ),
    )
    parser.add_argument(
        "--models",
        type=Path,
        nargs="+",
        default=[
            Path(
                "outputs/stage2/models_3fold_r4km_imagenet_noamp/"
                "stage2_gbm_3fold_r4km_imagenet_noamp_holdout0.joblib"
            ),
            Path(
                "outputs/stage2/models_3fold_r4km_imagenet_noamp/"
                "stage2_gbm_3fold_r4km_imagenet_noamp_holdout1.joblib"
            ),
            Path(
                "outputs/stage2/models_3fold_r4km_imagenet_noamp/"
                "stage2_gbm_3fold_r4km_imagenet_noamp_holdout2.joblib"
            ),
        ],
    )
    parser.add_argument("--hdf5-root", type=Path, default=Path("wsts_hdf5"))
    parser.add_argument(
        "--source-output-csv",
        type=Path,
        default=Path(
            "outputs/tanisha_followup_experiments/figures/"
            "01_top5_added_feature_group_shap.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/tanisha_followup_experiments/figures/"
            "01_top5_added_feature_shap_heatmap.png"
        ),
    )
    args = parser.parse_args()

    reject_tanisha_checkpoint_paths([args.scored_candidate_csv, *args.models])
    table = build_grouped_contribution_table(
        scored_candidate_csv=args.scored_candidate_csv,
        model_paths=args.models,
        hdf5_root=args.hdf5_root,
    )
    args.source_output_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.source_output_csv, index=False)
    render_heatmap(table, args.output)


def build_grouped_contribution_table(
    scored_candidate_csv: Path,
    model_paths: list[Path],
    hdf5_root: Path,
    top_n: int = 5,
) -> pd.DataFrame:
    """Return top 2020 test alerts with grouped LightGBM contribution values."""

    bundles = [joblib.load(path) for path in model_paths]
    feature_columns = list(bundles[0]["feature_columns"])
    for model_path, bundle in zip(model_paths, bundles, strict=True):
        if list(bundle["feature_columns"]) != feature_columns:
            raise ValueError(f"feature columns differ in {model_path}")

    top_rows = top_unique_event_rows(scored_candidate_csv, feature_columns, top_n)
    features = top_rows[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    contribution_stack = [
        bundle["model"].predict(features, pred_contrib=True)[:, :-1] for bundle in bundles
    ]
    contributions = np.mean(np.stack(contribution_stack), axis=0)
    contribution_frame = pd.DataFrame(contributions, columns=feature_columns)

    lon, lat = rows_cols_to_lon_lat(top_rows, hdf5_root)
    output = top_rows[METADATA_COLUMNS].copy()
    output.insert(0, "model_source", MODEL_SOURCE)
    output["latitude"] = lat
    output["longitude"] = lon
    output["burned_tomorrow"] = output.pop("label").astype(int)
    for group_name, group_features in FEATURE_GROUPS.items():
        missing = sorted(set(group_features) - set(contribution_frame.columns))
        if missing:
            raise ValueError(f"missing features for {group_name}: {missing}")
        output[group_name] = contribution_frame[group_features].sum(axis=1)
    output["Added feature total"] = output[FEATURE_COLUMNS].sum(axis=1)
    return output[
        [
            "model_source",
            "event_id",
            "target_date",
            "row",
            "col",
            "latitude",
            "longitude",
            "model_score",
            "burned_tomorrow",
            *FEATURE_COLUMNS,
            "Added feature total",
        ]
    ]


def top_unique_event_rows(
    scored_candidate_csv: Path,
    feature_columns: list[str],
    top_n: int,
) -> pd.DataFrame:
    """Find the top-scoring candidate for each event and return the best events."""

    usecols = [*METADATA_COLUMNS, *feature_columns]
    best_by_event: dict[str, pd.Series] = {}
    for chunk in pd.read_csv(scored_candidate_csv, usecols=usecols, chunksize=200_000):
        chunk_best = chunk.sort_values("model_score", ascending=False).groupby("event_id").head(1)
        for row in chunk_best.itertuples(index=False):
            event_id = str(row.event_id)
            if (
                event_id not in best_by_event
                or float(row.model_score) > float(best_by_event[event_id]["model_score"])
            ):
                best_by_event[event_id] = pd.Series(row._asdict())

    if len(best_by_event) < top_n:
        raise ValueError(f"only found {len(best_by_event)} unique events; need {top_n}")
    return (
        pd.DataFrame(best_by_event.values())
        .sort_values("model_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def rows_cols_to_lon_lat(rows: pd.DataFrame, hdf5_root: Path) -> tuple[list[float], list[float]]:
    """Convert selected raster row/column pixel centers to longitude/latitude."""

    lon_values: list[float] = []
    lat_values: list[float] = []
    for row in rows.itertuples(index=False):
        year = str(row.target_date)[:4]
        hdf5_path = hdf5_root / year / f"{row.event_id}.hdf5"
        if not hdf5_path.exists():
            raise FileNotFoundError(f"missing HDF5 event file: {hdf5_path}")
        with h5py.File(hdf5_path, "r") as handle:
            dataset = handle["data"]
            crs = str(dataset.attrs["crs"])
            transform = np.asarray(dataset.attrs["transform"], dtype=float)
        col = float(row.col)
        raster_row = float(row.row)
        x_coord = transform[0] * (col + 0.5) + transform[1] * (raster_row + 0.5) + transform[2]
        y_coord = transform[3] * (col + 0.5) + transform[4] * (raster_row + 0.5) + transform[5]
        lon, lat = rasterio.warp.transform(crs, "EPSG:4326", [x_coord], [y_coord])
        lon_values.append(float(lon[0]))
        lat_values.append(float(lat[0]))
    return lon_values, lat_values


def render_heatmap(table: pd.DataFrame, output_path: Path) -> Path:
    """Render grouped SHAP contributions with a readable risk-score column."""

    shap_values = table[FEATURE_COLUMNS].to_numpy(dtype=float)
    risk_values = table[["model_score"]].to_numpy(dtype=float)
    row_labels = [
        f"{row.event_id.replace('fire_', '')}  {row.target_date}\n"
        f"lat {row.latitude:.4f}, lon {row.longitude:.4f}"
        for row in table.itertuples(index=False)
    ]

    max_abs_shap = float(np.nanmax(np.abs(shap_values)))
    shap_norm = mcolors.TwoSlopeNorm(vmin=-max_abs_shap, vcenter=0.0, vmax=max_abs_shap)
    risk_norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    shap_cmap = plt.get_cmap("RdBu_r")
    risk_cmap = mcolors.LinearSegmentedColormap.from_list(
        "readable_risk",
        ["#eef7fb", "#9ad0e5", "#3b82b8", "#08306b"],
    )

    fig = plt.figure(figsize=(13.0, 6.4))
    grid = fig.add_gridspec(1, 3, width_ratios=[4.0, 0.42, 0.14], wspace=0.08)
    shap_axis = fig.add_subplot(grid[0, 0])
    risk_axis = fig.add_subplot(grid[0, 1], sharey=shap_axis)
    shap_cbar_axis = fig.add_subplot(grid[0, 2])

    shap_image = shap_axis.imshow(shap_values, cmap=shap_cmap, norm=shap_norm, aspect="auto")
    risk_axis.imshow(risk_values, cmap=risk_cmap, norm=risk_norm, aspect="auto")

    shap_axis.set_xticks(np.arange(len(FEATURE_COLUMNS)), FEATURE_COLUMNS)
    shap_axis.set_yticks(np.arange(len(row_labels)), row_labels)
    risk_axis.set_xticks([0], ["Risk"])
    risk_axis.tick_params(axis="y", left=False, labelleft=False)

    for axis in (shap_axis, risk_axis):
        axis.set_xticks(np.arange(-0.5, axis.images[0].get_array().shape[1], 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=2.0)
        axis.tick_params(which="minor", bottom=False, left=False)
        for spine in axis.spines.values():
            spine.set_visible(False)

    annotate_cells(shap_axis, shap_values, shap_cmap, shap_norm, signed=True)
    annotate_cells(risk_axis, risk_values, risk_cmap, risk_norm, signed=False)

    shap_cbar = fig.colorbar(shap_image, cax=shap_cbar_axis)
    shap_cbar.set_label("SHAP contribution to fire-risk score", rotation=90, labelpad=12)

    shap_axis.set_xlabel("Added feature group")
    risk_axis.set_xlabel("Risk score")
    fig.suptitle(
        "Top 5 2020 Test Fire Events: Added Feature SHAP Contributions",
        weight="bold",
        y=0.98,
    )
    fig.text(
        0.01,
        0.02,
        "Rows use the highest-scoring Stage 2 alert from each of five unique "
        "2020 test-set fire events.",
        fontsize=10,
        color="#444444",
    )
    fig.autofmt_xdate(rotation=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def reject_tanisha_checkpoint_paths(paths: list[Path]) -> None:
    """Prevent active follow-up figures from using Tanisha-checkpoint artifacts."""

    blocked = [str(path) for path in paths if "tanisha_ckpt" in str(path)]
    if blocked:
        joined = "\n  ".join(blocked)
        raise ValueError(
            "Active follow-up figures must use regenerated-checkpoint artifacts, "
            f"not Tanisha-checkpoint paths:\n  {joined}"
        )


def annotate_cells(
    axis: plt.Axes,
    values: np.ndarray,
    cmap: mcolors.Colormap,
    norm: mcolors.Normalize,
    *,
    signed: bool,
) -> None:
    """Write cell values with automatic black/white text contrast."""

    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = float(values[row, col])
            rgba = cmap(norm(value))
            text_color = "#ffffff" if relative_luminance(rgba[:3]) < 0.45 else "#1f1f1f"
            label = f"{value:+.2f}" if signed else f"{value:.3f}"
            axis.text(
                col,
                row,
                label,
                ha="center",
                va="center",
                color=text_color,
                fontsize=10,
                fontweight="bold" if signed else "normal",
            )


def relative_luminance(rgb: tuple[float, float, float]) -> float:
    """Return WCAG-style relative luminance for an RGB triplet in 0..1 space."""

    channels = []
    for channel in rgb:
        if channel <= 0.03928:
            channels.append(channel / 12.92)
        else:
            channels.append(((channel + 0.055) / 1.055) ** 2.4)
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


if __name__ == "__main__":
    main()
