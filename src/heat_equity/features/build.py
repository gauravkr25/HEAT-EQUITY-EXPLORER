"""Build analysis rows by aggregating calibrated raster pixels to grid cells."""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from scipy import ndimage
from shapely.geometry import mapping

from heat_equity.data.landsat import LandsatSceneArrays


def rasterize_grid_labels(
    analysis_grid: gpd.GeoDataFrame,
    *,
    out_shape: tuple[int, int],
    transform: Any,
    raster_crs: str,
) -> np.ndarray:
    """Rasterize grid cells to stable 1-based labels on a source raster grid."""
    projected_grid = analysis_grid.to_crs(raster_crs)
    shapes = (
        (mapping(geometry), index + 1)
        for index, geometry in enumerate(projected_grid.geometry)
        if geometry is not None and not geometry.is_empty
    )
    return rasterize(
        shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype="int32",
        all_touched=False,
    )


def aggregate_scene_to_grid(
    scene: LandsatSceneArrays,
    analysis_grid: gpd.GeoDataFrame,
    *,
    minimum_valid_fraction: float = 0.5,
) -> gpd.GeoDataFrame:
    """Summarize one scene's valid pixels and predictors into analysis-grid cells."""
    if analysis_grid.empty:
        raise ValueError("The analysis grid must contain at least one cell.")
    if "grid_id" not in analysis_grid.columns:
        raise ValueError("The analysis grid must include a stable grid_id column.")
    if not 0 <= minimum_valid_fraction <= 1:
        raise ValueError("minimum_valid_fraction must be between zero and one.")

    scene_crs = scene.grid.get("crs")
    transform_values = scene.grid.get("transform")
    width = scene.grid.get("width")
    height = scene.grid.get("height")
    if scene_crs is None or transform_values is None or width is None or height is None:
        raise ValueError("Scene grid metadata must include CRS, transform, width, and height.")
    if len(transform_values) < 6:
        raise ValueError("Scene transform metadata must contain at least six affine values.")

    try:
        from affine import Affine
    except ImportError as error:
        raise RuntimeError("The affine package is required for raster-grid aggregation.") from error

    scene_transform = Affine(*transform_values[:6])
    cell_labels = rasterize_grid_labels(
        analysis_grid,
        out_shape=(int(height), int(width)),
        transform=scene_transform,
        raster_crs=str(scene_crs),
    )

    expected_counts = np.bincount(cell_labels.ravel(), minlength=len(analysis_grid) + 1)
    valid_counts = np.bincount(
        cell_labels[scene.valid_mask].ravel(), minlength=len(analysis_grid) + 1
    )
    cell_slices = ndimage.find_objects(cell_labels, max_label=len(analysis_grid))
    records: list[dict[str, Any]] = []
    raster_features = {
        "lst_celsius": scene.lst_celsius,
        "ndvi": scene.ndvi,
        **{f"reflectance_{name.lower()}": values for name, values in scene.reflectance.items()},
    }

    for cell_index, (_, cell) in enumerate(analysis_grid.iterrows(), start=1):
        expected_count = int(expected_counts[cell_index])
        valid_count = int(valid_counts[cell_index])
        valid_fraction = valid_count / expected_count if expected_count else 0.0
        if expected_count == 0 or valid_fraction < minimum_valid_fraction:
            continue

        cell_slice = cell_slices[cell_index - 1]
        if cell_slice is None:
            continue
        local_labels = cell_labels[cell_slice]
        pixel_mask = (local_labels == cell_index) & scene.valid_mask[cell_slice]
        record: dict[str, Any] = {
            "grid_id": cell["grid_id"],
            "raster_cell_label": cell_index,
            "valid_pixel_fraction": valid_fraction,
            "x": float(cell.geometry.centroid.x),
            "y": float(cell.geometry.centroid.y),
            "geometry": cell.geometry,
        }
        for feature_name, values in raster_features.items():
            feature_values = values[cell_slice][pixel_mask]
            finite_values = feature_values[np.isfinite(feature_values)]
            record[feature_name] = (
                float(np.median(finite_values)) if finite_values.size else np.nan
            )
        records.append(record)

    if not records:
        raise ValueError(
            "No grid cells met the valid-pixel threshold. Check scene coverage and QA masks."
        )
    return gpd.GeoDataFrame(records, geometry="geometry", crs=analysis_grid.crs)


def add_land_cover_proportions(
    grid_table: gpd.GeoDataFrame,
    land_cover_values: np.ndarray,
    cell_labels: np.ndarray,
    *,
    class_codes: list[int],
) -> gpd.GeoDataFrame:
    """Add per-cell proportions for listed land-cover classes from aligned raster labels."""
    values = np.asarray(land_cover_values)
    labels = np.asarray(cell_labels)
    if values.shape != labels.shape:
        raise ValueError("Land-cover values and cell-label arrays must have matching shapes.")
    if not class_codes:
        raise ValueError("At least one land-cover class code is required.")

    output = grid_table.copy()
    cell_slices = ndimage.find_objects(labels, max_label=int(labels.max(initial=0)))
    for row_index, row in output.iterrows():
        cell_label = int(row["raster_cell_label"])
        cell_slice = cell_slices[cell_label - 1] if cell_label > 0 else None
        if cell_slice is None:
            for class_code in class_codes:
                output.loc[row_index, f"land_cover_{class_code}_fraction"] = np.nan
            continue
        local_labels = labels[cell_slice]
        local_values = values[cell_slice]
        valid = (local_labels == cell_label) & np.isfinite(local_values)
        denominator = int(valid.sum())
        for class_code in class_codes:
            output.loc[row_index, f"land_cover_{class_code}_fraction"] = (
                float(np.count_nonzero(local_values[valid] == class_code) / denominator)
                if denominator
                else np.nan
            )
    return output


def add_continuous_raster_summary(
    grid_table: gpd.GeoDataFrame,
    raster_values: np.ndarray,
    cell_labels: np.ndarray,
    *,
    feature_name: str,
    statistic: str = "median",
) -> gpd.GeoDataFrame:
    """Add a continuous-raster summary per cell using local raster windows."""
    values = np.asarray(raster_values)
    labels = np.asarray(cell_labels)
    if values.shape != labels.shape:
        raise ValueError("Raster values and cell-label arrays must have matching shapes.")
    if statistic not in {"mean", "median"}:
        raise ValueError("statistic must be 'mean' or 'median'.")

    cell_slices = ndimage.find_objects(labels, max_label=int(labels.max(initial=0)))
    output = grid_table.copy()
    summaries: list[float] = []
    for _, row in output.iterrows():
        cell_label = int(row["raster_cell_label"])
        cell_slice = cell_slices[cell_label - 1] if cell_label > 0 else None
        if cell_slice is None:
            summaries.append(np.nan)
            continue
        local_values = values[cell_slice]
        local_labels = labels[cell_slice]
        local_cell_values = local_values[local_labels == cell_label]
        finite_values = local_cell_values[np.isfinite(local_cell_values)]
        if finite_values.size == 0:
            summaries.append(np.nan)
        elif statistic == "mean":
            summaries.append(float(np.mean(finite_values)))
        else:
            summaries.append(float(np.median(finite_values)))
    output[feature_name] = summaries
    return output