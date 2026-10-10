"""Projected grid creation and leakage-resistant spatial assignments."""

from __future__ import annotations

import math

import geopandas as gpd
import pandas as pd
from shapely.geometry import box


def create_analysis_grid(
    area: gpd.GeoDataFrame,
    *,
    cell_size_meters: int = 500,
    target_crs: str = "EPSG:26912",
) -> gpd.GeoDataFrame:
    """Create clipped square cells over an input area with stable grid IDs."""
    if area.empty or area.geometry.is_empty.all():
        raise ValueError("The study area must contain at least one non-empty geometry.")
    if cell_size_meters <= 0:
        raise ValueError("cell_size_meters must be positive.")

    projected_area = area.to_crs(target_crs)
    boundary = projected_area.geometry.union_all()
    min_x, min_y, max_x, max_y = boundary.bounds
    first_x = math.floor(min_x / cell_size_meters) * cell_size_meters
    first_y = math.floor(min_y / cell_size_meters) * cell_size_meters
    columns = math.ceil((max_x - first_x) / cell_size_meters)
    rows = math.ceil((max_y - first_y) / cell_size_meters)

    records: list[dict[str, object]] = []
    for row_index in range(rows):
        bottom = first_y + row_index * cell_size_meters
        global_row = int(bottom // cell_size_meters)
        for column_index in range(columns):
            left = first_x + column_index * cell_size_meters
            global_column = int(left // cell_size_meters)
            cell = box(left, bottom, left + cell_size_meters, bottom + cell_size_meters)
            clipped_cell = cell.intersection(boundary)
            if clipped_cell.is_empty or clipped_cell.area == 0:
                continue
            records.append(
                {
                    "grid_id": f"r{global_row}_c{global_column}",
                    "grid_row": global_row,
                    "grid_column": global_column,
                    "geometry": clipped_cell,
                }
            )

    if not records:
        raise ValueError("The configured study area produced no grid cells.")
    return gpd.GeoDataFrame(records, geometry="geometry", crs=target_crs)


def assign_spatial_blocks(
    cells: pd.DataFrame,
    *,
    x_column: str = "x",
    y_column: str = "y",
    block_size_meters: int = 2000,
    holdout_fraction: float = 0.2,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Assign complete spatial blocks to train or final holdout deterministically."""
    if block_size_meters <= 0:
        raise ValueError("block_size_meters must be positive.")
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between 0 and 1.")
    missing_columns = {x_column, y_column}.difference(cells.columns)
    if missing_columns:
        raise ValueError(f"Missing coordinate columns: {', '.join(sorted(missing_columns))}.")
    if cells[[x_column, y_column]].isna().any().any():
        raise ValueError("Spatial coordinates must not contain missing values.")
    if len(cells) < 2:
        raise ValueError("At least two cells are required for a spatial holdout.")

    assignments = cells.copy()
    block_x = (assignments[x_column] // block_size_meters).astype("int64")
    block_y = (assignments[y_column] // block_size_meters).astype("int64")
    assignments["spatial_block_id"] = block_x.astype(str) + "_" + block_y.astype(str)
    block_ids = assignments["spatial_block_id"].drop_duplicates().to_numpy()
    if len(block_ids) < 2:
        raise ValueError("At least two distinct spatial blocks are required for holdout.")

    shuffled_blocks = pd.Series(block_ids).sample(frac=1, random_state=random_seed).tolist()
    holdout_count = min(len(shuffled_blocks) - 1, max(1, round(len(shuffled_blocks) * holdout_fraction)))
    holdout_blocks = set(shuffled_blocks[:holdout_count])
    assignments["split"] = assignments["spatial_block_id"].map(
        lambda block_id: "test" if block_id in holdout_blocks else "train"
    )
    return assignments