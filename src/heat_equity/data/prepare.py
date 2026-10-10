"""End-to-end preparation of one configured Landsat scene and its context layers."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np

from heat_equity.data.census import join_acs_to_tracts, load_acs_indicators, load_tract_boundaries
from heat_equity.data.landsat import load_landsat_scene, read_mtl_metadata
from heat_equity.data.mrlc import read_aligned_mrlc_raster
from heat_equity.features.build import (
    add_continuous_raster_summary,
    add_land_cover_proportions,
    aggregate_scene_to_grid,
    rasterize_grid_labels,
)
from heat_equity.features.grid import assign_spatial_blocks, create_analysis_grid


def _input_paths(config: dict[str, Any], project_root: Path) -> dict[str, Path]:
    paths = {
        str(item["name"]): project_root / str(item["path"])
        for item in config.get("inputs", [])
    }
    missing = [
        f"{name}: {path}"
        for name, path in paths.items()
        if path.is_file() is False
        and next(
            (bool(item.get("required", True)) for item in config.get("inputs", []) if item["name"] == name),
            True,
        )
    ]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n- " + "\n- ".join(missing))
    return paths


def prepare_scene(
    config: dict[str, Any],
    project_root: Path,
    *,
    minimum_valid_fraction: float = 0.5,
    block_size_meters: int = 2000,
    random_seed: int = 42,
) -> dict[str, Path]:
    """Create a single-scene model table and tract-level descriptive context."""
    inputs = _input_paths(config, project_root)
    by_name = {name: path for name, path in inputs.items()}
    required_names = {
        "Phoenix municipal boundary",
        "Maricopa County census tracts",
        "ACS 2017-2021 tract estimates",
        "Landsat surface temperature band",
        "Landsat blue reflectance band",
        "Landsat green reflectance band",
        "Landsat SWIR1 reflectance band",
        "Landsat SWIR2 reflectance band",
        "Landsat pixel QA band",
        "Landsat saturation QA band",
        "Landsat red reflectance band",
        "Landsat near-infrared reflectance band",
        "NLCD land-cover raster",
        "NLCD fractional imperviousness raster",
    }
    missing_names = required_names.difference(by_name)
    if missing_names:
        raise ValueError(f"Configuration is missing inputs: {', '.join(sorted(missing_names))}.")

    landsat_directory = by_name["Landsat surface temperature band"].parent
    metadata_candidates = sorted(landsat_directory.glob("LC08*_MTL.txt"))
    if len(metadata_candidates) != 1:
        raise ValueError(
            f"Expected exactly one Landsat *_MTL.txt file in {landsat_directory}; "
            f"found {len(metadata_candidates)}."
        )
    scene_metadata = read_mtl_metadata(metadata_candidates[0])
    acquired_date_value = scene_metadata.get("DATE_ACQUIRED")
    if acquired_date_value is None:
        raise ValueError("Landsat MTL metadata is missing DATE_ACQUIRED.")
    acquired_date = date.fromisoformat(acquired_date_value)
    start_date = date.fromisoformat(str(config.get("season_start", f"{config['year']}-06-01")))
    end_date = date.fromisoformat(str(config.get("season_end", f"{config['year']}-08-31")))
    if not start_date <= acquired_date <= end_date:
        raise ValueError(
            f"Landsat scene date {acquired_date} falls outside configured study window "
            f"{start_date} through {end_date}."
        )
    scene_band_paths = {
        "ST_B10": by_name["Landsat surface temperature band"],
        "QA_PIXEL": by_name["Landsat pixel QA band"],
        "QA_RADSAT": by_name["Landsat saturation QA band"],
        "SR_B2": by_name["Landsat blue reflectance band"],
        "SR_B3": by_name["Landsat green reflectance band"],
        "SR_B4": by_name["Landsat red reflectance band"],
        "SR_B5": by_name["Landsat near-infrared reflectance band"],
        "SR_B6": by_name["Landsat SWIR1 reflectance band"],
        "SR_B7": by_name["Landsat SWIR2 reflectance band"],
    }
    scene = load_landsat_scene(scene_band_paths, metadata_candidates[0])

    boundary = gpd.read_file(by_name["Phoenix municipal boundary"])
    if boundary.crs is None:
        raise ValueError("Phoenix boundary does not declare a CRS.")
    target_crs = str(config.get("target_crs", "EPSG:26912"))
    buffer_meters = float(config.get("study_area_buffer_meters", 0))
    if buffer_meters < 0:
        raise ValueError("study_area_buffer_meters must not be negative.")
    if buffer_meters:
        boundary = boundary.to_crs(target_crs)
        boundary["geometry"] = boundary.geometry.buffer(buffer_meters)
    analysis_grid = create_analysis_grid(
        boundary,
        cell_size_meters=int(config.get("grid_size_meters", 500)),
        target_crs=target_crs,
    )
    analysis_table = aggregate_scene_to_grid(
        scene,
        analysis_grid,
        minimum_valid_fraction=minimum_valid_fraction,
    )

    scene_grid_labels = rasterize_grid_labels(
        analysis_grid,
        out_shape=(int(scene.grid["height"]), int(scene.grid["width"])),
        transform=scene.grid["transform"],
        raster_crs=str(scene.grid["crs"]),
    )
    land_cover = read_aligned_mrlc_raster(
        by_name["NLCD land-cover raster"], scene, categorical=True
    )
    land_cover_codes = sorted(
        int(code) for code in np.unique(land_cover[np.isfinite(land_cover)]) if int(code) != 0
    )
    land_cover = np.where(scene.valid_mask, land_cover, np.nan)
    analysis_table = add_land_cover_proportions(
        analysis_table,
        land_cover,
        scene_grid_labels,
        class_codes=land_cover_codes,
    )

    impervious = read_aligned_mrlc_raster(
        by_name["NLCD fractional imperviousness raster"], scene, categorical=False
    )
    analysis_table = add_continuous_raster_summary(
        analysis_table,
        np.where(scene.valid_mask, impervious, np.nan),
        scene_grid_labels,
        feature_name="imperviousness_median",
    )

    tree_canopy_name = "NLCD tree-canopy raster"
    if tree_canopy_name in by_name and by_name[tree_canopy_name].is_file():
        canopy = read_aligned_mrlc_raster(by_name[tree_canopy_name], scene, categorical=False)
        analysis_table = add_continuous_raster_summary(
            analysis_table,
            np.where(scene.valid_mask, canopy, np.nan),
            scene_grid_labels,
            feature_name="tree_canopy_median",
        )

    tract_boundaries = load_tract_boundaries(by_name["Maricopa County census tracts"], target_crs=target_crs)
    acs = load_acs_indicators(by_name["ACS 2017-2021 tract estimates"])
    tract_context = join_acs_to_tracts(tract_boundaries, acs)
    cell_centers = analysis_table[["grid_id", "geometry"]].copy()
    cell_centers["geometry"] = cell_centers.geometry.centroid
    tract_lookup = gpd.sjoin(
        cell_centers,
        tract_context[["geoid", "geometry"]],
        how="left",
        predicate="within",
    )[["grid_id", "geoid"]].drop_duplicates("grid_id")
    analysis_table = analysis_table.merge(
        tract_lookup.rename(columns={"geoid": "tract_geoid"}),
        on="grid_id",
        how="left",
        validate="one_to_one",
    )

    split_table = assign_spatial_blocks(
        analysis_table[["grid_id", "x", "y"]],
        block_size_meters=block_size_meters,
        random_seed=random_seed,
    )[["grid_id", "spatial_block_id", "split"]]
    analysis_table = analysis_table.merge(split_table, on="grid_id", how="left", validate="one_to_one")
    analysis_table = gpd.GeoDataFrame(analysis_table, geometry="geometry", crs=target_crs)

    output_directory = project_root / "data" / "processed"
    output_directory.mkdir(parents=True, exist_ok=True)
    table_path = output_directory / "model_table.parquet"
    grid_path = output_directory / "analysis_grid.geojson"
    tract_path = output_directory / "tract_context.geojson"
    analysis_table.to_parquet(table_path, index=False)
    analysis_table.to_crs("EPSG:4326").to_file(grid_path, driver="GeoJSON")
    tract_context.to_crs("EPSG:4326").to_file(tract_path, driver="GeoJSON")

    provenance_path = output_directory / "preparation.json"
    provenance = {
        "project_name": config.get("project_name"),
        "study_area": config.get("study_area"),
        "year": config.get("year"),
        "target_crs": target_crs,
        "grid_size_meters": config.get("grid_size_meters"),
        "minimum_valid_fraction": minimum_valid_fraction,
        "block_size_meters": block_size_meters,
        "random_seed": random_seed,
        "scene_metadata": metadata_candidates[0].name,
        "scene_acquired_date": acquired_date.isoformat(),
        "study_area_buffer_meters": buffer_meters,
        "row_count": len(analysis_table),
        "feature_columns": [
            column
            for column in analysis_table.columns
            if column not in {"geometry", "grid_id", "grid_row", "grid_column", "x", "y", "lst_celsius", "split", "spatial_block_id", "raster_cell_label", "tract_geoid", "valid_pixel_fraction"}
        ],
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return {"table": table_path, "grid": grid_path, "tracts": tract_path, "provenance": provenance_path}