import unittest

import geopandas as gpd
import numpy as np
import pandas as pd
from affine import Affine
from shapely.geometry import box

from heat_equity.data.landsat import LandsatSceneArrays
from heat_equity.features.build import (
    add_land_cover_proportions,
    aggregate_scene_to_grid,
    rasterize_grid_labels,
)
from heat_equity.features.grid import assign_spatial_blocks, create_analysis_grid


class SpatialGridTests(unittest.TestCase):
    def test_grid_cells_are_clipped_and_have_stable_ids(self) -> None:
        area = gpd.GeoDataFrame(
            geometry=[box(500000, 3700000, 501000, 3700500)],
            crs="EPSG:26912",
        )

        first_grid = create_analysis_grid(area, cell_size_meters=500)
        second_grid = create_analysis_grid(area, cell_size_meters=500)

        self.assertEqual(len(first_grid), 2)
        self.assertEqual(first_grid["grid_id"].tolist(), second_grid["grid_id"].tolist())
        self.assertTrue((first_grid.geometry.area <= 500 * 500).all())

    def test_grid_ids_do_not_shift_when_study_area_expands(self) -> None:
        small_area = gpd.GeoDataFrame(
            geometry=[box(500000, 3700000, 501000, 3700500)],
            crs="EPSG:26912",
        )
        expanded_area = gpd.GeoDataFrame(
            geometry=[box(499000, 3699500, 501500, 3701000)],
            crs="EPSG:26912",
        )

        small_grid = create_analysis_grid(small_area, cell_size_meters=500)
        expanded_grid = create_analysis_grid(expanded_area, cell_size_meters=500)

        self.assertTrue(set(small_grid["grid_id"]).issubset(set(expanded_grid["grid_id"])))

    def test_spatial_split_keeps_each_block_in_one_split(self) -> None:
        cells = pd.DataFrame(
            {
                "x": [100, 200, 2100, 2200, 4100, 4200],
                "y": [100, 200, 100, 200, 100, 200],
            }
        )

        assigned = assign_spatial_blocks(cells, block_size_meters=2000, random_seed=7)
        split_counts = assigned.groupby("spatial_block_id")["split"].nunique()

        self.assertTrue((split_counts == 1).all())
        self.assertEqual(set(assigned["split"]), {"train", "test"})

    def test_spatial_split_is_reproducible(self) -> None:
        cells = pd.DataFrame({"x": [100, 2100, 4100, 6100], "y": [100, 100, 100, 100]})

        first = assign_spatial_blocks(cells, block_size_meters=2000, random_seed=19)
        second = assign_spatial_blocks(cells, block_size_meters=2000, random_seed=19)

        pd.testing.assert_series_equal(first["split"], second["split"])

    def test_invalid_holdout_fraction_is_rejected(self) -> None:
        cells = pd.DataFrame({"x": [100, 2100], "y": [100, 100]})

        with self.assertRaises(ValueError):
            assign_spatial_blocks(cells, holdout_fraction=1.0)

    def test_aggregates_scene_pixels_and_land_cover_by_stable_grid_label(self) -> None:
        transform = Affine(250, 0, 0, 0, -250, 1000)
        grid = gpd.GeoDataFrame(
            {
                "grid_id": ["west", "east"],
                "geometry": [box(0, 500, 500, 1000), box(500, 500, 1000, 1000)],
            },
            crs="EPSG:26912",
        )
        lst = np.arange(16, dtype=float).reshape(4, 4) + 30
        valid_mask = np.ones((4, 4), dtype=bool)
        valid_mask[0, 0] = False
        ndvi = np.full((4, 4), 0.5)
        reflectance = {"SR_B4": np.full((4, 4), 0.2), "SR_B5": np.full((4, 4), 0.6)}
        scene = LandsatSceneArrays(
            lst_celsius=np.where(valid_mask, lst, np.nan),
            reflectance=reflectance,
            ndvi=np.where(valid_mask, ndvi, np.nan),
            valid_mask=valid_mask,
            grid={
                "crs": "EPSG:26912",
                "width": 4,
                "height": 4,
                "transform": tuple(transform),
            },
        )

        table = aggregate_scene_to_grid(scene, grid, minimum_valid_fraction=0.5)
        labels = rasterize_grid_labels(
            grid,
            out_shape=(4, 4),
            transform=transform,
            raster_crs="EPSG:26912",
        )
        land_cover = np.where(labels == 1, 21, 22)
        table = add_land_cover_proportions(
            table,
            land_cover,
            labels,
            class_codes=[21, 22],
        )

        self.assertEqual(table["grid_id"].tolist(), ["west", "east"])
        self.assertAlmostEqual(table.loc[0, "valid_pixel_fraction"], 0.75)
        self.assertEqual(table.loc[0, "land_cover_21_fraction"], 1.0)
        self.assertEqual(table.loc[1, "land_cover_22_fraction"], 1.0)
        self.assertGreater(table.loc[0, "lst_celsius"], 30)


if __name__ == "__main__":
    unittest.main()