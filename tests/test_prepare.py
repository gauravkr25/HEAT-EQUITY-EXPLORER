import tempfile
import tomllib
import unittest
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from heat_equity.data.prepare import prepare_scene
from heat_equity.modeling.train import compare_regressors, save_model_runs


class PrepareSceneIntegrationTests(unittest.TestCase):
    def test_prepares_synthetic_scene_and_writes_expected_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "pyproject.toml").write_text("", encoding="utf-8")
            landsat_dir = root / "data/raw/landsat"
            mrlc_dir = root / "data/raw/mrlc"
            boundary_dir = root / "data/raw/boundaries"
            census_dir = root / "data/raw/census"
            for directory in (landsat_dir, mrlc_dir, boundary_dir, census_dir):
                directory.mkdir(parents=True)

            transform = from_origin(400000, 3704000, 250, 250)
            raster_values = {
                "ST_B10": np.full((16, 16), 44177, dtype=np.uint16),
                "QA_PIXEL": np.zeros((16, 16), dtype=np.uint16),
                "QA_RADSAT": np.zeros((16, 16), dtype=np.uint16),
                "SR_B2": np.full((16, 16), 18000, dtype=np.uint16),
                "SR_B3": np.full((16, 16), 20000, dtype=np.uint16),
                "SR_B4": np.full((16, 16), 15000, dtype=np.uint16),
                "SR_B5": np.full((16, 16), 25000, dtype=np.uint16),
                "SR_B6": np.full((16, 16), 23000, dtype=np.uint16),
                "SR_B7": np.full((16, 16), 21000, dtype=np.uint16),
            }
            for band_name, values in raster_values.items():
                with rasterio.open(
                    landsat_dir / f"{band_name}.tif",
                    "w",
                    driver="GTiff",
                    width=16,
                    height=16,
                    count=1,
                    dtype=values.dtype,
                    crs="EPSG:26912",
                    transform=transform,
                    nodata=0 if band_name.startswith(("ST_", "SR_")) else None,
                ) as raster:
                    raster.write(values, 1)

            mtl_values = [
                *(f"REFLECTANCE_MULT_BAND_{band} = 2.75E-05" for band in range(2, 8)),
                *(f"REFLECTANCE_ADD_BAND_{band} = -0.2" for band in range(2, 8)),
                "TEMPERATURE_MULT_BAND_ST_B10 = 3.41802E-03",
                "TEMPERATURE_ADD_BAND_ST_B10 = 149.0",
                "DATE_ACQUIRED = 2021-07-15",
            ]
            (landsat_dir / "LC08_synthetic_MTL.txt").write_text(
                "\n".join(mtl_values) + "\n",
                encoding="utf-8",
            )

            for file_name, value in (
                ("nlcd_land_cover_2021.tif", 21),
                ("nlcd_impervious_2021.tif", 60),
            ):
                with rasterio.open(
                    mrlc_dir / file_name,
                    "w",
                    driver="GTiff",
                    width=16,
                    height=16,
                    count=1,
                    dtype="uint8",
                    crs="EPSG:26912",
                    transform=transform,
                    nodata=0,
                ) as raster:
                    raster.write(np.full((16, 16), value, dtype=np.uint8), 1)

            study_area = gpd.GeoDataFrame(
                geometry=[box(400000, 3700000, 404000, 3704000)],
                crs="EPSG:26912",
            )
            study_area.to_file(boundary_dir / "phoenix_boundary.gpkg", driver="GPKG")
            tracts = gpd.GeoDataFrame(
                {"GEOID": ["04013000100"], "geometry": [box(400000, 3700000, 404000, 3704000)]},
                crs="EPSG:26912",
            )
            tracts.to_file(boundary_dir / "maricopa_tracts.gpkg", driver="GPKG")
            pd.DataFrame(
                {
                    "GEO_ID": ["1400000US04013000100"],
                    "B17001_001E": [1000],
                    "B17001_002E": [120],
                    "B17001_031E": [180],
                    "B19013_001E": [65000],
                }
            ).to_csv(census_dir / "acs_2017_2021.csv", index=False)

            config_path = root / "configs/phoenix.toml"
            config_path.parent.mkdir()
            config_path.write_text(
                'project_name="Synthetic test"\n'
                'study_area="Synthetic Phoenix"\n'
                'year=2021\n'
                'target_crs="EPSG:26912"\n'
                'grid_size_meters=500\n'
                'study_area_buffer_meters=0\n'
                + self._input_config("Phoenix municipal boundary", "data/raw/boundaries/phoenix_boundary.gpkg")
                + self._input_config("Maricopa County census tracts", "data/raw/boundaries/maricopa_tracts.gpkg")
                + self._input_config("ACS 2017-2021 tract estimates", "data/raw/census/acs_2017_2021.csv")
                + self._input_config("Landsat surface temperature band", "data/raw/landsat/ST_B10.tif")
                + self._input_config("Landsat scene metadata", "data/raw/landsat/LC08_synthetic_MTL.txt")
                + self._input_config("Landsat pixel QA band", "data/raw/landsat/QA_PIXEL.tif")
                + self._input_config("Landsat saturation QA band", "data/raw/landsat/QA_RADSAT.tif")
                + self._input_config("Landsat blue reflectance band", "data/raw/landsat/SR_B2.tif")
                + self._input_config("Landsat green reflectance band", "data/raw/landsat/SR_B3.tif")
                + self._input_config("Landsat red reflectance band", "data/raw/landsat/SR_B4.tif")
                + self._input_config("Landsat near-infrared reflectance band", "data/raw/landsat/SR_B5.tif")
                + self._input_config("Landsat SWIR1 reflectance band", "data/raw/landsat/SR_B6.tif")
                + self._input_config("Landsat SWIR2 reflectance band", "data/raw/landsat/SR_B7.tif")
                + self._input_config("NLCD land-cover raster", "data/raw/mrlc/nlcd_land_cover_2021.tif")
                + self._input_config("NLCD fractional imperviousness raster", "data/raw/mrlc/nlcd_impervious_2021.tif"),
                encoding="utf-8",
            )
            with config_path.open("rb") as config_file:
                config = tomllib.load(config_file)

            outputs = prepare_scene(
                config,
                root,
                minimum_valid_fraction=0.5,
                block_size_meters=2000,
                random_seed=42,
            )

            model_table = gpd.read_parquet(outputs["table"])
            self.assertEqual(len(model_table), 64)
            self.assertEqual(set(model_table["split"]), {"train", "test"})
            self.assertTrue(model_table["tract_geoid"].notna().all())
            self.assertTrue(outputs["grid"].is_file())
            self.assertTrue(outputs["tracts"].is_file())
            self.assertTrue(outputs["provenance"].is_file())

            runs = compare_regressors(model_table)
            artifact_directory = root / "models" / "integration-test"
            save_model_runs(runs, artifact_directory, model_table=model_table)
            self.assertTrue((artifact_directory / "metrics.json").is_file())
            self.assertTrue((artifact_directory / "surface_predictions.csv").is_file())

    @staticmethod
    def _input_config(name: str, path: str) -> str:
        return f'[[inputs]]\nname="{name}"\npath="{path}"\nrequired=true\n'


if __name__ == "__main__":
    unittest.main()