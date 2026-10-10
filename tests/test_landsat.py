import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

import numpy as np

from heat_equity.data.landsat import (
    BandCalibration,
    apply_mask,
    calibrate_band,
    calibration_from_metadata,
    load_landsat_scene,
    normalized_difference,
    read_mtl_metadata,
    valid_landsat_pixels,
    validate_raster_alignment,
)


class LandsatPreparationTests(unittest.TestCase):
    def test_calibration_uses_supplied_metadata_values(self) -> None:
        digital_numbers = np.array([0, 1000], dtype=np.uint16)

        calibrated = calibrate_band(
            digital_numbers,
            BandCalibration(scale=0.003, offset=149.0),
        )

        np.testing.assert_allclose(calibrated, [149.0, 152.0])

    def test_mtl_reader_and_reflectance_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata_path = Path(temp_dir) / "scene_MTL.txt"
            metadata_path.write_text(
                'GROUP = LEVEL2_SURFACE_REFLECTANCE\n'
                '  REFLECTANCE_MULT_BAND_4 = 2.75E-05\n'
                '  REFLECTANCE_ADD_BAND_4 = -0.2\n'
                'END_GROUP = LEVEL2_SURFACE_REFLECTANCE\n'
                'GROUP = LEVEL1_RADIOMETRIC_RESCALING\n'
                '  REFLECTANCE_MULT_BAND_4 = 2.0E-05\n'
                '  REFLECTANCE_ADD_BAND_4 = -0.1\n'
                'END_GROUP = LEVEL1_RADIOMETRIC_RESCALING\n'
                'END\n',
                encoding="utf-8",
            )

            metadata = read_mtl_metadata(metadata_path)
            calibration = calibration_from_metadata(metadata, "4")

        self.assertAlmostEqual(calibration.scale, 2.75e-5)
        self.assertAlmostEqual(calibration.offset, -0.2)

    def test_temperature_calibration_uses_st_band_key(self) -> None:
        calibration = calibration_from_metadata(
            {
                "TEMPERATURE_MULT_BAND_ST_B10": "0.00341802",
                "TEMPERATURE_ADD_BAND_ST_B10": "149.0",
            },
            "ST_B10",
        )

        self.assertAlmostEqual(calibration.scale, 0.00341802)
        self.assertAlmostEqual(calibration.offset, 149.0)

    def test_quality_mask_rejects_cloud_shadow_snow_and_selected_saturation(self) -> None:
        qa_pixel = np.array([[0, 1 << 3, 1 << 4, 1 << 5, 0]], dtype=np.uint16)
        qa_radsat = np.array([[0, 0, 0, 0, 1 << 4]], dtype=np.uint16)

        valid = valid_landsat_pixels(
            qa_pixel,
            qa_radsat,
            saturated_band_bits=(4,),
        )

        np.testing.assert_array_equal(valid, [[True, False, False, False, False]])

    def test_water_can_be_optionally_rejected(self) -> None:
        qa_pixel = np.array([[0, 1 << 7]], dtype=np.uint16)
        qa_radsat = np.zeros_like(qa_pixel)

        keep_water = valid_landsat_pixels(qa_pixel, qa_radsat, saturated_band_bits=())
        reject_water = valid_landsat_pixels(
            qa_pixel, qa_radsat, saturated_band_bits=(), reject_water=True
        )

        np.testing.assert_array_equal(keep_water, [[True, True]])
        np.testing.assert_array_equal(reject_water, [[True, False]])

    def test_ndvi_handles_zero_denominator_as_nan(self) -> None:
        nir = np.array([[0.8, 0.0]])
        red = np.array([[0.2, 0.0]])

        ndvi = normalized_difference(nir, red)

        self.assertAlmostEqual(ndvi[0, 0], 0.6 / 1.0)
        self.assertTrue(np.isnan(ndvi[0, 1]))

    def test_apply_mask_sets_invalid_pixels_to_nan(self) -> None:
        values = np.array([[290, 300]], dtype=np.uint16)
        valid_mask = np.array([[True, False]])

        masked = apply_mask(values, valid_mask)

        self.assertEqual(masked[0, 0], 290.0)
        self.assertTrue(np.isnan(masked[0, 1]))

    def test_mismatched_shapes_raise_value_error(self) -> None:
        with self.assertRaises(ValueError):
            normalized_difference(np.ones((2, 2)), np.ones((2, 1)))

    @unittest.skipUnless(find_spec("rasterio"), "Rasterio is unavailable in this interpreter.")
    def test_raster_alignment_accepts_same_grid(self) -> None:
        import rasterio
        from rasterio.transform import from_origin

        with tempfile.TemporaryDirectory() as temp_dir:
            raster_paths = {
                "first": Path(temp_dir) / "first.tif",
                "second": Path(temp_dir) / "second.tif",
            }
            for raster_path in raster_paths.values():
                with rasterio.open(
                    raster_path,
                    "w",
                    driver="GTiff",
                    width=2,
                    height=2,
                    count=1,
                    dtype="uint16",
                    crs="EPSG:32612",
                    transform=from_origin(500000, 3700000, 30, 30),
                ) as raster:
                    raster.write(np.ones((2, 2), dtype=np.uint16), 1)

            grid = validate_raster_alignment(raster_paths)

        self.assertEqual(grid["width"], 2)
        self.assertEqual(grid["height"], 2)
        self.assertEqual(grid["crs"], "EPSG:32612")

    @unittest.skipUnless(find_spec("rasterio"), "Rasterio is unavailable in this interpreter.")
    def test_scene_loader_calibrates_bands_and_masks_clouds(self) -> None:
        import rasterio
        from rasterio.transform import from_origin

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            metadata_path = root / "scene_MTL.txt"
            metadata_path.write_text(
                "REFLECTANCE_MULT_BAND_4 = 2.75E-05\n"
                "REFLECTANCE_ADD_BAND_4 = -0.2\n"
                "REFLECTANCE_MULT_BAND_5 = 2.75E-05\n"
                "REFLECTANCE_ADD_BAND_5 = -0.2\n"
                "TEMPERATURE_MULT_BAND_ST_B10 = 3.41802E-03\n"
                "TEMPERATURE_ADD_BAND_ST_B10 = 149.0\n",
                encoding="utf-8",
            )
            raster_values = {
                "ST_B10": np.array([[44177, 44177]], dtype=np.uint16),
                "SR_B4": np.array([[14545, 14545]], dtype=np.uint16),
                "SR_B5": np.array([[29091, 29091]], dtype=np.uint16),
                "QA_PIXEL": np.array([[0, 1 << 3]], dtype=np.uint16),
                "QA_RADSAT": np.zeros((1, 2), dtype=np.uint16),
            }
            band_paths: dict[str, Path] = {}
            for band_name, values in raster_values.items():
                raster_path = root / f"{band_name}.tif"
                with rasterio.open(
                    raster_path,
                    "w",
                    driver="GTiff",
                    width=2,
                    height=1,
                    count=1,
                    dtype="uint16",
                    crs="EPSG:32612",
                    transform=from_origin(500000, 3700000, 30, 30),
                    nodata=0 if band_name.startswith("SR_") or band_name == "ST_B10" else None,
                ) as raster:
                    raster.write(values, 1)
                band_paths[band_name] = raster_path

            scene = load_landsat_scene(band_paths, metadata_path)

        self.assertTrue(scene.valid_mask[0, 0])
        self.assertFalse(scene.valid_mask[0, 1])
        self.assertAlmostEqual(scene.lst_celsius[0, 0], 300.0 - 273.15, places=1)
        self.assertTrue(np.isnan(scene.lst_celsius[0, 1]))
        self.assertGreater(scene.ndvi[0, 0], 0)


if __name__ == "__main__":
    unittest.main()