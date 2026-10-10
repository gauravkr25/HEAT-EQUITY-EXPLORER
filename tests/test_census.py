import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from heat_equity.data.census import (
    join_acs_to_tracts,
    load_acs_indicators,
    normalize_geoid,
)


class CensusDataTests(unittest.TestCase):
    def test_normalizes_and_validates_tract_geoid(self) -> None:
        normalized = normalize_geoid(pd.Series(["01001020100", "01001020200"]))

        self.assertEqual(normalized.tolist(), ["01001020100", "01001020200"])
        with self.assertRaises(ValueError):
            normalize_geoid(pd.Series(["1234"]))

    def test_derives_poverty_rate_from_total_below_poverty_row_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "acs.csv"
            pd.DataFrame(
                {
                    "GEO_ID": ["1400000US04013000100"],
                    "B17001_001E": [1000],
                    "B17001_002E": [120],
                    "B17001_031E": [180],
                    "B19013_001E": [65000],
                }
            ).to_csv(csv_path, index=False)

            indicators = load_acs_indicators(csv_path)

        self.assertEqual(indicators.loc[0, "geoid"], "04013000100")
        self.assertAlmostEqual(indicators.loc[0, "poverty_rate"], 0.12)
        self.assertEqual(indicators.loc[0, "median_household_income"], 65000)

    def test_loads_official_bulk_table_column_names(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "acs.csv"
            pd.DataFrame(
                {
                    "GEO_ID": ["1400000US04013000100"],
                    "B17001_E001": [1000],
                    "B17001_E002": [120],
                    "B19013_E001": [65000],
                }
            ).to_csv(csv_path, index=False)

            indicators = load_acs_indicators(csv_path)

        self.assertAlmostEqual(indicators.loc[0, "poverty_rate"], 0.12)
        self.assertEqual(indicators.loc[0, "median_household_income"], 65000)

    def test_joins_indicators_to_tract_geometry_by_geoid(self) -> None:
        boundaries = gpd.GeoDataFrame(
            {"geoid": ["04013000100"], "geometry": [box(0, 0, 1, 1)]},
            crs="EPSG:26912",
        )
        indicators = pd.DataFrame(
            {"geoid": ["04013000100"], "poverty_rate": [0.3], "median_household_income": [65000]}
        )

        joined = join_acs_to_tracts(boundaries, indicators)

        self.assertEqual(joined.loc[0, "poverty_rate"], 0.3)
        self.assertEqual(joined.crs.to_string(), "EPSG:26912")