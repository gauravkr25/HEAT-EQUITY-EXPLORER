import unittest

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from heat_equity.priority import add_exploratory_priority_score, summarize_predictions_by_tract


class PrioritySummaryTests(unittest.TestCase):
    def test_summarizes_predictions_and_adds_transparent_score(self) -> None:
        tract_boundaries = gpd.GeoDataFrame(
            {
                "geoid": ["tract-a", "tract-b"],
                "poverty_rate": [0.1, 0.4],
                "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1)],
            },
            crs="EPSG:26912",
        )
        predictions = pd.DataFrame(
            {
                "tract_geoid": ["tract-a", "tract-a", "tract-b"],
                "predicted": [35.0, 39.0, 41.0],
            }
        )

        summaries = summarize_predictions_by_tract(predictions, tract_boundaries)
        scored = add_exploratory_priority_score(summaries, heat_weight=0.6)

        self.assertEqual(summaries.loc[0, "predicted_cell_count"], 2)
        self.assertEqual(summaries.loc[0, "predicted_lst_mean"], 37.0)
        self.assertLess(scored.loc[0, "exploratory_priority_score"], scored.loc[1, "exploratory_priority_score"])
        self.assertIn("heat percentile", scored.attrs["priority_formula"])

    def test_rejects_invalid_weight(self) -> None:
        tracts = pd.DataFrame({"predicted_lst_mean": [30], "poverty_rate": [0.2]})

        with self.assertRaises(ValueError):
            add_exploratory_priority_score(tracts, heat_weight=1.2)

    def test_leading_zero_census_geoids_join_as_strings(self) -> None:
        tract_boundaries = gpd.GeoDataFrame(
            {
                "geoid": ["04013000100"],
                "poverty_rate": [0.25],
                "geometry": [box(0, 0, 1, 1)],
            },
            crs="EPSG:26912",
        )
        predictions = pd.DataFrame(
            {
                "tract_geoid": pd.Series(["04013000100", "04013000100"], dtype="string"),
                "predicted": [38.0, 40.0],
            }
        )

        summaries = summarize_predictions_by_tract(predictions, tract_boundaries)
        scored = add_exploratory_priority_score(summaries)

        self.assertEqual(summaries.loc[0, "predicted_cell_count"], 2)
        self.assertEqual(summaries.loc[0, "predicted_lst_mean"], 39.0)
        self.assertTrue(pd.notna(scored.loc[0, "exploratory_priority_score"]))


if __name__ == "__main__":
    unittest.main()