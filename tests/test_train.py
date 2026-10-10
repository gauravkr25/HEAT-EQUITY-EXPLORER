import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from heat_equity.modeling.train import compare_regressors, save_model_runs


class RegressorComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.table = pd.DataFrame(
            {
                "grid_id": [f"cell-{index}" for index in range(40)],
                "x": np.arange(40) * 2000,
                "feature_green": np.linspace(0, 1, 40),
                "feature_impervious": np.linspace(1, 0, 40),
                "lst_celsius": 40 - np.linspace(0, 1, 40) * 8,
                "split": ["train"] * 32 + ["test"] * 8,
                "spatial_block_id": [f"train-{index // 4}" for index in range(32)]
                + [f"test-{index // 4}" for index in range(8)],
            }
        )

    def test_compares_classical_models_on_same_test_rows(self) -> None:
        runs = compare_regressors(self.table, excluded_columns={"x"}, random_seed=8)

        self.assertEqual(
            set(runs),
            {"median_baseline", "ridge", "random_forest", "hist_gradient_boosting"},
        )
        self.assertTrue(all(len(run.predictions) == 8 for run in runs.values()))
        self.assertTrue(all("rmse_celsius" in run.metrics for run in runs.values()))

    def test_rejects_coordinates_unless_explicitly_excluded(self) -> None:
        runs = compare_regressors(self.table, excluded_columns={"x"})

        self.assertEqual(len(runs), 4)

    def test_missing_test_split_is_rejected(self) -> None:
        table = self.table.loc[self.table["split"] == "train"]

        with self.assertRaises(ValueError):
            compare_regressors(table, excluded_columns={"x"})

    def test_overlapping_spatial_blocks_are_rejected(self) -> None:
        table = self.table.copy()
        table.loc[table["split"] == "test", "spatial_block_id"] = "train-0"

        with self.assertRaisesRegex(ValueError, "overlap"):
            compare_regressors(table)

    def test_saves_holdout_metrics_and_full_grid_prediction_artifacts(self) -> None:
        table = self.table.assign(metadata_only=np.arange(len(self.table)))
        runs = compare_regressors(
            table,
            excluded_columns={"x", "metadata_only"},
            random_seed=11,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_directory = Path(temp_dir)
            save_model_runs(runs, output_directory, model_table=table)

            metrics = json.loads((output_directory / "metrics.json").read_text(encoding="utf-8"))
            selected = json.loads(
                (output_directory / "model_selection.json").read_text(encoding="utf-8")
            )
            surface = pd.read_csv(output_directory / "surface_predictions.csv")

        self.assertIn("group_cv_mae_celsius_mean", metrics["ridge"])
        self.assertEqual(len(surface), len(self.table))
        self.assertEqual(selected["final_model_fit_scope"], "all rows after holdout evaluation")
        self.assertTrue(selected["holdout_metrics_remain_unchanged"])
        self.assertNotIn("metadata_only", runs[selected["selected_model"]].feature_columns)

    def test_target_is_never_treated_as_predictor(self) -> None:
        runs = compare_regressors(self.table, excluded_columns={"x"})
        first_run = runs["median_baseline"]
        test_row = self.table.loc[self.table["split"] == "test"].iloc[0]

        self.assertNotEqual(first_run.predictions.loc[test_row.name, "predicted"], test_row["lst_celsius"])


if __name__ == "__main__":
    unittest.main()