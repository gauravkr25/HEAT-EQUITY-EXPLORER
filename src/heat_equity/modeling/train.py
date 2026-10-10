"""Reproducible classical regression comparison for prepared grid-cell data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelRun:
    """Trained estimator with held-out predictions and comparable metrics."""

    name: str
    estimator: Any
    metrics: dict[str, float]
    predictions: pd.DataFrame
    cv_metrics: dict[str, float]
    feature_importance: pd.DataFrame
    feature_columns: list[str]


def _validate_model_table(
    table: pd.DataFrame,
    *,
    target_column: str,
    split_column: str,
    excluded_columns: set[str],
) -> list[str]:
    required_columns = {target_column, split_column}
    missing = required_columns.difference(table.columns)
    if missing:
        raise ValueError(f"Model table is missing columns: {', '.join(sorted(missing))}.")
    if table.empty:
        raise ValueError("Model table must contain at least one row.")
    if not table[split_column].isin(["train", "test"]).all():
        raise ValueError("Split column values must be either 'train' or 'test'.")
    if set(table[split_column].unique()) != {"train", "test"}:
        raise ValueError("Model table must contain both train and test rows.")
    feature_columns = [
        column
        for column in table.select_dtypes(include=[np.number]).columns
        if column not in excluded_columns and column != target_column
    ]
    if not feature_columns:
        raise ValueError("Model table contains no numeric predictor columns.")
    if table.loc[table[split_column] == "train", target_column].isna().any():
        raise ValueError("Training target values must not be missing.")
    if table.loc[table[split_column] == "test", target_column].isna().any():
        raise ValueError("Test target values must not be missing for evaluation.")
    return feature_columns


def compare_regressors(
    table: pd.DataFrame,
    *,
    target_column: str = "lst_celsius",
    split_column: str = "split",
    group_column: str = "spatial_block_id",
    excluded_columns: set[str] | None = None,
    random_seed: int = 42,
) -> dict[str, ModelRun]:
    """Fit baseline and tree regressors, evaluating each on the same spatial holdout."""
    excluded = set(excluded_columns or set()) | {
        split_column,
        group_column,
        "grid_id",
        "grid_row",
        "grid_column",
        "raster_cell_label",
        "x",
        "y",
        "tract_geoid",
        "valid_pixel_fraction",
    }
    feature_columns = _validate_model_table(
        table,
        target_column=target_column,
        split_column=split_column,
        excluded_columns=excluded,
    )
    training = table.loc[table[split_column] == "train"]
    testing = table.loc[table[split_column] == "test"]
    if group_column not in table.columns:
        raise ValueError(f"Model table must include spatial grouping column {group_column!r}.")
    if training[group_column].isna().any() or testing[group_column].isna().any():
        raise ValueError("Spatial block IDs must not be missing.")
    training_groups = set(training[group_column])
    testing_groups = set(testing[group_column])
    if training_groups.intersection(testing_groups):
        raise ValueError("Spatial blocks overlap between train and test; rebuild the spatial split.")
    if len(training_groups) < 2:
        raise ValueError("At least two training spatial blocks are required for grouped validation.")
    x_train = training[feature_columns]
    y_train = training[target_column]
    x_test = testing[feature_columns]
    y_test = testing[target_column]
    grouped_cv = GroupKFold(n_splits=min(5, len(training_groups)))

    estimators = {
        "median_baseline": make_pipeline(SimpleImputer(strategy="median"), DummyRegressor(strategy="median")),
        "ridge": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=1.0),
        ),
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                n_estimators=150,
                min_samples_leaf=2,
                random_state=random_seed,
                n_jobs=1,
            ),
        ),
        "hist_gradient_boosting": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(
                max_iter=150,
                l2_regularization=1.0,
                random_state=random_seed,
            ),
        ),
    }

    runs: dict[str, ModelRun] = {}
    for name, estimator in estimators.items():
        cv_result = cross_validate(
            estimator,
            x_train,
            y_train,
            groups=training[group_column],
            cv=grouped_cv,
            scoring={"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"},
            n_jobs=1,
            error_score="raise",
        )
        estimator.fit(x_train, y_train)
        predicted = estimator.predict(x_test)
        predictions = pd.DataFrame(
            {
                "observed": y_test.to_numpy(),
                "predicted": predicted,
                "residual": y_test.to_numpy() - predicted,
            },
            index=testing.index,
        )
        for metadata_column in ("grid_id", "x", "y", "tract_geoid"):
            if metadata_column in testing.columns:
                predictions[metadata_column] = testing[metadata_column].to_numpy()
        importance_result = permutation_importance(
            estimator,
            x_test,
            y_test,
            scoring="neg_mean_absolute_error",
            n_repeats=5,
            random_state=random_seed,
            n_jobs=1,
        )
        feature_importance = pd.DataFrame(
            {
                "feature": feature_columns,
                "importance_mean": importance_result.importances_mean,
                "importance_std": importance_result.importances_std,
            }
        ).sort_values("importance_mean", ascending=False, ignore_index=True)
        runs[name] = ModelRun(
            name=name,
            estimator=estimator,
            metrics={
                "mae_celsius": float(mean_absolute_error(y_test, predicted)),
                "rmse_celsius": float(np.sqrt(mean_squared_error(y_test, predicted))),
                "r_squared": float(r2_score(y_test, predicted)),
            },
            predictions=predictions,
            cv_metrics={
                "group_cv_mae_celsius_mean": float(-np.mean(cv_result["test_mae"])),
                "group_cv_mae_celsius_std": float(np.std(-cv_result["test_mae"])),
                "group_cv_rmse_celsius_mean": float(-np.mean(cv_result["test_rmse"])),
                "group_cv_rmse_celsius_std": float(np.std(-cv_result["test_rmse"])),
            },
            feature_importance=feature_importance,
            feature_columns=feature_columns,
        )
    return runs


def save_model_runs(
    runs: dict[str, ModelRun],
    output_directory: Path,
    *,
    model_table: pd.DataFrame,
    target_column: str = "lst_celsius",
    split_column: str = "split",
    group_column: str = "spatial_block_id",
) -> None:
    """Save holdout evaluation and a separate full-data model for prediction coverage."""
    output_directory.mkdir(parents=True, exist_ok=True)
    metrics = {name: {**run.metrics, **run.cv_metrics} for name, run in runs.items()}
    (output_directory / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for name, run in runs.items():
        joblib.dump(run.estimator, output_directory / f"{name}.joblib")
        run.predictions.to_csv(output_directory / f"predictions_{name}.csv", index=True)
        run.feature_importance.to_csv(output_directory / f"importance_{name}.csv", index=False)

    selected_name = min(
        runs,
        key=lambda name: runs[name].cv_metrics["group_cv_mae_celsius_mean"],
    )
    full_model = runs[selected_name].estimator
    feature_columns = runs[selected_name].feature_columns
    full_model.fit(model_table[feature_columns], model_table[target_column])
    full_predictions = pd.DataFrame(
        {
            "grid_id": model_table.get("grid_id", pd.Series(model_table.index, index=model_table.index)),
            "predicted": full_model.predict(model_table[feature_columns]),
        },
        index=model_table.index,
    )
    for metadata_column in ("x", "y", "tract_geoid", "valid_pixel_fraction"):
        if metadata_column in model_table.columns:
            full_predictions[metadata_column] = model_table[metadata_column].to_numpy()
    full_predictions.to_csv(output_directory / "surface_predictions.csv", index=False)
    joblib.dump(full_model, output_directory / "surface_model.joblib")
    (output_directory / "model_selection.json").write_text(
        json.dumps(
            {
                "selected_model": selected_name,
                "selection_metric": "group_cv_mae_celsius_mean",
                "selection_value": runs[selected_name].cv_metrics[
                    "group_cv_mae_celsius_mean"
                ],
                "final_model_fit_scope": "all rows after holdout evaluation",
                "holdout_metrics_remain_unchanged": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Matplotlib is required to save model evaluation figures.") from error

    for name, run in runs.items():
        figure, axis = plt.subplots(figsize=(6, 6))
        axis.scatter(run.predictions["observed"], run.predictions["predicted"], alpha=0.65)
        lower = min(run.predictions["observed"].min(), run.predictions["predicted"].min())
        upper = max(run.predictions["observed"].max(), run.predictions["predicted"].max())
        axis.plot([lower, upper], [lower, upper], linestyle="--", color="black")
        axis.set(xlabel="Observed LST (°C)", ylabel="Predicted LST (°C)", title=name)
        figure.tight_layout()
        figure.savefig(output_directory / f"observed_vs_predicted_{name}.png", dpi=140)
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(7, 4))
        axis.hist(run.predictions["residual"], bins=25)
        axis.set(xlabel="Observed - predicted (°C)", ylabel="Cell count", title=f"Residuals: {name}")
        figure.tight_layout()
        figure.savefig(output_directory / f"residuals_{name}.png", dpi=140)
        plt.close(figure)

        top_importance = run.feature_importance.head(15).sort_values("importance_mean")
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.barh(top_importance["feature"], top_importance["importance_mean"])
        axis.set(xlabel="Increase in holdout MAE when shuffled (°C)", title=f"Permutation importance: {name}")
        figure.tight_layout()
        figure.savefig(output_directory / f"importance_{name}.png", dpi=140)
        plt.close(figure)