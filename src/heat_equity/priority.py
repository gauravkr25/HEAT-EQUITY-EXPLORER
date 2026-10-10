"""Transparent, non-causal summaries for exploratory cooling-priority review."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd


def summarize_predictions_by_tract(
    predictions: pd.DataFrame,
    tract_boundaries: gpd.GeoDataFrame,
    *,
    tract_id_column: str = "tract_geoid",
    boundary_id_column: str = "geoid",
) -> gpd.GeoDataFrame:
    """Summarize grid predictions by tract and join the summaries to polygons."""
    required_columns = {tract_id_column, "predicted"}
    missing_columns = required_columns.difference(predictions.columns)
    if missing_columns:
        raise ValueError(f"Prediction data is missing: {', '.join(sorted(missing_columns))}.")
    if boundary_id_column not in tract_boundaries.columns:
        raise ValueError(f"Tract boundaries must include {boundary_id_column!r}.")

    summaries = predictions.groupby(tract_id_column, dropna=True).agg(
        predicted_lst_mean=("predicted", "mean"),
        predicted_lst_median=("predicted", "median"),
        predicted_cell_count=("predicted", "size"),
    )
    summaries.index = summaries.index.astype(str)
    summaries.index.name = boundary_id_column
    joined = tract_boundaries.merge(
        summaries.reset_index(),
        on=boundary_id_column,
        how="left",
        validate="one_to_one",
    )
    return gpd.GeoDataFrame(joined, geometry=tract_boundaries.geometry.name, crs=tract_boundaries.crs)


def add_exploratory_priority_score(
    tracts: gpd.GeoDataFrame,
    *,
    heat_column: str = "predicted_lst_mean",
    vulnerability_column: str = "poverty_rate",
    heat_weight: float = 0.5,
) -> gpd.GeoDataFrame:
    """Combine heat and vulnerability percentile ranks as a configurable screen.

    The score is the weighted sum of percentile ranks. It is not a validated risk
    estimate and must not be interpreted causally or as a policy determination.
    """
    if not 0 <= heat_weight <= 1:
        raise ValueError("heat_weight must be between zero and one.")
    missing_columns = {heat_column, vulnerability_column}.difference(tracts.columns)
    if missing_columns:
        raise ValueError(f"Tract table is missing: {', '.join(sorted(missing_columns))}.")

    output = tracts.copy()
    heat_percentile = output[heat_column].rank(pct=True, method="average")
    vulnerability_percentile = output[vulnerability_column].rank(pct=True, method="average")
    output["heat_percentile"] = heat_percentile
    output["vulnerability_percentile"] = vulnerability_percentile
    output["exploratory_priority_score"] = (
        heat_weight * heat_percentile + (1 - heat_weight) * vulnerability_percentile
    )
    output.attrs["priority_formula"] = (
        f"{heat_weight:.2f} * heat percentile + {1 - heat_weight:.2f} * "
        "vulnerability percentile"
    )
    output.attrs["priority_warning"] = (
        "Exploratory, non-causal screening aid; not a validated risk score or policy ranking."
    )
    return output