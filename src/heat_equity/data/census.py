"""Census geography and tract-level ACS indicator loading."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd


def normalize_geoid(values: pd.Series) -> pd.Series:
    """Normalize 11-digit tract GEOIDs without losing leading zeroes."""
    normalized = values.astype("string").str.replace(r"\.0$", "", regex=True).str.strip()
    invalid = normalized.isna() | ~normalized.str.fullmatch(r"\d{11}")
    if invalid.any():
        example_values = normalized[invalid].head(3).tolist()
        raise ValueError(f"Invalid 11-digit tract GEOID values found: {example_values}.")
    return normalized


def load_acs_indicators(csv_path: Path) -> pd.DataFrame:
    """Load 2017-2021 ACS detailed-table estimates and derive tract indicators.

    Accepts Census API-style columns or the official table-based summary-file
    columns after conversion to CSV. B17001 total-below-poverty is row 002;
    row 031 is at-or-above poverty and must not be added to that numerator.
    """
    table = pd.read_csv(csv_path, dtype={"GEO_ID": "string"}, comment="#")
    column_options = {
        "population_total": ("B17001_001E", "B17001_E001"),
        "below_poverty": ("B17001_002E", "B17001_E002"),
        "median_income": ("B19013_001E", "B19013_E001"),
    }
    resolved_columns = {
        role: next((candidate for candidate in candidates if candidate in table.columns), None)
        for role, candidates in column_options.items()
    }
    missing_columns = [role for role, column in resolved_columns.items() if column is None]
    if "GEO_ID" not in table.columns or missing_columns:
        missing = ["GEO_ID"] if "GEO_ID" not in table.columns else []
        missing.extend(missing_columns)
        raise ValueError(f"ACS table is missing required fields: {', '.join(missing)}.")

    geoid = table["GEO_ID"].astype("string").str.extract(r"US(\d{11})$", expand=False)
    if geoid.isna().any():
        geoid = table["GEO_ID"].astype("string").str.replace(r"^US", "", regex=True)
    table["geoid"] = normalize_geoid(geoid)
    for column in resolved_columns.values():
        table[column] = pd.to_numeric(table[column], errors="coerce")

    total_population = table[resolved_columns["population_total"]]
    below_poverty = table[resolved_columns["below_poverty"]]
    valid_denominator = total_population.where(total_population > 0)
    table["poverty_rate"] = (below_poverty / valid_denominator).clip(0, 1)
    income = table[resolved_columns["median_income"]]
    table["median_household_income"] = income.where(income >= 0)
    return table[["geoid", "poverty_rate", "median_household_income"]].drop_duplicates("geoid")


def load_tract_boundaries(boundary_path: Path, *, target_crs: str = "EPSG:26912") -> gpd.GeoDataFrame:
    """Load tract polygons and normalize the GEOID join field."""
    boundaries = gpd.read_file(boundary_path)
    geoid_column = next(
        (column for column in ("GEOID", "GEOID20", "GEOID10") if column in boundaries.columns),
        None,
    )
    if geoid_column is None:
        raise ValueError("Tract boundaries must include GEOID, GEOID20, or GEOID10.")
    if boundaries.crs is None:
        raise ValueError("Tract boundaries do not declare a CRS.")
    boundaries = boundaries.rename(columns={geoid_column: "geoid"})
    boundaries["geoid"] = normalize_geoid(boundaries["geoid"])
    return boundaries.to_crs(target_crs)


def join_acs_to_tracts(
    boundaries: gpd.GeoDataFrame,
    indicators: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """Join validated ACS indicators to tract polygons by normalized GEOID."""
    if "geoid" not in boundaries.columns or "geoid" not in indicators.columns:
        raise ValueError("Both tract boundaries and ACS indicators must contain geoid.")
    if indicators["geoid"].duplicated().any():
        raise ValueError("ACS indicators contain duplicate GEOIDs.")
    joined = boundaries.merge(indicators, on="geoid", how="left", validate="one_to_one")
    if joined["poverty_rate"].isna().all():
        raise ValueError("No tract geometries matched ACS poverty estimates by GEOID.")
    return gpd.GeoDataFrame(joined, geometry=boundaries.geometry.name, crs=boundaries.crs)