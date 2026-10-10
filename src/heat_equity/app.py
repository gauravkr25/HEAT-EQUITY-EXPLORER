"""Local Streamlit interface for saved Heat Equity Explorer model runs."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import geopandas as gpd
import pandas as pd
import plotly.express as px
import streamlit as st

from heat_equity.cli import validate_inputs
from heat_equity.priority import add_exploratory_priority_score, summarize_predictions_by_tract


def load_run_metrics(run_directory: Path) -> dict[str, dict[str, float]]:
    """Load model metrics from one local run directory."""
    metrics_path = run_directory / "metrics.json"
    with metrics_path.open("r", encoding="utf-8") as metrics_file:
        metrics = json.load(metrics_file)
    if not isinstance(metrics, dict):
        raise TypeError("metrics.json must contain a model-name to metric mapping.")
    return metrics


def main() -> None:
    st.set_page_config(page_title="Heat Equity Explorer", layout="wide")
    st.title("Heat Equity Explorer")
    project_root = Path(__file__).resolve().parents[2]
    preparation_path = project_root / "data" / "processed" / "preparation.json"
    scene_date = "2021 study period"
    if preparation_path.is_file():
        try:
            preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
            scene_date = str(preparation.get("scene_acquired_date", scene_date))
        except (OSError, json.JSONDecodeError):
            pass
    st.caption(f"Phoenix | Landsat 8 scene acquired {scene_date} | Classical regression | Local data")

    with st.sidebar:
        st.header("Run")
        artifact_input = Path(st.text_input("Artifact directory", "models/latest"))

    run_directory = (
        artifact_input if artifact_input.is_absolute() else project_root / artifact_input
    )

    metrics_path = run_directory / "metrics.json"
    if not metrics_path.is_file():
        st.error(f"No trained model artifacts found at `{run_directory}`.")
        config_path = project_root / "configs" / "phoenix-2021.toml"
        try:
            missing_required, missing_optional = validate_inputs(config_path)
        except (OSError, ValueError, TypeError, tomllib.TOMLDecodeError) as error:
            st.error(f"Could not check configured data inputs: {error}")
            missing_required, missing_optional = [], []

        if missing_required:
            st.warning(
                f"{len(missing_required)} required source files are missing. "
                "No real Phoenix temperature data can be prepared until these are downloaded."
            )
            with st.expander("Missing required source files", expanded=True):
                for missing_input in missing_required:
                    st.write(f"- {missing_input}")
                if missing_optional:
                    st.caption(f"Also missing optional inputs: {len(missing_optional)}")
        else:
            processed_table = project_root / "data" / "processed" / "model_table.parquet"
            if processed_table.is_file():
                st.success("Source data is prepared. The model-training step is still needed.")
            else:
                st.warning("Configured source files exist, but the processed model table is missing.")

        st.subheader("Run these steps in the project terminal")
        st.code(
            "uv sync --extra dev\n"
            "uv run heat-equity validate-inputs --config configs/phoenix-2021.toml\n"
            "# After manually downloading the required public data:\n"
            "uv run heat-equity prepare --config configs/phoenix-2021.toml\n"
            "uv run heat-equity train-table data/processed/model_table.parquet "
            "--output models/latest",
            language="powershell",
        )
        st.warning(
            "No sample predictions are shown as real Phoenix results. Any future output is "
            "satellite land-surface temperature, not air temperature or health-risk advice. "
            "Stage downloaded inputs under the exact filenames shown above; rename the Landsat "
            "metadata file to `LC08_scene_MTL.txt`."
        )
        return

    try:
        metrics_by_model = load_run_metrics(run_directory)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        st.error(f"Could not read model metrics: {error}")
        return

    model_names = sorted(metrics_by_model)
    if not model_names:
        st.warning("No model metrics were found in this run directory.")
        return

    selection_path = run_directory / "model_selection.json"
    selected_default = model_names[0]
    if selection_path.is_file():
        try:
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            if selection.get("selected_model") in model_names:
                selected_default = selection["selected_model"]
        except (OSError, json.JSONDecodeError):
            pass

    with st.sidebar:
        selected_model = st.selectbox(
            "Model",
            model_names,
            index=model_names.index(selected_default),
        )

    metrics = metrics_by_model[selected_model]
    metric_columns = st.columns(3)
    metric_columns[0].metric("MAE", f"{metrics['mae_celsius']:.2f} °C")
    metric_columns[1].metric("RMSE", f"{metrics['rmse_celsius']:.2f} °C")
    metric_columns[2].metric("R²", f"{metrics['r_squared']:.3f}")
    if "group_cv_mae_celsius_mean" in metrics:
        st.caption(
            "Training-only grouped CV MAE: "
            f"{metrics['group_cv_mae_celsius_mean']:.2f} ± "
            f"{metrics['group_cv_mae_celsius_std']:.2f} °C. "
            "Final metrics below are from the untouched spatial holdout."
        )

    prediction_path = run_directory / f"predictions_{selected_model}.csv"
    if not prediction_path.is_file():
        st.error(f"Prediction file is missing: {prediction_path}")
        return
    predictions = pd.read_csv(prediction_path)
    required_columns = {"observed", "predicted", "residual"}
    if not required_columns.issubset(predictions.columns):
        st.error("Prediction CSV must include observed, predicted, and residual columns.")
        return

    st.subheader("Held-out predictions")
    observed_plot = px.scatter(
        predictions,
        x="observed",
        y="predicted",
        hover_name="grid_id" if "grid_id" in predictions else None,
        labels={"observed": "Observed LST (°C)", "predicted": "Predicted LST (°C)"},
        title="Observed vs. predicted surface temperature",
    )
    observed_plot.add_shape(
        type="line",
        x0=predictions["observed"].min(),
        y0=predictions["observed"].min(),
        x1=predictions["observed"].max(),
        y1=predictions["observed"].max(),
        line={"dash": "dash", "color": "#6b7280"},
    )
    st.plotly_chart(observed_plot, width="stretch")

    left_column, right_column = st.columns(2)
    with left_column:
        residual_plot = px.histogram(
            predictions,
            x="residual",
            nbins=30,
            labels={"residual": "Observed - predicted (°C)"},
            title="Held-out residual distribution",
        )
        st.plotly_chart(residual_plot, width="stretch")
    with right_column:
        surface_path = run_directory / "surface_predictions.csv"
        spatial_predictions = (
            pd.read_csv(surface_path) if surface_path.is_file() else predictions
        )
        spatial_title = (
            "Full-grid prediction surface (final refit)"
            if surface_path.is_file()
            else "Held-out prediction cells only"
        )
        if {"x", "y"}.issubset(spatial_predictions.columns):
            spatial_plot = px.scatter(
                spatial_predictions,
                x="x",
                y="y",
                color="predicted",
                hover_name="grid_id" if "grid_id" in spatial_predictions else None,
                color_continuous_scale="YlOrRd",
                labels={"x": "Easting (m)", "y": "Northing (m)", "predicted": "Predicted LST (°C)"},
                title=spatial_title,
            )
            st.plotly_chart(spatial_plot, width="stretch")
        else:
            st.info("Spatial coordinates were not saved with this run; no spatial plot is available.")

    with st.expander("Held-out prediction table"):
        st.dataframe(predictions, width="stretch")
        st.download_button(
            "Download predictions CSV",
            predictions.to_csv(index=False),
            file_name=f"predictions_{selected_model}.csv",
            mime="text/csv",
        )

    importance_path = run_directory / f"importance_{selected_model}.csv"
    if importance_path.is_file():
        st.subheader("Permutation importance")
        st.caption(
            "Change in held-out MAE after shuffling one feature. Correlated inputs can share "
            "importance; these are associations, not causal effects."
        )
        st.dataframe(pd.read_csv(importance_path), width="stretch", hide_index=True)

    tract_path = project_root / "data" / "processed" / "tract_context.geojson"
    surface_path = run_directory / "surface_predictions.csv"
    if tract_path.is_file() and surface_path.is_file():
        tract_geometries = gpd.read_file(tract_path).to_crs("EPSG:26912")
        if "poverty_rate" in tract_geometries.columns:
            heat_weight = st.slider(
                "Heat contribution to exploratory score",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
            )
            surface_predictions = pd.read_csv(
                surface_path,
                dtype={"tract_geoid": "string"},
            )
            tract_summaries = summarize_predictions_by_tract(surface_predictions, tract_geometries)
            tract_summaries = add_exploratory_priority_score(
                tract_summaries,
                heat_weight=heat_weight,
            )
            tract_centers = tract_summaries.copy()
            tract_centers["geometry"] = tract_centers.geometry.centroid
            tract_centers["x"] = tract_centers.geometry.x
            tract_centers["y"] = tract_centers.geometry.y
            st.subheader("Exploratory tract-level screening")
            st.caption(
                f"Score: {tract_summaries.attrs['priority_formula']}. "
                f"{tract_summaries.attrs['priority_warning']}"
            )
            priority_plot = px.scatter(
                tract_centers.dropna(subset=["exploratory_priority_score"]),
                x="x",
                y="y",
                color="exploratory_priority_score",
                size="predicted_cell_count",
                hover_name="geoid",
                color_continuous_scale="YlOrRd",
                labels={"x": "Easting (m)", "y": "Northing (m)", "exploratory_priority_score": "Score"},
                title="Tract centroids; local coordinates, no basemap",
            )
            st.plotly_chart(priority_plot, width="stretch")
            st.download_button(
                "Download tract summaries CSV",
                tract_summaries.drop(columns="geometry").to_csv(index=False),
                file_name="tract_screening_summary.csv",
                mime="text/csv",
            )

    st.warning(
        "These are satellite land-surface temperature estimates, not air temperature, "
        "individual heat exposure, health risk, or a causal estimate."
    )


if __name__ == "__main__":
    main()