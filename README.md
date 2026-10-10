# Heat Equity Explorer

A local-first Python capstone for exploring Phoenix summer land-surface temperature with classical machine learning. It does not use neural networks, deep learning, LLMs, generative AI, paid integrations, or hosted model services.

## Current Status

The project includes input validation, a single-scene Landsat calibration and QA pipeline, spatial aggregation, tract context, spatial splits, classical regression comparison, and a local results dashboard. Public Phoenix inputs are downloaded under `data/raw/` and ignored by Git; their sources and checksums are recorded in `data/raw/downloads/source-manifest.json`. See `data/README.md` and `HEAT_EQUITY_EXPLORER_PLAN.md`.

## Setup

Python 3.11 is required. With `uv` installed:

```powershell
uv sync --extra dev
```

Validate required data files before processing:

```powershell
uv run heat-equity validate-inputs --config configs/phoenix-2021.toml
```

After downloading files into the configured locations, prepare the grid-cell model table:

```powershell
uv run heat-equity prepare --config configs/phoenix-2021.toml
```

Train and compare the classical models on the prepared spatial holdout:

```powershell
uv run heat-equity train-table data/processed/model_table.parquet --output models/latest
```

Launch the local dashboard:

```powershell
uv run streamlit run src/heat_equity/app.py
```

Run tests and lint:

```powershell
uv run pytest
uv run ruff check src tests
```

## Data Inputs

Public inputs currently staged locally are 2021 Arizona TIGER/Line place and tract boundaries, 2017–2021 ACS five-year tract estimates filtered to Maricopa County, one public Planetary Computer Landsat 8 Level-2 scene, and 2021 MRLC land-cover and imperviousness crops. The ACS data came from official pipe-delimited Census bulk tables, so no API key was needed. Source URLs and per-file checksums are in `data/raw/downloads/source-manifest.json`.

The current model uses one cloud-free Landsat 8 scene acquired on August 27, 2021, not a June–August composite. Treat all outputs as a snapshot of satellite-observed surface temperature on that overpass date. Optional tree canopy was not downloaded and is not included as a feature. The model has a spatial holdout; latest results are in `models/latest/metrics.json` and `models/latest/model_selection.json`.

## Scientific Scope

The target is observed satellite land-surface temperature, not near-surface air temperature, individual exposure, a health outcome, or a climate trend. Socioeconomic attributes are kept separate from the temperature model and are used only in descriptive tract-level views.