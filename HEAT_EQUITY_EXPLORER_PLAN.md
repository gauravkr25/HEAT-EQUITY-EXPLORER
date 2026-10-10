# Heat Equity Explorer: Implementation Plan

Build a reproducible, local-first Python capstone that estimates observed summer land-surface temperature (LST) patterns across Phoenix, Arizona, and overlays tract-level socioeconomic context to support transparent cooling-priority exploration. This plan is for an intermediate learner and a 4–8 week build. Interpret “all ts architecture” as “all technical architecture”; this is a Python project, so do not add TypeScript.

## Core Constraints and Scope

- Classical ML only: supervised regression and standard statistical baselines. No neural networks, deep learning, transfer learning, LLMs, generative AI, hosted model APIs, paid integrations, or required cloud/GPU compute.
- Run processing, model inference, tests, and dashboard locally. Public-source downloads can be manual. No live external API or map tile is required at runtime.
- Default study area: Phoenix city boundary plus a modest buffer in Maricopa County, configured so the area can be changed later. Use a projected CRS suitable for local distances, initially EPSG:26912 (NAD83 / UTM zone 12N); verify all inputs and transformations.
- Initial time period: summer 2021. The current downloaded baseline is a single cloud-free Landsat 8 scene acquired on 2021-08-27 (0% scene-level cloud reported by STAC), aligned with 2021 MRLC products and 2017–2021 ACS five-year estimates. This is a single-overpass snapshot, not a June–August composite. Multi-scene compositing remains a follow-up.
- Modeling unit: 500 m grid cell. One row per eligible cell. The target is median valid observed Landsat LST in the cell across selected scenes. Do not claim this is air temperature, human body exposure, or a health outcome.
- Census attributes are for a separate descriptive prioritization overlay, not predictors in the LST model. Do not infer causation or portray the overlay as an official environmental-justice index.

## Datasets

1. **Landsat Collection 2 Level-2, Landsat 8, Phoenix, June–August 2021.** Landsat 9 was not operating during this summer period. Use surface-temperature band `ST_B10` for the target; selected surface-reflectance bands (`SR_B2` through `SR_B7`) for predictors; and `QA_PIXEL` and `QA_RADSAT` for masking. Obtain files from [USGS EarthExplorer or LandsatLook](https://www.usgs.gov/landsat-missions/landsat-collection-2-level-2-science-products). These are public data sources; a free portal account may be required. Download only the scenes and bands needed. Read product metadata for scale factors, offsets, fill values, and QA bit definitions; never hard-code assumptions without checking metadata.
2. **MRLC Annual NLCD 2021, CONUS.** Use land-cover proportions and fractional imperviousness; use the 2021 tree-canopy product if its exact geographic coverage, date, and download terms are confirmed. MRLC provides direct downloadable products. Prefer the dataset portal’s 2021 bundle rather than a live map service. See the [MRLC data portal](https://www.mrlc.gov/data).
3. **U.S. Census TIGER/Line census-tract geometry** for Maricopa County and the Phoenix municipal boundary. Use a matching vintage where practical; record the vintage and CRS. TIGER/Line is a free bulk download.
4. **ACS 2017–2021 five-year tract estimates**, downloaded as a public bulk file or static table. The Census API currently requires an API key, so do not make it a project requirement. Candidate overlay fields: poverty rate derived from B17001 estimates and median household income from B19013. Verify variable definitions, denominator, annotations, missing/suppressed values, and the exact 2017–2021 file layout using ACS metadata before using them. See the [ACS 5-year data documentation](https://www.census.gov/data/developers/data-sets/acs-5year.html).
5. Document source URLs, product names, vintages, retrieval dates, licenses/terms, checksums where feasible, and required portal-account steps in `data/README.md`. Do not commit raw or large intermediate data.

## Feature and Target Design

- Convert Landsat digital numbers according to product metadata before deriving indices. Compute NDVI from corrected NIR and red reflectance; optionally compute NDBI and a water/vegetation index only if data quality and explanation remain clear. Guard zero denominators and masked pixels.
- Mask fill, cloud, cirrus, cloud shadow, snow, saturated pixels, and invalid/low-quality temperature using QA bands and documented product flags. Record valid-pixel coverage for every cell. Exclude cells below a configurable valid-coverage threshold; choose the threshold after inspecting the distribution and document it.
- Reproject/crop data to the configured AOI and common grid. Aggregate target temperature robustly (median); aggregate continuous predictors with documented statistics; aggregate land-cover classes as per-cell proportions; aggregate imperviousness/tree canopy as means or medians. Ensure all layers use consistent pixel alignment and nodata handling.
- Apply the `ST_B10` scale/offset from metadata to Kelvin and convert to Celsius. Verify the valid output range against product metadata and spot checks.
- Base predictors: corrected reflectance bands or selected indices, NLCD class proportions, imperviousness, and tree-canopy fraction. Do not include the thermal target band, target-derived summaries, ACS variables, or post-outcome features as model inputs. Treat coordinates as excluded in the primary model; optionally run a clearly labeled coordinate ablation only if it helps quantify spatial memorization.
- Output analysis table: stable grid-cell ID, geometry or centroid, target LST, predictors, valid-pixel fraction, scene/date provenance, and optional tract GEOID. Keep sensitive/small-population concerns out of the project; ACS is aggregate tract data.
- Summarize cell predictions to tracts. Create an optional transparent cooling-priority score by combining normalized tract mean predicted LST with an ACS vulnerability measure (initially poverty rate), with weights shown in the UI and documented. Label it exploratory and non-causal; expose its formula and do not call it a validated risk score.

## Classical ML and Evaluation

- Baseline: `DummyRegressor` predicting the training-set median. Also report a simple `Ridge` pipeline if feature preprocessing is straightforward.
- Candidate models: `RandomForestRegressor` and `HistGradientBoostingRegressor` from scikit-learn. Keep the comparison small, deterministic, and feasible on CPU. Use a `ColumnTransformer`/`Pipeline` where useful; one-hot encode land-cover categories only if categorical codes are used instead of class proportions.
- Split spatially, not randomly by row. Create fixed spatial blocks (initially 2 km, configurable), keep complete blocks together, reserve approximately 20% of blocks as final holdout, and use grouped folds inside training for comparison/tuning. Record split IDs and random seeds. Avoid target leakage across neighboring cells.
- Report MAE and RMSE in Celsius, R-squared as secondary context, prediction-vs-observation plots, residual summaries/maps, and baseline-relative improvement. Report metrics by held-out spatial fold and avoid presenting a single score without context. Do not promise a performance threshold before measuring data quality.
- Use scikit-learn permutation importance on held-out data for interpretation. Explain that correlated reflectance/land-cover features can share or distort importance; do not make causal claims. Optional residual analysis by land-cover and broad tract poverty groups is descriptive only.

## Technical Stack

- Python 3.11; `uv` for environment and lockfile management; `pyproject.toml` for package metadata and dependencies.
- Geospatial/data: `numpy`, `pandas`, `geopandas`, `shapely`, `pyproj`, `rasterio`, `pyarrow` for Parquet where appropriate.
- Classical ML: `scikit-learn`, `joblib`.
- Presentation: `streamlit`, `plotly`; use GeoJSON-based or locally rendered maps without paid basemap tokens or mandatory remote tiles. `matplotlib` for reproducible report figures.
- Quality: `pytest`, `ruff`. Keep dependencies minimal and generate a lockfile after installation validation on the target OS.
- Do not include TensorFlow, PyTorch, Keras, Transformers, OpenAI clients, or other neural/GenAI frameworks.

## Architecture and Proposed Repository Layout

- `pyproject.toml`, `uv.lock`, `.gitignore`, `README.md`, `data/README.md`, `configs/phoenix-2021.yaml`.
- `src/heat_equity/config.py`: typed configuration loading, path resolution, and defaults; no secrets/API keys.
- `src/heat_equity/data/manifest.py`: expected files, source metadata, checksums, and input existence checks.
- `src/heat_equity/data/landsat.py`: raster reading, metadata-based calibration, QA masking, cropping, reprojection, and multi-scene compositing.
- `src/heat_equity/data/mrlc.py`: NLCD layer loading and resampling/aggregation.
- `src/heat_equity/data/census.py`: boundary loading, ACS table validation, GEOID normalization, and tract joins.
- `src/heat_equity/features/grid.py`: AOI/grid creation and stable IDs; `src/heat_equity/features/build.py`: zonal aggregation, spectral indices, and model-table creation.
- `src/heat_equity/modeling/splits.py`: spatial block IDs and reproducible train/validation/test assignments.
- `src/heat_equity/modeling/train.py`: baseline and candidate pipelines, fitting, and artifact output.
- `src/heat_equity/modeling/evaluate.py`: metric calculation, fold reporting, residuals, and permutation importance.
- `src/heat_equity/priority.py`: transparent tract summaries and configurable exploratory priority calculation.
- `src/heat_equity/visualization/maps.py`: map-ready GeoJSON and report plots.
- `src/heat_equity/cli.py`: commands for validation, preparation, training, evaluation, and prediction; fail with actionable messages when inputs are missing.
- `src/heat_equity/app.py`: local Streamlit dashboard reading processed outputs and model artifacts. Do not perform expensive raster processing or network data downloads on app startup.
- `tests/`: unit tests for QA/calibration, indices, CRS/grid alignment, joins, spatial splits, scoring, and input validation; integration smoke test on tiny synthetic fixtures.
- `data/raw/`, `data/interim/`, `data/processed/`, `models/`, `reports/figures/`: generated/user-downloaded artifacts, ignored by git except small sample fixtures and explanatory README files.

## Step-by-Step Build Sequence

1. **Source and feasibility check:** Confirm Phoenix boundary, 2021 Landsat scene availability/cloud quality, 2021 NLCD downloads, 2021 tree-canopy coverage, TIGER/Line boundary, and ACS 2017–2021 variables. Record exact filenames and terms. If canopy coverage or ACS download is problematic, use the available NLCD land-cover/imperviousness features and keep ACS as an optional descriptive overlay; do not silently substitute years.
2. **Project scaffold:** Create the package layout, `pyproject.toml`, `uv` environment and lock, Ruff/Pytest settings, `.gitignore`, configuration schema, data manifest template, and README skeleton. Verify imports and CLI help before adding processing logic.
3. **Input validation:** Implement a command that checks required files, readable raster bands, CRS, bounds, nodata, expected ACS columns, tract GEOID format, and metadata. Emit actionable reports; do not download files automatically or request credentials.
4. **Geospatial preprocessing:** Load AOI and grids; crop/reproject consistently; calibrate and QA-mask Landsat; create summer composites; process NLCD and ACS; aggregate to grid/tract. Write intermediate outputs as GeoTIFF/GeoParquet/Parquet with provenance metadata. Add unit tests and inspect intermediate maps before modeling.
5. **Model table and exploratory analysis:** Build one stable row per valid grid cell, validate missingness/ranges/target coverage, inspect distributions and maps, create train/held-out spatial block assignments, and save the split manifest. Check for leakage and geography/vintage mismatch.
6. **Baseline and candidate models:** Fit Dummy and Ridge baselines, then Random Forest and HistGradientBoosting on identical spatial folds. Tune only a small set of interpretable parameters on training folds. Save model, feature schema, config, split IDs, software versions, and metrics as one reproducible run artifact.
7. **Evaluation and interpretation:** Evaluate once on untouched spatial holdout; produce MAE/RMSE/R-squared, per-fold metrics, observed-vs-predicted plot, residual map, and permutation importance. Investigate weak regions and coverage effects. Do not retune using holdout results.
8. **Equity-oriented summary:** Join cell predictions to census tracts and calculate tract summaries plus the documented optional priority score. Show the heat and socioeconomic layers separately as well as together; make weights and caveats visible.
9. **Dashboard:** Build local Streamlit views for observed LST, predicted LST, residuals/coverage, tract summaries, and optional priority rankings; include filters, legend/units, data-source panel, model metrics, and a clear limitations notice. Export selected summary CSV. The app must load precomputed outputs and work without paid services or live map tiles.
10. **Reproducibility and release polish:** Run the full pipeline from documented inputs, execute tests/lint, verify clean-environment setup on Windows, document data retrieval/manual prerequisites, include a model card/limitations section, and capture screenshots or figures for the README.

## Verification and Acceptance Criteria

1. Fresh environment can install dependencies from `pyproject.toml`/lockfile and run CLI help, tests, and the local dashboard without paid keys or cloud services.
2. Data validator identifies missing/mismatched files and reports source/vintage/CRS/metadata; preprocessing checks grid alignment, nodata, QA masks, temperature units/ranges, feature/target separation, and nonzero coverage.
3. Spatial holdout uses disjoint blocks and is reproducible. Baseline and candidate metrics are generated from a single documented command and saved with configuration and seeds.
4. No neural/DL/GenAI dependencies or model calls exist. All learned predictors are conventional scikit-learn regression models.
5. Dashboard uses local artifacts, clearly distinguishes observed from modeled LST, labels Celsius and vintage, exposes the exploratory score formula, and states that results are not causal, operational, or health advice.
6. Tests cover calibration and QA logic using synthetic arrays, feature calculations, missing/invalid inputs, GEOID joins, spatial grouping, metric calculations, priority score behavior, and app data loading.
7. README gives reproducible setup and execution steps, public dataset links/terms, manual download instructions, output descriptions, final held-out results, known limitations, and screenshots.

## Risks and Guardrails

- Landsat LST is affected by clouds, emissivity assumptions, overpass timing, and sensor/product processing; document these and avoid equating it with air temperature.
- A single summer composite is a snapshot, not a long-term climate trend. Additional years are a future extension only after the 2021 pipeline is validated.
- Census estimates have margins of error and temporal/spatial aggregation mismatch. Preserve source estimates and label tract-level comparisons as exploratory.
- Spatially adjacent cells are not independent; random row splits are prohibited for headline evaluation.
- Public data may require free portal registration and large downloads. Include manual download steps, file-size estimates after confirming them, and checksums; never store credentials.
- Do not overclaim fairness or equity from one vulnerability variable. Show temperature and ACS metrics separately; describe the combined score as a configurable screening aid only.
- Do not expand to multiple cities, real-time ingestion, health-outcome prediction, deep learning, GenAI, cloud deployment, authentication, or automated alerts in the capstone scope.

## Plan Decisions

- Concrete default geography/time is Phoenix summer 2021 to make the guide executable and keep product vintages aligned; make location and year configurable, but do not generalize until data QA works.
- Use only free public datasets, local CPU execution, classical supervised ML, and a local app. No paid integration/API is part of the deliverable.
- ACS data should be obtained as a bulk download/static file rather than relying on the Census API key requirement.
- Build the core physical-temperature model independently of ACS features; use socioeconomic data only in a separate descriptive tract-level overlay.
- This is a Python-only architecture; TypeScript is out of scope.
- **Implementation status:** The repository contains the Python package, input validator, Landsat 8 metadata-driven calibration/QA, MRLC grid-aligned numeric GeoTIFFs, spatial-grid aggregation, ACS tract context, leakage-resistant spatial splits, grouped-CV classical regressors, holdout evaluation and permutation importance, full-grid predictions, exploratory tract screening, and local dashboard. Public boundaries, ACS bulk tables, one Landsat 8 scene crop, and MRLC layers have been downloaded. Synthetic integration tests pass, and real Phoenix data preparation/training has completed. The run remains a single-scene snapshot; multi-scene compositing and product-specific QA review are future work. Document that limitation when sharing results.
