# Data Directory

Source files are staged under `raw/`. The configured required inputs are present; the optional 2021 tree-canopy raster is not currently included. Keep raw and processed datasets out of version control.

For each source, record the source URL, product and version, acquisition or estimate vintage, retrieval date, license/terms, portal-account requirements, and a checksum when feasible. Current downloaded source URLs and SHA-256 checksums are recorded in `raw/downloads/source-manifest.json`. No credentials are stored in this repository. The Landsat scene is a public Planetary Computer COG accessed with a short-lived read token; only the Phoenix boundary plus 1 km pixels were cropped locally. The MRLC WMS exports are numeric GeoTIFFs snapped to the cropped Landsat grid.