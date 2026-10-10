"""Load MRLC raster products onto a selected Landsat grid."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from heat_equity.data.landsat import LandsatSceneArrays


def read_aligned_mrlc_raster(
    raster_path: Path,
    target_scene: LandsatSceneArrays,
    *,
    categorical: bool,
) -> NDArray[np.float32]:
    """Reproject an MRLC layer onto a Landsat scene grid.

    Categorical layers use nearest-neighbor resampling. Continuous layers use
    average resampling so source-pixel proportions are retained where possible.
    """
    try:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.warp import reproject
    except ImportError as error:
        raise RuntimeError("Rasterio is required to align MRLC rasters.") from error

    grid = target_scene.grid
    target_crs = grid.get("crs")
    transform_values = grid.get("transform")
    width = grid.get("width")
    height = grid.get("height")
    if target_crs is None or transform_values is None or width is None or height is None:
        raise ValueError("Target scene grid metadata is incomplete.")

    from affine import Affine

    destination = np.full((int(height), int(width)), np.nan, dtype=np.float32)
    resampling: Literal["nearest", "average"] = "nearest" if categorical else "average"
    with rasterio.open(raster_path) as source:
        reproject(
            source=rasterio.band(source, 1),
            destination=destination,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=source.nodata,
            dst_transform=Affine(*transform_values[:6]),
            dst_crs=target_crs,
            dst_nodata=np.nan,
            resampling=getattr(Resampling, resampling),
            init_dest_nodata=True,
        )
    return destination