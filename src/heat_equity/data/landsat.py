"""Metadata-driven Landsat calibration and pixel-quality masking."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.integer]


@dataclass(frozen=True)
class BandCalibration:
    """Scale and offset values read from a Landsat scene's product metadata."""

    scale: float
    offset: float


@dataclass(frozen=True)
class LandsatSceneArrays:
    """Calibrated arrays and grid metadata for one Landsat scene."""

    lst_celsius: FloatArray
    reflectance: dict[str, FloatArray]
    ndvi: FloatArray
    valid_mask: NDArray[np.bool_]
    grid: dict[str, object]


def read_mtl_metadata(metadata_path: Path) -> dict[str, str]:
    """Read scalar key/value pairs from a Landsat MTL text file."""
    metadata: dict[str, str] = {}
    assignment_pattern = re.compile(r'^\s*([A-Z0-9_]+)\s*=\s*"?([^"\n]+?)"?\s*$')
    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        for line in metadata_file:
            match = assignment_pattern.match(line)
            if match:
                metadata.setdefault(match.group(1), match.group(2).strip())
    if not metadata:
        raise ValueError(f"No scalar metadata assignments were found in {metadata_path}.")
    return metadata


def calibration_from_metadata(
    metadata: Mapping[str, str], band_name: str
) -> BandCalibration:
    """Resolve a band's multiplier and additive offset from Landsat MTL metadata."""
    if band_name.startswith("ST_"):
        metadata_band_name = f"BAND_{band_name}"
        calibration_group = "TEMPERATURE"
    else:
        metadata_band_name = f"BAND_{band_name.removeprefix('BAND_')}"
        calibration_group = "REFLECTANCE"
    multiplier_key = f"{calibration_group}_MULT_{metadata_band_name}"
    offset_key = f"{calibration_group}_ADD_{metadata_band_name}"
    try:
        return BandCalibration(scale=float(metadata[multiplier_key]), offset=float(metadata[offset_key]))
    except KeyError as error:
        raise ValueError(f"Metadata is missing calibration field {error.args[0]}.") from error
    except ValueError as error:
        raise ValueError(f"Invalid numeric calibration value for {band_name}.") from error


def validate_raster_alignment(raster_paths: Mapping[str, Path]) -> dict[str, object]:
    """Validate raster CRS, dimensions, and affine transform against the first raster."""
    try:
        import rasterio
    except ImportError as error:
        raise RuntimeError(
            "Rasterio is required for raster-grid validation. Install project dependencies first."
        ) from error

    if not raster_paths:
        raise ValueError("At least one raster path is required.")

    reference_name = next(iter(raster_paths))
    with rasterio.open(raster_paths[reference_name]) as reference:
        reference_signature = (reference.crs, reference.width, reference.height, reference.transform)

    mismatches: list[str] = []
    for name, path in raster_paths.items():
        with rasterio.open(path) as raster:
            signature = (raster.crs, raster.width, raster.height, raster.transform)
        if signature != reference_signature:
            mismatches.append(name)

    if mismatches:
        raise ValueError(
            f"Raster grids differ from {reference_name}: {', '.join(mismatches)}. "
            "Reproject/resample inputs to a shared grid before feature construction."
        )

    return {
        "crs": str(reference_signature[0]),
        "width": reference_signature[1],
        "height": reference_signature[2],
        "transform": tuple(reference_signature[3]),
    }


def load_landsat_scene(
    band_paths: Mapping[str, Path],
    metadata_path: Path,
) -> LandsatSceneArrays:
    """Load and calibrate a Landsat scene using product metadata and QA bands.

    ``band_paths`` keys are Landsat band names such as ``ST_B10``, ``SR_B4``,
    ``SR_B5``, ``QA_PIXEL``, and ``QA_RADSAT``. Optional surface-reflectance
    bands are calibrated and returned when provided. The configured saturation
    bits correspond to Landsat 8/9 reflective bands B2-B7 and thermal band B10.
    """
    required_bands = {"ST_B10", "SR_B4", "SR_B5", "QA_PIXEL", "QA_RADSAT"}
    missing_bands = required_bands.difference(band_paths)
    if missing_bands:
        raise ValueError(f"Missing required Landsat bands: {', '.join(sorted(missing_bands))}.")

    try:
        import rasterio
    except ImportError as error:
        raise RuntimeError("Rasterio is required to read Landsat GeoTIFF bands.") from error

    metadata = read_mtl_metadata(metadata_path)
    grid = validate_raster_alignment(band_paths)
    arrays: dict[str, NDArray[np.number]] = {}
    nodata_masks: list[NDArray[np.bool_]] = []
    for band_name, path in band_paths.items():
        with rasterio.open(path) as raster:
            values = raster.read(1)
            arrays[band_name] = values
            if raster.nodata is not None:
                nodata_masks.append(values != raster.nodata)
            else:
                nodata_masks.append(np.ones(values.shape, dtype=bool))

    saturation_bits_by_band = {"SR_B2": 1, "SR_B3": 2, "SR_B4": 3, "SR_B5": 4,
                               "SR_B6": 5, "SR_B7": 6, "ST_B10": 9}
    saturation_bits = tuple(
        bit for band_name, bit in saturation_bits_by_band.items() if band_name in band_paths
    )
    valid_mask = valid_landsat_pixels(
        arrays["QA_PIXEL"],
        arrays["QA_RADSAT"],
        saturated_band_bits=saturation_bits,
    )
    for nodata_mask in nodata_masks:
        valid_mask &= nodata_mask

    thermal_kelvin = calibrate_band(
        arrays["ST_B10"], calibration_from_metadata(metadata, "ST_B10")
    )
    lst_celsius = apply_mask(thermal_kelvin - 273.15, valid_mask)

    reflectance: dict[str, FloatArray] = {}
    for band_name, values in arrays.items():
        if not band_name.startswith("SR_"):
            continue
        reflectance_number = band_name.removeprefix("SR_B")
        calibrated = calibrate_band(
            values,
            calibration_from_metadata(metadata, reflectance_number),
        )
        reflectance[band_name] = apply_mask(calibrated, valid_mask)

    ndvi = normalized_difference(reflectance["SR_B5"], reflectance["SR_B4"])
    ndvi[~valid_mask] = np.nan
    return LandsatSceneArrays(
        lst_celsius=lst_celsius,
        reflectance=reflectance,
        ndvi=ndvi,
        valid_mask=valid_mask,
        grid=grid,
    )


def calibrate_band(
    digital_numbers: NDArray[np.number], calibration: BandCalibration
) -> FloatArray:
    """Convert band digital numbers to physical values using metadata coefficients."""
    values = np.asarray(digital_numbers, dtype=np.float64)
    return values * calibration.scale + calibration.offset


def valid_landsat_pixels(
    qa_pixel: IntArray,
    qa_radsat: IntArray,
    *,
    saturated_band_bits: tuple[int, ...],
    reject_water: bool = False,
) -> NDArray[np.bool_]:
    """Return pixels that are clear of documented QA_PIXEL and selected saturation flags.

    QA_PIXEL bits 0-5 mark fill, dilated cloud, cirrus, cloud, cloud shadow, and snow.
    QA_RADSAT bit positions are sensor/product-specific, so callers must supply the
    selected bands' saturation bits from the scene documentation.
    """
    qa_pixel_values = np.asarray(qa_pixel)
    qa_radsat_values = np.asarray(qa_radsat)
    if qa_pixel_values.shape != qa_radsat_values.shape:
        raise ValueError("QA_PIXEL and QA_RADSAT arrays must have matching shapes.")

    invalid_pixel_mask = sum(1 << bit for bit in range(6))
    invalid = (qa_pixel_values.astype(np.uint64) & invalid_pixel_mask) != 0
    if reject_water:
        invalid |= (qa_pixel_values.astype(np.uint64) & (1 << 7)) != 0

    saturation_mask = 0
    for bit in saturated_band_bits:
        if bit < 0:
            raise ValueError("Saturation bit indexes must be non-negative.")
        saturation_mask |= 1 << bit
    invalid |= (qa_radsat_values.astype(np.uint64) & saturation_mask) != 0
    return ~invalid


def normalized_difference(
    first_band: NDArray[np.number], second_band: NDArray[np.number]
) -> FloatArray:
    """Calculate a normalized difference index, returning NaN for zero denominators."""
    first = np.asarray(first_band, dtype=np.float64)
    second = np.asarray(second_band, dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError("Input bands must have matching shapes.")

    denominator = first + second
    result = np.full(first.shape, np.nan, dtype=np.float64)
    np.divide(first - second, denominator, out=result, where=denominator != 0)
    return result


def apply_mask(values: NDArray[np.number], valid_mask: NDArray[np.bool_]) -> FloatArray:
    """Cast a raster band to float and mark invalid pixels as NaN."""
    band = np.asarray(values, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool)
    if band.shape != mask.shape:
        raise ValueError("Band and validity mask must have matching shapes.")
    return np.where(mask, band, np.nan)