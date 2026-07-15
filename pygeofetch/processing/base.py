"""Base classes, result types, and shared I/O helpers for the processing engine."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Result type
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class ProcessingResult:
    """Outcome of a single processing step."""

    success: bool
    output_path: Path | None = None
    input_path: Path | None = None
    operation: str = ""
    duration_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def path(self) -> Path | None:
        return self.output_path

    def __str__(self) -> str:
        status = "✓" if self.success else "✗"
        out = self.output_path.name if self.output_path else "none"
        return f"{status} {self.operation} → {out} ({self.duration_seconds:.2f}s)"


# ══════════════════════════════════════════════════════════════════════════════
# Dependency guards
# ══════════════════════════════════════════════════════════════════════════════


def _require_rasterio():
    try:
        import rasterio

        return rasterio
    except ImportError:
        msg = (
            "rasterio is required for raster processing.\n"
            "Install with: pip install rasterio\n"
            'Or:           pip install "PyGeoFetch[geo]"'
        )
        raise ImportError(msg)


def _require_numpy():
    try:
        import numpy as np

        return np
    except ImportError:
        msg = "numpy is required: pip install numpy"
        raise ImportError(msg)


def _require_scipy():
    try:
        from scipy import ndimage

        return ndimage
    except ImportError:
        msg = "scipy is required: pip install scipy"
        raise ImportError(msg)


def _require_geopandas():
    try:
        import geopandas as gpd

        return gpd
    except ImportError:
        msg = "geopandas is required for vector operations.\nInstall with: pip install geopandas"
        raise ImportError(msg)


def _require_shapely():
    try:
        import shapely

        return shapely
    except ImportError:
        msg = "shapely is required: pip install shapely"
        raise ImportError(msg)


# ══════════════════════════════════════════════════════════════════════════════
# _timed decorator — catches exceptions → returns ProcessingResult(success=False)
# ══════════════════════════════════════════════════════════════════════════════


def _timed(func):
    """
    Measure execution time.

    On success: sets result.duration_seconds and returns result.
    On exception: logs the error and returns ProcessingResult(success=False, error=...)
    so callers always get a result object, never a raw exception.
    """
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        try:
            result = func(*args, **kwargs)
            if isinstance(result, ProcessingResult):
                result.duration_seconds = time.time() - t0
            return result
        except Exception as exc:
            elapsed = time.time() - t0
            op = func.__name__.replace("_cmd", "")
            logger.error("%s failed after %.2fs: %s", op, elapsed, exc, exc_info=True)
            return ProcessingResult(
                success=False,
                operation=op,
                duration_seconds=elapsed,
                error=str(exc),
            )

    return wrapper


# ══════════════════════════════════════════════════════════════════════════════
# Output path resolution
# ══════════════════════════════════════════════════════════════════════════════


def _resolve_output(
    input_path: Path | None,
    output: str | None,
    suffix: str,
) -> Path:
    """Build an output path from input + suffix when output is not specified."""
    if output:
        out = Path(output)
        if out.is_dir() or str(output).endswith("/"):
            out.mkdir(parents=True, exist_ok=True)
            stem = input_path.stem if input_path else "output"
            return out / f"{stem}_{suffix}.tif"
        out.parent.mkdir(parents=True, exist_ok=True)
        return out
    p = input_path or Path("output.tif")
    return p.parent / f"{p.stem}_{suffix}.tif"


# ══════════════════════════════════════════════════════════════════════════════
# Shared raster I/O — robust to tiled / COG / compressed GeoTIFFs
# ══════════════════════════════════════════════════════════════════════════════


def _safe_read_band(
    path: str | Path,
    band: int = 1,
    out_shape: tuple[int, int] | None = None,
) -> tuple[Any, dict, Any]:
    """
    Read a single band from any GeoTIFF, including tiled and COG formats.

    Strategy:
    1. Try the fast path: ``src.read(band)`` — works for stripped TIFFs.
    2. If that fails (tiled COG, DEFLATE-encoded, JPEG-compressed tiles),
       fall back to block-by-block reading which rasterio handles tile by tile.
    3. Apply optional resampling to ``out_shape`` after reading.

    Args:
        path:      Path to a raster file.
        band:      1-based band index to read.
        out_shape: Optional (height, width) to resample to after reading.

    Returns:
        (data, profile, nodata)
        data    — float32 numpy array, NaN where nodata
        profile — rasterio profile dict for writing output
        nodata  — original nodata value (may be None)
    """
    rasterio = _require_rasterio()
    np = _require_numpy()

    p = Path(path)
    if not p.exists():
        msg = f"Raster not found: {p}"
        raise FileNotFoundError(msg)

    with rasterio.open(p) as src:
        if band > src.count:
            msg = (
                f"Band {band} requested but file has only {src.count} band(s): {p.name}"
            )
            raise ValueError(msg)

        nodata = src.nodata
        profile = src.profile.copy()
        h, w = src.height, src.width

        # ── attempt 1: fast whole-array read ──────────────────────────────
        try:
            data = src.read(band).astype(np.float32)
        except Exception as fast_exc:
            logger.debug(
                "Fast read failed for %s (band %d): %s — using block-by-block fallback",
                p.name,
                band,
                fast_exc,
            )
            # ── attempt 2: block-by-block read ────────────────────────────
            data = np.empty((h, w), dtype=np.float32)
            data[:] = np.nan

            try:
                for _, window in src.block_windows(band):
                    row_off = window.row_off
                    col_off = window.col_off
                    row_end = row_off + window.height
                    col_end = col_off + window.width
                    try:
                        block = src.read(band, window=window).astype(np.float32)
                        data[row_off:row_end, col_off:col_end] = block
                    except Exception as block_exc:
                        # Fill failed blocks with nodata/NaN — don't crash
                        logger.warning(
                            "Block read failed at (%d,%d) in %s: %s — filling NaN",
                            row_off,
                            col_off,
                            p.name,
                            block_exc,
                        )
                        fill = float(nodata) if nodata is not None else np.nan
                        data[row_off:row_end, col_off:col_end] = fill

            except Exception as block_exc2:
                msg = (
                    f"Both fast-read and block-read failed for {p.name} band {band}.\n"
                    f"Fast error: {fast_exc}\n"
                    f"Block error: {block_exc2}\n"
                    "The file may be corrupt. Try: gdalinfo -checksum <file>"
                )
                raise RuntimeError(msg) from block_exc2

    # ── nodata → NaN ──────────────────────────────────────────────────────
    if nodata is not None:
        nd = float(nodata)
        if np.isnan(nd):
            # nodata is NaN — already NaN in float32, nothing to do
            pass
        else:
            data = np.where(data == nd, np.nan, data)

    # ── optional resampling ───────────────────────────────────────────────
    if out_shape is not None and out_shape != (h, w):
        from scipy.ndimage import zoom

        zoom_factors = (out_shape[0] / h, out_shape[1] / w)
        data = zoom(data, zoom_factors, order=1, mode="nearest").astype(np.float32)

    return data, profile, nodata


def _safe_write_band(
    data: Any,
    profile: dict,
    out_path: Path,
    nodata: float = -9999.0,
    compress: str = "deflate",
    tiled: bool = True,
    blocksize: int = 256,
) -> None:
    """
    Write a float32 array to a GeoTIFF, always with valid compression settings.

    Cleans the inherited profile to remove settings incompatible with float32
    (e.g. JPEG compression, photometric=RGB, etc.).

    Args:
        data:      2-D or 3-D (bands, h, w) float32 numpy array.
        profile:   Base profile (from source raster). Will be updated safely.
        out_path:  Destination path.
        nodata:    Nodata value for NaN pixels.
        compress:  Compression: ``"deflate"`` (default), ``"lzw"``, ``"zstd"``, ``None``.
        tiled:     Write as tiled (COG-ready) GeoTIFF.
        blocksize: Tile size in pixels (must be power of 2).
    """
    np = _require_numpy()
    rasterio = _require_rasterio()

    if data.ndim == 2:
        data_3d = data[np.newaxis, :, :]
        n_bands = 1
    else:
        data_3d = data
        n_bands = data.shape[0]

    h, w = data_3d.shape[1], data_3d.shape[2]

    # Replace NaN with nodata value
    data_3d = np.where(np.isnan(data_3d), nodata, data_3d).astype(np.float32)

    # Build a clean profile — remove anything incompatible with float32
    clean = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": n_bands,
        "height": h,
        "width": w,
        "nodata": nodata,
        "crs": profile.get("crs"),
        "transform": profile.get("transform"),
    }

    # Only add compression if specified
    if compress:
        clean["compress"] = compress
        if compress in ("deflate", "zstd"):
            clean["predictor"] = 2  # horizontal differencing for float data

    # Tiling for COG-ready output
    if tiled and blocksize:
        clean["tiled"] = True
        clean["blockxsize"] = blocksize
        clean["blockysize"] = blocksize

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(out_path, "w", **clean) as dst:
        dst.write(data_3d)
