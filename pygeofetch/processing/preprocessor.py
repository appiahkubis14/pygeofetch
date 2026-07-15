"""
Preprocessor — A-H categories:
    A: Atmospheric & Radiometric Correction
    B: Cloud & Shadow Masking
    C: Geometric Correction
    D: Resolution & Resampling
    H: Mosaicking & Compositing
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pygeofetch.processing.base import (
    ProcessingResult,
    _require_numpy,
    _require_rasterio,
    _resolve_output,
    _safe_read_band,
    _timed,
)

logger = logging.getLogger(__name__)

BBox = tuple[float, float, float, float]  # minx, miny, maxx, maxy


def _safe_read_1(src, path=None):
    """Read band 1 safely with block fallback."""
    try:
        return _safe_read_1(src)
    except Exception:
        if path is None:
            raise
        from pygeofetch.processing.base import _safe_read_band

        data, _, _ = _safe_read_band(path, band=1)
        return data


class Preprocessor:
    """
    Complete preprocessing engine.

    All methods accept file paths as str or Path and return a
    :class:`ProcessingResult` with the output path.

    Example::

        from pygeofetch import PyGeoFetch
        client = PyGeoFetch()
        result = client.preprocess.clip("scene.tif", bbox=(-74.1, 40.6, -73.7, 40.9))
        result = client.preprocess.reproject("scene.tif", crs="EPSG:4326")
        result = client.preprocess.ndvi(red="B04.tif", nir="B08.tif")
    """

    def __init__(self) -> None:
        self._logger = logger

    # ──────────────────────────────────────────────────────────────────────
    # A: Atmospheric & Radiometric Correction
    # ──────────────────────────────────────────────────────────────────────

    @_timed
    def atmos(
        self,
        input: str | Path,
        method: str = "dos1",
        output: str | None = None,
        **kwargs: Any,
    ) -> ProcessingResult:
        """
        Atmospheric correction.

        Args:
            input:  Input GeoTIFF path.
            method: ``"dos1"`` (Dark Object Subtraction), ``"dos2"``,
                    ``"sen2cor"`` (Sentinel-2 specific), ``"flaash"``,
                    ``"6s"``, ``"iCOR"``.
            output: Output path (auto-generated if omitted).
            **kwargs: Method-specific parameters.

        Returns:
            ProcessingResult with corrected raster path.

        Example::

            result = client.preprocess.atmos("scene.tif", method="dos1")
            result = client.preprocess.atmos("scene.tif", method="sen2cor",
                                              aerosol_type="maritime")
        """
        rasterio = _require_rasterio()
        np = _require_numpy()

        inp = Path(input)
        out_path = _resolve_output(inp, output, f"atmos_{method}")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        data_bands, profile, nodata = _safe_read_band(inp, band=1)
        with rasterio.open(inp) as src:
            profile = src.profile.copy()
            nodata = src.nodata
            data = np.stack(
                [_safe_read_band(inp, band=b)[0] for b in range(1, src.count + 1)],
                axis=0,
            )

        method = method.lower()

        if method in ("dos1", "dos2"):
            # Dark Object Subtraction — subtract minimum valid DN per band
            corrected = np.zeros_like(data, dtype=np.float32)
            for b in range(data.shape[0]):
                band = data[b]
                valid = band > 0 if nodata is None else (band != nodata) & (band > 0)
                dark_pixel = (
                    float(np.percentile(band[valid], 1)) if valid.any() else 0.0
                )
                corrected[b] = np.where(valid, band - dark_pixel, nodata or 0)
            if method == "dos2":
                # DOS2: also apply path radiance correction
                for b in range(corrected.shape[0]):
                    valid = corrected[b] > 0
                    if valid.any():
                        corrected[b] = np.where(
                            valid, corrected[b] * 1.05, corrected[b]
                        )

        elif method == "sen2cor":
            # Simplified Sen2Cor: band-specific scale + offset
            # Real Sen2Cor requires the ESA tool; this applies the standard
            # Sentinel-2 L1C→L2A reflectance conversion (divide by 10000)
            quantification = kwargs.get("quantification_value", 10000.0)
            corrected = np.where(
                (data != nodata) if nodata else np.ones_like(data, dtype=bool),
                np.clip(data / quantification, 0, 1),
                nodata or 0,
            ).astype(np.float32)

        elif method in ("flaash", "6s", "icor"):
            # Placeholder for tools that require external executables
            # Returns DOS1 as best-effort fallback with warning
            logger.warning(
                f"{method} requires the external tool to be installed. "
                "Applying DOS1 as fallback. Install the tool and set "
                f"PYGEOFETCH_{method.upper()}_BIN=/path/to/binary."
            )
            corrected = data.copy()
            for b in range(corrected.shape[0]):
                band = corrected[b]
                valid = band > 0 if nodata is None else band != nodata
                if valid.any():
                    corrected[b] -= float(np.percentile(band[valid], 1))
        else:
            msg = (
                f"Unknown atmospheric correction method: {method!r}. "
                "Supported: dos1, dos2, sen2cor, flaash, 6s, icor"
            )
            raise ValueError(msg)

        profile.update(dtype="float32", nodata=nodata or None)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(corrected)

        logger.info(f"Atmospheric correction ({method}) → {out_path}")
        return ProcessingResult(
            success=True,
            operation=f"atmos:{method}",
            input_path=inp,
            output_path=out_path,
            metadata={"method": method},
        )

    @_timed
    def topo_correct(
        self,
        input: str | Path,
        dem: str | Path,
        method: str = "cosine",
        output: str | None = None,
    ) -> ProcessingResult:
        """
        Topographic correction to remove illumination effects from terrain.

        Args:
            input:  Input raster to correct.
            dem:    DEM raster (same CRS/resolution or auto-resampled).
            method: ``"cosine"``, ``"minnaert"``, ``"c_correction"``.
            output: Output path.

        Example::

            result = client.preprocess.topo_correct(
                "scene.tif", dem="srtm.tif", method="c_correction"
            )
        """
        rasterio = _require_rasterio()
        np = _require_numpy()

        inp = Path(input)
        dem_path = Path(dem)
        out_path = _resolve_output(inp, output, f"topo_{method}")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(inp) as src:
            profile = src.profile.copy()
            data = __import__("numpy").array(
                [_safe_read_1(src)]
                if src.count == 1
                else [
                    src.read(b).astype(__import__("numpy").float32)
                    for b in range(1, src.count + 1)
                ]
            )
            nodata = src.nodata

        with rasterio.open(dem_path) as dem_src:
            dem_data = _safe_read_1(dem_src)

        # Compute slope and aspect from DEM (simplified gradient-based)
        from numpy import arctan, cos, gradient, pi, sqrt, where

        dy, dx = gradient(dem_data)
        slope = arctan(sqrt(dx**2 + dy**2))
        # Solar zenith angle — use 45° as default (override via metadata)
        cos_z = cos(45 * pi / 180)

        # Resize dem arrays to match image shape
        if dem_data.shape != data[0].shape:
            import rasterio.transform
            from rasterio.enums import Resampling

            with rasterio.open(dem_path) as dem_src:
                dem_data = dem_src.read(
                    1,
                    out_shape=data[0].shape,
                    resampling=Resampling.bilinear,
                ).astype(np.float32)
            dy, dx = gradient(dem_data)
            slope = arctan(sqrt(dx**2 + dy**2))

        cos_i = cos(slope) * cos_z  # simplified illumination

        corrected = np.zeros_like(data)
        for b in range(data.shape[0]):
            band = data[b]
            if method == "cosine":
                with np.errstate(divide="ignore", invalid="ignore"):
                    corrected[b] = where(cos_i > 0.01, band * (cos_z / cos_i), band)
            elif method == "minnaert":
                k = 0.5  # Minnaert constant (could be estimated empirically)
                with np.errstate(divide="ignore", invalid="ignore"):
                    corrected[b] = where(
                        cos_i > 0.01,
                        band * (cos_z / cos_i) ** k,
                        band,
                    )
            elif method == "c_correction":
                # Regress band on cos_i to find c constant
                valid = (cos_i > 0.01) & (band > 0)
                if valid.sum() > 100:
                    x, y = cos_i[valid].ravel(), band[valid].ravel()
                    c_val = float(np.polyfit(x, y, 1)[1]) / (
                        float(np.polyfit(x, y, 1)[0]) + 1e-10
                    )
                else:
                    c_val = 1.0
                with np.errstate(divide="ignore", invalid="ignore"):
                    corrected[b] = where(
                        cos_i > 0.01,
                        band * (cos_z + c_val) / (cos_i + c_val),
                        band,
                    )
            else:
                msg = f"Unknown topo method: {method!r}"
                raise ValueError(msg)

        if nodata is not None:
            corrected = np.where(data == nodata, nodata, corrected)

        profile.update(dtype="float32")
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(corrected.astype(np.float32))

        logger.info(f"Topographic correction ({method}) → {out_path}")
        return ProcessingResult(
            success=True,
            operation=f"topo:{method}",
            input_path=inp,
            output_path=out_path,
        )

    # ──────────────────────────────────────────────────────────────────────
    # B: Cloud & Shadow Masking
    # ──────────────────────────────────────────────────────────────────────

    @_timed
    def cloud_mask(
        self,
        input: str | Path,
        method: str = "scl",
        output: str | None = None,
        scl_band: str | Path | None = None,
        cloud_classes: list[int] | None = None,
    ) -> ProcessingResult:
        """
        Cloud masking — sets cloud pixels to NoData.

        Args:
            input:          Input raster.
            method:         ``"scl"`` (Sentinel-2 SCL), ``"fmask"``,
                            ``"ndsi"`` (snow removal), ``"threshold"``.
            output:         Output path.
            scl_band:       Path to SCL band file (required for ``"scl"``).
            cloud_classes:  SCL class values to mask
                            (default: [3,8,9,10,11] — cloud shadow, cloud).

        Example::

            # Sentinel-2 SCL-based cloud masking
            result = client.preprocess.cloud_mask(
                "scene.tif", method="scl", scl_band="SCL.tif"
            )
            # Simple threshold (any band 1 value > 8000 = cloud)
            result = client.preprocess.cloud_mask(
                "scene.tif", method="threshold"
            )
        """
        rasterio = _require_rasterio()
        np = _require_numpy()

        inp = Path(input)
        out_path = _resolve_output(inp, output, f"cloudmask_{method}")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(inp) as src:
            profile = src.profile.copy()
            data = __import__("numpy").array(
                [_safe_read_1(src)]
                if src.count == 1
                else [
                    src.read(b).astype(__import__("numpy").float32)
                    for b in range(1, src.count + 1)
                ]
            )
            nodata_val = src.nodata if src.nodata is not None else 0.0

        cloud_classes = cloud_classes or [3, 8, 9, 10, 11]  # SCL cloud classes

        if method == "scl":
            if scl_band is None:
                msg = "scl_band path is required for method='scl'"
                raise ValueError(msg)
            with rasterio.open(Path(scl_band)) as scl_src:
                scl = scl_src.read(1)
                if scl.shape != data[0].shape:
                    scl = scl_src.read(
                        1,
                        out_shape=data[0].shape,
                        resampling=rasterio.enums.Resampling.nearest,
                    )
            cloud_mask_arr = np.isin(scl, cloud_classes)
            for b in range(data.shape[0]):
                data[b] = np.where(cloud_mask_arr, nodata_val, data[b])

        elif method == "fmask":
            # Simplified FMask heuristic using band reflectance ratios
            if data.shape[0] >= 4:
                blue = data[0].astype(float) / 10000
                green = data[1].astype(float) / 10000
                red = data[2].astype(float) / 10000
                nir = data[3].astype(float) / 10000
                # Brightness + whiteness test
                brightness = (blue + green + red + nir) / 4
                whiteness = (abs(blue - green) + abs(blue - red) + abs(green - red)) / (
                    brightness + 1e-6
                )
                cloud_mask_arr = (brightness > 0.3) & (whiteness < 0.7)
            else:
                # Fallback: simple brightness threshold on band 1
                cloud_mask_arr = data[0] > 0.8
            for b in range(data.shape[0]):
                data[b] = np.where(cloud_mask_arr, nodata_val, data[b])

        elif method == "threshold":
            # Generic high-reflectance threshold (assumes scaled to 0-1 or 0-10000)
            scale = 10000.0 if data.max() > 10 else 1.0
            threshold = 0.8 * scale
            cloud_mask_arr = data[0] > threshold
            for b in range(data.shape[0]):
                data[b] = np.where(cloud_mask_arr, nodata_val, data[b])

        elif method == "ndsi":
            # Snow/ice mask using NDSI — assumes bands[0]=green, bands[3]=swir
            if data.shape[0] >= 4:
                green = data[1].astype(float)
                swir1 = data[3].astype(float)
                with np.errstate(divide="ignore", invalid="ignore"):
                    ndsi = np.where(
                        green + swir1 > 0, (green - swir1) / (green + swir1), 0
                    )
                snow_mask = ndsi > 0.4
                for b in range(data.shape[0]):
                    data[b] = np.where(snow_mask, nodata_val, data[b])
            else:
                logger.warning("NDSI requires at least 4 bands; no masking applied")
        else:
            msg = f"Unknown cloud mask method: {method!r}"
            raise ValueError(msg)

        profile.update(dtype="float32", nodata=nodata_val)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data)

        masked_pct = 100 * np.mean(data[0] == nodata_val) if data.size > 0 else 0
        logger.info(
            f"Cloud mask ({method}) masked {masked_pct:.1f}% pixels → {out_path}"
        )
        return ProcessingResult(
            success=True,
            operation=f"cloud_mask:{method}",
            input_path=inp,
            output_path=out_path,
            metadata={"masked_pct": round(masked_pct, 2)},
        )

    @_timed
    def cloud_fill(
        self,
        input: str | Path,
        time_series: list[str | Path],
        output: str | None = None,
        method: str = "interpolate",
    ) -> ProcessingResult:
        """
        Fill cloud gaps using neighbouring dates from a time series.

        Args:
            input:       Target scene with cloud gaps (NoData pixels).
            time_series: List of other scenes (same AOI) to use as fill sources.
            method:      ``"interpolate"`` (linear), ``"nearest"`` (nearest valid date).
            output:      Output path.

        Example::

            result = client.preprocess.cloud_fill(
                "scene_cloudy.tif",
                time_series=["scene_jan.tif", "scene_mar.tif"],
                method="interpolate"
            )
        """
        rasterio = _require_rasterio()
        np = _require_numpy()

        inp = Path(input)
        out_path = _resolve_output(inp, output, "filled")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(inp) as src:
            profile = src.profile.copy()
            target = __import__("numpy").array(
                [_safe_read_1(src)]
                if src.count == 1
                else [
                    src.read(b).astype(__import__("numpy").float32)
                    for b in range(1, src.count + 1)
                ]
            )
            nodata_val = src.nodata if src.nodata is not None else 0.0

        # Build stack of fill candidates
        stack = [target]
        for ts_path in time_series:
            with rasterio.open(Path(ts_path)) as ts_src:
                ts_data = ts_src.read(
                    out_shape=target.shape,
                    resampling=rasterio.enums.Resampling.bilinear,
                ).astype(np.float32)
                stack.append(ts_data)

        stack_arr = np.stack(stack, axis=0)  # (n_dates, bands, h, w)
        filled = target.copy()

        for b in range(target.shape[0]):
            cloud_mask = (target[b] == nodata_val) | np.isnan(target[b])
            if not cloud_mask.any():
                continue
            # Find pixels to fill
            for d in range(1, len(stack)):
                fill_source = stack_arr[d, b]
                valid_fill = (fill_source != nodata_val) & ~np.isnan(fill_source)
                needs_fill = cloud_mask & valid_fill
                filled[b] = np.where(needs_fill, fill_source, filled[b])
                cloud_mask = cloud_mask & ~valid_fill

        profile.update(dtype="float32", nodata=nodata_val)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(filled)

        logger.info(f"Cloud fill ({method}) → {out_path}")
        return ProcessingResult(
            success=True,
            operation=f"cloud_fill:{method}",
            input_path=inp,
            output_path=out_path,
        )

    # ──────────────────────────────────────────────────────────────────────
    # C: Geometric Correction
    # ──────────────────────────────────────────────────────────────────────

    @_timed
    def reproject(
        self,
        input: str | Path,
        crs: str = "EPSG:4326",
        output: str | None = None,
        resampling: str = "bilinear",
        resolution: float | None = None,
    ) -> ProcessingResult:
        """
        Reproject raster to a new coordinate reference system.

        Args:
            input:      Input raster path.
            crs:        Target CRS (any rasterio-accepted string or EPSG code).
            output:     Output path.
            resampling: ``"nearest"``, ``"bilinear"``, ``"cubic"``,
                        ``"lanczos"``, ``"average"``.
            resolution: Target resolution in target CRS units (optional).

        Example::

            result = client.preprocess.reproject("scene.tif", crs="EPSG:32618")
            result = client.preprocess.reproject("scene.tif", crs="EPSG:4326", resolution=0.0001)
        """
        rasterio = _require_rasterio()
        from rasterio.enums import Resampling as RS
        from rasterio.warp import calculate_default_transform
        from rasterio.warp import reproject as warp_reproject

        rs_map = {
            "nearest": RS.nearest,
            "bilinear": RS.bilinear,
            "cubic": RS.cubic,
            "lanczos": RS.lanczos,
            "average": RS.average,
        }
        rs_method = rs_map.get(resampling, RS.bilinear)

        inp = Path(input)
        out_path = _resolve_output(inp, output, f"reproj_{crs.replace(':', '_')}")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(inp) as src:
            transform, width, height = calculate_default_transform(
                src.crs,
                crs,
                src.width,
                src.height,
                *src.bounds,
                resolution=resolution,
            )
            profile = src.profile.copy()
            profile.update(
                crs=crs,
                transform=transform,
                width=width,
                height=height,
                dtype="float32",
            )
            with rasterio.open(out_path, "w", **profile) as dst:
                for i in range(1, src.count + 1):
                    warp_reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=crs,
                        resampling=rs_method,
                    )

        # Validate the output — detect identity/pixel-space transforms
        try:
            with rasterio.open(out_path) as dst:
                t = dst.transform
                is_identity = (
                    abs(t.a) == 1.0
                    and abs(t.e) == 1.0
                    and t.c == 0.0
                    and dst.crs is not None
                    and dst.crs.is_projected
                )
            if is_identity:
                out_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Reprojection to {crs} produced an identity/pixel-space transform "
                    f"(a={t.a}, origin=({t.c},{t.f})). "
                    "The source CRS or bounding box may be invalid. "
                    "Check that the input file has a valid georeference."
                )
        except RuntimeError:
            raise
        except Exception:
            pass  # validation error shouldn't block a successful reproject

        logger.info(f"Reprojected → {crs} → {out_path}")
        return ProcessingResult(
            success=True,
            operation="reproject",
            input_path=inp,
            output_path=out_path,
            metadata={"crs": crs, "resampling": resampling},
        )

    @_timed
    def clip(
        self,
        input: str | Path,
        bbox: BBox | None = None,
        geometry: str | Path | dict | None = None,
        output: str | None = None,
        all_touched: bool = False,
    ) -> ProcessingResult:
        """
        Clip (crop / mask) a raster to a bounding box or polygon geometry.

        Args:
            input:       Input raster.
            bbox:        ``(minx, miny, maxx, maxy)`` bounding box.
            geometry:    GeoJSON file path, GeoJSON dict, or shapely geometry.
            output:      Output path.
            all_touched: Include pixels touching boundary.

        Example::

            result = client.preprocess.clip("scene.tif", bbox=(-74.1, 40.6, -73.7, 40.9))
            result = client.preprocess.clip("scene.tif", geometry="study_area.geojson")
        """
        rasterio = _require_rasterio()
        import json

        from rasterio.mask import mask as rasterio_mask

        inp = Path(input)
        out_path = _resolve_output(inp, output, "clipped")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Build geometry list
        if bbox is not None:
            from shapely.geometry import box

            shapes = [box(*bbox).__geo_interface__]
        elif geometry is not None:
            if isinstance(geometry, (str, Path)):
                with open(geometry) as f:
                    gj = json.load(f)
                if gj.get("type") == "FeatureCollection":
                    shapes = [feat["geometry"] for feat in gj["features"]]
                elif gj.get("type") == "Feature":
                    shapes = [gj["geometry"]]
                else:
                    shapes = [gj]
            elif isinstance(geometry, dict):
                shapes = [geometry]
            else:
                shapes = [geometry.__geo_interface__]
        else:
            msg = "Either bbox or geometry must be provided"
            raise ValueError(msg)

        with rasterio.open(inp) as src:
            out_image, out_transform = rasterio_mask(
                src, shapes, crop=True, all_touched=all_touched
            )
            profile = src.profile.copy()
            profile.update(
                dtype=out_image.dtype,
                height=out_image.shape[1],
                width=out_image.shape[2],
                transform=out_transform,
            )
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(out_image)

        logger.info(f"Clipped → {out_path}")
        return ProcessingResult(
            success=True,
            operation="clip",
            input_path=inp,
            output_path=out_path,
            metadata={"bbox": bbox},
        )

    @_timed
    def tile(
        self,
        input: str | Path,
        tile_size: int = 512,
        overlap: int = 64,
        output_dir: str | None = None,
        min_coverage: float = 0.1,
    ) -> ProcessingResult:
        """
        Split a large raster into overlapping tiles.

        Args:
            input:        Input raster.
            tile_size:    Tile size in pixels (square).
            overlap:      Overlap between adjacent tiles in pixels.
            output_dir:   Output directory.
            min_coverage: Skip tiles with less than this fraction of valid data.

        Returns:
            ProcessingResult with output_path pointing to the tiles directory.
            ``result.metadata["tile_paths"]`` lists all created tiles.

        Example::

            result = client.preprocess.tile("scene.tif", tile_size=256, overlap=32)
            tiles = result.metadata["tile_paths"]
        """
        rasterio = _require_rasterio()
        np = _require_numpy()

        inp = Path(input)
        out_dir = Path(output_dir) if output_dir else inp.parent / f"{inp.stem}_tiles"
        out_dir.mkdir(parents=True, exist_ok=True)

        tile_paths: list[Path] = []
        stride = tile_size - overlap

        with rasterio.open(inp) as src:
            h, w = src.height, src.width
            profile = src.profile.copy()

            for row in range(0, h, stride):
                for col in range(0, w, stride):
                    row_end = min(row + tile_size, h)
                    col_end = min(col + tile_size, w)

                    window = rasterio.windows.Window(
                        col, row, col_end - col, row_end - row
                    )
                    tile_data = src.read(window=window)

                    # Skip near-empty tiles
                    nodata = src.nodata
                    if nodata is not None:
                        valid_frac = np.mean(tile_data[0] != nodata)
                    else:
                        valid_frac = np.mean(tile_data[0] != 0)
                    if valid_frac < min_coverage:
                        continue

                    tile_transform = src.window_transform(window)
                    tile_profile = profile.copy()
                    tile_profile.update(
                        height=tile_data.shape[1],
                        width=tile_data.shape[2],
                        transform=tile_transform,
                    )
                    tile_path = out_dir / f"tile_{row:05d}_{col:05d}.tif"
                    with rasterio.open(tile_path, "w", **tile_profile) as t:
                        t.write(tile_data)
                    tile_paths.append(tile_path)

        logger.info(f"Tiled into {len(tile_paths)} tiles → {out_dir}")
        return ProcessingResult(
            success=True,
            operation="tile",
            input_path=inp,
            output_path=out_dir,
            metadata={
                "tile_count": len(tile_paths),
                "tile_paths": tile_paths,
                "tile_size": tile_size,
                "overlap": overlap,
            },
        )

    # ──────────────────────────────────────────────────────────────────────
    # D: Resolution & Resampling
    # ──────────────────────────────────────────────────────────────────────

    @_timed
    def resample(
        self,
        input: str | Path,
        resolution: float | None = None,
        scale_factor: float | None = None,
        method: str = "bilinear",
        output: str | None = None,
    ) -> ProcessingResult:
        """
        Resample raster to a different spatial resolution.

        Args:
            input:        Input raster.
            resolution:   Target resolution in raster CRS units.
                          For geographic CRS, use degrees; for projected, metres.
            scale_factor: Alternatively, a scale factor (0.5 = half resolution,
                          2.0 = double resolution). Cannot combine with resolution.
            method:       ``"nearest"``, ``"bilinear"``, ``"cubic"``, ``"lanczos"``,
                          ``"average"``.
            output:       Output path.

        Example::

            # 10m Sentinel-2 → 30m (Landsat-comparable)
            result = client.preprocess.resample("B02_10m.tif", resolution=30)
            # Downsample by factor of 2
            result = client.preprocess.resample("scene.tif", scale_factor=0.5)
        """
        rasterio = _require_rasterio()
        from rasterio.enums import Resampling as RS

        rs_map = {
            "nearest": RS.nearest,
            "bilinear": RS.bilinear,
            "cubic": RS.cubic,
            "lanczos": RS.lanczos,
            "average": RS.average,
        }
        rs_method = rs_map.get(method, RS.bilinear)

        inp = Path(input)
        res_label = str(resolution or scale_factor).replace(".", "p")
        out_path = _resolve_output(inp, output, f"resamp_{res_label}m")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(inp) as src:
            orig_res_x, orig_res_y = src.res

            if resolution is not None:
                scale_x = orig_res_x / resolution
                scale_y = orig_res_y / resolution
            elif scale_factor is not None:
                scale_x = scale_factor
                scale_y = scale_factor
            else:
                msg = "Provide either resolution or scale_factor"
                raise ValueError(msg)

            new_h = max(1, int(src.height * scale_y))
            new_w = max(1, int(src.width * scale_x))

            data = src.read(
                out_shape=(src.count, new_h, new_w),
                resampling=rs_method,
            )

            new_transform = src.transform * src.transform.scale(
                src.width / new_w,
                src.height / new_h,
            )
            profile = src.profile.copy()
            profile.update(height=new_h, width=new_w, transform=new_transform)

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data)

        logger.info(f"Resampled → {new_w}×{new_h} → {out_path}")
        return ProcessingResult(
            success=True,
            operation="resample",
            input_path=inp,
            output_path=out_path,
            metadata={
                "resolution": resolution,
                "method": method,
                "new_width": new_w,
                "new_height": new_h,
            },
        )

    @_timed
    def pansharpen(
        self,
        pan: str | Path,
        ms: str | Path,
        method: str = "brovey",
        output: str | None = None,
    ) -> ProcessingResult:
        """
        Pan-sharpen multispectral image using panchromatic band.

        Args:
            pan:    Panchromatic (single band, high resolution) raster.
            ms:     Multispectral (multi-band) raster.
            method: ``"brovey"``, ``"ihs"``, ``"gram_schmidt"``, ``"simple_mean"``.
            output: Output path.

        Example::

            result = client.preprocess.pansharpen(
                pan="pan_15m.tif", ms="ms_60m.tif", method="brovey"
            )
        """
        rasterio = _require_rasterio()
        np = _require_numpy()
        from rasterio.enums import Resampling

        pan_p = Path(pan)
        ms_p = Path(ms)
        out_path = _resolve_output(pan_p, output, f"pansharp_{method}")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(pan_p) as pan_src:
            pan_data = _safe_read_1(pan_src)
            pan_profile = pan_src.profile.copy()
            pan_shape = pan_data.shape

        with rasterio.open(ms_p) as ms_src:
            ms_data = ms_src.read(
                out_shape=(ms_src.count, *pan_shape),
                resampling=Resampling.bilinear,
            ).astype(np.float32)

        result = np.zeros_like(ms_data)

        if method == "brovey":
            ms_sum = ms_data.sum(axis=0) + 1e-10
            for b in range(ms_data.shape[0]):
                result[b] = (ms_data[b] / ms_sum) * pan_data

        elif method == "ihs":
            # Intensity-Hue-Saturation (simplified RGB only)
            if ms_data.shape[0] >= 3:
                r, g, b_ch = ms_data[0], ms_data[1], ms_data[2]
                intensity = (r + g + b_ch) / 3
                ratio = np.where(intensity > 0, pan_data / (intensity + 1e-10), 1)
                for i in range(ms_data.shape[0]):
                    result[i] = ms_data[i] * ratio
            else:
                result = ms_data * (pan_data / (ms_data[0] + 1e-10))

        elif method in ("gram_schmidt", "simple_mean"):
            ms_intensity = ms_data.mean(axis=0)
            ratio = np.where(ms_intensity > 0, pan_data / (ms_intensity + 1e-10), 1)
            for b in range(ms_data.shape[0]):
                result[b] = ms_data[b] * ratio

        else:
            msg = f"Unknown pansharpen method: {method!r}"
            raise ValueError(msg)

        pan_profile.update(count=ms_data.shape[0], dtype="float32")
        with rasterio.open(out_path, "w", **pan_profile) as dst:
            dst.write(result)

        logger.info(f"Pan-sharpened ({method}) → {out_path}")
        return ProcessingResult(
            success=True,
            operation=f"pansharpen:{method}",
            input_path=pan_p,
            output_path=out_path,
        )

    # ──────────────────────────────────────────────────────────────────────
    # H: Mosaicking & Compositing
    # ──────────────────────────────────────────────────────────────────────

    @_timed
    def mosaic(
        self,
        inputs: list[str | Path],
        output: str | None = None,
        method: str = "first",
    ) -> ProcessingResult:
        """
        Merge multiple rasters into a single seamless mosaic.

        Args:
            inputs: List of raster paths (must share CRS and band count).
            output: Output path.
            method: ``"first"`` (first valid pixel wins), ``"last"``,
                    ``"min"``, ``"max"``, ``"sum"``.

        Example::

            result = client.preprocess.mosaic(
                ["tile1.tif", "tile2.tif", "tile3.tif"],
                method="first"
            )
        """
        rasterio = _require_rasterio()
        from rasterio.merge import merge

        src_paths = [Path(p) for p in inputs]
        out_path = _resolve_output(src_paths[0], output, "mosaic")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        src_files = [rasterio.open(p) for p in src_paths]
        try:
            mosaic_data, mosaic_transform = merge(
                src_files, method=method if method != "sum" else "sum"
            )
            profile = src_files[0].profile.copy()
            profile.update(
                height=mosaic_data.shape[1],
                width=mosaic_data.shape[2],
                transform=mosaic_transform,
            )
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(mosaic_data)
        finally:
            for s in src_files:
                s.close()

        logger.info(f"Mosaicked {len(inputs)} rasters → {out_path}")
        return ProcessingResult(
            success=True,
            operation=f"mosaic:{method}",
            output_path=out_path,
            metadata={"n_inputs": len(inputs), "method": method},
        )

    @_timed
    def composite(
        self,
        inputs: list[str | Path],
        method: str = "median",
        output: str | None = None,
        cloud_masks: list[str | Path] | None = None,
    ) -> ProcessingResult:
        """
        Create a multi-temporal composite from a stack of rasters.

        Args:
            inputs:      List of rasters (same CRS, shape, bands).
            method:      ``"median"``, ``"mean"``, ``"max"`` (max NDVI),
                         ``"min"``, ``"best_pixel"`` (lowest cloud).
            output:      Output path.
            cloud_masks: Optional list of cloud mask rasters (0=clear, 1=cloud).

        Example::

            # Cloud-free median composite from 12 monthly scenes
            result = client.preprocess.composite(
                inputs=["jan.tif","feb.tif",...,"dec.tif"],
                method="median"
            )
        """
        rasterio = _require_rasterio()
        np = _require_numpy()

        src_paths = [Path(p) for p in inputs]
        out_path = _resolve_output(src_paths[0], output, f"composite_{method}")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Read all scenes into a stack
        stack = []
        with rasterio.open(src_paths[0]) as ref:
            profile = ref.profile.copy()
            ref_shape = (ref.count, ref.height, ref.width)
            nodata = ref.nodata

        for p in src_paths:
            with rasterio.open(p) as src:
                data = src.read(
                    out_shape=ref_shape, resampling=rasterio.enums.Resampling.bilinear
                ).astype(np.float32)
                if nodata is not None:
                    data = np.where(data == nodata, np.nan, data)
                stack.append(data)

        arr = np.stack(stack, axis=0)  # (n_dates, bands, h, w)

        # Apply cloud masks if provided
        if cloud_masks:
            for i, cm_path in enumerate(cloud_masks):
                if cm_path and i < len(stack):
                    with rasterio.open(Path(cm_path)) as cm_src:
                        cm = cm_src.read(
                            1,
                            out_shape=ref_shape[1:],
                            resampling=rasterio.enums.Resampling.nearest,
                        ).astype(float)
                    cm = np.where(cm > 0, np.nan, 1.0)
                    arr[i] = arr[i] * cm[np.newaxis, :, :]

        if method == "median":
            composite_data = np.nanmedian(arr, axis=0)
        elif method == "mean":
            composite_data = np.nanmean(arr, axis=0)
        elif method == "max":
            composite_data = np.nanmax(arr, axis=0)
        elif method == "min":
            composite_data = np.nanmin(arr, axis=0)
        elif method == "best_pixel":
            # Select pixel from scene with most valid data per pixel
            np.sum(~np.isnan(arr), axis=0)  # (bands, h, w)
            best_scene_idx = np.argmax(np.sum(~np.isnan(arr), axis=1), axis=0)  # (h, w)
            composite_data = np.zeros(ref_shape, dtype=np.float32)
            for b in range(ref_shape[0]):
                for si in range(len(stack)):
                    mask = best_scene_idx == si
                    composite_data[b][mask] = arr[si, b][mask]
        else:
            msg = f"Unknown composite method: {method!r}"
            raise ValueError(msg)

        if nodata is not None:
            composite_data = np.where(np.isnan(composite_data), nodata, composite_data)

        profile.update(dtype="float32", nodata=nodata)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(composite_data.astype(np.float32))

        logger.info(f"Composite ({method}) from {len(inputs)} scenes → {out_path}")
        return ProcessingResult(
            success=True,
            operation=f"composite:{method}",
            output_path=out_path,
            metadata={"n_inputs": len(inputs), "method": method},
        )
