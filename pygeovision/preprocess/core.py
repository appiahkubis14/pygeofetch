"""
pygeovision.preprocess.core
===========================

Complete preprocessing pipeline for satellite imagery:
  - Band stacking (single-band files → multi-band GeoTIFF)
  - Spatial clipping to bounding box or GeoJSON polygon
  - Cloud and SCL masking
  - Pixel normalisation (min-max, z-score, percentile, scale-factor)
  - Resampling to a target resolution

All operations are rasterio-based and preserve the full GeoTIFF
spatial metadata (CRS, transform, nodata).  The ``pipeline()``
method chains all steps in the correct order with a single call.

Quick start::

    from pygeovision.preprocess import Preprocessor

    pre = Preprocessor()

    # Stack 6 Sentinel-2 bands into one file
    pre.stack_bands(
        ["B02.tif", "B03.tif", "B04.tif", "B08.tif", "B11.tif", "B12.tif"],
        output_path="sentinel2_6band.tif",
        band_names=["Blue","Green","Red","NIR","SWIR1","SWIR2"],
    )

    # Clip to study area
    pre.clip_to_bbox("sentinel2_6band.tif", bbox=(-74.1,40.6,-73.7,40.9))

    # One-shot pipeline
    result = pre.pipeline(
        input_path="sentinel2_6band.tif",
        bbox=(-74.1, 40.6, -73.7, 40.9),
        cloud_mask_path="cloud_mask.tif",
        normalise="minmax",
        output_path="ready_for_model.tif",
    )
"""
from __future__ import annotations

import logging
import os
import pathlib
import re
import tempfile
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# Band name patterns for auto-discovery inside a scene directory
_SENTINEL2_BAND_RE = re.compile(
    r"(?:^|[_\-])(?P<band>B0?[1-9]|B1[0-2A]|TCI|SCL|WVP|AOT|B8A)"
    r"(?:_[A-Z0-9]+)?\.tif$",
    re.IGNORECASE,
)
_LANDSAT_BAND_RE = re.compile(
    r"(?:^|[_\-])(?P<band>B[1-9]|B10|B11)"
    r"(?:_[A-Z0-9]+)?\.tif$",
    re.IGNORECASE,
)


def _require_rasterio():
    try:
        import rasterio
        return rasterio
    except ImportError:
        raise ImportError("pip install rasterio") from None


def _require_shapely():
    try:
        from shapely.geometry import shape, box, mapping
        return shape, box, mapping
    except ImportError:
        raise ImportError("pip install shapely") from None


def _safe_output(input_path: str, output_path: Optional[str], suffix: str) -> str:
    """Return output_path, or derive it from input_path + suffix."""
    if output_path:
        pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        return output_path
    p = pathlib.Path(input_path)
    out = p.parent / f"{p.stem}{suffix}{p.suffix}"
    return str(out)


class Preprocessor:
    """Complete satellite image preprocessing pipeline.

    All methods return the output path as a string so they can be
    chained or used in list comprehensions.

    Example::

        from pygeovision.preprocess import Preprocessor

        pre = Preprocessor()

        # Stack → clip → mask → normalise in one call
        result = pre.pipeline(
            input_path  = "scene_dir/",   # or a stacked .tif
            bands       = ["B02","B03","B04","B08","B11","B12"],
            bbox        = (-74.1, 40.6, -73.7, 40.9),
            scl_path    = "SCL.tif",
            normalise   = "minmax",
            output_path = "ready.tif",
        )
    """

    def __init__(self, validator=None):
        if validator is None:
            from pygeovision.data.validator import DataValidator
            validator = DataValidator(mode="fix")
        self._v = validator

    def validate(self, path: str, **kwargs):
        """Run the DataValidator on a GeoTIFF and return the report."""
        return self._v.validate(path, **kwargs)

    # ------------------------------------------------------------------
    # 1. Band Stacking
    # ------------------------------------------------------------------

    def stack_bands(
        self,
        band_paths: Union[List[str], Dict[str, str]],
        output_path: str,
        band_names: Optional[List[str]] = None,
    ) -> str:
        """Stack individual single-band GeoTIFFs into one multi-band raster.

        Args:
            band_paths: Ordered list of single-band .tif paths, or a
                dict mapping ``{band_name: path}``.
            output_path: Destination path for the stacked raster.
            band_names: Optional list of names to write as band
                descriptions (must match length of band_paths).

        Returns:
            ``output_path``

        Example::

            pre.stack_bands(
                ["B02.tif","B03.tif","B04.tif","B08.tif"],
                output_path="s2_4band.tif",
                band_names=["Blue","Green","Red","NIR"],
            )
        """
        rasterio = _require_rasterio()

        if isinstance(band_paths, dict):
            if band_names is None:
                band_names = list(band_paths.keys())
            band_paths = list(band_paths.values())

        if not band_paths:
            raise ValueError("band_paths is empty")

        pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Read the first band to get spatial metadata
        with rasterio.open(band_paths[0]) as src0:
            profile = src0.profile.copy()
            crs       = src0.crs
            transform = src0.transform
            width     = src0.width
            height    = src0.height
            dtype     = src0.dtypes[0]

        profile.update(count=len(band_paths), compress="lzw")

        with rasterio.open(output_path, "w", **profile) as dst:
            for idx, bp in enumerate(band_paths, start=1):
                with rasterio.open(bp) as src:
                    data = src.read(1)
                    # Resample to match first band if sizes differ
                    if data.shape != (height, width):
                        from rasterio.enums import Resampling
                        data = src.read(
                            1,
                            out_shape=(height, width),
                            resampling=Resampling.bilinear,
                        )
                dst.write(data, idx)
                if band_names and idx <= len(band_names):
                    dst.update_tags(idx, name=band_names[idx - 1])

        n = len(band_paths)
        logger.info("Stacked %d bands → %s", n, output_path)
        return output_path

    def stack_from_dir(
        self,
        scene_dir: str,
        band_names: List[str],
        output_path: str,
        pattern: str = "*.tif",
        sensor: str = "auto",
    ) -> str:
        """Discover band files inside a Sentinel-2 / Landsat scene directory
        and stack them.

        The method matches filenames against band names using flexible
        pattern matching — it works with both the SAFE folder layout
        (``T30UWD_B02_10m.tif``) and simple names (``B02.tif``).

        Args:
            scene_dir: Directory containing single-band .tif files.
            band_names: Ordered list of bands to include, e.g.
                ``["B02","B03","B04","B08","B11","B12"]``.
            output_path: Destination path for the stacked raster.
            pattern: Glob pattern to find candidate files.
            sensor: ``"sentinel2"`` | ``"landsat"`` | ``"auto"``
                (auto-detect from filenames).

        Returns:
            ``output_path``

        Example::

            pre.stack_from_dir(
                scene_dir  = "./downloads/S2C_20240628/",
                band_names = ["B02","B03","B04","B08","B11","B12"],
                output_path= "s2_6band.tif",
            )
        """
        scene_dir = pathlib.Path(scene_dir)
        candidates = sorted(scene_dir.rglob(pattern))

        if not candidates:
            raise FileNotFoundError(
                f"No .tif files found in {scene_dir}. "
                "Check that the scene was downloaded and the pattern is correct."
            )

        # Build a name→path lookup
        name_to_path: Dict[str, pathlib.Path] = {}
        for p in candidates:
            fname = p.name.upper()
            for bn in band_names:
                token = bn.upper()
                # Match "_B02_", "_B02.", "B02.tif", "B02_10m.tif" …
                if re.search(rf"(?:^|[_\-]){re.escape(token)}(?:[_\.\-]|$)", fname):
                    if token not in name_to_path:  # first match wins
                        name_to_path[token] = p

        ordered_paths = []
        missing = []
        for bn in band_names:
            key = bn.upper()
            if key in name_to_path:
                ordered_paths.append(str(name_to_path[key]))
                logger.debug("Band %s → %s", bn, name_to_path[key].name)
            else:
                missing.append(bn)

        if missing:
            found_names = [p.name for p in candidates]
            raise FileNotFoundError(
                f"Could not find band(s) {missing} in {scene_dir}.\n"
                f"Files present: {found_names[:10]}{'…' if len(found_names)>10 else ''}\n"
                "Tip: check that the downloaded scene contains the expected band files."
            )

        return self.stack_bands(ordered_paths, output_path, band_names=band_names)

    # ------------------------------------------------------------------
    # 2. Spatial Clipping
    # ------------------------------------------------------------------

    def clip_to_bbox(
        self,
        input_path: str,
        bbox: Tuple[float, float, float, float],
        output_path: Optional[str] = None,
        bbox_crs: str = "EPSG:4326",
        all_touched: bool = False,
    ) -> str:
        """Clip a raster to a bounding box.

        The bounding box is automatically reprojected to match the
        raster's CRS when they differ.

        Args:
            input_path: Source GeoTIFF path.
            bbox: ``(min_lon, min_lat, max_lon, max_lat)`` in WGS84
                (or ``bbox_crs`` if specified).
            output_path: Destination path. If ``None``, appends
                ``"_clipped"`` to the stem.
            bbox_crs: CRS of the supplied bbox (default ``EPSG:4326``).
            all_touched: If ``True``, include pixels touching the bbox
                boundary (rasterio ``all_touched`` option).

        Returns:
            Path of the clipped raster.

        Example::

            clipped = pre.clip_to_bbox(
                "s2_6band.tif",
                bbox=(-74.1, 40.6, -73.7, 40.9),
            )
        """
        rasterio = _require_rasterio()
        from rasterio.mask import mask as rio_mask
        from rasterio.crs import CRS
        import rasterio.warp as warp

        output_path = _safe_output(input_path, output_path, "_clipped")
        _shape, _box, _mapping = _require_shapely()

        with rasterio.open(input_path) as src:
            raster_crs = src.crs

            # Reproject bbox to raster CRS if they differ
            src_crs = CRS.from_user_input(bbox_crs)
            if src_crs != raster_crs:
                minx, miny, maxx, maxy = bbox
                xs, ys = warp.transform(src_crs, raster_crs,
                                         [minx, maxx], [miny, maxy])
                minx, maxx = min(xs), max(xs)
                miny, maxy = min(ys), max(ys)
                bbox_proj = (minx, miny, maxx, maxy)
            else:
                bbox_proj = bbox

            geom = [_mapping(_box(*bbox_proj))]
            data, transform = rio_mask(src, geom,
                                        crop=True, all_touched=all_touched)
            profile = src.profile.copy()
            profile.update(
                height=data.shape[1],
                width=data.shape[2],
                transform=transform,
                compress="lzw",
            )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data)

        orig_px = _raster_shape(input_path)
        new_px  = (data.shape[1], data.shape[2])
        pct     = 100 * new_px[0] * new_px[1] / max(orig_px[0] * orig_px[1], 1)
        logger.info(
            "Clipped to bbox: %dx%d → %dx%d (%.1f%% of original) → %s",
            orig_px[1], orig_px[0], new_px[1], new_px[0], pct, output_path,
        )
        return output_path

    def clip_to_polygon(
        self,
        input_path: str,
        geojson: Union[str, dict],
        output_path: Optional[str] = None,
        crop: bool = True,
        all_touched: bool = False,
        nodata: Optional[float] = None,
    ) -> str:
        """Clip a raster to one or more GeoJSON polygons.

        Args:
            input_path: Source GeoTIFF path.
            geojson: Path to a ``.geojson`` / ``.json`` file, or a
                Python dict (GeoJSON FeatureCollection or Geometry).
            output_path: Destination path. Defaults to ``"_clipped"``.
            crop: Crop the output extent to the geometry bounds.
            all_touched: Include pixels touching the polygon edge.
            nodata: Override nodata value for masked pixels.

        Returns:
            Path of the clipped raster.

        Example::

            pre.clip_to_polygon(
                "s2_6band.tif",
                geojson="study_area.geojson",
                output_path="s2_study.tif",
            )
        """
        rasterio = _require_rasterio()
        from rasterio.mask import mask as rio_mask
        _shape, _box, _mapping = _require_shapely()

        import json

        # Load GeoJSON
        if isinstance(geojson, str):
            with open(geojson) as f:
                gj = json.load(f)
        else:
            gj = geojson

        # Extract geometries (support FeatureCollection, Feature, or Geometry)
        if gj.get("type") == "FeatureCollection":
            geoms = [f["geometry"] for f in gj["features"]]
        elif gj.get("type") == "Feature":
            geoms = [gj["geometry"]]
        else:
            geoms = [gj]  # plain Geometry

        output_path = _safe_output(input_path, output_path, "_clipped")

        with rasterio.open(input_path) as src:
            _nodata = nodata if nodata is not None else src.nodata
            data, transform = rio_mask(
                src, geoms,
                crop=crop,
                all_touched=all_touched,
                nodata=_nodata,
            )
            profile = src.profile.copy()
            profile.update(
                height=data.shape[1],
                width=data.shape[2],
                transform=transform,
                nodata=_nodata,
                compress="lzw",
            )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data)

        logger.info("Clipped to polygon → %s (%dx%d px)",
                     output_path, data.shape[2], data.shape[1])
        return output_path

    # ------------------------------------------------------------------
    # 3. Cloud Masking
    # ------------------------------------------------------------------

    def apply_cloud_mask(
        self,
        input_path: str,
        mask_path: str,
        output_path: Optional[str] = None,
        nodata: float = 0.0,
        invert: bool = False,
    ) -> str:
        """Set cloud-contaminated pixels to nodata.

        Args:
            input_path: Source multi-band GeoTIFF.
            mask_path: Cloud mask GeoTIFF where **1 = cloud** (set to
                nodata) and **0 = clear** (kept). Set ``invert=True``
                to flip the convention.
            output_path: Destination path. Defaults to ``"_masked"``.
            nodata: Value written to masked pixels (default 0).
            invert: If ``True``, treat 0 as cloud and 1 as clear.

        Returns:
            Path of the masked raster.
        """
        rasterio = _require_rasterio()
        from rasterio.enums import Resampling

        output_path = _safe_output(input_path, output_path, "_masked")

        with rasterio.open(input_path) as src, \
             rasterio.open(mask_path) as msrc:

            data    = src.read().astype(np.float32)
            profile = src.profile.copy()

            # Read mask, resampling to match data if needed
            cloud   = msrc.read(
                1,
                out_shape=(data.shape[1], data.shape[2]),
                resampling=Resampling.nearest,
            ).astype(bool)

        if invert:
            cloud = ~cloud          # flip: 0=cloud, 1=clear → 0=clear, 1=cloud

        # Apply mask to all bands
        for b in range(data.shape[0]):
            data[b][cloud] = nodata

        profile.update(nodata=nodata, dtype="float32", compress="lzw")
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data)

        masked_pct = cloud.mean() * 100
        logger.info("Cloud mask applied: %.1f%% pixels masked → %s",
                     masked_pct, output_path)
        return output_path

    def apply_scl(
        self,
        input_path: str,
        scl_path: str,
        output_path: Optional[str] = None,
        keep_classes: Sequence[int] = (4, 5, 6),
        nodata: float = 0.0,
    ) -> str:
        """Mask clouds and shadows using the Sentinel-2 Scene Classification
        Layer (SCL).

        SCL class codes:
          1=saturated  2=dark  3=shadow  4=vegetation  5=not_vegetation
          6=water  7=unclassified  8=cloud_medium  9=cloud_high
          10=thin_cirrus  11=snow

        Args:
            input_path: Source multi-band GeoTIFF (Sentinel-2).
            scl_path: SCL band GeoTIFF (usually ``*_SCL_20m.tif``).
            output_path: Destination path. Defaults to ``"_scl"``.
            keep_classes: SCL values to *keep* unmasked.
                Default ``(4, 5, 6)`` keeps vegetation, bare soil, water.
                Use ``(4, 5, 6, 7, 11)`` to also keep unclassified and snow.
            nodata: Value for masked pixels.

        Returns:
            Path of the masked raster.

        Example::

            # Mask clouds, shadow, and snow — keep only clear land/water
            pre.apply_scl(
                "s2_6band.tif",
                scl_path="SCL.tif",
                keep_classes=[4, 5, 6],   # vegetation, bare soil, water
            )
        """
        rasterio = _require_rasterio()
        from rasterio.enums import Resampling

        output_path = _safe_output(input_path, output_path, "_scl")
        keep_set = set(keep_classes)

        with rasterio.open(input_path) as src, \
             rasterio.open(scl_path) as sclsrc:

            data    = src.read().astype(np.float32)
            profile = src.profile.copy()

            scl = sclsrc.read(
                1,
                out_shape=(data.shape[1], data.shape[2]),
                resampling=Resampling.nearest,
            ).astype(int)

        # Mask = True where pixel should be removed
        mask = ~np.isin(scl, sorted(keep_set))
        for b in range(data.shape[0]):
            data[b][mask] = nodata

        profile.update(nodata=nodata, dtype="float32", compress="lzw")
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data)

        masked_pct = mask.mean() * 100
        classes_removed = sorted(set(np.unique(scl).tolist()) - keep_set)
        logger.info(
            "SCL mask: %.1f%% pixels removed (classes %s) → %s",
            masked_pct, classes_removed, output_path,
        )
        return output_path

    # ------------------------------------------------------------------
    # 4. Normalisation
    # ------------------------------------------------------------------

    def normalise(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        method: str = "minmax",
        scale_factor: float = 10000.0,
        percentile: float = 2.0,
        out_dtype: str = "float32",
    ) -> str:
        """Normalise pixel values across all bands.

        Methods
        -------
        ``minmax``
            Scale each band independently to [0, 1] using its
            per-band min and max.

        ``zscore``
            Standardise each band to zero mean, unit variance
            (``(x - mean) / std``).

        ``percentile``
            Clip to the [p, 100-p] percentile range then scale to [0, 1].
            Robust to outliers and sensor saturation.

        ``scale_factor``
            Divide all bands by ``scale_factor`` (e.g. 10000 for HLS /
            Sentinel-2 L2A reflectance → [0, 1] range).

        Args:
            input_path: Source GeoTIFF.
            output_path: Destination path. Defaults to ``"_norm"``.
            method: Normalisation method (see above).
            scale_factor: Divisor for ``"scale_factor"`` method.
            percentile: Lower-tail percentile for ``"percentile"`` method
                (upper tail = 100 - percentile).
            out_dtype: NumPy dtype for the output file
                (``"float32"`` or ``"float64"``).

        Returns:
            Path of the normalised raster.

        Example::

            # Sentinel-2 L2A (DN → reflectance)
            pre.normalise("s2_6band.tif", method="scale_factor", scale_factor=10000.0)

            # Robust stretch for visualisation
            pre.normalise("s2_6band.tif", method="percentile", percentile=2.0)
        """
        rasterio = _require_rasterio()

        output_path = _safe_output(input_path, output_path, "_norm")
        VALID = {"minmax", "zscore", "percentile", "scale_factor"}
        if method not in VALID:
            raise ValueError(f"method must be one of {VALID}, got '{method}'")

        with rasterio.open(input_path) as src:
            data    = src.read().astype(np.float64)
            profile = src.profile.copy()
            nodata  = src.nodata

        out = np.zeros_like(data, dtype=np.float64)

        if method == "scale_factor":
            out = data / scale_factor

        elif method == "minmax":
            for b in range(data.shape[0]):
                band = data[b]
                if nodata is not None:
                    valid = band[band != nodata]
                else:
                    valid = band.ravel()
                mn, mx = valid.min(), valid.max()
                denom = mx - mn if mx - mn > 1e-10 else 1.0
                out[b] = (band - mn) / denom
                if nodata is not None:
                    out[b][band == nodata] = nodata

        elif method == "zscore":
            for b in range(data.shape[0]):
                band = data[b]
                if nodata is not None:
                    valid = band[band != nodata]
                else:
                    valid = band.ravel()
                mu, sigma = valid.mean(), valid.std()
                sigma = sigma if sigma > 1e-10 else 1.0
                out[b] = (band - mu) / sigma
                if nodata is not None:
                    out[b][band == nodata] = nodata

        elif method == "percentile":
            for b in range(data.shape[0]):
                band = data[b]
                if nodata is not None:
                    valid = band[band != nodata]
                else:
                    valid = band.ravel()
                lo = np.percentile(valid, percentile)
                hi = np.percentile(valid, 100.0 - percentile)
                denom = hi - lo if hi - lo > 1e-10 else 1.0
                out[b] = np.clip((band - lo) / denom, 0.0, 1.0)
                if nodata is not None:
                    out[b][band == nodata] = nodata

        out = out.astype(out_dtype)
        profile.update(dtype=out_dtype, compress="lzw")
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(out)

        logger.info("Normalised (%s) → %s", method, output_path)
        return output_path

    # ------------------------------------------------------------------
    # 5. Resampling
    # ------------------------------------------------------------------

    def resample(
        self,
        input_path: str,
        resolution_m: float,
        output_path: Optional[str] = None,
        resampling: str = "bilinear",
    ) -> str:
        """Resample a raster to a target ground sampling distance.

        Args:
            input_path: Source GeoTIFF.
            resolution_m: Target pixel size in metres.
            output_path: Destination path. Defaults to
                ``"_<res>m"``.
            resampling: Resampling algorithm —
                ``"nearest"`` | ``"bilinear"`` | ``"cubic"`` |
                ``"lanczos"`` | ``"average"`` | ``"mode"``

        Returns:
            Path of the resampled raster.

        Example::

            # Upsample Sentinel-2 20m bands to 10m
            pre.resample("s2_20m.tif", resolution_m=10.0)
        """
        rasterio = _require_rasterio()
        from rasterio.enums import Resampling as _R
        import rasterio.transform as rt

        _ALG = {
            "nearest":  _R.nearest,
            "bilinear": _R.bilinear,
            "cubic":    _R.cubic,
            "lanczos":  _R.lanczos,
            "average":  _R.average,
            "mode":     _R.mode,
        }
        if resampling not in _ALG:
            raise ValueError(f"resampling must be one of {list(_ALG)}")

        suffix = f"_{int(resolution_m)}m"
        output_path = _safe_output(input_path, output_path, suffix)

        with rasterio.open(input_path) as src:
            # Current pixel size in the file's CRS (assumed metres for projected)
            src_res_x = abs(src.transform.a)
            scale     = src_res_x / resolution_m

            new_h = max(1, int(round(src.height * scale)))
            new_w = max(1, int(round(src.width  * scale)))

            data = src.read(
                out_shape=(src.count, new_h, new_w),
                resampling=_ALG[resampling],
            )

            new_transform = src.transform * src.transform.scale(
                src.width  / new_w,
                src.height / new_h,
            )
            profile = src.profile.copy()
            profile.update(
                height=new_h, width=new_w,
                transform=new_transform,
                compress="lzw",
            )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data)

        logger.info(
            "Resampled %.1f m → %.1f m (%dx%d → %dx%d) → %s",
            src_res_x, resolution_m,
            src.width, src.height, new_w, new_h,
            output_path,
        )
        return output_path

    # ------------------------------------------------------------------
    # 6. Full Pipeline
    # ------------------------------------------------------------------

    def pipeline(
        self,
        input_path: str,
        output_path: str,
        *,
        # stacking
        stack_bands: Optional[List[str]] = None,
        stack_dir: Optional[str] = None,
        sensor: str = "auto",
        # spatial
        bbox: Optional[Tuple[float, float, float, float]] = None,
        bbox_crs: str = "EPSG:4326",
        clip_geojson: Optional[Union[str, dict]] = None,
        # cloud masking
        cloud_mask_path: Optional[str] = None,
        scl_path: Optional[str] = None,
        scl_keep_classes: Sequence[int] = (4, 5, 6),
        nodata: float = 0.0,
        # normalisation
        normalise: Optional[str] = None,
        scale_factor: float = 10000.0,
        percentile: float = 2.0,
        # resampling
        resample_m: Optional[float] = None,
        resampling: str = "bilinear",
        # output
        out_dtype: str = "float32",
        keep_intermediates: bool = False,
    ) -> Dict:
        """Run a full preprocessing pipeline in the correct order.

        Pipeline order (each step is skipped when its argument is None):

        1. **Stack** — assemble individual band files into a multi-band raster
        2. **Clip** — crop to ``bbox`` or ``clip_geojson`` study area
        3. **Cloud mask** — apply binary cloud mask or Sentinel-2 SCL
        4. **Normalise** — scale pixel values (minmax / zscore / percentile / scale_factor)
        5. **Resample** — change pixel resolution

        Args:
            input_path: A stacked multi-band GeoTIFF **or** a scene
                directory (when ``stack_bands`` is set).
            output_path: Final output path for the fully preprocessed raster.
            stack_bands: Band names to discover and stack from
                ``input_path`` (treated as a directory).
                E.g. ``["B02","B03","B04","B08","B11","B12"]``.
            stack_dir: Alternative scene directory for stacking
                (overrides ``input_path`` for this step).
            sensor: Hint for band discovery (``"auto"``, ``"sentinel2"``,
                ``"landsat"``).
            bbox: ``(min_lon, min_lat, max_lon, max_lat)`` bounding box
                (WGS84 by default).
            bbox_crs: CRS of the supplied ``bbox``.
            clip_geojson: GeoJSON path or dict for polygon clipping.
                Used instead of ``bbox`` when both are supplied.
            cloud_mask_path: Binary cloud mask (1=cloud, 0=clear).
            scl_path: Sentinel-2 SCL band for cloud/shadow masking.
                Applied after ``cloud_mask_path`` when both are given.
            scl_keep_classes: SCL class codes to keep (default 4,5,6).
            nodata: Value for masked / invalid pixels.
            normalise: Normalisation method — ``"minmax"``, ``"zscore"``,
                ``"percentile"``, ``"scale_factor"``, or ``None`` to skip.
            scale_factor: Divisor for ``"scale_factor"`` normalisation
                (10000 for Sentinel-2 L2A).
            percentile: Tail percentile for ``"percentile"`` normalisation.
            resample_m: Target resolution in metres. ``None`` to skip.
            resampling: Resampling algorithm (see :meth:`resample`).
            out_dtype: NumPy dtype for the final output.
            keep_intermediates: If ``True``, intermediate files are kept
                alongside the final output; otherwise they are deleted.

        Returns:
            Dict with keys:
              - ``"output_path"`` — final file path
              - ``"steps_applied"`` — ordered list of steps that ran
              - ``"shape"`` — ``(bands, height, width)``
              - ``"resolution_m"`` — pixel size of the output

        Example::

            result = pre.pipeline(
                input_path   = "./downloads/S2C_20240628/",
                output_path  = "ready.tif",
                stack_bands  = ["B02","B03","B04","B08","B11","B12"],
                bbox         = (-74.1, 40.6, -73.7, 40.9),
                scl_path     = "./downloads/S2C_20240628/SCL.tif",
                normalise    = "scale_factor",
                scale_factor = 10000.0,
            )
            print(result["shape"], result["resolution_m"])
        """
        tmpdir = tempfile.mkdtemp(prefix="pgv_preprocess_")
        tmp    = pathlib.Path(tmpdir)
        intermediates: List[str] = []
        steps_applied: List[str] = []

        def _tmp(suffix: str) -> str:
            return str(tmp / f"step_{len(steps_applied):02d}_{suffix}.tif")

        current = input_path

        # ── Step 1: Stack ───────────────────────────────────────────────
        if stack_bands:
            src_dir = stack_dir or (
                input_path if pathlib.Path(input_path).is_dir() else
                str(pathlib.Path(input_path).parent)
            )
            stacked = _tmp("stacked")
            self.stack_from_dir(
                scene_dir=src_dir,
                band_names=stack_bands,
                output_path=stacked,
                sensor=sensor,
            )
            intermediates.append(stacked)
            current = stacked
            steps_applied.append(f"stack({len(stack_bands)} bands)")

        # ── Step 2: Clip ────────────────────────────────────────────────
        if clip_geojson is not None:
            clipped = _tmp("clipped")
            self.clip_to_polygon(current, clip_geojson, output_path=clipped, nodata=nodata)
            intermediates.append(clipped)
            current = clipped
            steps_applied.append("clip_polygon")
        elif bbox is not None:
            clipped = _tmp("clipped")
            self.clip_to_bbox(current, bbox, output_path=clipped, bbox_crs=bbox_crs)
            intermediates.append(clipped)
            current = clipped
            steps_applied.append(f"clip_bbox{bbox}")

        # ── Step 3: Cloud masking ───────────────────────────────────────
        if cloud_mask_path is not None:
            masked = _tmp("masked")
            self.apply_cloud_mask(current, cloud_mask_path,
                                    output_path=masked, nodata=nodata)
            intermediates.append(masked)
            current = masked
            steps_applied.append("cloud_mask")

        if scl_path is not None:
            scl_out = _tmp("scl")
            self.apply_scl(current, scl_path, output_path=scl_out,
                            keep_classes=scl_keep_classes, nodata=nodata)
            intermediates.append(scl_out)
            current = scl_out
            steps_applied.append(f"scl(keep={list(scl_keep_classes)})")

        # ── Step 4: Normalise ───────────────────────────────────────────
        if normalise:
            normed = _tmp("norm")
            self.normalise(
                current, output_path=normed,
                method=normalise,
                scale_factor=scale_factor,
                percentile=percentile,
                out_dtype=out_dtype,
            )
            intermediates.append(normed)
            current = normed
            steps_applied.append(f"normalise({normalise})")

        # ── Step 5: Resample ────────────────────────────────────────────
        if resample_m is not None:
            resampled = _tmp("resampled")
            self.resample(current, resample_m,
                           output_path=resampled, resampling=resampling)
            intermediates.append(resampled)
            current = resampled
            steps_applied.append(f"resample({resample_m}m)")

        # ── Write final output ──────────────────────────────────────────
        pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        if current != output_path:
            import shutil
            shutil.copy2(current, output_path)

        # Clean up intermediates
        if not keep_intermediates:
            for f in intermediates:
                try:
                    os.remove(f)
                except OSError:
                    pass
            try:
                os.rmdir(tmpdir)
            except OSError:
                pass

        # Read final metadata
        rasterio = _require_rasterio()
        with rasterio.open(output_path) as dst:
            shape_out    = (dst.count, dst.height, dst.width)
            resolution_m = abs(dst.transform.a)

        if not steps_applied:
            steps_applied.append("passthrough (no steps configured)")

        logger.info(
            "Pipeline complete: %s | shape=%s | res=%.1fm | steps=%s",
            output_path, shape_out, resolution_m, steps_applied,
        )
        return {
            "output_path":  output_path,
            "steps_applied": steps_applied,
            "shape":         shape_out,
            "resolution_m":  resolution_m,
        }

    # ------------------------------------------------------------------
    # 7. Convenience helpers
    # ------------------------------------------------------------------

    def info(self, path: str) -> Dict:
        """Return a summary dict for a GeoTIFF.

        Useful for quick inspection before and after preprocessing.

        Returns:
            Dict with ``crs``, ``shape``, ``bands``, ``resolution_m``,
            ``bbox``, ``dtype``, ``nodata``, ``value_range``.
        """
        rasterio = _require_rasterio()
        with rasterio.open(path) as src:
            data  = src.read()
            valid = data[data != src.nodata] if src.nodata is not None else data.ravel()
            return {
                "path":         path,
                "crs":          str(src.crs),
                "shape":        (src.count, src.height, src.width),
                "bands":        src.count,
                "resolution_m": abs(src.transform.a),
                "bbox":         list(src.bounds),
                "dtype":        src.dtypes[0],
                "nodata":       src.nodata,
                "value_range":  (float(valid.min()), float(valid.max()))
                                 if valid.size else (None, None),
            }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _raster_shape(path: str) -> Tuple[int, int]:
    rasterio = _require_rasterio()
    with rasterio.open(path) as src:
        return src.height, src.width
