"""
SARProcessor — F: SAR-specific processing.
Speckle filtering, radiometric calibration, flood mapping.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

from pygeofetch.processing.base import (
    ProcessingResult, _require_rasterio, _require_numpy,
    _resolve_output, _timed,
)

logger = logging.getLogger(__name__)


class SARProcessor:
    """
    SAR (Synthetic Aperture Radar) processing engine.

    Example::

        from pygeofetch import PyGeoFetch
        client = PyGeoFetch()
        result = client.sar.despeckle("sentinel1_vv.tif", filter="lee")
        result = client.sar.flood_map("sentinel1.tif", threshold=-15.0)
    """

    @_timed
    def despeckle(
        self,
        input: Union[str, Path],
        filter: str = "lee",
        window: int = 5,
        output: Optional[str] = None,
        num_looks: int = 1,
    ) -> ProcessingResult:
        """
        SAR speckle filtering.

        Args:
            input:     Input SAR GeoTIFF (linear amplitude or power).
            filter:    ``"lee"``, ``"frost"``, ``"gamma"``, ``"enhanced_lee"``,
                       ``"boxcar"``.
            window:    Filter window size (must be odd).
            output:    Output path.
            num_looks: Number of looks (affects Lee filter threshold).

        Example::

            result = client.sar.despeckle("sentinel1_vv.tif", filter="enhanced_lee")
        """
        rasterio = _require_rasterio()
        np = _require_numpy()
        from scipy.ndimage import uniform_filter, generic_filter, gaussian_filter

        inp = Path(input)
        out_path = _resolve_output(inp, output, f"despeckle_{filter}")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(inp) as src:
            profile = src.profile.copy()
            data = src.read(1).astype(np.float32)
            nodata = src.nodata

        valid = (data != nodata) & np.isfinite(data) if nodata is not None else np.isfinite(data)
        result = np.where(valid, data, 0)

        if filter == "boxcar":
            result = uniform_filter(result, size=window)

        elif filter == "lee":
            # Lee filter: estimate local variance to determine adaptive weight
            mean  = uniform_filter(result, size=window)
            sq_mean = uniform_filter(result**2, size=window)
            var   = sq_mean - mean**2
            # Equivalent number of looks ENL estimate
            enl  = mean**2 / (var + 1e-10)
            weight = var / (var + mean**2 / (num_looks + 1e-10) + 1e-10)
            result = mean + weight * (result - mean)

        elif filter == "enhanced_lee":
            mean  = uniform_filter(result, size=window)
            sq_mean = uniform_filter(result**2, size=window)
            var   = np.maximum(sq_mean - mean**2, 0)
            cv_local  = np.sqrt(var) / (mean + 1e-10)
            cv_thresh = 1.0 / np.sqrt(num_looks + 1e-10)
            # Pixels below CV threshold → use local mean; above → use Frost-like
            weight = np.where(cv_local <= cv_thresh, 0.0,
                              np.where(cv_local > 3 * cv_thresh, 1.0,
                                       (cv_local - cv_thresh) / (2 * cv_thresh)))
            result = (1 - weight) * mean + weight * result

        elif filter == "frost":
            # Frost filter: exponential damping
            k = 1.0  # damping factor
            mean  = uniform_filter(result, size=window)
            sq_mean = uniform_filter(result**2, size=window)
            var   = sq_mean - mean**2
            cv    = np.sqrt(np.maximum(var, 0)) / (mean + 1e-10)
            alpha = k * cv
            # Approximation using Gaussian with alpha-based sigma
            from scipy.ndimage import gaussian_filter as gf
            smoothed = gf(result, sigma=window / (2 * (1 + alpha.mean()) + 1e-10))
            weight = np.exp(-alpha * window)
            result = weight * result + (1 - weight) * smoothed

        elif filter == "gamma":
            # Gamma MAP filter
            mean  = uniform_filter(result, size=window)
            sq_mean = uniform_filter(result**2, size=window)
            var   = np.maximum(sq_mean - mean**2, 0)
            b     = (var - mean**2 / num_looks) / (var + mean**2 / (num_looks + 1e-10) + 1e-10)
            b     = np.clip(b, 0, 1)
            result = mean + b * (result - mean)

        else:
            raise ValueError(f"Unknown SAR filter: {filter!r}")

        if nodata is not None:
            result = np.where(valid, result, nodata)

        profile.update(dtype="float32", nodata=nodata)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(result[np.newaxis, :, :].astype(np.float32))

        logger.info(f"SAR despeckle ({filter}) → {out_path}")
        return ProcessingResult(
            success=True, operation=f"despeckle:{filter}",
            input_path=inp, output_path=out_path,
            metadata={"filter": filter, "window": window},
        )

    @_timed
    def calibrate(
        self,
        input: Union[str, Path],
        output_type: str = "sigma0",
        output: Optional[str] = None,
        in_db: bool = True,
    ) -> ProcessingResult:
        """
        Radiometric calibration — convert SAR DN to backscatter coefficient.

        Args:
            input:       Input SAR raster in DN (digital numbers).
            output_type: ``"sigma0"`` (normalized radar cross-section),
                         ``"gamma0"`` (terrain-flattened), ``"beta0"`` (radar brightness).
            output:      Output path.
            in_db:       If True, convert result to dB scale.

        Example::

            result = client.sar.calibrate("sentinel1_dn.tif", output_type="sigma0")
        """
        rasterio = _require_rasterio()
        np = _require_numpy()

        inp = Path(input)
        out_path = _resolve_output(inp, output, f"cal_{output_type}")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(inp) as src:
            profile = src.profile.copy()
            data = src.read(1).astype(np.float64)
            nodata = src.nodata

        # Sentinel-1 calibration constants (defaults — real values from metadata)
        A = 1.0  # calibration constant from Sentinel-1 annotation XML
        calibrated = (data**2) / (A**2)

        if output_type == "gamma0":
            # Approximate terrain flattening (simplified)
            calibrated *= np.cos(np.deg2rad(38))  # Nominal incidence angle

        if in_db:
            with np.errstate(divide="ignore", invalid="ignore"):
                calibrated = 10 * np.log10(calibrated + 1e-10)

        if nodata is not None:
            calibrated = np.where(data == nodata, nodata, calibrated)

        profile.update(dtype="float32", nodata=nodata if nodata else -9999.0)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(calibrated.astype(np.float32)[np.newaxis, :, :])

        logger.info(f"SAR calibration ({output_type}, dB={in_db}) → {out_path}")
        return ProcessingResult(
            success=True, operation=f"calibrate:{output_type}",
            input_path=inp, output_path=out_path,
            metadata={"output_type": output_type, "in_db": in_db},
        )

    @_timed
    def flood_map(
        self,
        input: Union[str, Path],
        threshold: float = -15.0,
        output: Optional[str] = None,
        reference: Optional[Union[str, Path]] = None,
    ) -> ProcessingResult:
        """
        Flood mapping from SAR backscatter using threshold.

        Args:
            input:     SAR raster (dB scale, VV or VH polarization).
            threshold: Backscatter threshold below which pixels = water (dB).
                       Typical: -15 to -20 dB for C-band over water.
            output:    Output binary mask (1=water, 0=land).
            reference: Optional pre-event reference image for change-based detection.

        Example::

            result = client.sar.flood_map("sentinel1_post.tif",
                                          reference="sentinel1_pre.tif")
        """
        rasterio = _require_rasterio()
        np = _require_numpy()

        inp = Path(input)
        out_path = _resolve_output(inp, output, "flood_map")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(inp) as src:
            profile = src.profile.copy()
            data = src.read(1).astype(np.float32)
            nodata = src.nodata

        valid = np.isfinite(data) & ((data != nodata) if nodata is not None else np.ones_like(data, bool))

        if reference is not None:
            # Change-based detection
            with rasterio.open(Path(reference)) as ref_src:
                ref_data = ref_src.read(
                    1, out_shape=data.shape, resampling=rasterio.enums.Resampling.bilinear
                ).astype(np.float32)
            change = ref_data - data  # Decrease in backscatter = potential flood
            water_mask = (change > abs(threshold * 0.5)) & valid
        else:
            # Simple threshold
            water_mask = (data < threshold) & valid

        result = water_mask.astype(np.uint8)

        water_pct = 100 * np.sum(water_mask) / (np.sum(valid) + 1e-10)

        profile.update(count=1, dtype="uint8", nodata=255)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(result[np.newaxis, :, :])

        logger.info(f"Flood map: {water_pct:.1f}% water → {out_path}")
        return ProcessingResult(
            success=True, operation="flood_map",
            input_path=inp, output_path=out_path,
            metadata={"threshold_db": threshold, "water_pct": round(water_pct, 2)},
        )

    @_timed
    def coherence(
        self,
        image1: Union[str, Path],
        image2: Union[str, Path],
        window: int = 7,
        output: Optional[str] = None,
    ) -> ProcessingResult:
        """
        Compute interferometric coherence between two co-registered SLC images.
        Coherence = |<s1·s2*>| / sqrt(<|s1|²><|s2|²>) — range 0 to 1.
        High coherence = stable surface; low = change/movement.

        Args:
            image1, image2: Two co-registered complex SLC rasters.
            window:         Estimation window size.
            output:         Output coherence raster.

        Example::

            result = client.sar.coherence("slc_20240101.tif", "slc_20240113.tif")
        """
        rasterio = _require_rasterio()
        np = _require_numpy()
        from scipy.ndimage import uniform_filter

        inp1, inp2 = Path(image1), Path(image2)
        out_path = _resolve_output(inp1, output, "coherence")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(inp1) as s:
            profile = s.profile.copy()
            s1 = s.read(1).astype(np.complex64) if s.dtypes[0] == 'complex64' else s.read(1).astype(np.float32) + 0j

        with rasterio.open(inp2) as s:
            s2 = s.read(1).astype(np.complex64) if s.dtypes[0] == 'complex64' else s.read(1).astype(np.float32) + 0j

        # Interferogram
        inter = s1 * np.conj(s2)
        numerator   = np.abs(uniform_filter(inter.real,   window) + 1j * uniform_filter(inter.imag, window))
        denominator = np.sqrt(
            uniform_filter(np.abs(s1)**2, window) *
            uniform_filter(np.abs(s2)**2, window) + 1e-10
        )
        coh = np.clip(numerator / denominator, 0, 1)

        profile.update(count=1, dtype="float32", nodata=-1.0)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(coh.astype(np.float32)[np.newaxis, :, :])

        logger.info(f"Coherence mean={coh.mean():.3f} → {out_path}")
        return ProcessingResult(
            success=True, operation="coherence",
            output_path=out_path,
            metadata={"mean_coherence": round(float(coh.mean()), 4), "window": window},
        )
