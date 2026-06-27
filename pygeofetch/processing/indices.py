"""
SpectralIndices — 20+ spectral indices and transformations (E1-E20).
All return a :class:`ProcessingResult` with the computed raster.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Union

from pygeofetch.processing.base import (
    ProcessingResult, _require_rasterio, _require_numpy,
    _resolve_output, _timed,
)

logger = logging.getLogger(__name__)


class SpectralIndices:
    """
    Spectral index computation engine.

    All methods accept per-band file paths and return a
    :class:`ProcessingResult` with the index raster path.
    Values are float32, range typically -1 to +1 for normalized indices.

    Example::

        from pygeofetch import PyGeoFetch
        client = PyGeoFetch()
        ndvi = client.indices.ndvi(red="B04.tif", nir="B08.tif")
        evi  = client.indices.evi(blue="B02.tif", red="B04.tif", nir="B08.tif")
    """

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _read_band(path: Union[str, Path], shape=None):
        rasterio = _require_rasterio()
        np = _require_numpy()
        p = Path(path)
        with rasterio.open(p) as src:
            if shape:
                data = src.read(1, out_shape=shape, resampling=rasterio.enums.Resampling.bilinear).astype(np.float32)
            else:
                data = src.read(1).astype(np.float32)
            nodata = src.nodata
            profile = src.profile.copy()
        data = np.where(data == nodata, np.nan, data) if nodata is not None else data
        return data, profile

    @staticmethod
    def _norm_diff(a, b):
        """(a - b) / (a + b) with safe division."""
        np = _require_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(a + b != 0, (a - b) / (a + b), np.nan)

    def _save_index(
        self, data, profile, out_path: Path, nodata: float = -9999.0
    ) -> None:
        rasterio = _require_rasterio()
        np = _require_numpy()
        profile.update(count=1, dtype="float32", nodata=nodata)
        result = np.where(np.isnan(data), nodata, data).astype(np.float32)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(result[np.newaxis, :, :])

    # ── E1: NDVI ─────────────────────────────────────────────────────────

    @_timed
    def ndvi(
        self,
        red: Union[str, Path],
        nir: Union[str, Path],
        output: Optional[str] = None,
    ) -> ProcessingResult:
        """
        NDVI — Normalized Difference Vegetation Index.
        Formula: (NIR - Red) / (NIR + Red).  Range: -1 to +1.
        Values > 0.3 indicate healthy vegetation.
        """
        red_data, profile = self._read_band(red)
        nir_data, _       = self._read_band(nir, shape=red_data.shape)
        result = self._norm_diff(nir_data, red_data)
        out_path = _resolve_output(Path(red), output, "ndvi")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_index(result, profile, out_path)
        logger.info(f"NDVI → {out_path}")
        return ProcessingResult(success=True, operation="ndvi",
                                output_path=out_path, input_path=Path(red))

    # ── E2: EVI ──────────────────────────────────────────────────────────

    @_timed
    def evi(
        self,
        blue: Union[str, Path],
        red: Union[str, Path],
        nir: Union[str, Path],
        G: float = 2.5, C1: float = 6.0, C2: float = 7.5, L: float = 1.0,
        output: Optional[str] = None,
    ) -> ProcessingResult:
        """
        EVI — Enhanced Vegetation Index.
        Formula: G * (NIR-Red) / (NIR + C1*Red - C2*Blue + L).
        Reduces atmospheric and canopy background effects vs NDVI.
        """
        np = _require_numpy()
        blue_data, profile = self._read_band(blue)
        red_data,  _       = self._read_band(red,  shape=blue_data.shape)
        nir_data,  _       = self._read_band(nir,  shape=blue_data.shape)
        denom = nir_data + C1 * red_data - C2 * blue_data + L
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.where(denom != 0, G * (nir_data - red_data) / denom, np.nan)
        out_path = _resolve_output(Path(red), output, "evi")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_index(result, profile, out_path)
        logger.info(f"EVI → {out_path}")
        return ProcessingResult(success=True, operation="evi",
                                output_path=out_path, input_path=Path(red))

    # ── E3: SAVI ─────────────────────────────────────────────────────────

    @_timed
    def savi(
        self,
        red: Union[str, Path],
        nir: Union[str, Path],
        L: float = 0.5,
        output: Optional[str] = None,
    ) -> ProcessingResult:
        """
        SAVI — Soil Adjusted Vegetation Index.
        Formula: (NIR-Red)/(NIR+Red+L) * (1+L).
        L=0.5 is standard; L=1 → no adjustment (= NDVI scaled).
        """
        np = _require_numpy()
        red_data, profile = self._read_band(red)
        nir_data, _       = self._read_band(nir, shape=red_data.shape)
        denom = nir_data + red_data + L
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.where(denom != 0, (nir_data - red_data) / denom * (1 + L), np.nan)
        out_path = _resolve_output(Path(red), output, "savi")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_index(result, profile, out_path)
        return ProcessingResult(success=True, operation="savi",
                                output_path=out_path, input_path=Path(red))

    # ── E4: NDWI ─────────────────────────────────────────────────────────

    @_timed
    def ndwi(
        self,
        green: Union[str, Path],
        nir: Union[str, Path],
        output: Optional[str] = None,
    ) -> ProcessingResult:
        """
        NDWI — Normalized Difference Water Index (McFeeters 1996).
        Formula: (Green - NIR) / (Green + NIR).  Water > 0, land < 0.
        """
        green_data, profile = self._read_band(green)
        nir_data,   _       = self._read_band(nir, shape=green_data.shape)
        result = self._norm_diff(green_data, nir_data)
        out_path = _resolve_output(Path(green), output, "ndwi")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_index(result, profile, out_path)
        return ProcessingResult(success=True, operation="ndwi",
                                output_path=out_path, input_path=Path(green))

    # ── E5: MNDWI ────────────────────────────────────────────────────────

    @_timed
    def mndwi(
        self,
        green: Union[str, Path],
        swir1: Union[str, Path],
        output: Optional[str] = None,
    ) -> ProcessingResult:
        """
        MNDWI — Modified NDWI (Xu 2006).
        Formula: (Green - SWIR1) / (Green + SWIR1).
        Better separation of water from built-up areas than NDWI.
        """
        green_data, profile = self._read_band(green)
        swir_data,  _       = self._read_band(swir1, shape=green_data.shape)
        result = self._norm_diff(green_data, swir_data)
        out_path = _resolve_output(Path(green), output, "mndwi")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_index(result, profile, out_path)
        return ProcessingResult(success=True, operation="mndwi",
                                output_path=out_path, input_path=Path(green))

    # ── E6: NDBI ─────────────────────────────────────────────────────────

    @_timed
    def ndbi(
        self,
        nir: Union[str, Path],
        swir1: Union[str, Path],
        output: Optional[str] = None,
    ) -> ProcessingResult:
        """
        NDBI — Normalized Difference Built-up Index (Zha 2003).
        Formula: (SWIR1 - NIR) / (SWIR1 + NIR).  Urban > 0, vegetation < 0.
        """
        swir_data, profile = self._read_band(swir1)
        nir_data,  _       = self._read_band(nir, shape=swir_data.shape)
        result = self._norm_diff(swir_data, nir_data)
        out_path = _resolve_output(Path(nir), output, "ndbi")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_index(result, profile, out_path)
        return ProcessingResult(success=True, operation="ndbi",
                                output_path=out_path, input_path=Path(nir))

    # ── E7: NDSI ─────────────────────────────────────────────────────────

    @_timed
    def ndsi(
        self,
        green: Union[str, Path],
        swir1: Union[str, Path],
        output: Optional[str] = None,
    ) -> ProcessingResult:
        """
        NDSI — Normalized Difference Snow Index (Hall 1995).
        Formula: (Green - SWIR1) / (Green + SWIR1).  Snow > 0.4.
        """
        green_data, profile = self._read_band(green)
        swir_data,  _       = self._read_band(swir1, shape=green_data.shape)
        result = self._norm_diff(green_data, swir_data)
        out_path = _resolve_output(Path(green), output, "ndsi")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_index(result, profile, out_path)
        return ProcessingResult(success=True, operation="ndsi",
                                output_path=out_path, input_path=Path(green))

    # ── E8: NDMI ─────────────────────────────────────────────────────────

    @_timed
    def ndmi(
        self,
        nir: Union[str, Path],
        swir1: Union[str, Path],
        output: Optional[str] = None,
    ) -> ProcessingResult:
        """
        NDMI — Normalized Difference Moisture Index (Wilson & Sader 2002).
        Formula: (NIR - SWIR1) / (NIR + SWIR1).
        Sensitive to canopy water content; positive = moist vegetation.
        """
        nir_data,  profile = self._read_band(nir)
        swir_data, _       = self._read_band(swir1, shape=nir_data.shape)
        result = self._norm_diff(nir_data, swir_data)
        out_path = _resolve_output(Path(nir), output, "ndmi")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_index(result, profile, out_path)
        return ProcessingResult(success=True, operation="ndmi",
                                output_path=out_path, input_path=Path(nir))

    # ── E9: NBR / dNBR ───────────────────────────────────────────────────

    @_timed
    def nbr(
        self,
        nir: Union[str, Path],
        swir2: Union[str, Path],
        output: Optional[str] = None,
    ) -> ProcessingResult:
        """
        NBR — Normalized Burn Ratio.
        Formula: (NIR - SWIR2) / (NIR + SWIR2).
        Use dNBR = pre_NBR - post_NBR for burn severity.
        """
        nir_data,   profile = self._read_band(nir)
        swir2_data, _       = self._read_band(swir2, shape=nir_data.shape)
        result = self._norm_diff(nir_data, swir2_data)
        out_path = _resolve_output(Path(nir), output, "nbr")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_index(result, profile, out_path)
        return ProcessingResult(success=True, operation="nbr",
                                output_path=out_path, input_path=Path(nir))

    @_timed
    def dnbr(
        self,
        pre_nir: Union[str, Path],
        pre_swir2: Union[str, Path],
        post_nir: Union[str, Path],
        post_swir2: Union[str, Path],
        output: Optional[str] = None,
    ) -> ProcessingResult:
        """
        dNBR — differenced Normalized Burn Ratio (burn severity).
        Formula: NBR_pre - NBR_post.
        Range: < -0.25 = regrowth; > 0.66 = high severity burn.
        """
        np = _require_numpy()
        pre_nir_d,   profile = self._read_band(pre_nir)
        pre_swir_d,  _       = self._read_band(pre_swir2, shape=pre_nir_d.shape)
        post_nir_d,  _       = self._read_band(post_nir,  shape=pre_nir_d.shape)
        post_swir_d, _       = self._read_band(post_swir2, shape=pre_nir_d.shape)
        nbr_pre  = self._norm_diff(pre_nir_d,  pre_swir_d)
        nbr_post = self._norm_diff(post_nir_d, post_swir_d)
        result = nbr_pre - nbr_post
        out_path = _resolve_output(Path(pre_nir), output, "dnbr")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_index(result, profile, out_path)
        return ProcessingResult(success=True, operation="dnbr",
                                output_path=out_path, input_path=Path(pre_nir))

    # ── E9: TCT — Tasseled Cap Transformation ────────────────────────────

    @_timed
    def tct(
        self,
        blue: Union[str, Path],
        green: Union[str, Path],
        red: Union[str, Path],
        nir: Union[str, Path],
        swir1: Union[str, Path],
        swir2: Union[str, Path],
        output: Optional[str] = None,
        sensor: str = "sentinel2",
    ) -> ProcessingResult:
        """
        Tasseled Cap Transformation.
        Produces 3-band output: Brightness, Greenness, Wetness.

        Args:
            blue, green, red, nir, swir1, swir2: Input band paths.
            sensor: ``"sentinel2"`` or ``"landsat8"`` — selects coefficients.
            output: Output path.
        """
        np = _require_numpy()
        rasterio = _require_rasterio()

        bands = []
        profile = None
        for b_path in [blue, green, red, nir, swir1, swir2]:
            data, p = self._read_band(b_path)
            if profile is None:
                profile = p
                ref_shape = data.shape
            else:
                if data.shape != ref_shape:
                    from rasterio.enums import Resampling
                    with rasterio.open(Path(b_path)) as src:
                        data = src.read(1, out_shape=ref_shape, resampling=Resampling.bilinear).astype(np.float32)
            bands.append(data)

        B = np.stack(bands, axis=0)  # (6, h, w)

        # Coefficients from published literature
        if sensor == "sentinel2":
            # Nedkov (2017) Sentinel-2 TCT coefficients
            coefs = np.array([
                [0.3510, 0.3813, 0.3437, 0.7196, 0.2396, 0.1949],  # Brightness
                [-0.3599, -0.3533, -0.4734, 0.6633, 0.0087, -0.2856],  # Greenness
                [0.2578, 0.2305, 0.0883, 0.1071, -0.7611, -0.5308],   # Wetness
            ])
        else:  # landsat8
            # Baig et al. (2014) Landsat-8 OLI coefficients
            coefs = np.array([
                [0.3029, 0.2786, 0.4733, 0.5599, 0.5080, 0.1872],
                [-0.2941, -0.2430, -0.5424, 0.7276, 0.0713, -0.1608],
                [0.1511, 0.1973, 0.3283, 0.3407, -0.7117, -0.4559],
            ])

        tct_data = np.tensordot(coefs, B, axes=([1], [0]))  # (3, h, w)

        out_path = _resolve_output(Path(red), output, "tct")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        profile.update(count=3, dtype="float32", nodata=-9999.0)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(np.nan_to_num(tct_data, nan=-9999.0).astype(np.float32))

        logger.info(f"TCT (Brightness, Greenness, Wetness) → {out_path}")
        return ProcessingResult(
            success=True, operation="tct", output_path=out_path,
            metadata={"sensor": sensor, "bands": ["Brightness", "Greenness", "Wetness"]},
        )

    # ── E10: PCA ─────────────────────────────────────────────────────────

    @_timed
    def pca(
        self,
        inputs: List[Union[str, Path]],
        n_components: int = 3,
        output: Optional[str] = None,
    ) -> ProcessingResult:
        """
        Principal Component Analysis — dimensionality reduction.

        Args:
            inputs:       List of single-band rasters (or multi-band path).
            n_components: Number of PCs to retain.
            output:       Output path (n_components-band GeoTIFF).

        Example::

            result = client.indices.pca(
                inputs=["B02.tif","B03.tif","B04.tif","B08.tif"],
                n_components=3
            )
        """
        np = _require_numpy()
        rasterio = _require_rasterio()

        bands = []
        profile = None
        for p in inputs:
            data, prof = self._read_band(p)
            if profile is None:
                profile = prof
            bands.append(data.ravel())

        X = np.stack(bands, axis=0).T  # (pixels, bands)
        valid = ~np.any(np.isnan(X), axis=1)
        X_valid = X[valid]

        # Standardize
        mean = X_valid.mean(axis=0)
        std  = X_valid.std(axis=0) + 1e-10
        X_std = (X_valid - mean) / std

        # Covariance + eigen decomposition
        cov = np.cov(X_std.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]

        n_components = min(n_components, len(inputs))
        W = eigenvectors[:, :n_components]

        explained = eigenvalues[idx[:n_components]] / eigenvalues.sum() * 100

        pcs_valid = X_std @ W  # (valid_pixels, n_components)

        # Reconstruct spatial arrays
        h, w = bands[0].reshape(profile["height"], profile["width"]).shape
        pcs = np.full((n_components, h * w), np.nan)
        pcs[:, valid] = pcs_valid.T
        pcs = pcs.reshape(n_components, h, w)

        out_path = _resolve_output(Path(inputs[0]), output, f"pca_{n_components}comp")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        profile.update(count=n_components, dtype="float32", nodata=-9999.0)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(np.nan_to_num(pcs, nan=-9999.0).astype(np.float32))

        logger.info(f"PCA {n_components} components → {out_path}")
        return ProcessingResult(
            success=True, operation="pca", output_path=out_path,
            metadata={
                "n_components": n_components,
                "explained_variance_pct": [round(float(e), 2) for e in explained],
            },
        )

    # ── E11: Texture (GLCM) ───────────────────────────────────────────────

    @_timed
    def texture(
        self,
        input: Union[str, Path],
        window: int = 5,
        features: Optional[List[str]] = None,
        output: Optional[str] = None,
    ) -> ProcessingResult:
        """
        GLCM Texture Features — contrast, correlation, energy, homogeneity.

        Args:
            input:    Input single-band raster (e.g. NIR).
            window:   Sliding window size (must be odd).
            features: Subset of ``["contrast","dissimilarity","homogeneity",
                      "energy","correlation","ASM"]``. Default: all 6.
            output:   Output multi-band path (one band per feature).

        Example::

            result = client.indices.texture(
                "B08.tif", window=7, features=["contrast","homogeneity"]
            )
        """
        np = _require_numpy()
        rasterio = _require_rasterio()

        features = features or ["contrast", "dissimilarity", "homogeneity",
                                 "energy", "correlation", "ASM"]

        inp = Path(input)
        data, profile = self._read_band(inp)
        out_path = _resolve_output(inp, output, "texture")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Normalize to 0-255 for GLCM
        valid = ~np.isnan(data)
        data_norm = np.zeros_like(data, dtype=np.uint8)
        if valid.any():
            d_min, d_max = data[valid].min(), data[valid].max()
            data_norm[valid] = ((data[valid] - d_min) / (d_max - d_min + 1e-10) * 255).astype(np.uint8)

        h, w = data_norm.shape
        pad = window // 2
        padded = np.pad(data_norm, pad, mode="reflect")
        n_feats = len(features)
        texture_maps = np.zeros((n_feats, h, w), dtype=np.float32)

        feat_funcs = {
            "contrast": lambda P: np.sum(P * np.arange(P.shape[-1])**2, axis=(-1,-2)),
            "dissimilarity": lambda P: np.sum(P * np.abs(np.arange(P.shape[-1])), axis=(-1,-2)),
            "homogeneity": lambda P: np.sum(P / (1 + np.arange(P.shape[-1])**2), axis=(-1,-2)),
            "energy": lambda P: np.sqrt(np.sum(P**2, axis=(-1,-2))),
            "ASM": lambda P: np.sum(P**2, axis=(-1,-2)),
            "correlation": lambda P: _glcm_correlation(P),
        }

        # Sliding window GLCM (simplified row/col offset=1)
        for i in range(h):
            for j in range(w):
                patch = padded[i:i+window, j:j+window].astype(np.int32)
                # Build 256x256 GLCM
                glcm = np.zeros((256, 256), dtype=np.float32)
                for di in range(window - 1):
                    for dj in range(window - 1):
                        glcm[patch[di, dj], patch[di+1, dj+1]] += 1
                # Symmetrize and normalize
                glcm = (glcm + glcm.T) / 2
                total = glcm.sum() + 1e-10
                glcm /= total
                idx_arr = np.arange(256)
                for fi, feat in enumerate(features):
                    func = feat_funcs.get(feat)
                    if func:
                        try:
                            texture_maps[fi, i, j] = float(func(glcm))
                        except Exception:
                            pass

        profile.update(count=n_feats, dtype="float32", nodata=-9999.0)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(texture_maps)

        logger.info(f"GLCM texture ({features}) → {out_path}")
        return ProcessingResult(
            success=True, operation="texture",
            input_path=inp, output_path=out_path,
            metadata={"features": features, "window": window},
        )

    # ── E17: Land Surface Temperature ────────────────────────────────────

    @_timed
    def lst(
        self,
        thermal: Union[str, Path],
        emissivity: float = 0.97,
        output: Optional[str] = None,
        sensor: str = "landsat8",
    ) -> ProcessingResult:
        """
        Land Surface Temperature from thermal band.

        Args:
            thermal:    Thermal band raster (e.g. Landsat B10 in DN).
            emissivity: Surface emissivity (0.97 = vegetation, 0.98 = water).
            sensor:     ``"landsat8"``, ``"landsat9"``, ``"modis"``.
            output:     Output path (Kelvin + Celsius).

        Returns:
            Two-band raster: Band 1 = LST in Kelvin, Band 2 = Celsius.
        """
        np = _require_numpy()
        rasterio = _require_rasterio()

        inp = Path(thermal)
        out_path = _resolve_output(inp, output, "lst")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        thermal_data, profile = self._read_band(inp)

        # Conversion constants (Landsat Collection 2 defaults)
        if sensor in ("landsat8", "landsat9"):
            K1, K2 = 774.8853, 1321.0789  # Band 10 constants
            ML, AL = 3.3420e-4, 0.1       # Radiance rescaling factors
        elif sensor == "modis":
            K1, K2 = 607.76, 1260.56      # MODIS Band 31
            ML, AL = 1.0, 0.0
        else:
            K1, K2 = 774.8853, 1321.0789
            ML, AL = 3.3420e-4, 0.1

        # DN → Radiance → Brightness Temperature → LST
        radiance = ML * thermal_data + AL
        with np.errstate(divide="ignore", invalid="ignore"):
            T_brightness = K2 / np.log(K1 / (radiance + 1e-10) + 1)

        # Stefan-Boltzmann constant correction for emissivity
        rho = 1.438e-2  # m*K
        lambda_t = 10.8e-6  # Effective wavelength in m
        lst_kelvin = T_brightness / (1 + (lambda_t * T_brightness / rho) * np.log(emissivity))
        lst_celsius = lst_kelvin - 273.15

        result = np.stack([lst_kelvin, lst_celsius], axis=0)
        profile.update(count=2, dtype="float32", nodata=-9999.0)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(np.nan_to_num(result, nan=-9999.0).astype(np.float32))

        logger.info(f"LST → Band1=Kelvin, Band2=Celsius → {out_path}")
        return ProcessingResult(
            success=True, operation="lst",
            input_path=inp, output_path=out_path,
            metadata={"emissivity": emissivity, "sensor": sensor,
                      "bands": ["LST_Kelvin", "LST_Celsius"]},
        )

    # ── E18: Albedo ───────────────────────────────────────────────────────

    @_timed
    def albedo(
        self,
        inputs: List[Union[str, Path]],
        output: Optional[str] = None,
        sensor: str = "sentinel2",
    ) -> ProcessingResult:
        """
        Narrowband-to-broadband surface albedo estimate.

        Args:
            inputs: Band paths — order must match sensor band order:
                    Sentinel-2: [B02, B03, B04, B08, B11, B12]
                    Landsat-8:  [B2,  B3,  B4,  B5,  B6,  B7]
            sensor: ``"sentinel2"`` or ``"landsat8"``.
            output: Output path.
        """
        np = _require_numpy()

        # Liang (2001) coefficients for narrowband-to-broadband conversion
        if sensor == "sentinel2":
            # Coefficients for B2,B3,B4,B8,B11,B12
            coefs = [0.160, 0.291, 0.243, 0.116, 0.112, 0.081]
            intercept = -0.0015
        else:  # landsat8
            coefs = [0.356, 0.130, 0.373, 0.085, 0.072, -0.0018]
            intercept = -0.0018

        albedo_sum = None
        profile = None
        for i, (b_path, coef) in enumerate(zip(inputs, coefs)):
            data, prof = self._read_band(b_path)
            if albedo_sum is None:
                albedo_sum = np.zeros_like(data)
                profile = prof
            data_scaled = data / 10000.0 if data.max() > 10 else data
            albedo_sum += coef * data_scaled

        albedo_sum += intercept
        albedo_sum = np.clip(albedo_sum, 0, 1)

        out_path = _resolve_output(Path(inputs[0]), output, "albedo")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_index(albedo_sum, profile, out_path)
        return ProcessingResult(
            success=True, operation="albedo", output_path=out_path,
            metadata={"sensor": sensor},
        )

    # ── Band operations ───────────────────────────────────────────────────

    @_timed
    def band_math(
        self,
        inputs: List[Union[str, Path]],
        expression: str,
        output: Optional[str] = None,
    ) -> ProcessingResult:
        """
        Arbitrary band arithmetic using a Python expression.

        Args:
            inputs:     List of raster paths. Accessible as B[0], B[1], ...
            expression: Python expression using ``B[i]`` arrays, e.g.
                        ``"(B[3] - B[2]) / (B[3] + B[2])"`` for NDVI.
            output:     Output path.

        Example::

            # Custom ratio
            result = client.indices.band_math(
                inputs=["B04.tif", "B08.tif"],
                expression="B[1] / (B[0] + B[1] + 1e-10)"
            )
        """
        np = _require_numpy()
        rasterio = _require_rasterio()

        B = []
        profile = None
        for p in inputs:
            data, prof = self._read_band(p)
            if profile is None:
                profile = prof
                ref_shape = data.shape
            else:
                if data.shape != ref_shape:
                    with rasterio.open(Path(p)) as src:
                        data = src.read(1, out_shape=ref_shape, resampling=rasterio.enums.Resampling.bilinear).astype(np.float32)
            B.append(data)

        result = eval(expression, {"B": B, "np": np})

        out_path = _resolve_output(Path(inputs[0]), output, "band_math")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._save_index(result, profile, out_path)
        return ProcessingResult(
            success=True, operation="band_math", output_path=out_path,
            metadata={"expression": expression},
        )

    @_timed
    def stack(
        self,
        inputs: List[Union[str, Path]],
        output: Optional[str] = None,
    ) -> ProcessingResult:
        """
        Stack multiple single-band rasters into a multi-band GeoTIFF.

        Example::

            result = client.indices.stack(["B02.tif","B03.tif","B04.tif"])
        """
        rasterio = _require_rasterio()
        np = _require_numpy()

        bands = []
        profile = None
        for p in inputs:
            data, prof = self._read_band(p)
            if profile is None:
                profile = prof
                ref_shape = data.shape
            else:
                if data.shape != ref_shape:
                    with rasterio.open(Path(p)) as src:
                        data = src.read(1, out_shape=ref_shape, resampling=rasterio.enums.Resampling.bilinear).astype(np.float32)
            bands.append(data)

        stacked = np.stack(bands, axis=0)
        out_path = _resolve_output(Path(inputs[0]), output, f"stack_{len(inputs)}bands")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        profile.update(count=len(bands), dtype="float32")
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(np.nan_to_num(stacked, nan=profile.get("nodata", 0)).astype(np.float32))

        return ProcessingResult(
            success=True, operation="stack", output_path=out_path,
            metadata={"n_bands": len(bands)},
        )


def _glcm_correlation(P):
    """Compute GLCM correlation feature."""
    import numpy as np
    n = P.shape[-1] if P.ndim == 2 else P.shape[0]
    i = np.arange(n)
    mu_i = np.sum(i * P.sum(axis=1))
    mu_j = np.sum(i * P.sum(axis=0))
    sig_i = np.sqrt(np.sum((i - mu_i)**2 * P.sum(axis=1)) + 1e-10)
    sig_j = np.sqrt(np.sum((i - mu_j)**2 * P.sum(axis=0)) + 1e-10)
    ii, jj = np.meshgrid(i, i, indexing="ij")
    corr = np.sum((ii - mu_i) * (jj - mu_j) * P) / (sig_i * sig_j)
    return float(corr)
