"""
pygeovision.data.indices
========================

22 spectral and spatial indices, each validated before and after
computation.  All indices work on both numpy arrays and GeoTIFF paths.

Supported indices
-----------------
Vegetation  : NDVI, EVI, SAVI, MSAVI, ARVI, NDRE, RVI, WDRVI, VARI, ExG
Water       : NDWI, MNDWI, LSWI, WRI
Built-up    : NDBI, BSI
Fire/Burn   : NBR, BAI
Snow        : NDSI
Transform   : TCT (Brightness, Greenness, Wetness), PCA

Usage::

    from pygeovision import PyGeoVision
    client = PyGeoVision()

    # Compute NDVI from a stacked GeoTIFF
    ndvi_path = client.indices.ndvi("s2_6band.tif", red_band=3, nir_band=4)

    # Compute all indices at once
    results = client.indices.all_vegetation("s2_6band.tif")

    # From numpy array directly
    ndvi_arr = client.indices.ndvi_array(red=red_arr, nir=nir_arr)
"""
from __future__ import annotations

import logging
import pathlib
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

_EPS = 1e-10   # avoid divide-by-zero


def _require_rasterio():
    try:
        import rasterio
        return rasterio
    except ImportError:
        raise ImportError("pip install rasterio") from None


def _load_band(source: Union[str, np.ndarray], band_idx: int = 1) -> np.ndarray:
    """Load a single band from a GeoTIFF or return the array slice."""
    if isinstance(source, (str, pathlib.Path)):
        r = _require_rasterio()
        with r.open(str(source)) as src:
            arr = src.read(band_idx).astype(np.float32)
    elif isinstance(source, np.ndarray):
        arr = source.astype(np.float32)
    else:
        raise TypeError(f"Expected path or ndarray, got {type(source)}")
    return arr


def _load_multiband(path: str) -> Tuple[np.ndarray, dict]:
    r = _require_rasterio()
    with r.open(path) as src:
        return src.read().astype(np.float32), dict(src.profile)


def _save_index(arr: np.ndarray, ref_path: str, output_path: str, nodata: float = -9999.0):
    r = _require_rasterio()
    with r.open(ref_path) as ref:
        profile = ref.profile.copy()
    profile.update(count=1, dtype="float32", nodata=nodata, compress="lzw")
    with r.open(output_path, "w", **profile) as dst:
        dst.write(arr.astype(np.float32)[np.newaxis])
    return output_path


def _validate_arr(arr: np.ndarray, name: str) -> np.ndarray:
    """Replace NaN/Inf with 0 and clip to [-1, 1] for normalised indices."""
    arr = np.where(np.isfinite(arr), arr, 0.0).astype(np.float32)
    return arr


# ===========================================================================
# SpectralIndices class
# ===========================================================================

class SpectralIndices:
    """Validated spectral and spatial indices for satellite imagery.

    All methods accept either:
    - A stacked GeoTIFF path (use ``*_band`` parameters to specify which band)
    - Individual numpy arrays for each input band

    Every result is validated (NaN, Inf, range) before being returned.

    Args:
        validator: Optional :class:`DataValidator` instance. Created
            automatically with mode="fix" if not supplied.

    Example::

        from pygeovision import PyGeoVision
        client = PyGeoVision()

        # From a stacked 6-band GeoTIFF [B02,B03,B04,B08,B11,B12]
        # (1-based band indices)
        ndvi = client.indices.ndvi("stack.tif", red_band=3, nir_band=4)
        evi  = client.indices.evi("stack.tif",  blue_band=1, red_band=3, nir_band=4)
        ndwi = client.indices.ndwi("stack.tif", green_band=2, nir_band=4)
    """

    def __init__(self, validator=None):
        if validator is None:
            from pygeovision.data.validator import DataValidator
            validator = DataValidator(mode="fix")
        self._v = validator

    # ------------------------------------------------------------------
    # VEGETATION INDICES
    # ------------------------------------------------------------------

    def ndvi(
        self,
        source: Union[str, np.ndarray],
        red_band: int = 3,
        nir_band: int = 4,
        output_path: Optional[str] = None,
    ) -> Union[str, np.ndarray]:
        """Normalized Difference Vegetation Index.

        NDVI = (NIR - Red) / (NIR + Red)

        Range: [-1, 1].  Healthy vegetation > 0.3.

        Args:
            source: Stacked GeoTIFF path or 2-element list/array ``[red, nir]``.
            red_band: 1-based band index of the Red channel.
            nir_band: 1-based band index of the NIR channel.
            output_path: If set, saves the result and returns the path.

        Returns:
            float32 numpy array or path string.
        """
        red, nir, ref = self._get_two_bands(source, red_band, nir_band)
        result = (nir - red) / (nir + red + _EPS)
        result = np.clip(_validate_arr(result, "NDVI"), -1.0, 1.0)
        return self._out(result, ref, output_path, "ndvi")

    def evi(
        self,
        source: Union[str, np.ndarray],
        blue_band: int = 1,
        red_band:  int = 3,
        nir_band:  int = 4,
        G: float = 2.5, C1: float = 6.0, C2: float = 7.5, L: float = 1.0,
        output_path: Optional[str] = None,
    ) -> Union[str, np.ndarray]:
        """Enhanced Vegetation Index (Huete et al. 2002).

        EVI = G × (NIR − Red) / (NIR + C1×Red − C2×Blue + L)

        Reduces atmospheric and soil noise vs NDVI.
        """
        data, _, ref = self._get_three_bands(source, blue_band, red_band, nir_band)
        blue, red, nir = data
        result = G * (nir - red) / (nir + C1*red - C2*blue + L + _EPS)
        result = np.clip(_validate_arr(result, "EVI"), -1.0, 1.0)
        return self._out(result, ref, output_path, "evi")

    def savi(
        self,
        source: Union[str, np.ndarray],
        red_band: int = 3,
        nir_band: int = 4,
        L: float = 0.5,
        output_path: Optional[str] = None,
    ) -> Union[str, np.ndarray]:
        """Soil-Adjusted Vegetation Index (Huete 1988).

        SAVI = (NIR − Red) × (1 + L) / (NIR + Red + L)

        L=0.5 is optimal for intermediate vegetation density.
        """
        red, nir, ref = self._get_two_bands(source, red_band, nir_band)
        result = (nir - red) * (1 + L) / (nir + red + L + _EPS)
        result = np.clip(_validate_arr(result, "SAVI"), -1.5, 1.5)
        return self._out(result, ref, output_path, "savi")

    def msavi(
        self,
        source: Union[str, np.ndarray],
        red_band: int = 3,
        nir_band: int = 4,
        output_path: Optional[str] = None,
    ) -> Union[str, np.ndarray]:
        """Modified SAVI (Qi et al. 1994) — no soil factor needed.

        MSAVI = (2×NIR + 1 − sqrt((2×NIR+1)² − 8×(NIR−Red))) / 2
        """
        red, nir, ref = self._get_two_bands(source, red_band, nir_band)
        inner = np.maximum(0.0, (2*nir + 1)**2 - 8*(nir - red))
        result = (2*nir + 1 - np.sqrt(inner)) / 2.0
        result = _validate_arr(result, "MSAVI")
        return self._out(result, ref, output_path, "msavi")

    def arvi(
        self,
        source: Union[str, np.ndarray],
        blue_band: int = 1,
        red_band:  int = 3,
        nir_band:  int = 4,
        gamma: float = 1.0,
        output_path: Optional[str] = None,
    ) -> Union[str, np.ndarray]:
        """Atmospherically Resistant Vegetation Index (Kaufman & Tanré 1992).

        rb = Red − γ × (Blue − Red)
        ARVI = (NIR − rb) / (NIR + rb)
        """
        data, _, ref = self._get_three_bands(source, blue_band, red_band, nir_band)
        blue, red, nir = data
        rb = red - gamma * (blue - red)
        result = (nir - rb) / (nir + rb + _EPS)
        result = np.clip(_validate_arr(result, "ARVI"), -1.0, 1.0)
        return self._out(result, ref, output_path, "arvi")

    def ndre(
        self,
        source: Union[str, np.ndarray],
        red_edge_band: int = 5,
        nir_band: int = 4,
        output_path: Optional[str] = None,
    ) -> Union[str, np.ndarray]:
        """Normalized Difference Red Edge (Gitelson & Merzlyak 1994).

        NDRE = (NIR − RedEdge) / (NIR + RedEdge)

        Sensitive to chlorophyll in dense canopies where NDVI saturates.
        Requires Sentinel-2 Band 5 (705 nm).
        """
        re, nir, ref = self._get_two_bands(source, red_edge_band, nir_band)
        result = (nir - re) / (nir + re + _EPS)
        result = np.clip(_validate_arr(result, "NDRE"), -1.0, 1.0)
        return self._out(result, ref, output_path, "ndre")

    def rvi(
        self,
        source: Union[str, np.ndarray],
        red_band: int = 3,
        nir_band: int = 4,
        output_path: Optional[str] = None,
    ) -> Union[str, np.ndarray]:
        """Ratio Vegetation Index (Jordan 1969).  RVI = NIR / Red."""
        red, nir, ref = self._get_two_bands(source, red_band, nir_band)
        result = nir / (red + _EPS)
        result = np.clip(_validate_arr(result, "RVI"), 0.0, 30.0)
        return self._out(result, ref, output_path, "rvi")

    def wdrvi(
        self,
        source: Union[str, np.ndarray],
        red_band: int = 3,
        nir_band: int = 4,
        alpha: float = 0.1,
        output_path: Optional[str] = None,
    ) -> Union[str, np.ndarray]:
        """Wide Dynamic Range Vegetation Index (Gitelson 2004).

        WDRVI = (α×NIR − Red) / (α×NIR + Red)

        Better dynamic range than NDVI in high biomass areas.
        """
        red, nir, ref = self._get_two_bands(source, red_band, nir_band)
        result = (alpha*nir - red) / (alpha*nir + red + _EPS)
        result = np.clip(_validate_arr(result, "WDRVI"), -1.0, 1.0)
        return self._out(result, ref, output_path, "wdrvi")

    def vari(
        self,
        source: Union[str, np.ndarray],
        blue_band: int = 1,
        green_band: int = 2,
        red_band:   int = 3,
        output_path: Optional[str] = None,
    ) -> Union[str, np.ndarray]:
        """Visible Atmospherically Resistant Index (Gitelson et al. 2002).

        VARI = (Green − Red) / (Green + Red − Blue)

        Works with RGB imagery — no NIR required.
        """
        data, _, ref = self._get_three_bands(source, blue_band, green_band, red_band)
        blue, green, red = data
        result = (green - red) / (green + red - blue + _EPS)
        result = np.clip(_validate_arr(result, "VARI"), -1.0, 1.0)
        return self._out(result, ref, output_path, "vari")

    def exg(
        self,
        source: Union[str, np.ndarray],
        blue_band: int = 1,
        green_band: int = 2,
        red_band:   int = 3,
        output_path: Optional[str] = None,
    ) -> Union[str, np.ndarray]:
        """Excess Green Index (Woebbecke et al. 1995).

        ExG = 2×g − r − b,  where r,g,b are normalised RGB channels.

        Used for vegetation detection in drone / RGB imagery.
        """
        data, _, ref = self._get_three_bands(source, blue_band, green_band, red_band)
        blue, green, red = data
        total = red + green + blue + _EPS
        r_, g_, b_ = red/total, green/total, blue/total
        result = _validate_arr(2*g_ - r_ - b_, "ExG")
        return self._out(result, ref, output_path, "exg")

    # ------------------------------------------------------------------
    # WATER INDICES
    # ------------------------------------------------------------------

    def ndwi(
        self,
        source: Union[str, np.ndarray],
        green_band: int = 2,
        nir_band:   int = 4,
        output_path: Optional[str] = None,
    ) -> Union[str, np.ndarray]:
        """Normalized Difference Water Index — Gao (1996).

        NDWI = (Green − NIR) / (Green + NIR)

        Detects open water bodies.  Values > 0 indicate water.
        """
        green, nir, ref = self._get_two_bands(source, green_band, nir_band)
        result = (green - nir) / (green + nir + _EPS)
        result = np.clip(_validate_arr(result, "NDWI"), -1.0, 1.0)
        return self._out(result, ref, output_path, "ndwi")

    def mndwi(
        self,
        source: Union[str, np.ndarray],
        green_band: int = 2,
        swir1_band: int = 5,
        output_path: Optional[str] = None,
    ) -> Union[str, np.ndarray]:
        """Modified NDWI — McFeeters (1996) / Xu (2006).

        MNDWI = (Green − SWIR1) / (Green + SWIR1)

        Better suppresses built-up / soil noise than NDWI.
        """
        green, swir1, ref = self._get_two_bands(source, green_band, swir1_band)
        result = (green - swir1) / (green + swir1 + _EPS)
        result = np.clip(_validate_arr(result, "MNDWI"), -1.0, 1.0)
        return self._out(result, ref, output_path, "mndwi")

    def lswi(
        self,
        source: Union[str, np.ndarray],
        nir_band:  int = 4,
        swir1_band: int = 5,
        output_path: Optional[str] = None,
    ) -> Union[str, np.ndarray]:
        """Land Surface Water Index (Xiao et al. 2004).

        LSWI = (NIR − SWIR1) / (NIR + SWIR1)

        Sensitive to canopy and soil moisture.
        """
        nir, swir1, ref = self._get_two_bands(source, nir_band, swir1_band)
        result = (nir - swir1) / (nir + swir1 + _EPS)
        result = np.clip(_validate_arr(result, "LSWI"), -1.0, 1.0)
        return self._out(result, ref, output_path, "lswi")

    def wri(
        self,
        source: Union[str, np.ndarray],
        green_band: int = 2,
        red_band:   int = 3,
        nir_band:   int = 4,
        swir1_band: int = 5,
        output_path: Optional[str] = None,
    ) -> Union[str, np.ndarray]:
        """Water Ratio Index (Shen & Li 2010).

        WRI = (Green + Red) / (NIR + SWIR1)

        Values > 1 typically indicate water.
        """
        data, _, ref = self._get_three_bands(source, green_band, red_band, nir_band)
        green, red, nir = data
        swir1 = self._load_one(source, swir1_band)
        result = (green + red) / (nir + swir1 + _EPS)
        result = np.clip(_validate_arr(result, "WRI"), 0.0, 10.0)
        return self._out(result, ref, output_path, "wri")

    # ------------------------------------------------------------------
    # BUILT-UP & BARE SOIL
    # ------------------------------------------------------------------

    def ndbi(
        self,
        source: Union[str, np.ndarray],
        swir1_band: int = 5,
        nir_band:   int = 4,
        output_path: Optional[str] = None,
    ) -> Union[str, np.ndarray]:
        """Normalized Difference Built-up Index (Zha et al. 2003).

        NDBI = (SWIR1 − NIR) / (SWIR1 + NIR)

        Positive values indicate urban / built-up areas.
        """
        swir1, nir, ref = self._get_two_bands(source, swir1_band, nir_band)
        result = (swir1 - nir) / (swir1 + nir + _EPS)
        result = np.clip(_validate_arr(result, "NDBI"), -1.0, 1.0)
        return self._out(result, ref, output_path, "ndbi")

    def bsi(
        self,
        source: Union[str, np.ndarray],
        blue_band:  int = 1,
        red_band:   int = 3,
        nir_band:   int = 4,
        swir1_band: int = 5,
        output_path: Optional[str] = None,
    ) -> Union[str, np.ndarray]:
        """Bare Soil Index (Rikimaru et al. 2002).

        BSI = ((SWIR1+Red) − (NIR+Blue)) / ((SWIR1+Red) + (NIR+Blue))

        High values = exposed bare soil.
        """
        data, _, ref = self._get_three_bands(source, blue_band, red_band, nir_band)
        blue, red, nir = data
        swir1 = self._load_one(source, swir1_band)
        result = ((swir1+red) - (nir+blue)) / ((swir1+red) + (nir+blue) + _EPS)
        result = np.clip(_validate_arr(result, "BSI"), -1.0, 1.0)
        return self._out(result, ref, output_path, "bsi")

    # ------------------------------------------------------------------
    # FIRE & BURN
    # ------------------------------------------------------------------

    def nbr(
        self,
        source: Union[str, np.ndarray],
        nir_band:  int = 4,
        swir2_band: int = 6,
        output_path: Optional[str] = None,
    ) -> Union[str, np.ndarray]:
        """Normalized Burn Ratio (Key & Benson 1999).

        NBR = (NIR − SWIR2) / (NIR + SWIR2)

        dNBR (pre − post) used for burn severity mapping (USFS protocol).
        """
        nir, swir2, ref = self._get_two_bands(source, nir_band, swir2_band)
        result = (nir - swir2) / (nir + swir2 + _EPS)
        result = np.clip(_validate_arr(result, "NBR"), -1.0, 1.0)
        return self._out(result, ref, output_path, "nbr")

    def bai(
        self,
        source: Union[str, np.ndarray],
        red_band: int = 3,
        nir_band: int = 4,
        output_path: Optional[str] = None,
    ) -> Union[str, np.ndarray]:
        """Burn Area Index (Martín 1998).

        BAI = 1 / ((0.1 − Red)² + (0.06 − NIR)²)

        High values indicate charcoal / burned area.
        """
        red, nir, ref = self._get_two_bands(source, red_band, nir_band)
        result = 1.0 / ((0.1 - red)**2 + (0.06 - nir)**2 + _EPS)
        result = np.clip(_validate_arr(result, "BAI"), 0.0, 5000.0)
        return self._out(result, ref, output_path, "bai")

    # ------------------------------------------------------------------
    # SNOW
    # ------------------------------------------------------------------

    def ndsi(
        self,
        source: Union[str, np.ndarray],
        green_band: int = 2,
        swir1_band: int = 5,
        output_path: Optional[str] = None,
    ) -> Union[str, np.ndarray]:
        """Normalized Difference Snow Index (Hall et al. 1995).

        NDSI = (Green − SWIR1) / (Green + SWIR1)

        Values > 0.4 typically indicate snow / ice cover.
        """
        green, swir1, ref = self._get_two_bands(source, green_band, swir1_band)
        result = (green - swir1) / (green + swir1 + _EPS)
        result = np.clip(_validate_arr(result, "NDSI"), -1.0, 1.0)
        return self._out(result, ref, output_path, "ndsi")

    # ------------------------------------------------------------------
    # TRANSFORMS
    # ------------------------------------------------------------------

    def tct(
        self,
        source: str,
        blue_band:  int = 1,
        green_band: int = 2,
        red_band:   int = 3,
        nir_band:   int = 4,
        swir1_band: int = 5,
        swir2_band: int = 6,
        sensor: str = "sentinel2",
        output_prefix: Optional[str] = None,
    ) -> Dict[str, Union[str, np.ndarray]]:
        """Tasseled Cap Transform — returns Brightness, Greenness, Wetness.

        Coefficients for Sentinel-2 (Nedkov 2017).

        Args:
            source: Stacked GeoTIFF path (must have all 6 bands).
            sensor: ``"sentinel2"`` | ``"landsat8"``
            output_prefix: If set, saves TCT_B.tif, TCT_G.tif, TCT_W.tif.

        Returns:
            Dict with keys ``"brightness"``, ``"greenness"``, ``"wetness"``.
        """
        # Sentinel-2 TCT coefficients (Nedkov 2017)
        COEF = {
            "sentinel2": {
                "brightness": [0.3510, 0.3813, 0.3437, 0.7196, 0.2396, 0.1949],
                "greenness":  [-0.3599, -0.3533, -0.4734, 0.6633, 0.0087, -0.2856],
                "wetness":    [0.2578, 0.2305, 0.0883, 0.1071, -0.7611, -0.5308],
            },
            "landsat8": {
                "brightness": [0.3029, 0.2786, 0.4733, 0.5599, 0.5080, 0.1872],
                "greenness":  [-0.2941, -0.2430, -0.5424, 0.7276, 0.0713, -0.1608],
                "wetness":    [0.1511, 0.1973, 0.3283, 0.3407, -0.7117, -0.4559],
            },
        }
        if sensor not in COEF:
            raise ValueError(f"sensor must be 'sentinel2' or 'landsat8', got '{sensor}'")

        data, profile, ref = self._load_multiband_indexed(
            source,
            [blue_band, green_band, red_band, nir_band, swir1_band, swir2_band],
        )
        coef = COEF[sensor]
        results = {}
        for comp_name, weights in coef.items():
            w = np.array(weights, dtype=np.float32)[:, None, None]
            plane = (data * w).sum(axis=0)
            plane = _validate_arr(plane, f"TCT_{comp_name}")
            if output_prefix:
                out_path = f"{output_prefix}_tct_{comp_name[0].upper()}.tif"
                _save_index(plane, source, out_path)
                results[comp_name] = out_path
            else:
                results[comp_name] = plane
        return results

    def pca(
        self,
        source: str,
        n_components: int = 3,
        output_prefix: Optional[str] = None,
    ) -> Union[Dict[str, str], np.ndarray]:
        """Principal Component Analysis — decorrelate multi-band imagery.

        Args:
            source: Stacked GeoTIFF path.
            n_components: Number of PCA components to compute.
            output_prefix: If set, saves PC1.tif, PC2.tif, … and returns paths.

        Returns:
            ``(n_components, H, W)`` array or dict of paths.
        """
        data, profile = _load_multiband(source)   # (C, H, W)
        C, H, W = data.shape
        n_components = min(n_components, C)

        # Reshape to (pixels, bands)
        X = data.reshape(C, -1).T.astype(np.float64)
        X -= X.mean(axis=0)

        # SVD-based PCA
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        Vt = Vt[:n_components]
        pcs = (X @ Vt.T).T.reshape(n_components, H, W).astype(np.float32)
        pcs = _validate_arr(pcs, "PCA")

        if output_prefix:
            results = {}
            for i, pc in enumerate(pcs, start=1):
                out = f"{output_prefix}_pc{i}.tif"
                _save_index(pc, source, out)
                results[f"PC{i}"] = out
            return results
        return pcs

    # ------------------------------------------------------------------
    # Batch / convenience
    # ------------------------------------------------------------------

    def compute_all(
        self,
        source: str,
        indices: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        band_map: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Union[str, np.ndarray]]:
        """Compute a set of indices in one call.

        Args:
            source: Stacked 6-band GeoTIFF (B02,B03,B04,B08,B11,B12).
            indices: List of index names. Default: all supported indices.
            output_dir: If set, saves each index as a GeoTIFF.
            band_map: Override default band indices, e.g.
                ``{"red": 3, "nir": 4, "swir1": 5, "swir2": 6}``.

        Returns:
            Dict mapping index name → array or path.

        Example::

            results = client.indices.compute_all(
                "s2_6band.tif",
                indices=["ndvi","ndwi","ndbi","nbr"],
                output_dir="./indices/",
            )
        """
        ALL = ["ndvi","evi","savi","msavi","ndre","rvi","wdrvi",
               "ndwi","mndwi","lswi","ndbi","bsi","nbr","bai","ndsi"]
        targets = indices or ALL
        bm = band_map or {}

        # Default Sentinel-2 band map (1-based)
        B = {
            "blue":     bm.get("blue",  1),
            "green":    bm.get("green", 2),
            "red":      bm.get("red",   3),
            "nir":      bm.get("nir",   4),
            "swir1":    bm.get("swir1", 5),
            "swir2":    bm.get("swir2", 6),
            "red_edge": bm.get("red_edge", 5),
        }

        if output_dir:
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

        def _path(name: str) -> Optional[str]:
            return str(pathlib.Path(output_dir) / f"{name}.tif") if output_dir else None

        _MAP = {
            "ndvi":  lambda: self.ndvi( source, B["red"],  B["nir"],  _path("ndvi")),
            "evi":   lambda: self.evi(  source, B["blue"], B["red"],  B["nir"],  output_path=_path("evi")),
            "savi":  lambda: self.savi( source, B["red"],  B["nir"],  _path("savi")),
            "msavi": lambda: self.msavi(source, B["red"],  B["nir"],  _path("msavi")),
            "arvi":  lambda: self.arvi( source, B["blue"], B["red"],  B["nir"],  output_path=_path("arvi")),
            "ndre":  lambda: self.ndre( source, B["red_edge"], B["nir"], _path("ndre")),
            "rvi":   lambda: self.rvi(  source, B["red"],  B["nir"],  _path("rvi")),
            "wdrvi": lambda: self.wdrvi(source, B["red"],  B["nir"],  _path("wdrvi")),
            "ndwi":  lambda: self.ndwi( source, B["green"],B["nir"],  _path("ndwi")),
            "mndwi": lambda: self.mndwi(source, B["green"],B["swir1"],_path("mndwi")),
            "lswi":  lambda: self.lswi( source, B["nir"],  B["swir1"],_path("lswi")),
            "ndbi":  lambda: self.ndbi( source, B["swir1"],B["nir"],  _path("ndbi")),
            "bsi":   lambda: self.bsi(  source, B["blue"], B["red"],  B["nir"],  output_path=_path("bsi")),
            "nbr":   lambda: self.nbr(  source, B["nir"],  B["swir2"],_path("nbr")),
            "bai":   lambda: self.bai(  source, B["red"],  B["nir"],  _path("bai")),
            "ndsi":  lambda: self.ndsi( source, B["green"],B["swir1"],_path("ndsi")),
        }

        results = {}
        for name in targets:
            if name in _MAP:
                try:
                    results[name] = _MAP[name]()
                    logger.debug("Computed %s", name)
                except Exception as e:
                    logger.warning("Failed to compute %s: %s", name, e)
            else:
                logger.warning("Unknown index '%s', skipping", name)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_two_bands(
        self, source, b1: int, b2: int
    ) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
        ref = str(source) if isinstance(source, (str, pathlib.Path)) else None
        arr1 = _load_band(source, b1)
        arr2 = _load_band(source, b2)
        return arr1, arr2, ref

    def _get_three_bands(
        self, source, b1: int, b2: int, b3: int
    ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], None, Optional[str]]:
        ref = str(source) if isinstance(source, (str, pathlib.Path)) else None
        return ((_load_band(source, b1),
                  _load_band(source, b2),
                  _load_band(source, b3)), None, ref)

    def _load_one(self, source, band: int) -> np.ndarray:
        return _load_band(source, band)

    def _load_multiband_indexed(
        self, source: str, band_idxs: List[int]
    ) -> Tuple[np.ndarray, dict, str]:
        r = _require_rasterio()
        with r.open(source) as src:
            arrs = [src.read(b).astype(np.float32) for b in band_idxs]
            profile = dict(src.profile)
        return np.stack(arrs, axis=0), profile, source

    def _out(
        self,
        arr: np.ndarray,
        ref_path: Optional[str],
        output_path: Optional[str],
        name: str,
    ) -> Union[str, np.ndarray]:
        if output_path and ref_path:
            pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            _save_index(arr, ref_path, output_path)
            logger.info("Saved %s → %s", name.upper(), output_path)
            return output_path
        return arr
