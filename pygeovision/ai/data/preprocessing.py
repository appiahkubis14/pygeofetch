"""
Geospatial image preprocessing.

Provides atmospheric correction, pansharpening, normalization, and
band computation (NDVI, NDWI) for satellite imagery before AI processing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from pygeovision.core.exceptions import PreprocessingError

logger = logging.getLogger(__name__)


class GeoPreprocessor:
    """
    Preprocessing pipeline for satellite imagery.

    Applies a configurable sequence of preprocessing steps to prepare raw
    satellite imagery for AI model training and inference.

    Parameters
    ----------
    steps : list of str
        Ordered preprocessing steps to apply. Available:
        - ``"normalize"`` — Scale pixel values to [0, 1]
        - ``"standardize"`` — Zero-mean, unit-variance normalisation
        - ``"clip_percentile"`` — Clip to 2nd–98th percentile range
        - ``"atmospheric_dos"`` — Dark Object Subtraction atmospheric correction
        - ``"ndvi"`` — Compute and append NDVI band
        - ``"ndwi"`` — Compute and append NDWI band
        - ``"pansharpen"`` — Panchromatic sharpening (requires pan band)
    config : dict, optional
        Per-step configuration overrides.
    nodata_value : float, optional
        Pixel value representing no-data. Masked during processing.

    Examples
    --------
    >>> preprocessor = GeoPreprocessor(steps=["clip_percentile", "normalize", "ndvi"])
    >>> processed = preprocessor.process(image_array, red_band=3, nir_band=4)
    """

    AVAILABLE_STEPS = frozenset(
        [
            "normalize",
            "standardize",
            "clip_percentile",
            "atmospheric_dos",
            "ndvi",
            "ndwi",
            "pansharpen",
        ]
    )

    def __init__(
        self,
        steps: Optional[list[str]] = None,
        config: Optional[dict[str, Any]] = None,
        nodata_value: Optional[float] = None,
    ) -> None:
        self.steps = steps or ["clip_percentile", "normalize"]
        self.config = config or {}
        self.nodata_value = nodata_value

        invalid = set(self.steps) - self.AVAILABLE_STEPS
        if invalid:
            raise PreprocessingError(
                f"Unknown preprocessing steps: {invalid}. "
                f"Available: {self.AVAILABLE_STEPS}"
            )

        # Per-band statistics (computed during fit())
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._percentile_min: Optional[np.ndarray] = None
        self._percentile_max: Optional[np.ndarray] = None

        logger.debug("GeoPreprocessor steps: %s", self.steps)

    def fit(
        self,
        images: Union[np.ndarray, list[np.ndarray]],
    ) -> "GeoPreprocessor":
        """
        Compute per-band statistics for standardisation/normalisation.

        Parameters
        ----------
        images : np.ndarray or list of np.ndarray
            Sample images in CHW format. Used to compute mean, std, percentiles.

        Returns
        -------
        GeoPreprocessor
            Self (for chaining).
        """
        if isinstance(images, np.ndarray):
            images = [images]

        all_data = np.concatenate(
            [img.reshape(img.shape[0], -1) for img in images], axis=1
        ).astype(np.float64)

        if self.nodata_value is not None:
            mask = all_data != self.nodata_value
        else:
            mask = np.ones_like(all_data, dtype=bool)

        n_bands = all_data.shape[0]
        self._mean = np.zeros(n_bands)
        self._std = np.ones(n_bands)
        self._percentile_min = np.zeros(n_bands)
        self._percentile_max = np.ones(n_bands)

        for b in range(n_bands):
            valid = all_data[b][mask[b]]
            if len(valid) > 0:
                self._mean[b] = float(np.mean(valid))
                self._std[b] = float(np.std(valid)) or 1.0
                self._percentile_min[b] = float(np.percentile(valid, 2))
                self._percentile_max[b] = float(np.percentile(valid, 98))

        logger.debug(
            "GeoPreprocessor fitted on %d images, %d bands", len(images), n_bands
        )
        return self

    def process(
        self,
        image: np.ndarray,
        red_band: int = 3,
        nir_band: int = 4,
        green_band: int = 2,
        pan_band: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Apply the preprocessing pipeline to an image.

        Parameters
        ----------
        image : np.ndarray
            Input image in CHW format (bands × height × width).
        red_band : int
            1-indexed red band number (for NDVI, pansharpening).
        nir_band : int
            1-indexed NIR band number (for NDVI).
        green_band : int
            1-indexed green band number (for NDWI).
        pan_band : np.ndarray, optional
            High-resolution panchromatic band for pansharpening.

        Returns
        -------
        np.ndarray
            Preprocessed image in CHW format.
        """
        result = image.astype(np.float32)

        for step in self.steps:
            try:
                result = self._apply_step(
                    step=step,
                    image=result,
                    red_band=red_band,
                    nir_band=nir_band,
                    green_band=green_band,
                    pan_band=pan_band,
                )
            except Exception as exc:
                raise PreprocessingError(
                    f"Preprocessing step '{step}' failed: {exc}"
                ) from exc

        return result

    def process_batch(
        self,
        images: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Apply preprocessing to a batch of images.

        Parameters
        ----------
        images : np.ndarray
            Batch in NCHW format.
        **kwargs
            Forwarded to :meth:`process`.

        Returns
        -------
        np.ndarray
            Preprocessed batch in NCHW format.
        """
        return np.stack(
            [self.process(images[i], **kwargs) for i in range(images.shape[0])],
            axis=0,
        )

    # ------------------------------------------------------------------
    # Individual step implementations
    # ------------------------------------------------------------------

    def _apply_step(
        self,
        step: str,
        image: np.ndarray,
        red_band: int,
        nir_band: int,
        green_band: int,
        pan_band: Optional[np.ndarray],
    ) -> np.ndarray:
        """Dispatch to individual step method."""
        if step == "normalize":
            return self._normalize(image)
        if step == "standardize":
            return self._standardize(image)
        if step == "clip_percentile":
            return self._clip_percentile(image)
        if step == "atmospheric_dos":
            return self._atmospheric_dos(image)
        if step == "ndvi":
            return self._append_ndvi(image, red_band=red_band, nir_band=nir_band)
        if step == "ndwi":
            return self._append_ndwi(image, green_band=green_band, nir_band=nir_band)
        if step == "pansharpen":
            return self._pansharpen(image, pan_band=pan_band)
        raise PreprocessingError(f"Unknown step: {step}")

    @staticmethod
    def _normalize(image: np.ndarray) -> np.ndarray:
        """Scale each band to [0, 1] using band-wise min/max."""
        result = image.copy()
        for c in range(result.shape[0]):
            band = result[c]
            b_min, b_max = float(band.min()), float(band.max())
            if b_max > b_min:
                result[c] = (band - b_min) / (b_max - b_min)
            else:
                result[c] = 0.0
        return result

    def _standardize(self, image: np.ndarray) -> np.ndarray:
        """Zero-mean, unit-variance standardisation using fitted statistics."""
        if self._mean is None or self._std is None:
            logger.warning("GeoPreprocessor not fitted — using per-image statistics for standardize")
            return self._normalize(image)
        result = image.copy()
        for c in range(min(result.shape[0], len(self._mean))):
            result[c] = (result[c] - self._mean[c]) / (self._std[c] or 1.0)
        return result

    def _clip_percentile(self, image: np.ndarray) -> np.ndarray:
        """Clip values to fitted 2nd–98th percentile range."""
        cfg = self.config.get("clip_percentile", {})
        low_pct = cfg.get("low", 2)
        high_pct = cfg.get("high", 98)

        result = image.copy()
        for c in range(result.shape[0]):
            band = result[c]
            lo = float(np.percentile(band, low_pct))
            hi = float(np.percentile(band, high_pct))
            result[c] = np.clip(band, lo, hi)
        return result

    @staticmethod
    def _atmospheric_dos(image: np.ndarray) -> np.ndarray:
        """
        Dark Object Subtraction (DOS) atmospheric correction.

        Subtracts the per-band minimum value (dark object value) to remove
        additive atmospheric scattering contribution.
        """
        result = image.copy()
        for c in range(result.shape[0]):
            dark_object = float(result[c].min())
            result[c] = np.maximum(result[c] - dark_object, 0)
        return result

    @staticmethod
    def _append_ndvi(
        image: np.ndarray,
        red_band: int = 3,
        nir_band: int = 4,
    ) -> np.ndarray:
        """
        Compute NDVI and append as a new band.

        NDVI = (NIR - Red) / (NIR + Red)

        Parameters
        ----------
        image : np.ndarray
            Input image (C, H, W).
        red_band : int
            1-indexed red band number.
        nir_band : int
            1-indexed NIR band number.

        Returns
        -------
        np.ndarray
            Image with NDVI appended as last band. Shape: (C+1, H, W).
        """
        r_idx = red_band - 1
        nir_idx = nir_band - 1
        if r_idx >= image.shape[0] or nir_idx >= image.shape[0]:
            logger.warning(
                "NDVI: band index out of range (red=%d, nir=%d, bands=%d) — skipping",
                red_band,
                nir_band,
                image.shape[0],
            )
            return image

        red = image[r_idx].astype(np.float32)
        nir = image[nir_idx].astype(np.float32)
        denominator = nir + red
        ndvi = np.where(denominator != 0, (nir - red) / denominator, 0.0)
        ndvi = np.clip(ndvi, -1.0, 1.0)
        # Normalise NDVI to [0, 1] for compatibility with normalised imagery
        ndvi_norm = (ndvi + 1.0) / 2.0
        return np.concatenate([image, ndvi_norm[np.newaxis, ...]], axis=0)

    @staticmethod
    def _append_ndwi(
        image: np.ndarray,
        green_band: int = 2,
        nir_band: int = 4,
    ) -> np.ndarray:
        """
        Compute NDWI and append as a new band.

        NDWI = (Green - NIR) / (Green + NIR)

        Parameters
        ----------
        image : np.ndarray
        green_band : int
            1-indexed green band.
        nir_band : int
            1-indexed NIR band.

        Returns
        -------
        np.ndarray
            Image with NDWI appended. Shape: (C+1, H, W).
        """
        g_idx = green_band - 1
        nir_idx = nir_band - 1
        if g_idx >= image.shape[0] or nir_idx >= image.shape[0]:
            logger.warning(
                "NDWI: band index out of range — skipping"
            )
            return image

        green = image[g_idx].astype(np.float32)
        nir = image[nir_idx].astype(np.float32)
        denominator = green + nir
        ndwi = np.where(denominator != 0, (green - nir) / denominator, 0.0)
        ndwi = np.clip(ndwi, -1.0, 1.0)
        ndwi_norm = (ndwi + 1.0) / 2.0
        return np.concatenate([image, ndwi_norm[np.newaxis, ...]], axis=0)

    @staticmethod
    def _pansharpen(
        image: np.ndarray,
        pan_band: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Apply Brovey pansharpening to increase spatial resolution.

        Parameters
        ----------
        image : np.ndarray
            Multispectral image (C, H, W) — should be upsampled to pan resolution.
        pan_band : np.ndarray, optional
            Panchromatic band (H, W) at target resolution.

        Returns
        -------
        np.ndarray
            Pansharpened image at pan resolution.
        """
        if pan_band is None:
            logger.warning("pansharpen step requires pan_band — skipping")
            return image

        pan = pan_band.astype(np.float32)
        ms = image.astype(np.float32)

        # Brovey transform: multiply each band by pan / sum(all bands)
        band_sum = np.sum(ms, axis=0) + 1e-10
        result = np.stack(
            [ms[c] * (pan / band_sum) for c in range(ms.shape[0])],
            axis=0,
        )
        return np.clip(result, 0, None)
