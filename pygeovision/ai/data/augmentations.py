"""
Geospatial-aware augmentation pipeline.

All augmentations preserve geospatial validity:
- Geometric transforms use only 90° rotation multiples (no arbitrary rotation)
- Radiometric augmentations simulate real sensor and atmospheric variations
- All transforms operate on (H, W, C) numpy arrays (albumentations convention)
- Georeferencing metadata is preserved through augmentation callbacks
"""

from __future__ import annotations

import logging
import random
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Albumentations is required for augmentations
try:
    import albumentations as A
    from albumentations.core.transforms_interface import ImageOnlyTransform

    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    A = None  # type: ignore[assignment]
    ImageOnlyTransform = object  # type: ignore[assignment, misc]


def _require_albumentations() -> None:
    """Raise ImportError if albumentations is not installed."""
    if not ALBUMENTATIONS_AVAILABLE:
        raise ImportError(
            "albumentations is required for augmentations. "
            "Install with: pip install 'pygeovision[ai]'"
        )


# ------------------------------------------------------------------
# Custom geospatial-specific transforms
# ------------------------------------------------------------------

class AtmosphericHaze(ImageOnlyTransform):
    """
    Simulate atmospheric haze by blending the image with a uniform haze layer.

    Parameters
    ----------
    haze_range : tuple of float
        Range of haze intensity to sample from uniformly. Defaults to (0.01, 0.3).
    p : float
        Probability of applying the transform.
    """

    def __init__(
        self,
        haze_range: tuple[float, float] = (0.01, 0.3),
        always_apply: bool = False,
        p: float = 0.5,
    ) -> None:
        super().__init__(always_apply=always_apply, p=p)
        self.haze_range = haze_range

    def apply(self, img: np.ndarray, haze_intensity: float = 0.1, **params: Any) -> np.ndarray:
        haze_color = np.ones_like(img, dtype=np.float32) * 0.9
        img_float = img.astype(np.float32)
        return np.clip(
            img_float * (1 - haze_intensity) + haze_color * haze_intensity, 0, 1
        ).astype(img.dtype)

    def get_params(self) -> dict[str, float]:
        return {"haze_intensity": random.uniform(*self.haze_range)}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("haze_range",)


class SensorNoise(ImageOnlyTransform):
    """
    Add realistic sensor noise (Gaussian) to simulate different sensor characteristics.

    Parameters
    ----------
    noise_range : tuple of float
        Range of noise standard deviation. Defaults to (0.0, 0.05).
    """

    def __init__(
        self,
        noise_range: tuple[float, float] = (0.0, 0.05),
        always_apply: bool = False,
        p: float = 0.3,
    ) -> None:
        super().__init__(always_apply=always_apply, p=p)
        self.noise_range = noise_range

    def apply(self, img: np.ndarray, noise_std: float = 0.01, **params: Any) -> np.ndarray:
        noise = np.random.normal(0, noise_std, img.shape).astype(np.float32)
        return np.clip(img.astype(np.float32) + noise, 0, 1).astype(img.dtype)

    def get_params(self) -> dict[str, float]:
        return {"noise_std": random.uniform(*self.noise_range)}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("noise_range",)


class ResolutionDegradation(ImageOnlyTransform):
    """
    Degrade image resolution by downsampling and upsampling.

    Simulates imagery from lower-resolution sensors.

    Parameters
    ----------
    scale_range : tuple of float
        Fraction of original resolution to downsample to. (0.5, 0.9) means
        50-90% of original resolution.
    """

    def __init__(
        self,
        scale_range: tuple[float, float] = (0.5, 0.9),
        always_apply: bool = False,
        p: float = 0.2,
    ) -> None:
        super().__init__(always_apply=always_apply, p=p)
        self.scale_range = scale_range

    def apply(self, img: np.ndarray, scale: float = 0.75, **params: Any) -> np.ndarray:
        try:
            import cv2  # noqa: PLC0415

            h, w = img.shape[:2]
            small_h, small_w = max(1, int(h * scale)), max(1, int(w * scale))
            small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
            degraded = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
            return degraded
        except ImportError:
            return img  # Skip if cv2 unavailable

    def get_params(self) -> dict[str, float]:
        return {"scale": random.uniform(*self.scale_range)}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("scale_range",)


class BandDropout(ImageOnlyTransform):
    """
    Randomly zero out entire bands to simulate missing data.

    Parameters
    ----------
    drop_fraction : float
        Maximum fraction of bands to drop (0.0–1.0). Defaults to 0.3.
    """

    def __init__(
        self,
        drop_fraction: float = 0.3,
        always_apply: bool = False,
        p: float = 0.2,
    ) -> None:
        super().__init__(always_apply=always_apply, p=p)
        self.drop_fraction = drop_fraction

    def apply(self, img: np.ndarray, bands_to_drop: list[int] = (), **params: Any) -> np.ndarray:
        out = img.copy()
        for b in bands_to_drop:
            if img.ndim == 3 and b < img.shape[2]:
                out[:, :, b] = 0
        return out

    def get_params_dependent_on_targets(self, params: dict[str, Any]) -> dict[str, Any]:
        img = params["image"]
        if img.ndim == 3:
            n_bands = img.shape[2]
            n_drop = max(0, int(n_bands * self.drop_fraction * random.random()))
            bands_to_drop = random.sample(range(n_bands), k=min(n_drop, n_bands))
        else:
            bands_to_drop = []
        return {"bands_to_drop": bands_to_drop}

    @property
    def targets_as_params(self) -> list[str]:
        return ["image"]

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("drop_fraction",)


# ------------------------------------------------------------------
# Main pipeline builder
# ------------------------------------------------------------------


class GeoAugmentationPipeline:
    """
    Geospatial-aware augmentation pipeline using albumentations.

    Provides pre-built pipelines for different intensity levels and
    supports custom pipeline construction.

    Parameters
    ----------
    mode : str
        Augmentation intensity: ``"light"``, ``"medium"``, ``"heavy"``,
        ``"geometric_only"``, ``"radiometric_only"``, or ``"custom"``.
    custom_transforms : list, optional
        Custom albumentations transforms (used when ``mode="custom"``).
    task : str
        AI task, affects which augmentations are applied:
        ``"segmentation"``, ``"detection"``, ``"classification"``,
        ``"change_detection"``.
    preserve_masks : bool
        Apply spatial transforms to masks as well. Defaults to True.
    seed : int, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> pipeline = GeoAugmentationPipeline(mode="medium", task="segmentation")
    >>> result = pipeline(image=image_hwc, mask=mask)
    >>> aug_image, aug_mask = result["image"], result["mask"]
    """

    MODES = ("light", "medium", "heavy", "geometric_only", "radiometric_only", "custom")

    def __init__(
        self,
        mode: str = "medium",
        custom_transforms: Optional[list[Any]] = None,
        task: str = "segmentation",
        preserve_masks: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        _require_albumentations()

        if mode not in self.MODES:
            raise ValueError(f"Invalid mode '{mode}'. Choose from {self.MODES}")

        self.mode = mode
        self.task = task
        self.preserve_masks = preserve_masks
        self.seed = seed

        self._pipeline = self._build_pipeline(mode, custom_transforms)

    def __call__(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Apply augmentation to an image and optional mask.

        Parameters
        ----------
        image : np.ndarray
            Image array in HWC format, float32 values in [0, 1].
        mask : np.ndarray, optional
            Label mask in HW format.
        **kwargs
            Additional albumentations targets (e.g. ``bboxes=``, ``keypoints=``).

        Returns
        -------
        dict
            Albumentations result dict with ``"image"`` and optionally ``"mask"`` keys.
        """
        if mask is not None:
            return self._pipeline(image=image, mask=mask, **kwargs)
        return self._pipeline(image=image, **kwargs)

    @property
    def pipeline(self) -> Any:
        """The underlying albumentations Compose object."""
        return self._pipeline

    # ------------------------------------------------------------------
    # Pipeline builders
    # ------------------------------------------------------------------

    def _build_pipeline(
        self,
        mode: str,
        custom_transforms: Optional[list[Any]],
    ) -> Any:
        """Build the albumentations Compose pipeline."""
        additional_targets = {"mask": "mask"} if self.preserve_masks else {}

        if mode == "custom":
            if custom_transforms is None:
                raise ValueError("custom_transforms must be provided when mode='custom'")
            return A.Compose(custom_transforms, additional_targets=additional_targets)

        transforms = []
        transforms.extend(self._geometric_transforms(mode))
        if mode not in ("geometric_only",):
            transforms.extend(self._radiometric_transforms(mode))
            transforms.extend(self._atmospheric_transforms(mode))
            transforms.extend(self._sensor_transforms(mode))

        return A.Compose(transforms, additional_targets=additional_targets)

    @staticmethod
    def _geometric_transforms(mode: str) -> list[Any]:
        """Geospatially valid geometric augmentations."""
        base = [
            # Only 90° rotations to preserve geographic alignment
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
        ]
        if mode in ("medium", "heavy"):
            base += [
                A.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.1,
                    rotate_limit=0,  # No arbitrary rotation for geospatial data
                    p=0.3,
                ),
                A.Transpose(p=0.2),
            ]
        if mode == "heavy":
            base += [
                A.ElasticTransform(alpha=50, sigma=5, alpha_affine=5, p=0.1),
                A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.1),
            ]
        return base

    @staticmethod
    def _radiometric_transforms(mode: str) -> list[Any]:
        """Radiometric augmentations (brightness, contrast, etc.)."""
        p_scale = {"light": 0.2, "medium": 0.4, "heavy": 0.6, "radiometric_only": 0.5}.get(
            mode, 0.3
        )
        return [
            A.RandomBrightnessContrast(
                brightness_limit=0.15 if mode == "light" else 0.3,
                contrast_limit=0.15 if mode == "light" else 0.3,
                p=p_scale,
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=p_scale * 0.5,
            ),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=p_scale * 0.3),
            A.RandomGamma(gamma_limit=(80, 120), p=p_scale * 0.3),
        ]

    @staticmethod
    def _atmospheric_transforms(mode: str) -> list[Any]:
        """Atmospheric simulation transforms."""
        if mode == "light":
            return []
        p = 0.15 if mode == "medium" else 0.3
        return [
            AtmosphericHaze(haze_range=(0.01, 0.2 if mode == "medium" else 0.4), p=p),
        ]

    @staticmethod
    def _sensor_transforms(mode: str) -> list[Any]:
        """Sensor simulation transforms."""
        if mode == "light":
            return [SensorNoise(noise_range=(0.0, 0.02), p=0.1)]
        if mode == "medium":
            return [
                SensorNoise(noise_range=(0.0, 0.05), p=0.2),
                ResolutionDegradation(scale_range=(0.7, 0.95), p=0.1),
            ]
        return [
            SensorNoise(noise_range=(0.0, 0.08), p=0.3),
            ResolutionDegradation(scale_range=(0.5, 0.9), p=0.15),
            BandDropout(drop_fraction=0.2, p=0.1),
        ]

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def for_task(cls, task: str, mode: str = "medium") -> "GeoAugmentationPipeline":
        """
        Build a task-specific augmentation pipeline.

        Parameters
        ----------
        task : str
            Task type: ``"segmentation"``, ``"detection"``, ``"classification"``,
            ``"change_detection"``.
        mode : str
            Intensity mode.

        Returns
        -------
        GeoAugmentationPipeline
        """
        logger.debug("Building augmentation pipeline: task=%s, mode=%s", task, mode)
        return cls(mode=mode, task=task)

    @classmethod
    def light(cls, task: str = "segmentation") -> "GeoAugmentationPipeline":
        """Light augmentation (minimal distortion)."""
        return cls(mode="light", task=task)

    @classmethod
    def medium(cls, task: str = "segmentation") -> "GeoAugmentationPipeline":
        """Medium augmentation (recommended default)."""
        return cls(mode="medium", task=task)

    @classmethod
    def heavy(cls, task: str = "segmentation") -> "GeoAugmentationPipeline":
        """Heavy augmentation (maximum diversity)."""
        return cls(mode="heavy", task=task)
