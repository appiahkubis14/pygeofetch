"""
Segment Anything Model (SAM) Labeler.

Generates segmentation masks using Meta AI's Segment Anything Model (SAM)
for zero-shot, prompt-driven segmentation of satellite imagery. Supports
automatic mask generation (no prompt), point prompts, and bounding box prompts.

Model reference: https://github.com/facebookresearch/segment-anything

Example:
    >>> from pygeovision.ai.labeling.sam_labeler import SAMLabeler
    >>> labeler = SAMLabeler(model_type="vit_h", device="cuda")
    >>> results = labeler.label_tiles(tiles, output_dir="./labels/")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from pygeovision.ai.labeling.base_labeler import BaseLabeler, LabelingResult
from pygeovision.core.exceptions import LabelingError

logger = logging.getLogger(__name__)

# SAM model checkpoints available from Meta
_SAM_CHECKPOINTS: Dict[str, Dict[str, str]] = {
    "vit_h": {
        "filename": "sam_vit_h_4b8939.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    },
    "vit_l": {
        "filename": "sam_vit_l_0b3195.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    },
    "vit_b": {
        "filename": "sam_vit_b_01ec64.pth",
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    },
}

# MobileSAM: lightweight alternative for faster inference
_MOBILE_SAM_CHECKPOINT = {
    "filename": "mobile_sam.pt",
    "url": "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt",
}


@dataclass
class SAMConfig:
    """Configuration for the SAM labeler.

    Attributes:
        model_type: SAM model variant ('vit_h', 'vit_l', 'vit_b', 'mobile').
        device: Compute device ('cuda', 'mps', or 'cpu').
        checkpoint_dir: Directory for cached model weights.
        points_per_side: Grid density for automatic mask generation.
        pred_iou_thresh: Minimum predicted IoU for mask quality filtering.
        stability_score_thresh: Minimum stability score for mask filtering.
        box_nms_thresh: NMS threshold for suppressing duplicate boxes.
        min_mask_region_area: Minimum mask area in pixels (smaller filtered out).
        points_prompt: Optional list of (x, y) point prompts.
        boxes_prompt: Optional list of [x1, y1, x2, y2] box prompts.
        multimask_output: Whether to return multiple masks per prompt.
        rgb_bands: Band indices (0-based) to use as RGB for SAM (3 required).
    """

    model_type: str = "vit_b"
    device: str = "cpu"
    checkpoint_dir: Optional[Path] = None
    points_per_side: int = 32
    pred_iou_thresh: float = 0.88
    stability_score_thresh: float = 0.95
    box_nms_thresh: float = 0.7
    min_mask_region_area: int = 100
    points_prompt: Optional[List[Tuple[int, int]]] = None
    boxes_prompt: Optional[List[List[int]]] = None
    multimask_output: bool = False
    rgb_bands: Tuple[int, int, int] = (0, 1, 2)


class SAMLabeler(BaseLabeler):
    """Zero-shot segmentation labeler using Meta's Segment Anything Model.

    SAM segments anything in satellite imagery without task-specific training.
    Useful for generating high-quality object masks from any imagery,
    optionally guided by point or bounding box prompts.

    Args:
        model_type: SAM variant to use: 'vit_h' (best), 'vit_l', 'vit_b'
            (fastest), or 'mobile' (MobileSAM, lightest).
        device: Compute device. Auto-detects CUDA/MPS if not specified.
        checkpoint_dir: Directory for storing model weights.
        points_per_side: Grid density for automatic mask generation
            (higher = more masks, slower). Default 32.
        pred_iou_thresh: Minimum predicted IoU quality threshold (0-1).
        stability_score_thresh: Minimum mask stability score (0-1).
        min_mask_region_area: Filter out masks smaller than this area (pixels).
        rgb_bands: Band indices (0-based) to extract RGB for SAM inference.
            SAM requires 3-band uint8 RGB input.
        num_workers: Number of parallel tile workers.
        skip_existing: Skip tiles that already have labels.

    Example:
        >>> labeler = SAMLabeler(model_type="vit_h", device="cuda")
        >>> # With bounding box prompt for buildings:
        >>> labeler = SAMLabeler(
        ...     model_type="vit_b",
        ...     points_per_side=16,
        ...     min_mask_region_area=200,
        ... )
    """

    def __init__(
        self,
        model_type: str = "vit_b",
        device: Optional[str] = None,
        checkpoint_dir: Optional[Path] = None,
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        min_mask_region_area: int = 100,
        rgb_bands: Tuple[int, int, int] = (0, 1, 2),
        num_workers: int = 1,  # SAM is GPU-heavy; default 1
        skip_existing: bool = True,
    ) -> None:
        super().__init__(
            labeler_name="sam",
            num_workers=num_workers,
            skip_existing=skip_existing,
        )
        if model_type not in (*_SAM_CHECKPOINTS.keys(), "mobile"):
            raise ValueError(
                f"model_type must be one of {list(_SAM_CHECKPOINTS.keys()) + ['mobile']}, "
                f"got {model_type!r}"
            )

        resolved_device = device or self._detect_device()
        self.config = SAMConfig(
            model_type=model_type,
            device=resolved_device,
            checkpoint_dir=checkpoint_dir
            or Path.home() / ".pygeovision" / "cache" / "sam",
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            min_mask_region_area=min_mask_region_area,
            rgb_bands=rgb_bands,
        )
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._sam_model: Optional[Any] = None
        self._mask_generator: Optional[Any] = None

    # ------------------------------------------------------------------
    # BaseLabeler abstract properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return 'sam'

    @property
    def supported_tasks(self) -> list:
        return ['segmentation']

    # ------------------------------------------------------------------
    # BaseLabeler interface
    # ------------------------------------------------------------------

    def label_tile(
        self,
        tile_path: Path,
        tile_metadata: Any,
        output_path: Path,
    ) -> LabelingResult:
        """Generate SAM segmentation masks for a single tile.

        Each unique segment gets a unique integer ID in the output mask.
        Background (unsegmented pixels) = 0.

        Args:
            tile_path: Path to the GeoTIFF tile.
            tile_metadata: TileMetadata (bounds, CRS, shape).
            output_path: Destination for the label GeoTIFF.

        Returns:
            LabelingResult with segment count and coverage statistics.
        """
        try:
            import rasterio
        except ImportError as exc:
            raise LabelingError(
                "sam labeler requires rasterio. Install: pip install rasterio"
            ) from exc

        try:
            # Load SAM model (lazy, cached)
            self._ensure_model_loaded()

            with rasterio.open(tile_path) as src:
                height, width = src.height, src.width
                crs = src.crs
                transform = src.transform
                bands = src.count

            # Extract RGB image for SAM
            rgb = self._load_rgb(tile_path, self.config.rgb_bands)

            # Run SAM mask generation
            masks_data = self._generate_masks(rgb)

            # Compose instance mask (each segment = unique ID)
            instance_mask = self._compose_instance_mask(
                masks_data, height, width
            )

            # Write output
            meta = {
                "driver": "GTiff",
                "dtype": "uint16",  # Supports up to 65535 instances
                "width": width,
                "height": height,
                "count": 1,
                "crs": crs,
                "transform": transform,
                "compress": "lzw",
            }
            self._write_label_geotiff(instance_mask.astype(np.uint16), output_path, meta)

            num_segments = int(instance_mask.max())
            coverage = float(np.sum(instance_mask > 0)) / instance_mask.size

            return LabelingResult(
                tile_path=tile_path,
                label_path=output_path,
                success=True,
                labeler="sam",
                class_distribution={
                    "background": float(np.sum(instance_mask == 0)) / instance_mask.size,
                    "segmented": coverage,
                },
                metadata={
                    "model_type": self.config.model_type,
                    "num_segments": num_segments,
                    "coverage": coverage,
                    "device": self.config.device,
                },
            )

        except Exception as exc:
            logger.error("SAMLabeler failed for %s: %s", tile_path, exc)
            return LabelingResult(
                tile_path=tile_path,
                label_path=output_path,
                success=False,
                labeler="sam",
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_device() -> str:
        """Auto-detect the best available compute device.

        Returns:
            Device string: 'cuda', 'mps', or 'cpu'.
        """
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def _ensure_model_loaded(self) -> None:
        """Load SAM model into memory if not already loaded."""
        if self._sam_model is not None:
            return

        checkpoint_path = self._download_checkpoint()
        logger.info(
            "Loading SAM model '%s' on %s…",
            self.config.model_type,
            self.config.device,
        )

        try:
            if self.config.model_type == "mobile":
                from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator  # type: ignore
                self._sam_model = sam_model_registry["vit_t"](checkpoint=str(checkpoint_path))
            else:
                from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
                self._sam_model = sam_model_registry[self.config.model_type](
                    checkpoint=str(checkpoint_path)
                )

            self._sam_model.to(device=self.config.device)

            self._mask_generator = SamAutomaticMaskGenerator(
                model=self._sam_model,
                points_per_side=self.config.points_per_side,
                pred_iou_thresh=self.config.pred_iou_thresh,
                stability_score_thresh=self.config.stability_score_thresh,
                box_nms_thresh=self.config.box_nms_thresh,
                min_mask_region_area=self.config.min_mask_region_area,
            )
            logger.info("SAM model loaded successfully.")

        except ImportError as exc:
            raise LabelingError(
                "SAM labeler requires segment-anything. "
                "Install via: pip install segment-anything\n"
                "Or MobileSAM: pip install mobile-sam"
            ) from exc

    def _download_checkpoint(self) -> Path:
        """Download the SAM checkpoint if not already cached.

        Returns:
            Local path to the model checkpoint.
        """
        import requests

        if self.config.model_type == "mobile":
            info = _MOBILE_SAM_CHECKPOINT
        else:
            info = _SAM_CHECKPOINTS[self.config.model_type]

        local_path = self.config.checkpoint_dir / info["filename"]
        if local_path.exists():
            return local_path

        logger.info(
            "Downloading SAM checkpoint '%s' (~%.0f MB)…",
            info["filename"],
            {"vit_h": 2500, "vit_l": 1250, "vit_b": 375, "mobile": 39}.get(
                self.config.model_type, 0
            ),
        )
        resp = requests.get(info["url"], stream=True, timeout=600)
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(local_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                fh.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = 100 * downloaded / total
                    logger.debug("SAM download: %.1f%%", pct)

        logger.info("SAM checkpoint saved to %s", local_path)
        return local_path

    def _load_rgb(
        self, tile_path: Path, rgb_bands: Tuple[int, int, int]
    ) -> np.ndarray:
        """Load and prepare an RGB image for SAM.

        SAM expects a uint8 HxWx3 array in [0, 255].

        Args:
            tile_path: Path to the source GeoTIFF.
            rgb_bands: 0-based band indices for R, G, B.

        Returns:
            uint8 numpy array of shape (H, W, 3).
        """
        import rasterio

        with rasterio.open(tile_path) as src:
            # Read bands (rasterio uses 1-based indexing)
            r = src.read(rgb_bands[0] + 1).astype(np.float32)
            g = src.read(rgb_bands[1] + 1).astype(np.float32)
            b = src.read(rgb_bands[2] + 1).astype(np.float32)

        def _normalize_band(arr: np.ndarray) -> np.ndarray:
            p2, p98 = np.percentile(arr[arr > 0], (2, 98)) if np.any(arr > 0) else (0, 1)
            arr = np.clip(arr, p2, p98)
            rng = max(p98 - p2, 1e-6)
            return ((arr - p2) / rng * 255).astype(np.uint8)

        rgb = np.stack([_normalize_band(r), _normalize_band(g), _normalize_band(b)], axis=-1)
        return rgb  # (H, W, 3) uint8

    def _generate_masks(self, rgb: np.ndarray) -> List[Dict[str, Any]]:
        """Run SAM automatic mask generation on an RGB image.

        Args:
            rgb: uint8 HxWx3 array.

        Returns:
            List of SAM mask dicts with 'segmentation', 'area', 'bbox', etc.
        """
        if self._mask_generator is None:
            raise LabelingError("SAM model not loaded. Call _ensure_model_loaded() first.")

        masks = self._mask_generator.generate(rgb)
        logger.debug("SAM generated %d masks", len(masks))
        return masks

    def _compose_instance_mask(
        self,
        masks_data: List[Dict[str, Any]],
        height: int,
        width: int,
    ) -> np.ndarray:
        """Compose individual SAM masks into a single instance mask.

        Assigns each segment a unique integer ID (1-based). Larger segments
        are drawn first so smaller segments on top take priority.

        Args:
            masks_data: SAM mask list sorted by area.
            height: Output mask height.
            width: Output mask width.

        Returns:
            uint16 instance mask where 0=background, 1..N=segment IDs.
        """
        instance_mask = np.zeros((height, width), dtype=np.uint16)

        # Sort by area descending so large segments are drawn first
        sorted_masks = sorted(masks_data, key=lambda m: m["area"], reverse=True)

        for idx, mask_info in enumerate(sorted_masks, start=1):
            seg = mask_info["segmentation"]  # bool HxW
            instance_mask[seg] = idx

        return instance_mask
