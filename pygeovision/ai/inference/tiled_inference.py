"""
Tiled Inference Engine for Large Satellite Scenes.

Runs model inference on arbitrarily large satellite GeoTIFFs by:
1. Splitting the scene into overlapping tiles.
2. Running model inference tile-by-tile (or in batches).
3. Stitching predictions back with overlap averaging (soft blending).
4. Writing a georeferenced output GeoTIFF.

Example:
    >>> from pygeovision.ai.inference.tiled_inference import TiledInference
    >>> engine = TiledInference(model, tile_size=512, overlap=64)
    >>> engine.run("scene.tif", "predictions.tif", num_classes=10)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import numpy as np
try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


@dataclass
class TiledInferenceConfig:
    """Configuration for tiled inference.

    Attributes:
        tile_size: Spatial size of each tile (pixels).
        overlap: Overlap between adjacent tiles (pixels).
        batch_size: Number of tiles per inference batch.
        device: Compute device.
        mixed_precision: Enable AMP during inference.
        num_classes: Number of output classes.
        activation: Post-prediction activation ('softmax', 'sigmoid', None).
        blend_mode: Overlap blending strategy ('average' or 'gaussian').
    """
    tile_size: int = 512
    overlap: int = 64
    batch_size: int = 8
    device: str = "cpu"
    mixed_precision: bool = True
    num_classes: int = 2
    activation: Optional[str] = "softmax"
    blend_mode: str = "gaussian"


class TiledInference:
    """Inference engine for arbitrarily large satellite GeoTIFFs.

    Handles tiling, batched model inference, and soft-blended stitching
    to produce seamless predictions over large scenes.

    Args:
        model: PyTorch model (should be in eval mode).
        tile_size: Tile size in pixels (height = width).
        overlap: Tile overlap in pixels. Larger = smoother seams.
        batch_size: Tiles per GPU batch.
        device: Compute device string.
        mixed_precision: Use AMP for faster inference.
        activation: 'softmax' for multi-class, 'sigmoid' for binary, None for raw.
        blend_mode: 'gaussian' (smooth blend) or 'average'.

    Example:
        >>> model.eval()
        >>> engine = TiledInference(model, tile_size=512, overlap=128)
        >>> mask = engine.run("big_scene.tif", "output_mask.tif", num_classes=15)
    """

    def __init__(
        self,
        model: nn.Module,
        tile_size: int = 512,
        overlap: int = 64,
        batch_size: int = 8,
        device: Optional[str] = None,
        mixed_precision: bool = True,
        activation: Optional[str] = "softmax",
        blend_mode: str = "gaussian",
    ) -> None:
        self.config = TiledInferenceConfig(
            tile_size=tile_size,
            overlap=overlap,
            batch_size=batch_size,
            device=device or self._detect_device(),
            mixed_precision=mixed_precision,
            activation=activation,
            blend_mode=blend_mode,
        )
        self.model = model.to(self.config.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        input_path: str | Path,
        output_path: str | Path,
        num_classes: int = 2,
        band_indices: Optional[list] = None,
        preprocessing_fn: Optional[Callable] = None,
    ) -> np.ndarray:
        """Run tiled inference on a GeoTIFF and write a georeferenced output.

        Args:
            input_path: Path to input GeoTIFF scene.
            output_path: Path for output prediction GeoTIFF.
            num_classes: Number of output classes.
            band_indices: 1-based band indices to read (default: all).
            preprocessing_fn: Optional callable applied to each tile array.

        Returns:
            Numpy array with class predictions (H, W).
        """
        try:
            import rasterio
            from rasterio.transform import from_bounds
        except ImportError:
            raise ImportError("TiledInference requires rasterio. pip install rasterio")

        self.config.num_classes = num_classes
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(input_path) as src:
            height, width = src.height, src.width
            band_count = src.count
            profile = src.profile.copy()
            crs = src.crs
            transform = src.transform

            bands = band_indices or list(range(1, band_count + 1))
            image = src.read(bands).astype(np.float32)  # (C, H, W)

        logger.info(
            "Running tiled inference on %s (%dx%d, %d bands)…",
            input_path.name, height, width, len(bands),
        )

        # Normalize per-band to [0, 1]
        image = self._normalize(image)

        # Compute tile grid
        tiles_info = self._compute_tile_grid(height, width)

        # Accumulation buffers
        accum = np.zeros((num_classes, height, width), dtype=np.float32)
        weight = np.zeros((height, width), dtype=np.float32)
        blend_kernel = self._make_blend_kernel(self.config.tile_size, self.config.blend_mode)

        # Batch inference
        batch_tiles = []
        batch_coords = []
        for (row, col, r1, r2, c1, c2, pr1, pr2, pc1, pc2) in tiles_info:
            tile = image[:, r1:r2, c1:c2]
            if preprocessing_fn:
                tile = preprocessing_fn(tile)
            batch_tiles.append(tile)
            batch_coords.append((r1, r2, c1, c2, pr1, pr2, pc1, pc2))

            if len(batch_tiles) == self.config.batch_size:
                self._infer_batch(batch_tiles, batch_coords, accum, weight, blend_kernel, num_classes)
                batch_tiles, batch_coords = [], []

        if batch_tiles:
            self._infer_batch(batch_tiles, batch_coords, accum, weight, blend_kernel, num_classes)

        # Normalize accumulation
        eps = 1e-7
        weight = np.maximum(weight, eps)
        accum /= weight[np.newaxis, ...]

        # Get final class predictions
        if num_classes == 1:
            pred = (accum[0] > 0.5).astype(np.uint8)
        else:
            pred = accum.argmax(axis=0).astype(np.uint8)

        # Write output GeoTIFF
        out_profile = {
            "driver": "GTiff",
            "dtype": "uint8",
            "width": width,
            "height": height,
            "count": 1,
            "crs": crs,
            "transform": transform,
            "compress": "lzw",
        }
        with rasterio.open(output_path, "w", **out_profile) as dst:
            dst.write(pred[np.newaxis, ...])

        logger.info("Inference complete → %s", output_path)
        return pred

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_tile_grid(self, height: int, width: int):
        """Compute tile coordinates covering the full image with overlap.

        Returns list of (row, col, r1, r2, c1, c2, pr1, pr2, pc1, pc2) where
        r1:r2, c1:c2 are the source crop and pr1:pr2, pc1:pc2 are the
        paste coordinates in the output (excluding overlap padding).
        """
        ts = self.config.tile_size
        ov = self.config.overlap
        stride = ts - ov
        tiles = []
        for row, r_start in enumerate(range(0, height, stride)):
            for col, c_start in enumerate(range(0, width, stride)):
                r1 = r_start
                r2 = min(r_start + ts, height)
                c1 = c_start
                c2 = min(c_start + ts, width)
                tiles.append((row, col, r1, r2, c1, c2, r1, r2, c1, c2))
        return tiles

    def _infer_batch(
        self,
        tiles: list,
        coords: list,
        accum: np.ndarray,
        weight: np.ndarray,
        blend_kernel: np.ndarray,
        num_classes: int,
    ) -> None:
        """Run model inference on a batch of tiles and accumulate results."""
        # Pad tiles to uniform size
        ts = self.config.tile_size
        padded = []
        orig_shapes = []
        for tile in tiles:
            h, w = tile.shape[1], tile.shape[2]
            orig_shapes.append((h, w))
            if h < ts or w < ts:
                pad = np.zeros((tile.shape[0], ts, ts), dtype=np.float32)
                pad[:, :h, :w] = tile
                padded.append(pad)
            else:
                padded.append(tile)

        batch_np = np.stack(padded, axis=0)  # (N, C, H, W)
        batch_tensor = torch.from_numpy(batch_np).to(self.config.device)

        dev_type = "cuda" if "cuda" in self.config.device else "cpu"
        with torch.no_grad(), torch.autocast(
            device_type=dev_type,
            enabled=self.use_amp,
        ):
            outputs = self.model(batch_tensor)
            if hasattr(outputs, "logits"):
                outputs = outputs.logits
            if outputs.shape[-2:] != (ts, ts):
                import torch.nn.functional as F
                outputs = F.interpolate(outputs, size=(ts, ts), mode="bilinear", align_corners=False)
            if self.config.activation == "softmax":
                outputs = torch.softmax(outputs, dim=1)
            elif self.config.activation == "sigmoid":
                outputs = torch.sigmoid(outputs)

        preds_np = outputs.float().cpu().numpy()  # (N, C, H, W)

        for pred, (r1, r2, c1, c2, pr1, pr2, pc1, pc2), (oh, ow) in zip(
            preds_np, coords, orig_shapes
        ):
            pred_crop = pred[:, :oh, :ow]
            kernel_crop = blend_kernel[:oh, :ow]
            accum[:, r1:r2, c1:c2] += pred_crop * kernel_crop[np.newaxis, ...]
            weight[r1:r2, c1:c2] += kernel_crop

    @property
    def use_amp(self) -> bool:
        return self.config.mixed_precision and "cuda" in self.config.device

    @staticmethod
    def _normalize(image: np.ndarray) -> np.ndarray:
        """Per-band percentile normalization to [0, 1]."""
        for i in range(image.shape[0]):
            band = image[i]
            valid = band[band > 0]
            if valid.size > 0:
                p2, p98 = np.percentile(valid, (2, 98))
                image[i] = np.clip((band - p2) / max(p98 - p2, 1e-6), 0, 1)
        return image

    @staticmethod
    def _make_blend_kernel(tile_size: int, mode: str) -> np.ndarray:
        """Create a blending weight kernel for tile overlap averaging.

        Args:
            tile_size: Tile spatial dimension.
            mode: 'gaussian' for smooth blend, 'average' for uniform.

        Returns:
            float32 (tile_size, tile_size) weight array.
        """
        if mode == "average":
            return np.ones((tile_size, tile_size), dtype=np.float32)

        # Gaussian kernel
        sigma = tile_size / 4.0
        center = tile_size / 2.0
        x = np.arange(tile_size) - center
        y = np.arange(tile_size) - center
        xx, yy = np.meshgrid(x, y)
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2)).astype(np.float32)
        return kernel / kernel.max()

    @staticmethod
    def _detect_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
