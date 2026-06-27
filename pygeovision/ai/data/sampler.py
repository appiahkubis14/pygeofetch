"""
Balanced sampling strategies for geospatial datasets.

Imbalanced class distributions are the norm in geospatial AI (e.g. buildings
cover only ~5% of pixels globally). These samplers ensure the training loop
sees a balanced view of the data.
"""

from __future__ import annotations

import logging
import math
from typing import Iterator, Optional, Sized

import numpy as np

logger = logging.getLogger(__name__)

# PyTorch samplers
try:
    from torch.utils.data import Sampler

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Sampler = object  # type: ignore[assignment, misc]


class ClassBalancedSampler(Sampler if TORCH_AVAILABLE else object):  # type: ignore[misc]
    """
    Sampler that balances class frequencies by oversampling minority classes.

    Each epoch, samples are drawn such that each class appears approximately
    equally often. Tiles with no positive labels are optionally up-weighted.

    Parameters
    ----------
    class_counts : list of int
        Number of pixels (or instances) per class for each tile.
        Shape: ``[n_tiles, n_classes]``.
    num_samples : int, optional
        Total samples per epoch. Defaults to dataset size.
    replacement : bool
        Sample with replacement. Defaults to True.
    background_weight : float
        Weight for tiles with only background pixels (class 0).
        Defaults to 0.1 (reduces background-only tiles in training).

    Examples
    --------
    >>> sampler = ClassBalancedSampler(class_counts=class_pixel_counts)
    >>> loader = DataLoader(dataset, batch_size=8, sampler=sampler)
    """

    def __init__(
        self,
        class_counts: list[list[int]],
        num_samples: Optional[int] = None,
        replacement: bool = True,
        background_weight: float = 0.1,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for ClassBalancedSampler")

        self.class_counts = np.array(class_counts, dtype=np.float64)  # [N, C]
        self.replacement = replacement
        self.background_weight = background_weight
        self._num_samples = num_samples or len(class_counts)

        self._weights = self._compute_weights()
        logger.debug(
            "ClassBalancedSampler: %d tiles, %d classes, %d samples/epoch",
            len(class_counts),
            self.class_counts.shape[1],
            self._num_samples,
        )

    def _compute_weights(self) -> np.ndarray:
        """Compute per-tile sampling weights for class balance."""
        n_tiles, n_classes = self.class_counts.shape

        # Class frequencies across all tiles
        class_totals = self.class_counts.sum(axis=0) + 1e-10
        class_weights = 1.0 / class_totals
        class_weights /= class_weights.sum()

        # Per-tile weight = sum over classes of (class_count × class_weight)
        tile_weights = (self.class_counts * class_weights[np.newaxis, :]).sum(axis=1)

        # Down-weight background-only tiles
        has_only_background = (self.class_counts[:, 0] > 0) & (
            self.class_counts[:, 1:].sum(axis=1) == 0
        )
        tile_weights[has_only_background] *= self.background_weight

        # Normalise
        total = tile_weights.sum()
        if total > 0:
            tile_weights = tile_weights / total
        else:
            tile_weights = np.ones(n_tiles) / n_tiles

        return tile_weights.astype(np.float64)

    def __iter__(self) -> Iterator[int]:
        import torch  # noqa: PLC0415

        weights_tensor = torch.from_numpy(self._weights)
        indices = torch.multinomial(
            weights_tensor,
            num_samples=self._num_samples,
            replacement=self.replacement,
        )
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self._num_samples


class GeographicBlockSampler(Sampler if TORCH_AVAILABLE else object):  # type: ignore[misc]
    """
    Sampler that trains on spatially contiguous blocks to reduce autocorrelation.

    Geospatial data is spatially autocorrelated — adjacent tiles look similar.
    Training on spatially separated mini-batches improves generalisation.

    Parameters
    ----------
    tile_bounds : list of tuple
        Bounds of each tile as ``(min_x, min_y, max_x, max_y)``.
    batch_size : int
        Target batch size.
    block_size : int
        Number of geographically nearby tiles to group per block.
    shuffle : bool
        Shuffle block order each epoch.

    Examples
    --------
    >>> bounds = [(tile.bounds) for tile in dataset.tiles]
    >>> sampler = GeographicBlockSampler(tile_bounds=bounds, batch_size=8)
    """

    def __init__(
        self,
        tile_bounds: list[tuple[float, float, float, float]],
        batch_size: int = 8,
        block_size: int = 64,
        shuffle: bool = True,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for GeographicBlockSampler")

        self.tile_bounds = tile_bounds
        self.batch_size = batch_size
        self.block_size = block_size
        self.shuffle = shuffle

        # Build spatial blocks by sorting tiles by centroid
        self._blocks = self._build_blocks()
        logger.debug(
            "GeographicBlockSampler: %d tiles → %d blocks",
            len(tile_bounds),
            len(self._blocks),
        )

    def _build_blocks(self) -> list[list[int]]:
        """Sort tiles geographically and group into blocks."""
        # Compute centroids
        centroids = np.array(
            [
                ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
                for b in self.tile_bounds
            ]
        )

        # Sort by Hilbert curve approximation: sort by (x + y) then (x - y)
        primary = centroids[:, 0] + centroids[:, 1]
        secondary = centroids[:, 0] - centroids[:, 1]
        order = np.lexsort((secondary, primary))

        # Group into blocks
        n = len(order)
        blocks = [
            list(order[i : i + self.block_size])
            for i in range(0, n, self.block_size)
        ]
        return blocks

    def __iter__(self) -> Iterator[int]:
        import torch  # noqa: PLC0415

        blocks = self._blocks.copy()
        if self.shuffle:
            import random  # noqa: PLC0415

            random.shuffle(blocks)

        indices = []
        for block in blocks:
            block = block.copy()
            if self.shuffle:
                import random  # noqa: PLC0415

                random.shuffle(block)
            indices.extend(block)

        return iter(indices)

    def __len__(self) -> int:
        return len(self.tile_bounds)


class StratifiedTileSampler(Sampler if TORCH_AVAILABLE else object):  # type: ignore[misc]
    """
    Stratified sampler ensuring each mini-batch contains tiles from each class.

    Parameters
    ----------
    tile_labels : list of int
        Dominant class label for each tile (simplified to one class per tile).
    num_classes : int
        Total number of classes.
    batch_size : int
        Batch size.
    num_samples : int, optional
        Total samples per epoch.

    Examples
    --------
    >>> sampler = StratifiedTileSampler(tile_labels=dominant_classes, num_classes=5)
    """

    def __init__(
        self,
        tile_labels: list[int],
        num_classes: int,
        batch_size: int = 8,
        num_samples: Optional[int] = None,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for StratifiedTileSampler")

        self.tile_labels = np.array(tile_labels)
        self.num_classes = num_classes
        self.batch_size = batch_size
        self._num_samples = num_samples or len(tile_labels)

        # Group indices by class
        self._class_indices: list[list[int]] = []
        for c in range(num_classes):
            self._class_indices.append(
                list(np.where(self.tile_labels == c)[0])
            )

    def __iter__(self) -> Iterator[int]:
        import random  # noqa: PLC0415

        all_indices: list[int] = []
        per_class = max(1, self.batch_size // self.num_classes)

        # Sample equally from each class
        for c in range(self.num_classes):
            class_idx = self._class_indices[c]
            if not class_idx:
                continue
            sampled = random.choices(class_idx, k=per_class * (self._num_samples // self.batch_size + 1))
            all_indices.extend(sampled)

        random.shuffle(all_indices)
        return iter(all_indices[: self._num_samples])

    def __len__(self) -> int:
        return self._num_samples
