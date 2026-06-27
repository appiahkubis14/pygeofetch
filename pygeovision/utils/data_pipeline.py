"""Data pipeline optimisation utilities (Phase 8.2)."""
from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import Any, Callable, Iterator, List, Optional, Tuple
logger = logging.getLogger(__name__)


def optimal_num_workers(safety_factor: float = 0.75) -> int:
    """Return the recommended num_workers for DataLoader."""
    n_cpu = os.cpu_count() or 1
    return max(1, int(n_cpu * safety_factor))


def prefetch_dataloader(
    dataset: Any,
    batch_size: int = 16,
    num_workers: Optional[int] = None,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    shuffle: bool = True,
) -> Any:
    """Build a DataLoader with optimal prefetching settings."""
    try:
        import torch
        nw = num_workers if num_workers is not None else optimal_num_workers()
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=nw,
            pin_memory=pin_memory and torch.cuda.is_available(),
            persistent_workers=persistent_workers and nw > 0,
            prefetch_factor=prefetch_factor if nw > 0 else None,
            drop_last=True,
        )
    except ImportError:
        raise ImportError("torch required for DataLoader")


def parallel_raster_read(
    paths: List[str],
    n_workers: int = 4,
    fn: Optional[Callable] = None,
) -> List[Any]:
    """Read multiple raster files in parallel using ThreadPoolExecutor."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    def _read(p: str) -> Any:
        if fn is not None:
            return fn(p)
        try:
            import rasterio
            with rasterio.open(p) as src:
                return src.read()
        except ImportError:
            raise ImportError("rasterio required: pip install rasterio")

    results = [None] * len(paths)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_read, p): i for i, p in enumerate(paths)}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as exc:
                logger.warning("Failed to read %s: %s", paths[idx], exc)
    return results


class StreamingRasterDataset:
    """Iterable-style streaming dataset for large GeoTIFF files (Phase 8.2).

    Tiles large rasters on-the-fly without loading the whole file into RAM,
    enabling training on imagery that exceeds available memory.
    """

    def __init__(
        self,
        image_path: str,
        label_path: Optional[str] = None,
        chip_size: int = 512,
        overlap: int = 64,
        transform: Optional[Callable] = None,
    ) -> None:
        self.image_path = image_path
        self.label_path = label_path
        self.chip_size = chip_size
        self.overlap = overlap
        self.transform = transform
        self._chips: Optional[List[Tuple[int, int, int, int]]] = None
        self._compute_chips()

    def _compute_chips(self) -> None:
        """Compute chip bounding boxes for the full raster."""
        try:
            import rasterio
            with rasterio.open(self.image_path) as src:
                H, W = src.height, src.width
            stride = self.chip_size - self.overlap
            chips = []
            for row in range(0, H, stride):
                for col in range(0, W, stride):
                    r2 = min(row + self.chip_size, H)
                    c2 = min(col + self.chip_size, W)
                    chips.append((col, row, c2 - col, r2 - row))
            self._chips = chips
        except (ImportError, Exception) as exc:
            logger.warning("Could not compute chips: %s", exc)
            self._chips = []

    def __len__(self) -> int:
        return len(self._chips or [])

    def __getitem__(self, idx: int) -> Any:
        try:
            import rasterio
            import numpy as np
            from rasterio.windows import Window
            col_off, row_off, width, height = self._chips[idx]
            window = Window(col_off, row_off, width, height)
            with rasterio.open(self.image_path) as src:
                image = src.read(window=window).astype(np.float32)
            item = {"image": image, "chip_idx": idx, "window": self._chips[idx]}
            if self.label_path:
                with rasterio.open(self.label_path) as src:
                    mask = src.read(1, window=window).astype(np.int64)
                item["mask"] = mask
            if self.transform:
                item = self.transform(item)
            return item
        except ImportError:
            raise ImportError("rasterio required: pip install rasterio")

    def __iter__(self) -> Iterator[Any]:
        for i in range(len(self)):
            yield self[i]
