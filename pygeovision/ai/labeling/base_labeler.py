"""
Base labeler interface.

All labeling strategies inherit from BaseLabeler and implement
the ``label_tiles`` method, which takes a list of TileMetadata objects
and writes label GeoTIFFs to an output directory.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from pygeovision.ai.data.dataset import TileMetadata
from pygeovision.core.exceptions import LabelingError

logger = logging.getLogger(__name__)


@dataclass
class LabelingResult:
    """
    Result from a labeling operation.

    Attributes
    ----------
    tile_id : str
        Tile identifier.
    label_path : Path or None
        Path to generated label GeoTIFF.
    confidence : float
        Label confidence score (0–1).
    source : str
        Labeling strategy used.
    class_distribution : dict of str → float
        Fraction of pixels per class.
    skipped : bool
        Whether this tile was skipped (insufficient data coverage).
    error : str or None
        Error message if labeling failed.
    """

    tile_id: str
    label_path: Optional[Path] = None
    confidence: float = 1.0
    source: str = ""
    class_distribution: dict[str, float] = field(default_factory=dict)
    skipped: bool = False
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """True if labeling succeeded."""
        return not self.skipped and self.error is None and self.label_path is not None


class BaseLabeler(ABC):
    """
    Abstract base class for all labeling strategies.

    Subclasses must implement:
    - :meth:`label_tile` — label a single tile
    - :attr:`name` — strategy identifier string
    - :attr:`supported_tasks` — list of supported task types

    Parameters
    ----------
    confidence_threshold : float
        Minimum confidence to include a label. Defaults to 0.5.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        labeler_name: str = "",
        num_workers: int = 4,
        skip_existing: bool = True,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.labeler_name = labeler_name or self.__class__.__name__.lower()
        self.num_workers = num_workers
        self.skip_existing = skip_existing
        self._logger = logging.getLogger(
            f"pygeovision.ai.labeling.{self.__class__.__name__}"
        )

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name (e.g. ``"openstreetmap"``)."""
        ...

    @property
    @abstractmethod
    def supported_tasks(self) -> list[str]:
        """List of supported task types."""
        ...

    @abstractmethod
    def label_tile(
        self,
        tile_path: Path,
        tile_metadata: Any,
        output_path: Path,
        **kwargs: Any,
    ) -> LabelingResult:
        """
        Generate labels for a single tile.

        Parameters
        ----------
        tile : TileMetadata
            Tile to label.
        output_dir : Path
            Directory to write the label GeoTIFF.
        task : str
            AI task type.
        **kwargs
            Strategy-specific parameters.

        Returns
        -------
        LabelingResult
        """
        ...

    def label_tiles(
        self,
        tiles: list[TileMetadata],
        output_dir: Path,
        task: str = "segmentation",
        max_workers: int = 4,
        show_progress: bool = True,
        **kwargs: Any,
    ) -> list[LabelingResult]:
        """
        Generate labels for a list of tiles (parallelised).

        Parameters
        ----------
        tiles : list of TileMetadata
        output_dir : Path
        task : str
        max_workers : int
            Thread pool size for parallel labeling.
        show_progress : bool
            Show a progress bar.
        **kwargs

        Returns
        -------
        list of LabelingResult
        """
        if task not in self.supported_tasks:
            raise LabelingError(
                f"Labeler '{self.name}' does not support task '{task}'. "
                f"Supported: {self.supported_tasks}"
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results: list[LabelingResult] = []
        success_count = 0
        skip_count = 0
        error_count = 0

        self._logger.info(
            "Labeling %d tiles with '%s' strategy", len(tiles), self.name
        )

        # Process tiles (sequential for stability; subclasses can override with threads)
        for i, tile in enumerate(tiles):
            try:
                result = self.label_tile(tile=tile, output_dir=output_dir, task=task, **kwargs)
                if result.success:
                    success_count += 1
                elif result.skipped:
                    skip_count += 1
                else:
                    error_count += 1
            except Exception as exc:  # noqa: BLE001
                self._logger.warning("Failed to label tile %s: %s", tile.tile_id, exc)
                result = LabelingResult(
                    tile_id=tile.tile_id,
                    source=self.name,
                    error=str(exc),
                )
                error_count += 1

            results.append(result)

            if show_progress and (i + 1) % 100 == 0:
                self._logger.info(
                    "Progress: %d/%d tiles (✓=%d skip=%d ✗=%d)",
                    i + 1,
                    len(tiles),
                    success_count,
                    skip_count,
                    error_count,
                )

        self._logger.info(
            "Labeling complete: ✓=%d skip=%d ✗=%d / %d tiles",
            success_count,
            skip_count,
            error_count,
            len(tiles),
        )
        return results

    # ------------------------------------------------------------------
    # Shared utilities for subclasses
    # ------------------------------------------------------------------

    def _write_label_geotiff(
        self,
        mask: np.ndarray,
        tile: TileMetadata,
        output_dir: Path,
        nodata: int = 255,
    ) -> Path:
        """
        Write a label mask as a GeoTIFF aligned to the tile.

        Parameters
        ----------
        mask : np.ndarray
            Label mask array of shape (H, W) with integer class IDs.
        tile : TileMetadata
            Source tile metadata.
        output_dir : Path
            Output directory.
        nodata : int
            No-data value for the label raster.

        Returns
        -------
        Path
            Written GeoTIFF path.
        """
        try:
            import rasterio  # noqa: PLC0415
            from affine import Affine  # noqa: PLC0415
        except ImportError as exc:
            raise LabelingError(
                "rasterio is required for writing labels. "
                "Install with: pip install rasterio"
            ) from exc

        output_path = output_dir / f"{tile.tile_id}.tif"

        transform = Affine(*tile.transform) if len(tile.transform) == 6 else Affine.identity()

        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=mask.shape[0],
            width=mask.shape[1],
            count=1,
            dtype=np.uint8,
            crs=tile.crs,
            transform=transform,
            nodata=nodata,
            compress="lzw",
        ) as dst:
            dst.write(mask.astype(np.uint8), 1)

        return output_path

    @staticmethod
    def _compute_class_distribution(
        mask: np.ndarray,
        class_names: Optional[list[str]] = None,
        nodata_value: int = 255,
    ) -> dict[str, float]:
        """
        Compute the fraction of pixels per class.

        Parameters
        ----------
        mask : np.ndarray
        class_names : list of str, optional
        nodata_value : int

        Returns
        -------
        dict
        """
        valid_mask = mask[mask != nodata_value]
        if len(valid_mask) == 0:
            return {}

        result = {}
        for class_id in np.unique(valid_mask):
            fraction = float(np.mean(valid_mask == class_id))
            name = (
                class_names[class_id]
                if class_names and class_id < len(class_names)
                else str(class_id)
            )
            result[name] = fraction
        return result

    def _skip_tile(self, tile: TileMetadata, reason: str) -> LabelingResult:
        """Return a skipped LabelingResult."""
        self._logger.debug("Skipping tile %s: %s", tile.tile_id, reason)
        return LabelingResult(
            tile_id=tile.tile_id,
            source=self.name,
            skipped=True,
            error=reason,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(strategy={self.name!r})"
