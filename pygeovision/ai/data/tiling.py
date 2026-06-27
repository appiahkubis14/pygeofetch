"""
TilingEngine — Large geospatial image tiling.

Splits arbitrarily large rasters into fixed-size tiles with configurable overlap,
streaming them to avoid loading entire rasters into memory.  All geospatial
metadata (CRS, affine transform, tile bounds) is preserved per tile.
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator, Optional, Union

import numpy as np

from pygeovision.ai.data.dataset import TileMetadata
from pygeovision.core.exceptions import TilingError

logger = logging.getLogger(__name__)


@dataclass
class TilingConfig:
    """
    Configuration for the tiling engine.

    Attributes
    ----------
    tile_size : int
        Side length of each square tile in pixels. Defaults to 512.
    overlap : int
        Overlap between adjacent tiles in pixels. Defaults to 64.
    min_valid_fraction : float
        Minimum fraction of valid (non-nodata) pixels required to keep a tile.
        Tiles with less valid data are discarded. Defaults to 0.1.
    nodata_value : float or None
        Value representing no-data. If None, auto-detected from raster metadata.
    padding_mode : str
        How to handle edge tiles that don't fill the full tile size:
        ``"reflect"`` — reflect padding; ``"constant"`` — fill with nodata;
        ``"edge"`` — replicate edge pixels; ``"discard"`` — drop incomplete tiles.
    save_tiles : bool
        Whether to save tiles to disk as GeoTIFFs. Defaults to True.
    output_dir : Path or None
        Where to save tiles. Required if ``save_tiles=True``.
    compress : str
        Compression codec for output tiles. Defaults to ``"lzw"``.
    """

    tile_size: int = 512
    overlap: int = 64
    min_valid_fraction: float = 0.1
    nodata_value: Optional[float] = None
    padding_mode: str = "reflect"
    save_tiles: bool = True
    output_dir: Optional[Path] = None
    compress: str = "lzw"

    def __post_init__(self) -> None:
        if self.tile_size <= 0:
            raise TilingError(f"tile_size must be positive, got {self.tile_size}")
        if self.overlap < 0:
            raise TilingError(f"overlap must be >= 0, got {self.overlap}")
        if self.overlap >= self.tile_size:
            raise TilingError(
                f"overlap ({self.overlap}) must be less than tile_size ({self.tile_size})"
            )
        valid_modes = {"reflect", "constant", "edge", "discard"}
        if self.padding_mode not in valid_modes:
            raise TilingError(
                f"Invalid padding_mode '{self.padding_mode}'. Choose from {valid_modes}"
            )
        if self.save_tiles and self.output_dir is None:
            raise TilingError("output_dir must be set when save_tiles=True")


class TilingEngine:
    """
    Tiles large geospatial rasters into fixed-size tiles.

    Streams tiles from disk without loading the entire raster into memory.
    Handles edge tiles, overlap blending, irregular boundaries, and
    multi-band imagery.

    Parameters
    ----------
    config : TilingConfig
        Tiling configuration.

    Examples
    --------
    >>> engine = TilingEngine(TilingConfig(tile_size=512, overlap=64,
    ...                                    output_dir=Path("./tiles/")))
    >>> tiles = engine.tile(raster_path=Path("sentinel2_scene.tif"))
    >>> print(f"Generated {len(tiles)} tiles")

    Or as a generator (memory-efficient):

    >>> for tile_meta, tile_data in engine.tile_stream(Path("large_scene.tif")):
    ...     process(tile_data)
    """

    def __init__(self, config: TilingConfig) -> None:
        self.config = config
        if config.output_dir:
            Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def tile(
        self,
        raster_path: Union[str, Path],
        bands: Optional[list[int]] = None,
        mask_path: Optional[Union[str, Path]] = None,
        provider: str = "",
        satellite: str = "",
        datetime: str = "",
        cloud_cover: float = 0.0,
    ) -> list[TileMetadata]:
        """
        Tile a raster and optionally save tiles to disk.

        Parameters
        ----------
        raster_path : str or Path
            Path to the source raster file.
        bands : list of int, optional
            Band indices to read (1-indexed). None = all bands.
        mask_path : str or Path, optional
            Path to a binary validity mask (1 = valid, 0 = nodata).
        provider : str
            Data provider ID (forwarded to tile metadata).
        satellite : str
            Satellite name.
        datetime : str
            Acquisition datetime (ISO 8601).
        cloud_cover : float
            Cloud cover percentage.

        Returns
        -------
        list of TileMetadata
            Metadata for all generated tiles.

        Raises
        ------
        TilingError
            If the raster cannot be opened or tiling fails.
        """
        tiles: list[TileMetadata] = []
        for meta, _ in self.tile_stream(
            raster_path=raster_path,
            bands=bands,
            mask_path=mask_path,
            provider=provider,
            satellite=satellite,
            datetime=datetime,
            cloud_cover=cloud_cover,
        ):
            tiles.append(meta)
        logger.info(
            "Tiled %s → %d tiles (size=%d, overlap=%d)",
            raster_path,
            len(tiles),
            self.config.tile_size,
            self.config.overlap,
        )
        return tiles

    def tile_stream(
        self,
        raster_path: Union[str, Path],
        bands: Optional[list[int]] = None,
        mask_path: Optional[Union[str, Path]] = None,
        provider: str = "",
        satellite: str = "",
        datetime: str = "",
        cloud_cover: float = 0.0,
    ) -> Generator[tuple[TileMetadata, np.ndarray], None, None]:
        """
        Stream tiles from a raster without loading it all into memory.

        Parameters
        ----------
        raster_path : str or Path
            Source raster file.
        bands : list of int, optional
            Band indices.
        mask_path : str or Path, optional
            Validity mask raster.
        provider : str
            Provider ID.
        satellite : str
            Satellite name.
        datetime : str
            Acquisition datetime.
        cloud_cover : float
            Cloud cover percentage.

        Yields
        ------
        tuple of (TileMetadata, np.ndarray)
            Tile metadata and pixel data ``[C, tile_size, tile_size]``.
        """
        try:
            import rasterio  # noqa: PLC0415
            from rasterio.windows import Window  # noqa: PLC0415
        except ImportError as exc:
            raise TilingError(
                "rasterio is required for tiling. Install with: pip install rasterio"
            ) from exc

        raster_path = Path(raster_path)
        if not raster_path.exists():
            raise TilingError(f"Raster not found: {raster_path}")

        try:
            with rasterio.open(raster_path) as src:
                raster_height = src.height
                raster_width = src.width
                src_crs = str(src.crs) if src.crs else "EPSG:4326"
                src_transform = list(src.transform)
                resolution_m = abs(src.transform.a)  # pixel width in CRS units
                nodata = self.config.nodata_value or src.nodata
                band_indices = bands or list(range(1, src.count + 1))

                step = self.config.tile_size - self.config.overlap
                n_rows = math.ceil((raster_height - self.config.overlap) / step)
                n_cols = math.ceil((raster_width - self.config.overlap) / step)

                tile_count = 0
                skipped_count = 0

                for row_idx in range(n_rows):
                    for col_idx in range(n_cols):
                        row_off = row_idx * step
                        col_off = col_idx * step

                        # Clamp to raster bounds
                        actual_height = min(self.config.tile_size, raster_height - row_off)
                        actual_width = min(self.config.tile_size, raster_width - col_off)

                        if actual_height <= 0 or actual_width <= 0:
                            continue

                        window = Window(
                            col_off=col_off,
                            row_off=row_off,
                            width=actual_width,
                            height=actual_height,
                        )

                        # Read tile data
                        data = src.read(band_indices, window=window)

                        # Handle edge tiles: pad to full tile_size
                        if actual_height < self.config.tile_size or actual_width < self.config.tile_size:
                            if self.config.padding_mode == "discard":
                                skipped_count += 1
                                continue
                            data = self._pad_tile(data, nodata)

                        # Filter by valid pixel fraction
                        if nodata is not None:
                            valid_frac = float(np.mean(data[0] != nodata))
                            if valid_frac < self.config.min_valid_fraction:
                                skipped_count += 1
                                continue

                        # Compute tile bounds
                        tile_transform = src.window_transform(window)
                        bounds = rasterio.transform.array_bounds(
                            actual_height, actual_width, tile_transform
                        )

                        # Generate stable tile ID
                        tile_id = self._make_tile_id(raster_path, row_off, col_off)

                        meta = TileMetadata(
                            tile_id=tile_id,
                            source_file=raster_path,
                            bounds=tuple(bounds),  # type: ignore[arg-type]
                            crs=src_crs,
                            transform=list(tile_transform),
                            row_off=row_off,
                            col_off=col_off,
                            height=self.config.tile_size,
                            width=self.config.tile_size,
                            bands=band_indices,
                            resolution_m=resolution_m,
                            overlap=self.config.overlap,
                            provider=provider,
                            satellite=satellite,
                            datetime=datetime,
                            cloud_cover=cloud_cover,
                        )

                        # Optionally save tile GeoTIFF
                        if self.config.save_tiles and self.config.output_dir:
                            self._save_tile(
                                data=data,
                                meta=meta,
                                src_profile=src.profile,
                                band_indices=band_indices,
                                nodata=nodata,
                            )

                        tile_count += 1
                        yield meta, data

        except rasterio.errors.RasterioIOError as exc:
            raise TilingError(f"Failed to open raster {raster_path}: {exc}") from exc

    def compute_grid(
        self,
        raster_height: int,
        raster_width: int,
    ) -> list[tuple[int, int, int, int]]:
        """
        Compute tile (row_off, col_off, height, width) without reading data.

        Useful for pre-planning memory requirements.

        Parameters
        ----------
        raster_height : int
            Source raster height in pixels.
        raster_width : int
            Source raster width in pixels.

        Returns
        -------
        list of tuple
            Each tuple: ``(row_off, col_off, tile_height, tile_width)``.
        """
        step = self.config.tile_size - self.config.overlap
        n_rows = math.ceil((raster_height - self.config.overlap) / step)
        n_cols = math.ceil((raster_width - self.config.overlap) / step)

        grid = []
        for row_idx in range(n_rows):
            for col_idx in range(n_cols):
                row_off = row_idx * step
                col_off = col_idx * step
                h = min(self.config.tile_size, raster_height - row_off)
                w = min(self.config.tile_size, raster_width - col_off)
                if h > 0 and w > 0:
                    grid.append((row_off, col_off, h, w))
        return grid

    def estimate_tile_count(
        self,
        raster_height: int,
        raster_width: int,
    ) -> int:
        """
        Estimate the number of tiles that will be generated.

        Parameters
        ----------
        raster_height : int
        raster_width : int

        Returns
        -------
        int
        """
        return len(self.compute_grid(raster_height, raster_width))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _pad_tile(
        self,
        data: np.ndarray,
        nodata: Optional[float],
    ) -> np.ndarray:
        """Pad a tile to ``(C, tile_size, tile_size)``."""
        c, h, w = data.shape
        ts = self.config.tile_size
        if h == ts and w == ts:
            return data

        padded = np.full(
            (c, ts, ts),
            fill_value=nodata if nodata is not None else 0.0,
            dtype=data.dtype,
        )
        padded[:, :h, :w] = data

        if self.config.padding_mode == "reflect" and h > 1 and w > 1:
            if h < ts:
                reflect_rows = min(ts - h, h - 1)
                padded[:, h : h + reflect_rows, :w] = data[:, -reflect_rows - 1 : -1, :][:, ::-1, :]
            if w < ts:
                reflect_cols = min(ts - w, w - 1)
                padded[:, :h, w : w + reflect_cols] = data[:, :, -reflect_cols - 1 : -1][:, :, ::-1]
        elif self.config.padding_mode == "edge":
            if h < ts:
                padded[:, h:, :w] = data[:, -1:, :]
            if w < ts:
                padded[:, :, w:] = padded[:, :, w - 1 : w]

        return padded

    def _save_tile(
        self,
        data: np.ndarray,
        meta: TileMetadata,
        src_profile: dict[str, Any],
        band_indices: list[int],
        nodata: Optional[float],
    ) -> Path:
        """Save a tile as a GeoTIFF."""
        try:
            import rasterio  # noqa: PLC0415
            from affine import Affine  # noqa: PLC0415
        except ImportError:
            return Path()  # Skip saving if rasterio unavailable

        output_dir = Path(self.config.output_dir)  # type: ignore[arg-type]
        output_path = output_dir / f"{meta.tile_id}.tif"

        profile = src_profile.copy()
        profile.update(
            {
                "driver": "GTiff",
                "height": self.config.tile_size,
                "width": self.config.tile_size,
                "count": len(band_indices),
                "crs": meta.crs,
                "transform": Affine(*meta.transform),
                "compress": self.config.compress,
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
            }
        )
        if nodata is not None:
            profile["nodata"] = nodata

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data)

        return output_path

    @staticmethod
    def _make_tile_id(raster_path: Path, row_off: int, col_off: int) -> str:
        """Generate a stable, unique tile ID."""
        raw = f"{raster_path.stem}_{row_off}_{col_off}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]
