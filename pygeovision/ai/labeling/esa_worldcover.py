"""
ESA WorldCover Labeler.

Generates land cover labels using the ESA WorldCover 10m global land cover map
(2020 and 2021 editions). Provides 11 land cover classes at 10m resolution
covering the entire globe.

Reference: https://worldcover2021.esa.int/

Example:
    >>> from pygeovision.ai.labeling.esa_worldcover import ESAWorldCoverLabeler
    >>> labeler = ESAWorldCoverLabeler(year=2021)
    >>> results = labeler.label_tiles(tiles, output_dir="./labels/")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pygeovision.ai.labeling.base_labeler import BaseLabeler, LabelingResult
from pygeovision.core.exceptions import LabelingError

logger = logging.getLogger(__name__)

# ESA WorldCover class definitions (value → name)
WORLDCOVER_CLASSES: Dict[int, str] = {
    10: "tree_cover",
    20: "shrubland",
    30: "grassland",
    40: "cropland",
    50: "built_up",
    60: "bare_sparse_vegetation",
    70: "snow_ice",
    80: "permanent_water",
    90: "herbaceous_wetland",
    95: "mangroves",
    100: "moss_lichen",
}

# AWS Open Data Registry: ESA WorldCover
_WORLDCOVER_S3_BASE = "s3://esa-worldcover/v200/{year}/map"
_WORLDCOVER_HTTP_BASE = (
    "https://esa-worldcover.s3.amazonaws.com/v200/{year}/map"
)

# Tile grid: 3°×3° tiles named ESA_WorldCover_10m_{year}_v200_N{lat}E{lon}_Map.tif
_TILE_SIZE_DEG = 3.0


@dataclass
class WorldCoverConfig:
    """Configuration for the ESA WorldCover labeler.

    Attributes:
        year: WorldCover edition year (2020 or 2021).
        remap: Optional dict mapping WorldCover class values to custom values.
        cache_dir: Local cache directory for downloaded tiles.
        request_timeout: HTTP timeout in seconds.
    """

    year: int = 2021
    remap: Optional[Dict[int, int]] = None
    cache_dir: Optional[Path] = None
    request_timeout: int = 120


class ESAWorldCoverLabeler(BaseLabeler):
    """Labeler using ESA WorldCover 10m global land cover map.

    Downloads ESA WorldCover tiles as needed and extracts class labels
    for any location on Earth. Supports both the 2020 and 2021 editions.

    Args:
        year: WorldCover edition year. One of 2020 or 2021.
        remap: Optional mapping from WorldCover class values to custom class IDs.
            Example: {10: 1, 20: 1, 30: 1, 40: 2, 50: 3}  # simplified 3-class
        cache_dir: Local directory for caching tiles.
        num_workers: Number of parallel tile workers.
        skip_existing: Skip tiles that already have labels.

    Raises:
        ValueError: If year is not 2020 or 2021.

    Example:
        >>> labeler = ESAWorldCoverLabeler(year=2021)
        >>> # Use simplified 3-class system: vegetation, built-up, water
        >>> labeler = ESAWorldCoverLabeler(
        ...     remap={10: 1, 20: 1, 30: 1, 40: 1, 50: 2, 80: 3, 90: 3}
        ... )
    """

    VALID_YEARS = (2020, 2021)

    def __init__(
        self,
        year: int = 2021,
        remap: Optional[Dict[int, int]] = None,
        cache_dir: Optional[Path] = None,
        num_workers: int = 4,
        skip_existing: bool = True,
    ) -> None:
        super().__init__(
            labeler_name="esa_worldcover",
            num_workers=num_workers,
            skip_existing=skip_existing,
        )
        if year not in self.VALID_YEARS:
            raise ValueError(
                f"year must be one of {self.VALID_YEARS}, got {year}"
            )

        self.config = WorldCoverConfig(
            year=year,
            remap=remap,
            cache_dir=cache_dir or Path.home() / ".pygeovision" / "cache" / "esa_worldcover",
        )
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # BaseLabeler abstract properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return 'esa_worldcover'

    @property
    def supported_tasks(self) -> list:
        return ['land_cover', 'segmentation']

    # ------------------------------------------------------------------
    # BaseLabeler interface
    # ------------------------------------------------------------------

    def label_tile(
        self,
        tile_path: Path,
        tile_metadata: Any,
        output_path: Path,
    ) -> LabelingResult:
        """Generate a land cover mask for a single imagery tile.

        Args:
            tile_path: Path to the GeoTIFF tile.
            tile_metadata: TileMetadata with bounds, CRS, and shape info.
            output_path: Destination path for the label GeoTIFF.

        Returns:
            LabelingResult with class distribution statistics.
        """
        try:
            import rasterio
            from rasterio.warp import reproject, Resampling
            import pyproj
            from shapely.geometry import box
            from shapely.ops import transform as shapely_transform
        except ImportError as exc:
            raise LabelingError(
                "esa_worldcover labeler requires rasterio, pyproj, and shapely. "
                "Install via: pip install rasterio pyproj shapely"
            ) from exc

        try:
            with rasterio.open(tile_path) as src:
                height, width = src.height, src.width
                crs = src.crs
                bounds = src.bounds
                transform = src.transform

            # Convert bounds to WGS-84 for tile discovery
            wgs84 = pyproj.CRS("EPSG:4326")
            project_to_wgs84 = pyproj.Transformer.from_crs(
                crs, wgs84, always_xy=True
            ).transform
            tile_box = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
            from shapely.ops import transform as shp_transform
            tile_box_wgs84 = shp_transform(project_to_wgs84, tile_box)
            bbox_wgs84 = tile_box_wgs84.bounds

            # Find and download covering WorldCover tiles
            wc_tile_paths = self._get_worldcover_tiles(bbox_wgs84)

            if not wc_tile_paths:
                # Return background mask if no tiles found
                logger.warning("No WorldCover tiles found for bbox %s", bbox_wgs84)
                mask = np.zeros((height, width), dtype=np.uint8)
                self._write_label_geotiff(
                    mask, output_path,
                    {"driver": "GTiff", "dtype": "uint8", "width": width,
                     "height": height, "count": 1, "crs": crs,
                     "transform": transform, "compress": "lzw"}
                )
                return LabelingResult(
                    tile_path=tile_path, label_path=output_path,
                    success=True, labeler="esa_worldcover",
                    metadata={"warning": "no_worldcover_tiles_found"},
                )

            # Mosaic and reproject WorldCover data to tile CRS/extent
            mask = self._extract_worldcover_data(
                wc_tile_paths=wc_tile_paths,
                dst_crs=crs,
                dst_transform=transform,
                dst_height=height,
                dst_width=width,
            )

            # Apply optional class remapping
            if self.config.remap:
                mask = self._remap_classes(mask, self.config.remap)

            # Write output
            meta = {
                "driver": "GTiff",
                "dtype": "uint8",
                "width": width,
                "height": height,
                "count": 1,
                "crs": crs,
                "transform": transform,
                "compress": "lzw",
            }
            self._write_label_geotiff(mask, output_path, meta)

            active_classes = {
                v: WORLDCOVER_CLASSES.get(v, f"class_{v}")
                for v in np.unique(mask)
                if v != 0
            }
            if self.config.remap:
                active_classes = {v: f"class_{v}" for v in np.unique(mask)}
            stats = self._compute_class_distribution(mask, active_classes)

            return LabelingResult(
                tile_path=tile_path,
                label_path=output_path,
                success=True,
                labeler="esa_worldcover",
                class_distribution=stats,
                metadata={
                    "year": self.config.year,
                    "num_classes": len(active_classes),
                    "remapped": self.config.remap is not None,
                    "bbox_wgs84": bbox_wgs84,
                },
            )

        except Exception as exc:
            logger.error(
                "ESAWorldCoverLabeler failed for %s: %s", tile_path, exc
            )
            return LabelingResult(
                tile_path=tile_path,
                label_path=output_path,
                success=False,
                labeler="esa_worldcover",
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_worldcover_tiles(
        self, bbox_wgs84: Tuple[float, float, float, float]
    ) -> List[Path]:
        """Identify and download WorldCover tiles covering a bounding box.

        WorldCover uses a 3°×3° tile grid with filenames like:
        ESA_WorldCover_10m_2021_v200_N03E012_Map.tif

        Args:
            bbox_wgs84: (minx, miny, maxx, maxy) in WGS-84 degrees.

        Returns:
            List of local paths to downloaded (or cached) WorldCover tiles.
        """
        minx, miny, maxx, maxy = bbox_wgs84
        tile_size = _TILE_SIZE_DEG

        # Find all 3°×3° tile origins that overlap the bbox
        lat_starts = []
        lat = int(np.floor(miny / tile_size) * tile_size)
        while lat <= maxy:
            lat_starts.append(lat)
            lat += tile_size

        lon_starts = []
        lon = int(np.floor(minx / tile_size) * tile_size)
        while lon <= maxx:
            lon_starts.append(lon)
            lon += tile_size

        local_paths: List[Path] = []
        for lat_s in lat_starts:
            for lon_s in lon_starts:
                path = self._download_tile(lat_s, lon_s)
                if path is not None:
                    local_paths.append(path)

        return local_paths

    def _download_tile(self, lat: int, lon: int) -> Optional[Path]:
        """Download a single WorldCover tile if not already cached.

        Args:
            lat: Tile origin latitude (south edge, integer degrees).
            lon: Tile origin longitude (west edge, integer degrees).

        Returns:
            Local path to the tile, or None if unavailable.
        """
        import requests

        lat_str = f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}"
        lon_str = f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}"
        filename = (
            f"ESA_WorldCover_10m_{self.config.year}_v200"
            f"_{lat_str}{lon_str}_Map.tif"
        )
        local_path = self.config.cache_dir / filename

        if local_path.exists():
            return local_path

        url = f"{_WORLDCOVER_HTTP_BASE.format(year=self.config.year)}/{filename}"
        logger.info("Downloading WorldCover tile: %s", filename)
        try:
            resp = requests.get(url, timeout=self.config.request_timeout, stream=True)
            resp.raise_for_status()
            with open(local_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    fh.write(chunk)
            logger.info("Cached WorldCover tile at %s", local_path)
            return local_path
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                logger.debug("WorldCover tile not found (ocean/no data): %s", url)
                return None
            logger.error("Failed to download WorldCover tile %s: %s", url, exc)
            return None

    def _extract_worldcover_data(
        self,
        wc_tile_paths: List[Path],
        dst_crs: Any,
        dst_transform: Any,
        dst_height: int,
        dst_width: int,
    ) -> np.ndarray:
        """Mosaic and reproject WorldCover tiles to match the imagery tile.

        Args:
            wc_tile_paths: Paths to downloaded WorldCover tiles.
            dst_crs: Target CRS (imagery tile CRS).
            dst_transform: Target affine transform.
            dst_height: Target height in pixels.
            dst_width: Target width in pixels.

        Returns:
            Uint8 label array with WorldCover class values.
        """
        import rasterio
        from rasterio.merge import merge
        from rasterio.warp import reproject, Resampling

        # Open all source tiles
        src_files = [rasterio.open(p) for p in wc_tile_paths]
        try:
            if len(src_files) > 1:
                merged_data, merged_transform = merge(src_files)
                merged_crs = src_files[0].crs
            else:
                merged_data = src_files[0].read(1, out_dtype=np.uint8)
                merged_data = merged_data[np.newaxis, ...]
                merged_transform = src_files[0].transform
                merged_crs = src_files[0].crs

            # Reproject to destination CRS
            dst_array = np.zeros((1, dst_height, dst_width), dtype=np.uint8)
            reproject(
                source=merged_data.astype(np.uint8),
                destination=dst_array,
                src_transform=merged_transform,
                src_crs=merged_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest,
            )

            return dst_array[0]
        finally:
            for f in src_files:
                f.close()

    @staticmethod
    def _remap_classes(
        mask: np.ndarray, remap: Dict[int, int]
    ) -> np.ndarray:
        """Apply class ID remapping to a label mask.

        Args:
            mask: Input label array with original class IDs.
            remap: Mapping from original to new class IDs.
                   Values not in the map are set to 0.

        Returns:
            Remapped label array.
        """
        new_mask = np.zeros_like(mask)
        for src_val, dst_val in remap.items():
            new_mask[mask == src_val] = dst_val
        return new_mask

    @property
    def class_names(self) -> Dict[int, str]:
        """Return the WorldCover class name mapping."""
        return WORLDCOVER_CLASSES
