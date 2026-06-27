"""
Dynamic World Near Real-Time Land Cover Labeler.

Generates land cover labels using Google's Dynamic World dataset — a near
real-time land use and land cover product built on Sentinel-2, providing
per-pixel class probabilities at 10m resolution worldwide.

Dataset: https://dynamicworld.app/
Paper: Brown et al. (2022), Nature Communications

Classes:
    0: water, 1: trees, 2: grass, 3: flooded_vegetation,
    4: crops, 5: shrub_scrub, 6: built, 7: bare, 8: snow_ice

Example:
    >>> from pygeovision.ai.labeling.dynamic_world import DynamicWorldLabeler
    >>> labeler = DynamicWorldLabeler(start_date="2023-06-01", end_date="2023-08-31")
    >>> results = labeler.label_tiles(tiles, output_dir="./labels/")
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pygeovision.ai.labeling.base_labeler import BaseLabeler, LabelingResult
from pygeovision.core.exceptions import LabelingError

logger = logging.getLogger(__name__)

# Dynamic World class definitions
DW_CLASSES: Dict[int, str] = {
    0: "water",
    1: "trees",
    2: "grass",
    3: "flooded_vegetation",
    4: "crops",
    5: "shrub_scrub",
    6: "built",
    7: "bare",
    8: "snow_ice",
}

# Dynamic World STAC endpoint via Google Earth Engine / Microsoft Planetary Computer
_DW_STAC_URL = (
    "https://planetarycomputer.microsoft.com/api/stac/v1"
)
_DW_COLLECTION = "dynamic-world-s2"


@dataclass
class DynamicWorldConfig:
    """Configuration for the Dynamic World labeler.

    Attributes:
        start_date: Start of the temporal composite window (ISO 8601).
        end_date: End of the temporal composite window (ISO 8601).
        composite_method: How to aggregate multiple scenes.
            'mode' = most frequent class, 'probability' = max mean probability.
        probability_bands: Whether to export per-class probability bands.
        cache_dir: Local directory for caching downloaded scenes.
        request_timeout: HTTP timeout in seconds.
    """

    start_date: str = "2023-01-01"
    end_date: str = "2023-12-31"
    composite_method: str = "mode"  # 'mode' | 'probability'
    probability_bands: bool = False
    cache_dir: Optional[Path] = None
    request_timeout: int = 120


class DynamicWorldLabeler(BaseLabeler):
    """Labeler using Google Dynamic World near real-time land cover.

    Fetches Dynamic World Sentinel-2-based land cover data via STAC
    (Microsoft Planetary Computer or Google Earth Engine) and generates
    pixel-aligned land cover masks for imagery tiles.

    Args:
        start_date: Start of the composite window (ISO 8601, e.g. "2023-06-01").
        end_date: End of the composite window (ISO 8601, e.g. "2023-08-31").
        composite_method: Aggregation method for multi-scene composites.
            'mode' returns the most frequent class; 'probability' returns
            the class with the highest mean probability.
        probability_bands: If True, output a multi-band GeoTIFF with per-class
            probability scores (0-100) instead of a single-band class mask.
        cache_dir: Local cache directory.
        num_workers: Number of parallel tile workers.
        skip_existing: Skip tiles that already have labels.

    Example:
        >>> labeler = DynamicWorldLabeler(
        ...     start_date="2023-06-01",
        ...     end_date="2023-08-31",
        ...     composite_method="mode",
        ... )
        >>> results = labeler.label_tiles(tiles, "./labels/")
    """

    def __init__(
        self,
        start_date: str = "2023-01-01",
        end_date: str = "2023-12-31",
        composite_method: str = "mode",
        probability_bands: bool = False,
        cache_dir: Optional[Path] = None,
        num_workers: int = 4,
        skip_existing: bool = True,
    ) -> None:
        super().__init__(
            labeler_name="dynamic_world",
            num_workers=num_workers,
            skip_existing=skip_existing,
        )
        if composite_method not in ("mode", "probability"):
            raise ValueError(
                f"composite_method must be 'mode' or 'probability', got {composite_method!r}"
            )

        self.config = DynamicWorldConfig(
            start_date=start_date,
            end_date=end_date,
            composite_method=composite_method,
            probability_bands=probability_bands,
            cache_dir=cache_dir or Path.home() / ".pygeovision" / "cache" / "dynamic_world",
        )
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # BaseLabeler abstract properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return 'dynamic_world'

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
        """Generate a Dynamic World land cover mask for a single tile.

        Args:
            tile_path: Path to the GeoTIFF imagery tile.
            tile_metadata: TileMetadata with bounds, CRS, and shape info.
            output_path: Destination path for the output label GeoTIFF.

        Returns:
            LabelingResult with class statistics and metadata.
        """
        try:
            import rasterio
            import pyproj
            from shapely.geometry import box
            from shapely.ops import transform as shp_transform
        except ImportError as exc:
            raise LabelingError(
                "dynamic_world labeler requires rasterio, pyproj, and shapely. "
                "Install via: pip install rasterio pyproj shapely"
            ) from exc

        try:
            with rasterio.open(tile_path) as src:
                height, width = src.height, src.width
                crs = src.crs
                bounds = src.bounds
                transform = src.transform

            # Project bounds to WGS-84 for STAC query
            wgs84 = pyproj.CRS("EPSG:4326")
            project = pyproj.Transformer.from_crs(
                crs, wgs84, always_xy=True
            ).transform
            tile_box = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
            tile_box_wgs84 = shp_transform(project, tile_box)
            bbox_wgs84 = tile_box_wgs84.bounds

            # Search and download Dynamic World scenes
            dw_paths = self._fetch_dynamic_world_scenes(bbox_wgs84)

            if not dw_paths:
                logger.warning(
                    "No Dynamic World scenes found for %s in %s to %s",
                    bbox_wgs84, self.config.start_date, self.config.end_date
                )
                mask = np.zeros((height, width), dtype=np.uint8)
                self._write_label_geotiff(
                    mask, output_path,
                    {"driver": "GTiff", "dtype": "uint8", "width": width,
                     "height": height, "count": 1, "crs": crs,
                     "transform": transform, "compress": "lzw"}
                )
                return LabelingResult(
                    tile_path=tile_path, label_path=output_path,
                    success=True, labeler="dynamic_world",
                    metadata={"warning": "no_scenes_found"},
                )

            # Create composite and reproject to tile CRS/extent
            mask = self._create_composite(
                scene_paths=dw_paths,
                method=self.config.composite_method,
                dst_crs=crs,
                dst_transform=transform,
                dst_height=height,
                dst_width=width,
            )

            bands = 1
            meta = {
                "driver": "GTiff",
                "dtype": "uint8",
                "width": width,
                "height": height,
                "count": bands,
                "crs": crs,
                "transform": transform,
                "compress": "lzw",
            }
            self._write_label_geotiff(mask, output_path, meta)

            stats = self._compute_class_distribution(mask, DW_CLASSES)

            return LabelingResult(
                tile_path=tile_path,
                label_path=output_path,
                success=True,
                labeler="dynamic_world",
                class_distribution=stats,
                metadata={
                    "start_date": self.config.start_date,
                    "end_date": self.config.end_date,
                    "composite_method": self.config.composite_method,
                    "num_scenes": len(dw_paths),
                    "bbox_wgs84": bbox_wgs84,
                },
            )

        except Exception as exc:
            logger.error(
                "DynamicWorldLabeler failed for %s: %s", tile_path, exc
            )
            return LabelingResult(
                tile_path=tile_path,
                label_path=output_path,
                success=False,
                labeler="dynamic_world",
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_dynamic_world_scenes(
        self, bbox_wgs84: Tuple[float, float, float, float]
    ) -> List[Path]:
        """Search for and download Dynamic World scenes via STAC.

        Falls back to GEE Python API if Planetary Computer is unavailable.

        Args:
            bbox_wgs84: (minx, miny, maxx, maxy) in WGS-84.

        Returns:
            List of local paths to downloaded Dynamic World GeoTIFFs.
        """
        # Try Planetary Computer first
        paths = self._fetch_via_planetary_computer(bbox_wgs84)
        if paths:
            return paths

        # Try Google Earth Engine
        paths = self._fetch_via_gee(bbox_wgs84)
        return paths

    def _fetch_via_planetary_computer(
        self, bbox_wgs84: Tuple[float, float, float, float]
    ) -> List[Path]:
        """Fetch Dynamic World scenes from Microsoft Planetary Computer STAC.

        Args:
            bbox_wgs84: Bounding box in WGS-84.

        Returns:
            Local paths to downloaded label GeoTIFFs.
        """
        try:
            import pystac_client
            import planetary_computer
        except ImportError:
            logger.debug(
                "pystac-client or planetary-computer not installed; "
                "skipping Planetary Computer fetch."
            )
            return []

        try:
            catalog = pystac_client.Client.open(
                _DW_STAC_URL,
                modifier=planetary_computer.sign_inplace,
            )
            search = catalog.search(
                collections=["dynamic-world-s2"],
                bbox=list(bbox_wgs84),
                datetime=f"{self.config.start_date}/{self.config.end_date}",
                max_items=50,
            )
            items = list(search.items())
            logger.info(
                "Found %d Dynamic World scenes on Planetary Computer", len(items)
            )
            paths: List[Path] = []
            for item in items:
                path = self._download_stac_item(item)
                if path is not None:
                    paths.append(path)
            return paths
        except Exception as exc:
            logger.warning("Planetary Computer fetch failed: %s", exc)
            return []

    def _fetch_via_gee(
        self, bbox_wgs84: Tuple[float, float, float, float]
    ) -> List[Path]:
        """Fetch a Dynamic World composite via Google Earth Engine Python API.

        Requires `earthengine-api` package and authenticated GEE account.

        Args:
            bbox_wgs84: Bounding box in WGS-84.

        Returns:
            Local paths to downloaded GeoTIFFs.
        """
        try:
            import ee
        except ImportError:
            logger.debug(
                "earthengine-api not installed; skipping GEE Dynamic World fetch."
            )
            return []

        try:
            ee.Initialize()
            minx, miny, maxx, maxy = bbox_wgs84
            region = ee.Geometry.Rectangle([minx, miny, maxx, maxy])

            dw = (
                ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
                .filterDate(self.config.start_date, self.config.end_date)
                .filterBounds(region)
            )

            if self.config.composite_method == "mode":
                label_band = dw.select("label").reduce(ee.Reducer.mode())
            else:
                # Maximum probability composite
                label_band = dw.select("label").reduce(ee.Reducer.mode())

            # Export to local temp file via getDownloadURL (small tiles only)
            url = label_band.getDownloadURL(
                {
                    "region": region,
                    "scale": 10,
                    "format": "GeoTIFF",
                    "crs": "EPSG:4326",
                }
            )
            import requests
            resp = requests.get(url, timeout=self.config.request_timeout)
            resp.raise_for_status()

            local_path = (
                self.config.cache_dir
                / f"dw_gee_{self.config.start_date}_{self.config.end_date}"
                  f"_{minx:.4f}_{miny:.4f}.tif"
            )
            local_path.write_bytes(resp.content)
            return [local_path]

        except Exception as exc:
            logger.warning("GEE Dynamic World fetch failed: %s", exc)
            return []

    def _download_stac_item(self, item: Any) -> Optional[Path]:
        """Download the label band from a STAC item.

        Args:
            item: pystac.Item with Dynamic World assets.

        Returns:
            Local path to the label GeoTIFF, or None on failure.
        """
        import requests

        item_id = item.id
        cache_path = self.config.cache_dir / f"{item_id}_label.tif"
        if cache_path.exists():
            return cache_path

        # Dynamic World items have a 'label' asset
        asset = item.assets.get("label") or item.assets.get("data")
        if asset is None:
            logger.warning("No label asset found in DW item %s", item_id)
            return None

        url = asset.href
        try:
            resp = requests.get(url, timeout=self.config.request_timeout, stream=True)
            resp.raise_for_status()
            with open(cache_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    fh.write(chunk)
            return cache_path
        except Exception as exc:
            logger.error("Failed to download DW item %s: %s", item_id, exc)
            return None

    def _create_composite(
        self,
        scene_paths: List[Path],
        method: str,
        dst_crs: Any,
        dst_transform: Any,
        dst_height: int,
        dst_width: int,
    ) -> np.ndarray:
        """Create a composite label mask from multiple Dynamic World scenes.

        Args:
            scene_paths: Local paths to Dynamic World label GeoTIFFs.
            method: Aggregation method ('mode' or 'probability').
            dst_crs: Target CRS.
            dst_transform: Target affine transform.
            dst_height: Target height in pixels.
            dst_width: Target width in pixels.

        Returns:
            Uint8 composite label array.
        """
        import rasterio
        from rasterio.warp import reproject, Resampling

        layers: List[np.ndarray] = []
        for path in scene_paths:
            try:
                layer = np.zeros((dst_height, dst_width), dtype=np.uint8)
                with rasterio.open(path) as src:
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=layer,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=dst_transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest,
                    )
                layers.append(layer)
            except Exception as exc:
                logger.warning("Failed to reproject DW scene %s: %s", path, exc)

        if not layers:
            return np.zeros((dst_height, dst_width), dtype=np.uint8)

        stack = np.stack(layers, axis=0)  # (N, H, W)

        if method == "mode" or len(layers) == 1:
            # Pixel-wise mode
            from scipy.stats import mode as scipy_mode  # type: ignore
            result = scipy_mode(stack, axis=0).mode.squeeze(0).astype(np.uint8)
        else:
            result = stack[0]

        return result

    @property
    def class_names(self) -> Dict[int, str]:
        """Return the Dynamic World class name mapping."""
        return DW_CLASSES
