"""
Microsoft Building Footprints labeler.

Accesses the Microsoft Building Footprints dataset (1.3 billion buildings globally)
and generates binary segmentation masks for satellite imagery tiles.

Dataset: https://github.com/microsoft/GlobalMLBuildingFootprints
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from pygeovision.ai.data.dataset import TileMetadata
from pygeovision.ai.labeling.base_labeler import BaseLabeler, LabelingResult
from pygeovision.core.exceptions import BuildingFootprintError

logger = logging.getLogger(__name__)

# Microsoft Building Footprints are distributed as Azure Blob Storage files
# partitioned by quadkey. Coverage index:
MS_BUILDINGS_INDEX_URL = (
    "https://minedbuildings.blob.core.windows.net/global-buildings/dataset-links.csv"
)


class MicrosoftBuildingsLabeler(BaseLabeler):
    """
    Generate building segmentation labels using Microsoft Building Footprints.

    The Microsoft Global ML Building Footprints dataset covers 1.3 billion
    building footprints across the globe, derived from Bing Maps imagery
    using semantic segmentation AI.

    Parameters
    ----------
    cache_dir : Path, optional
        Directory for caching downloaded footprint data.
    confidence_threshold : float
        Minimum confidence for keeping a label.
    quadkey_zoom : int
        QuadKey zoom level for spatial indexing. Defaults to 9.

    Examples
    --------
    >>> labeler = MicrosoftBuildingsLabeler(cache_dir=Path("./ms_buildings_cache/"))
    >>> results = labeler.label_tiles(tiles=tiles, output_dir=Path("./labels/"))
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        confidence_threshold: float = 0.5,
        quadkey_zoom: int = 9,
    ) -> None:
        super().__init__(confidence_threshold=confidence_threshold)
        self.cache_dir = Path(cache_dir) if cache_dir else Path(".pygeovision_cache/ms_buildings")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.quadkey_zoom = quadkey_zoom
        self._index_cache: Optional[dict[str, str]] = None

    @property
    def name(self) -> str:
        return "microsoft_buildings"

    @property
    def supported_tasks(self) -> list[str]:
        return ["segmentation", "detection"]

    def label_tile(
        self,
        tile: TileMetadata,
        output_dir: Path,
        task: str = "segmentation",
        **kwargs: Any,
    ) -> LabelingResult:
        """
        Generate a building mask for a tile using Microsoft Building Footprints.

        Parameters
        ----------
        tile : TileMetadata
        output_dir : Path
        task : str
        **kwargs

        Returns
        -------
        LabelingResult
        """
        bounds_wgs84 = self._ensure_wgs84_bounds(tile)
        if bounds_wgs84 is None:
            return self._skip_tile(tile, "Cannot reproject bounds to WGS84")

        # Get quadkeys covering this tile
        quadkeys = self._bounds_to_quadkeys(bounds_wgs84)

        # Download and load footprint data for each quadkey
        geometries: list[Any] = []
        for qk in quadkeys:
            try:
                geoms = self._load_footprints_for_quadkey(qk, bounds_wgs84)
                geometries.extend(geoms)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Failed to load footprints for quadkey %s: %s", qk, exc)

        if not geometries:
            return self._skip_tile(tile, "No building footprints found in tile bounds")

        # Rasterize footprints to mask
        try:
            mask = self._rasterize_buildings(
                geometries=geometries,
                bounds=bounds_wgs84,
                height=tile.height,
                width=tile.width,
            )
        except Exception as exc:
            return LabelingResult(
                tile_id=tile.tile_id,
                source=self.name,
                error=f"Rasterization failed: {exc}",
            )

        if float(np.mean(mask > 0)) < 0.001:
            return self._skip_tile(tile, "No building pixels after rasterization")

        label_path = self._write_label_geotiff(mask, tile, output_dir)
        distribution = self._compute_class_distribution(
            mask, ["background", "building"]
        )

        return LabelingResult(
            tile_id=tile.tile_id,
            label_path=label_path,
            confidence=0.9,  # ML-derived buildings have high but imperfect accuracy
            source=self.name,
            class_distribution=distribution,
        )

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _ensure_wgs84_bounds(
        self,
        tile: TileMetadata,
    ) -> Optional[tuple[float, float, float, float]]:
        """Reproject tile bounds to WGS84 if necessary."""
        bounds = tile.bounds
        if tile.crs == "EPSG:4326":
            return bounds
        try:
            from pyproj import Transformer  # noqa: PLC0415

            t = Transformer.from_crs(tile.crs, "EPSG:4326", always_xy=True)
            min_x, min_y = t.transform(bounds[0], bounds[1])
            max_x, max_y = t.transform(bounds[2], bounds[3])
            return (min_x, min_y, max_x, max_y)
        except Exception as exc:  # noqa: BLE001
            logger.warning("CRS reprojection failed: %s", exc)
            return None

    def _bounds_to_quadkeys(
        self,
        bounds: tuple[float, float, float, float],
    ) -> list[str]:
        """Convert WGS84 bounds to covering QuadKeys at zoom level."""
        min_lon, min_lat, max_lon, max_lat = bounds
        try:
            import mercantile  # noqa: PLC0415

            tiles = list(
                mercantile.tiles(min_lon, min_lat, max_lon, max_lat, zooms=self.quadkey_zoom)
            )
            return [mercantile.quadkey(t) for t in tiles]
        except ImportError:
            # Fallback: return a single approximate quadkey
            logger.debug("mercantile not installed — using fallback quadkey computation")
            return [self._latlon_to_quadkey(
                (min_lat + max_lat) / 2,
                (min_lon + max_lon) / 2,
                self.quadkey_zoom,
            )]

    @staticmethod
    def _latlon_to_quadkey(lat: float, lon: float, zoom: int) -> str:
        """Simple lat/lon to QuadKey conversion."""
        x = int((lon + 180) / 360 * (2**zoom))
        lat_rad = lat * 3.14159 / 180
        import math  # noqa: PLC0415

        y = int((1 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2 * (2**zoom))
        qk = ""
        for i in range(zoom, 0, -1):
            digit = 0
            mask = 1 << (i - 1)
            if x & mask:
                digit += 1
            if y & mask:
                digit += 2
            qk += str(digit)
        return qk

    def _load_footprints_for_quadkey(
        self,
        quadkey: str,
        bounds: tuple[float, float, float, float],
    ) -> list[Any]:
        """Load building footprints for a quadkey from cache or download."""
        cache_file = self.cache_dir / f"{quadkey}.geojsonl.gz"

        if not cache_file.exists():
            self._download_quadkey(quadkey, cache_file)

        if not cache_file.exists():
            return []

        return self._parse_footprints(cache_file, bounds)

    def _download_quadkey(self, quadkey: str, output_path: Path) -> None:
        """Download footprint data for a quadkey from Microsoft Azure."""
        import urllib.request  # noqa: PLC0415

        # Construct URL (Microsoft publishes footprints per quadkey)
        url = (
            f"https://minedbuildings.blob.core.windows.net/global-buildings/"
            f"{quadkey}.geojsonl.gz"
        )
        try:
            urllib.request.urlretrieve(url, str(output_path))
            logger.debug("Downloaded MS buildings for quadkey %s", quadkey)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Download failed for quadkey %s: %s", quadkey, exc)

    @staticmethod
    def _parse_footprints(
        filepath: Path,
        bounds: tuple[float, float, float, float],
    ) -> list[Any]:
        """Parse GeoJSONL file and filter by bounds."""
        import gzip  # noqa: PLC0415
        import json  # noqa: PLC0415

        try:
            import shapely.geometry as sg  # noqa: PLC0415

            bbox_geom = sg.box(*bounds)
        except ImportError:
            return []

        geometries = []
        try:
            with gzip.open(filepath, "rt", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        feature = json.loads(line)
                        geom = sg.shape(feature["geometry"])
                        if geom.intersects(bbox_geom):
                            geometries.append(sg.mapping(geom.intersection(bbox_geom)))
                    except Exception:  # noqa: BLE001
                        continue
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to parse footprints from %s: %s", filepath, exc)

        return geometries

    @staticmethod
    def _rasterize_buildings(
        geometries: list[Any],
        bounds: tuple[float, float, float, float],
        height: int,
        width: int,
    ) -> np.ndarray:
        """Rasterize building footprints to binary mask."""
        try:
            from rasterio.features import rasterize  # noqa: PLC0415
            from rasterio.transform import from_bounds  # noqa: PLC0415
        except ImportError as exc:
            raise BuildingFootprintError(
                "rasterio is required. Install with: pip install rasterio"
            ) from exc

        min_lon, min_lat, max_lon, max_lat = bounds
        transform = from_bounds(min_lon, min_lat, max_lon, max_lat, width, height)
        shapes = [(geom, 1) for geom in geometries]

        return rasterize(
            shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8,
        )
