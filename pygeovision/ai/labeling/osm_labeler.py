"""
OpenStreetMap labeler.

Queries OSM via Overpass API to extract vector features (buildings, roads,
land use, water bodies, etc.) and rasterizes them to label masks aligned
with imagery tiles.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from pygeovision.ai.data.dataset import TileMetadata
from pygeovision.ai.labeling.base_labeler import BaseLabeler, LabelingResult
from pygeovision.core.exceptions import OSMLabelingError

logger = logging.getLogger(__name__)

# Default Overpass API endpoint
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_TIMEOUT = 60  # seconds

# OSM feature type → class ID mapping for common tasks
OSM_BUILDING_CLASSES = {
    "background": 0,
    "building": 1,
}

OSM_LANDUSE_CLASSES = {
    "background": 0,
    "residential": 1,
    "commercial": 2,
    "industrial": 3,
    "farmland": 4,
    "forest": 5,
    "water": 6,
    "meadow": 7,
    "park": 8,
}

OSM_ROAD_CLASSES = {
    "background": 0,
    "motorway": 1,
    "primary": 2,
    "secondary": 3,
    "tertiary": 4,
    "residential": 5,
    "footway": 6,
}


class OSMLabeler(BaseLabeler):
    """
    Generate segmentation labels from OpenStreetMap data.

    Queries the Overpass API for the tile's bounding box and rasterizes
    the returned features to a label mask aligned with the tile.

    Parameters
    ----------
    feature_type : str
        OSM feature category to label:
        ``"buildings"``, ``"roads"``, ``"landuse"``, ``"water"``, ``"all"``.
    overpass_url : str
        Overpass API endpoint. Defaults to the public endpoint.
    request_delay : float
        Delay between API requests (seconds) to avoid rate limiting.
    confidence_threshold : float
        Minimum confidence for keeping a label. Defaults to 0.5.

    Examples
    --------
    >>> labeler = OSMLabeler(feature_type="buildings")
    >>> results = labeler.label_tiles(tiles=tile_list, output_dir=Path("./labels/"))
    """

    def __init__(
        self,
        feature_type: str = "buildings",
        overpass_url: str = OVERPASS_URL,
        request_delay: float = 1.0,
        confidence_threshold: float = 0.5,
    ) -> None:
        super().__init__(confidence_threshold=confidence_threshold)
        valid_types = {"buildings", "roads", "landuse", "water", "all"}
        if feature_type not in valid_types:
            raise OSMLabelingError(
                f"Invalid feature_type '{feature_type}'. Choose from {valid_types}"
            )
        self.feature_type = feature_type
        self.overpass_url = overpass_url
        self.request_delay = request_delay
        self._last_request_time: float = 0.0

    @property
    def name(self) -> str:
        return "openstreetmap"

    @property
    def supported_tasks(self) -> list[str]:
        return ["segmentation", "detection", "classification"]

    def label_tile(
        self,
        tile: TileMetadata,
        output_dir: Path,
        task: str = "segmentation",
        **kwargs: Any,
    ) -> LabelingResult:
        """
        Generate an OSM label mask for a single tile.

        Parameters
        ----------
        tile : TileMetadata
            Tile to label.
        output_dir : Path
            Where to write the label GeoTIFF.
        task : str
            Task type (affects which classes are included).
        **kwargs
            Additional parameters (``feature_type`` override, etc.).

        Returns
        -------
        LabelingResult
        """
        feature_type = kwargs.get("feature_type", self.feature_type)

        # Build bounding box (min_lon, min_lat, max_lon, max_lat)
        # Handle both legacy (bounds obj) and new TileMetadata (tuple)
        raw_bounds = tile.bounds
        if hasattr(raw_bounds, 'left'):
            bounds = (raw_bounds.left, raw_bounds.bottom, raw_bounds.right, raw_bounds.top)
        else:
            bounds = tuple(raw_bounds)
        tile_crs = tile.crs if isinstance(tile.crs, str) else str(tile.crs)
        if "4326" not in tile_crs:
            bounds = self._reproject_bounds(bounds, tile_crs, "EPSG:4326")

        # Rate limit
        self._throttle()

        # Query Overpass API
        try:
            features = self._query_overpass(bounds, feature_type)
        except Exception as exc:
            return LabelingResult(
                tile_id=tile.tile_id,
                source=self.name,
                error=f"Overpass query failed: {exc}",
            )

        if not features:
            return self._skip_tile(tile, "No OSM features found in tile bounds")

        # Rasterize features to a label mask
        try:
            mask = self._rasterize_features(
                features=features,
                bounds=bounds,
                height=tile.height,
                width=tile.width,
                feature_type=feature_type,
            )
        except Exception as exc:
            return LabelingResult(
                tile_id=tile.tile_id,
                source=self.name,
                error=f"Rasterization failed: {exc}",
            )

        # Filter by coverage
        positive_fraction = float(np.mean(mask > 0))
        if positive_fraction < 0.001:
            return self._skip_tile(tile, f"Positive pixel fraction too low: {positive_fraction:.4f}")

        # Write label GeoTIFF
        label_path = self._write_label_geotiff(mask, tile, output_dir)

        class_names = self._get_class_names(feature_type)
        distribution = self._compute_class_distribution(mask, class_names)

        return LabelingResult(
            tile_id=tile.tile_id,
            label_path=label_path,
            confidence=1.0,  # OSM data is authoritative
            source=self.name,
            class_distribution=distribution,
        )

    # ------------------------------------------------------------------
    # Overpass API
    # ------------------------------------------------------------------

    def _query_overpass(
        self,
        bounds: tuple[float, float, float, float],
        feature_type: str,
    ) -> list[dict[str, Any]]:
        """
        Query the Overpass API for features within bounds.

        Parameters
        ----------
        bounds : tuple
            (min_lon, min_lat, max_lon, max_lat) in WGS84.
        feature_type : str
            OSM feature type.

        Returns
        -------
        list of dict
            GeoJSON-like feature dicts.
        """
        import urllib.request  # noqa: PLC0415
        import urllib.parse  # noqa: PLC0415

        query = self._build_overpass_query(bounds, feature_type)
        data = urllib.parse.urlencode({"data": query}).encode("utf-8")

        req = urllib.request.Request(
            self.overpass_url,
            data=data,
            method="POST",
        )
        req.add_header("Content-Type", "application/x-www-form-urlencoded")
        req.add_header("User-Agent", "PyGeoVision-OSMLabeler/1.0")

        try:
            with urllib.request.urlopen(req, timeout=OVERPASS_TIMEOUT) as resp:
                response_data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            raise OSMLabelingError(f"Overpass request failed: {exc}") from exc

        return response_data.get("elements", [])

    @staticmethod
    def _build_overpass_query(
        bounds: tuple[float, float, float, float],
        feature_type: str,
    ) -> str:
        """Build an Overpass QL query string."""
        min_lon, min_lat, max_lon, max_lat = bounds
        bbox = f"{min_lat},{min_lon},{max_lat},{max_lon}"

        if feature_type == "buildings":
            filters = 'way["building"]'
            relations = 'relation["building"]'
        elif feature_type == "roads":
            filters = 'way["highway"]'
            relations = ""
        elif feature_type == "landuse":
            filters = 'way["landuse"]'
            relations = 'relation["landuse"]'
        elif feature_type == "water":
            filters = 'way["natural"="water"]'
            relations = 'relation["natural"="water"]'
        else:  # "all"
            filters = "way"
            relations = "relation"

        parts = [f"{filters}({bbox});"]
        if relations:
            parts.append(f"{relations}({bbox});")

        query = f"""
[out:json][timeout:{OVERPASS_TIMEOUT}];
(
  {''.join(parts)}
);
out body;
>;
out skel qt;
""".strip()
        return query

    def _throttle(self) -> None:
        """Enforce rate limiting between API requests."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self._last_request_time = time.monotonic()

    # ------------------------------------------------------------------
    # Rasterization
    # ------------------------------------------------------------------

    def _rasterize_features(
        self,
        features: list[dict[str, Any]],
        bounds: tuple[float, float, float, float],
        height: int,
        width: int,
        feature_type: str,
    ) -> np.ndarray:
        """
        Rasterize OSM features to a label mask.

        Parameters
        ----------
        features : list of dict
            OSM elements returned by Overpass.
        bounds : tuple
            (min_lon, min_lat, max_lon, max_lat).
        height, width : int
            Output mask dimensions.
        feature_type : str
            Determines class assignment.

        Returns
        -------
        np.ndarray
            Label mask of shape (H, W) with integer class IDs.
        """
        try:
            from rasterio.features import rasterize as rio_rasterize  # noqa: PLC0415
            from rasterio.transform import from_bounds  # noqa: PLC0415
            import shapely.geometry as sg  # noqa: PLC0415
        except ImportError as exc:
            raise OSMLabelingError(
                "rasterio and shapely are required for OSM rasterization. "
                "Install with: pip install rasterio shapely"
            ) from exc

        min_lon, min_lat, max_lon, max_lat = bounds
        transform = from_bounds(min_lon, min_lat, max_lon, max_lat, width, height)

        # Build node lookup for way reconstruction
        nodes: dict[int, tuple[float, float]] = {}
        for elem in features:
            if elem["type"] == "node":
                nodes[elem["id"]] = (elem["lon"], elem["lat"])

        shapes: list[tuple[Any, int]] = []
        for elem in features:
            if elem["type"] != "way":
                continue
            coords = [nodes[nid] for nid in elem.get("nd", []) if nid in nodes]
            if len(coords) < 3:
                continue

            try:
                geom = sg.Polygon(coords)
                if not geom.is_valid:
                    geom = geom.buffer(0)
            except Exception:  # noqa: BLE001
                continue

            class_id = self._get_class_id(elem.get("tags", {}), feature_type)
            shapes.append((sg.mapping(geom), class_id))

        if not shapes:
            return np.zeros((height, width), dtype=np.uint8)

        mask = rio_rasterize(
            shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8,
        )
        return mask

    @staticmethod
    def _get_class_id(tags: dict[str, Any], feature_type: str) -> int:
        """Map OSM tags to a class ID."""
        if feature_type == "buildings":
            return 1  # All buildings → class 1
        if feature_type == "roads":
            highway = tags.get("highway", "")
            return OSM_ROAD_CLASSES.get(highway, 1)
        if feature_type in ("landuse", "all"):
            landuse = tags.get("landuse", tags.get("natural", ""))
            return OSM_LANDUSE_CLASSES.get(landuse, 1)
        return 1

    @staticmethod
    def _get_class_names(feature_type: str) -> list[str]:
        """Return class names for a feature type."""
        if feature_type == "buildings":
            return ["background", "building"]
        if feature_type == "roads":
            return list(OSM_ROAD_CLASSES.keys())
        if feature_type in ("landuse", "all"):
            return list(OSM_LANDUSE_CLASSES.keys())
        return ["background", "foreground"]

    @staticmethod
    def _reproject_bounds(
        bounds: tuple[float, float, float, float],
        src_crs: str,
        dst_crs: str,
    ) -> tuple[float, float, float, float]:
        """Reproject bounding box from src_crs to dst_crs."""
        try:
            from pyproj import Transformer  # noqa: PLC0415

            transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
            min_x, min_y = transformer.transform(bounds[0], bounds[1])
            max_x, max_y = transformer.transform(bounds[2], bounds[3])
            return (min_x, min_y, max_x, max_y)
        except ImportError:
            logger.warning("pyproj not installed — using bounds as-is (may be wrong CRS)")
            return bounds
