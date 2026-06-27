"""
OSM Auto-Labeler (E1, E2) — Generate training labels from OpenStreetMap.

Uses the Overpass API to fetch building footprints, roads, water bodies,
and land use polygons, then rasterises them to GeoTIFF label masks.
No GeoAI dependency.
"""
from __future__ import annotations
import logging, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

OSM_CATEGORIES = {
    "buildings": {
        "query": 'way["building"]{bbox};relation["building"]{bbox};',
        "value": 1,
        "description": "All building footprints",
    },
    "roads": {
        "query": 'way["highway"~"primary|secondary|tertiary|residential|service"]{bbox};',
        "value": 2,
        "description": "Primary to service-level roads",
    },
    "water": {
        "query": 'way["natural"="water"]{bbox};relation["natural"="water"]{bbox};way["waterway"]{bbox};',
        "value": 3,
        "description": "Water bodies and waterways",
    },
    "vegetation": {
        "query": 'way["landuse"~"forest|meadow|grass|orchard|vineyard"]{bbox};way["natural"~"wood|scrub|heath"]{bbox};',
        "value": 4,
        "description": "Vegetation and land cover",
    },
    "agricultural": {
        "query": 'way["landuse"~"farmland|farmyard|greenhouse_horticulture"]{bbox};',
        "value": 5,
        "description": "Agricultural areas",
    },
    "commercial": {
        "query": 'way["landuse"~"commercial|retail|industrial"]{bbox};',
        "value": 6,
        "description": "Commercial and industrial zones",
    },
    "residential": {
        "query": 'way["landuse"="residential"]{bbox};',
        "value": 7,
        "description": "Residential zones",
    },
    "solar_panels": {
        "query": 'way["generator:source"="solar"]{bbox};node["generator:source"="solar"]{bbox};',
        "value": 8,
        "description": "Solar energy installations",
    },
    "parking": {
        "query": 'way["amenity"="parking"]{bbox};',
        "value": 9,
        "description": "Parking lots and structures",
    },
    "sports": {
        "query": 'way["leisure"~"pitch|track|sports_centre"]{bbox};',
        "value": 10,
        "description": "Sports facilities",
    },
}

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.fr/api/interpreter",
]


class OSMLabeler:
    """Generate geospatial training labels from OpenStreetMap via Overpass API.

    Downloads vector features for specified categories within a bounding box,
    then rasterises them to a multi-class GeoTIFF label mask.

    Supports:
        - Buildings, roads, water, vegetation, agriculture, commercial
        - Solar panels, parking, sports facilities
        - Multi-class or binary label output
        - Configurable resolution (default: 0.5m = ~Sentinel-2 super-res)

    Example::

        labeler = OSMLabeler()
        result = labeler.label(
            bbox=(-74.05, 40.70, -73.95, 40.80),
            categories=["buildings", "roads", "water"],
            output_path="./labels/nyc_osm.tif",
            reference_raster="./data/nyc_sentinel2.tif",  # optional: match resolution/CRS
        )
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        timeout: int = 120,
        retry_attempts: int = 3,
        crs: str = "EPSG:4326",
    ) -> None:
        self.endpoint = endpoint or OVERPASS_ENDPOINTS[0]
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.crs = crs

    # ── Overpass query ────────────────────────────────────────────────────────
    def _build_overpass_query(self, bbox: Tuple[float, ...], categories: List[str]) -> str:
        lon_min, lat_min, lon_max, lat_max = bbox
        # Overpass uses lat_min,lon_min,lat_max,lon_max
        bbox_str = f"({lat_min},{lon_min},{lat_max},{lon_max})"
        parts = []
        for cat in categories:
            if cat not in OSM_CATEGORIES:
                logger.warning("Unknown OSM category '%s'; skipping.", cat)
                continue
            raw = OSM_CATEGORIES[cat]["query"].replace("{bbox}", bbox_str)
            parts.append(raw)
        query = "[out:json][timeout:{}];\n(\n{}\n);\nout body geom;".format(
            self.timeout, "\n".join(parts)
        )
        return query

    def _fetch_overpass(self, query: str) -> Optional[Dict]:
        for attempt in range(1, self.retry_attempts + 1):
            for endpoint in OVERPASS_ENDPOINTS:
                try:
                    import requests
                    resp = requests.post(endpoint, data={"data": query},
                                         timeout=self.timeout)
                    if resp.status_code == 200:
                        return resp.json()
                    logger.warning("Overpass %s HTTP %d (attempt %d)", endpoint, resp.status_code, attempt)
                except Exception as exc:
                    logger.warning("Overpass request failed (%s): %s", endpoint, exc)
            if attempt < self.retry_attempts:
                time.sleep(2 ** attempt)
        return None

    # ── GeoJSON conversion ────────────────────────────────────────────────────
    def _osm_to_geojson(self, data: Dict, categories: List[str]) -> Dict:
        """Convert Overpass JSON to a categorised GeoJSON FeatureCollection."""
        features = []
        cat_by_name = {c: OSM_CATEGORIES[c] for c in categories if c in OSM_CATEGORIES}

        for elem in data.get("elements", []):
            if elem.get("type") not in ("way", "relation"):
                continue
            tags = elem.get("tags", {})
            # Determine category from tags
            assigned_cat, assigned_val = "unknown", 0
            if tags.get("building"):
                assigned_cat, assigned_val = "buildings", 1
            elif tags.get("highway"):
                assigned_cat, assigned_val = "roads", 2
            elif tags.get("natural") in ("water",) or tags.get("waterway"):
                assigned_cat, assigned_val = "water", 3
            elif tags.get("landuse") in ("forest", "meadow", "grass", "orchard"):
                assigned_cat, assigned_val = "vegetation", 4
            elif tags.get("landuse") in ("farmland", "farmyard"):
                assigned_cat, assigned_val = "agricultural", 5
            elif tags.get("landuse") in ("commercial", "retail", "industrial"):
                assigned_cat, assigned_val = "commercial", 6
            elif tags.get("landuse") == "residential":
                assigned_cat, assigned_val = "residential", 7
            elif tags.get("generator:source") == "solar":
                assigned_cat, assigned_val = "solar_panels", 8
            elif tags.get("amenity") == "parking":
                assigned_cat, assigned_val = "parking", 9
            elif tags.get("leisure") in ("pitch", "track"):
                assigned_cat, assigned_val = "sports", 10

            if assigned_cat not in categories:
                continue

            # Build geometry from nodes (way)
            if elem.get("type") == "way" and "geometry" in elem:
                coords = [[n["lon"], n["lat"]] for n in elem["geometry"]]
                if len(coords) >= 4:
                    geom = {"type": "Polygon", "coordinates": [coords]}
                    features.append({
                        "type": "Feature",
                        "geometry": geom,
                        "properties": {
                            "category": assigned_cat,
                            "label_value": assigned_val,
                            "osm_id": elem.get("id"),
                            **tags,
                        },
                    })

        return {"type": "FeatureCollection", "features": features}

    # ── Rasterisation ─────────────────────────────────────────────────────────
    def _rasterise(
        self,
        geojson: Dict,
        bbox: Tuple[float, ...],
        output_path: Path,
        resolution_m: float = 10.0,
        reference_raster: Optional[str] = None,
    ) -> Path:
        """Rasterise GeoJSON features to a labelled GeoTIFF."""
        try:
            import numpy as np
            import rasterio
            from rasterio.transform import from_bounds
            from rasterio.features import rasterize
            from shapely.geometry import shape
        except ImportError as exc:
            raise ImportError(f"rasterio + shapely required: {exc}")

        lon_min, lat_min, lon_max, lat_max = bbox

        if reference_raster:
            with rasterio.open(reference_raster) as ref:
                transform = ref.transform
                width, height = ref.width, ref.height
                crs = ref.crs
        else:
            # Approximate pixel count from resolution
            deg_per_m = 1 / 111320
            px_per_deg = 1 / (resolution_m * deg_per_m)
            width  = int((lon_max - lon_min) * px_per_deg)
            height = int((lat_max - lat_min) * px_per_deg)
            transform = from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)
            import rasterio.crs
            crs = rasterio.crs.CRS.from_epsg(4326)

        label = np.zeros((height, width), dtype=np.uint8)

        # Sort features by label value (higher = more specific)
        features = sorted(geojson.get("features", []),
                          key=lambda f: f["properties"].get("label_value", 0))
        shapes_vals = [(shape(f["geometry"]), f["properties"].get("label_value", 1))
                       for f in features if f.get("geometry")]

        if shapes_vals:
            burned = rasterize(
                shapes_vals,
                out_shape=(height, width),
                transform=transform,
                fill=0,
                dtype=np.uint8,
                merge_alg=rasterio.enums.MergeAlg.replace,
            )
            label = burned

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            str(output_path), "w",
            driver="GTiff", height=height, width=width,
            count=1, dtype="uint8",
            crs=crs, transform=transform,
            compress="lzw",
        ) as dst:
            dst.write(label[np.newaxis])
            dst.update_tags(
                source="OpenStreetMap",
                provider="Overpass API",
                n_features=str(len(shapes_vals)),
            )
        return output_path

    # ── Public API ────────────────────────────────────────────────────────────
    def label(
        self,
        bbox: Tuple[float, ...],
        categories: Optional[List[str]] = None,
        output_path: Union[str, Path] = "./labels/osm_labels.tif",
        reference_raster: Optional[str] = None,
        resolution_m: float = 10.0,
        save_vector: bool = True,
        vector_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a labelled GeoTIFF from OSM features.

        Args:
            bbox: (lon_min, lat_min, lon_max, lat_max)
            categories: List of OSM categories to include (default: all)
            output_path: Output label raster path (.tif)
            reference_raster: If provided, match its CRS/transform/resolution
            resolution_m: Label raster resolution in metres (ignored if reference_raster given)
            save_vector: Also save the raw GeoJSON features
            vector_path: Path for the GeoJSON vector output

        Returns:
            Dict with n_features, categories, output_path, vector_path
        """
        categories = categories or list(OSM_CATEGORIES.keys())
        output_path = Path(output_path)

        logger.info("OSMLabeler: querying Overpass for bbox=%s cats=%s", bbox, categories)
        query = self._build_overpass_query(bbox, categories)
        data = self._fetch_overpass(query)
        if data is None:
            logger.error("Overpass query failed; check connectivity and retry.")
            return {"success": False, "error": "Overpass request failed"}

        geojson = self._osm_to_geojson(data, categories)
        n_features = len(geojson["features"])
        logger.info("OSMLabeler: %d features fetched", n_features)

        # Save vector
        v_path: Optional[Path] = None
        if save_vector:
            import json
            v_path = Path(vector_path or str(output_path).replace(".tif", "_vectors.geojson"))
            v_path.parent.mkdir(parents=True, exist_ok=True)
            with open(v_path, "w") as f:
                json.dump(geojson, f)

        # Rasterise
        self._rasterise(geojson, bbox, output_path, resolution_m, reference_raster)
        logger.info("OSMLabeler: label raster → %s", output_path)

        # Count per category
        cat_counts = {}
        for feat in geojson["features"]:
            cat = feat["properties"].get("category", "unknown")
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

        return {
            "success": True,
            "n_features": n_features,
            "categories": cat_counts,
            "output_path": str(output_path),
            "vector_path": str(v_path) if v_path else None,
        }

    def list_categories(self) -> Dict[str, str]:
        """Return all supported OSM categories and their descriptions."""
        return {k: v["description"] for k, v in OSM_CATEGORIES.items()}

    def preview_query(self, bbox: Tuple[float, ...], categories: Optional[List[str]] = None) -> str:
        """Return the Overpass QL query without executing it."""
        return self._build_overpass_query(bbox, categories or list(OSM_CATEGORIES.keys()))
