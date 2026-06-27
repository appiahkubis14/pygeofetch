"""
Microsoft and Google Buildings Auto-Labelers (E1, E2).
Generate building footprint labels from global open building datasets.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
logger = logging.getLogger(__name__)

_MS_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
_GOOGLE_BUILDINGS_URL = "https://storage.googleapis.com/open-buildings-data/v3/polygons_s2_level_4_gzip"


class MicrosoftBuildingsLabeler:
    """Generate building labels from Microsoft's Global ML Buildings Dataset.

    174M+ buildings globally at ~cm accuracy, available via Planetary Computer.
    Covers: Africa, Americas, Asia, Europe, Oceania.

    Example::

        labeler = MicrosoftBuildingsLabeler()
        result = labeler.label(
            bbox=(-74.05, 40.70, -73.95, 40.80),
            output_path="./labels/ms_buildings.tif",
            reference_raster="./data/sentinel2.tif",
        )
    """

    SOURCE_URL = "https://minedbuildings.z5.web.core.windows.net/global-buildings/"

    def __init__(self, min_confidence: float = 0.5) -> None:
        self.min_confidence = min_confidence

    def _download_quadkey_tiles(self, bbox: Tuple[float, ...]) -> List[Dict]:
        """Download Microsoft building footprints for the bbox via Planetary Computer."""
        try:
            import requests
            lon_min, lat_min, lon_max, lat_max = bbox
            # Query planetary computer STAC for MS buildings
            params = {
                "collections": ["ms-buildings"],
                "bbox": f"{lon_min},{lat_min},{lon_max},{lat_max}",
                "limit": 100,
            }
            resp = requests.get(f"{_MS_STAC}/search", params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json().get("features", [])
        except Exception as exc:
            logger.warning("MS Buildings STAC fetch failed: %s", exc)
        return []

    def _fetch_buildings_geojson(self, bbox: Tuple[float, ...]) -> Dict:
        """Fetch MS buildings as GeoJSON using the quadkey tiles API."""
        import math, requests
        lon_min, lat_min, lon_max, lat_max = bbox
        features = []

        # Use the CSV/GeoJSON download endpoint (public, no auth)
        # QuadKey zoom 9 tiles covering the bbox
        def deg2num(lat, lon, zoom):
            n = 2 ** zoom
            x = int((lon + 180) / 360 * n)
            y = int((1 - math.log(math.tan(math.radians(lat)) + 1/math.cos(math.radians(lat))) / math.pi) / 2 * n)
            return x, y

        zoom = 9
        x0, y1 = deg2num(lat_min, lon_min, zoom)
        x1, y0 = deg2num(lat_max, lon_max, zoom)

        for tx in range(x0, x1 + 1):
            for ty in range(y0, y1 + 1):
                try:
                    url = f"{self.SOURCE_URL}{zoom}/{tx}/{ty}.geojson.gz"
                    resp = requests.get(url, timeout=20)
                    if resp.status_code != 200:
                        continue
                    import gzip, json
                    data = json.loads(gzip.decompress(resp.content))
                    for f in data.get("features", []):
                        conf = f.get("properties", {}).get("confidence", 1.0)
                        if conf >= self.min_confidence:
                            f["properties"]["label_value"] = 1
                            f["properties"]["source"] = "Microsoft"
                            features.append(f)
                except Exception as exc:
                    logger.debug("Tile %d/%d/%d: %s", zoom, tx, ty, exc)

        if not features:
            logger.warning("No MS Buildings found for bbox=%s (confidence>=%.1f)", bbox, self.min_confidence)
        return {"type": "FeatureCollection", "features": features}

    def label(
        self,
        bbox: Tuple[float, ...],
        output_path: Union[str, Path] = "./labels/ms_buildings.tif",
        reference_raster: Optional[str] = None,
        resolution_m: float = 1.0,
        save_vector: bool = True,
    ) -> Dict[str, Any]:
        """Generate a building label raster from Microsoft ML Buildings."""
        output_path = Path(output_path)
        geojson = self._fetch_buildings_geojson(bbox)
        n = len(geojson["features"])
        logger.info("Microsoft Buildings: %d footprints for bbox=%s", n, bbox)

        if save_vector:
            import json
            v = Path(str(output_path).replace(".tif", "_ms_buildings.geojson"))
            v.parent.mkdir(parents=True, exist_ok=True)
            with open(v, "w") as f:
                json.dump(geojson, f)

        # Rasterise buildings = 1
        self._rasterise_buildings(geojson, bbox, output_path, resolution_m, reference_raster)
        return {"success": True, "n_buildings": n, "output_path": str(output_path), "source": "Microsoft"}

    def _rasterise_buildings(self, geojson, bbox, output_path, resolution_m, reference_raster):
        try:
            import numpy as np, rasterio
            from rasterio.transform import from_bounds
            from rasterio.features import rasterize
            from shapely.geometry import shape

            lon_min, lat_min, lon_max, lat_max = bbox
            if reference_raster:
                with rasterio.open(reference_raster) as ref:
                    transform, width, height, crs = ref.transform, ref.width, ref.height, ref.crs
            else:
                deg_per_m = 1/111320
                px = 1/(resolution_m * deg_per_m)
                width, height = int((lon_max-lon_min)*px), int((lat_max-lat_min)*px)
                transform = from_bounds(lon_min, lat_min, lon_max, lat_max, width, height)
                import rasterio.crs; crs = rasterio.crs.CRS.from_epsg(4326)

            shapes = [(shape(f["geometry"]), 1) for f in geojson["features"] if f.get("geometry")]
            label = rasterize(shapes, out_shape=(height, width), transform=transform,
                              fill=0, dtype=np.uint8) if shapes else np.zeros((height, width), np.uint8)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(str(output_path), "w", driver="GTiff", height=height,
                                width=width, count=1, dtype="uint8", crs=crs,
                                transform=transform, compress="lzw") as dst:
                dst.write(label[np.newaxis])
                dst.update_tags(source="Microsoft_ML_Buildings")
        except ImportError as exc:
            raise ImportError(f"rasterio + shapely required: {exc}")


class GoogleBuildingsLabeler:
    """Generate building labels from Google Open Buildings v3.

    1B+ building footprints across Africa, South/Southeast Asia, Oceania.
    Based on satellite imagery from 2016–2022.

    Example::

        labeler = GoogleBuildingsLabeler()
        result = labeler.label(
            bbox=(3.35, 6.45, 3.45, 6.55),   # Lagos, Nigeria
            output_path="./labels/google_buildings_lagos.tif",
        )
    """

    S2_CELL_URL = "https://storage.googleapis.com/open-buildings-data/v3/polygons_s2_level_4_gzip/{cell_id}_buildings.csv.gz"

    def __init__(self, min_confidence: float = 0.7) -> None:
        self.min_confidence = min_confidence

    def _bbox_to_s2_cells(self, bbox: Tuple[float, ...]) -> List[str]:
        """Approximate S2 cell IDs for the bbox at level 4."""
        try:
            import s2sphere
            lon_min, lat_min, lon_max, lat_max = bbox
            region = s2sphere.LatLngRect(
                s2sphere.LatLng.from_degrees(lat_min, lon_min),
                s2sphere.LatLng.from_degrees(lat_max, lon_max),
            )
            coverer = s2sphere.RegionCoverer()
            coverer.min_level = coverer.max_level = 4
            covering = coverer.get_covering(region)
            return [str(c.id()) for c in covering]
        except ImportError:
            logger.warning("s2sphere not installed; pip install s2sphere for Google Buildings")
            return []

    def label(
        self,
        bbox: Tuple[float, ...],
        output_path: Union[str, Path] = "./labels/google_buildings.tif",
        reference_raster: Optional[str] = None,
        resolution_m: float = 1.0,
    ) -> Dict[str, Any]:
        """Generate building labels from Google Open Buildings."""
        output_path = Path(output_path)
        cells = self._bbox_to_s2_cells(bbox)
        if not cells:
            return {"success": False, "error": "pip install s2sphere for Google Buildings"}

        import requests, gzip, io
        all_features = []
        for cell in cells[:4]:  # limit to 4 cells
            url = self.S2_CELL_URL.format(cell_id=cell)
            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code != 200:
                    continue
                import pandas as pd
                df = pd.read_csv(io.BytesIO(gzip.decompress(resp.content)))
                df = df[df["confidence"] >= self.min_confidence]
                for _, row in df.iterrows():
                    import json
                    geom = json.loads(row["geometry"])
                    all_features.append({
                        "type": "Feature",
                        "geometry": geom,
                        "properties": {"confidence": row["confidence"], "label_value": 1, "source": "Google"},
                    })
            except Exception as exc:
                logger.debug("Google Buildings cell %s: %s", cell, exc)

        logger.info("Google Buildings: %d footprints", len(all_features))
        geojson = {"type": "FeatureCollection", "features": all_features}

        # Use MS-style rasterise
        ms = MicrosoftBuildingsLabeler()
        ms._rasterise_buildings(geojson, bbox, output_path, resolution_m, reference_raster)
        return {"success": True, "n_buildings": len(all_features), "output_path": str(output_path), "source": "Google"}
