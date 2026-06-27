"""
Vectorization utilities — convert raster prediction masks to vector formats.

Converts uint8 GeoTIFF label masks to GeoJSON FeatureCollections or
Shapefiles, with optional simplification, CRS reprojection, and
per-class attribute enrichment.

Example:
    >>> from pygeovision.ai.inference.vectorization import Vectorizer
    >>> v = Vectorizer(simplify_tolerance=1.0, min_area_m2=50)
    >>> geojson = v.raster_to_geojson("mask.tif", class_names=["bg", "building"])
    >>> v.raster_to_shapefile("mask.tif", "buildings.shp", class_filter=[1])
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VectorizationConfig:
    """Configuration for raster-to-vector conversion.

    Attributes:
        simplify_tolerance: Douglas-Peucker simplification in map units (0 = disabled).
        min_area_m2: Minimum polygon area in m² to keep (0 = keep all).
        output_crs: Target CRS for output (e.g. 'EPSG:4326'). None = source CRS.
        class_filter: Only vectorize these class IDs. None = all classes.
        multipart: If True, merge touching polygons of the same class into multipolygons.
        connectivity: Pixel connectivity for connected components (4 or 8).
    """

    simplify_tolerance: float = 1.0
    min_area_m2: float = 0.0
    output_crs: Optional[str] = "EPSG:4326"
    class_filter: Optional[List[int]] = None
    multipart: bool = False
    connectivity: int = 8


class Vectorizer:
    """Convert raster prediction masks to vector polygon formats.

    Takes a classified GeoTIFF (uint8 class labels) and produces
    GeoJSON or Shapefile polygon outputs with class attributes.

    Args:
        simplify_tolerance: Geometry simplification in source CRS units.
            Larger values = fewer vertices, smaller files.
        min_area_m2: Drop polygons smaller than this area.
        output_crs: Reproject output to this CRS (default: WGS84 EPSG:4326).
        class_filter: Only export specific class IDs.
        connectivity: Connected component connectivity (4 or 8).

    Example:
        >>> v = Vectorizer(simplify_tolerance=2.0, min_area_m2=100)
        >>> # From a GeoTIFF mask file
        >>> geojson = v.raster_to_geojson(
        ...     "building_mask.tif",
        ...     class_names=["background", "building"],
        ...     class_filter=[1],
        ... )
        >>> print(f"Found {len(geojson['features'])} buildings")
        >>>
        >>> # From a numpy array + transform
        >>> import numpy as np
        >>> mask = np.random.randint(0, 3, (512, 512), dtype=np.uint8)
        >>> geojson = v.array_to_geojson(mask, transform, crs="EPSG:32636")
    """

    def __init__(
        self,
        simplify_tolerance: float = 1.0,
        min_area_m2: float = 0.0,
        output_crs: Optional[str] = "EPSG:4326",
        class_filter: Optional[List[int]] = None,
        connectivity: int = 8,
    ) -> None:
        self.config = VectorizationConfig(
            simplify_tolerance=simplify_tolerance,
            min_area_m2=min_area_m2,
            output_crs=output_crs,
            class_filter=class_filter,
            connectivity=connectivity,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def raster_to_geojson(
        self,
        mask_path: Union[str, Path],
        class_names: Optional[List[str]] = None,
        class_filter: Optional[List[int]] = None,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Vectorize a classified GeoTIFF mask to GeoJSON.

        Args:
            mask_path: Path to the uint8 label mask GeoTIFF.
            class_names: List mapping class ID → name (index 0 = class 0).
            class_filter: Only export these class IDs (overrides init setting).
            output_path: If provided, write GeoJSON to this file.

        Returns:
            GeoJSON FeatureCollection dict.

        Raises:
            ImportError: If rasterio or shapely is not installed.
        """
        try:
            import rasterio
        except ImportError as exc:
            raise ImportError("Vectorizer requires rasterio. pip install rasterio") from exc

        mask_path = Path(mask_path)
        with rasterio.open(mask_path) as src:
            mask = src.read(1)
            transform = src.transform
            crs = src.crs

        return self.array_to_geojson(
            mask=mask,
            transform=transform,
            crs=str(crs),
            class_names=class_names,
            class_filter=class_filter or self.config.class_filter,
            output_path=output_path,
        )

    def array_to_geojson(
        self,
        mask: np.ndarray,
        transform: Any,
        crs: str,
        class_names: Optional[List[str]] = None,
        class_filter: Optional[List[int]] = None,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Vectorize a numpy label mask array to GeoJSON.

        Args:
            mask: uint8 array (H, W) of class labels.
            transform: Affine transform from source raster (rasterio/affine).
            crs: Source CRS string (e.g. 'EPSG:32636').
            class_names: Class ID → name mapping.
            class_filter: Only export these class IDs.
            output_path: Write output to this file if provided.

        Returns:
            GeoJSON FeatureCollection dict.
        """
        try:
            from rasterio.features import shapes
            from shapely.geometry import shape, mapping
        except ImportError as exc:
            raise ImportError("array_to_geojson requires rasterio + shapely. "
                              "pip install rasterio shapely") from exc

        _filter = set(class_filter or self.config.class_filter or [])
        _names = class_names or {}
        features = []

        classes_to_process = [c for c in np.unique(mask) if c != 0]
        if _filter:
            classes_to_process = [c for c in classes_to_process if c in _filter]

        for cls_id in classes_to_process:
            binary = (mask == cls_id).astype(np.uint8)
            class_name = (
                _names[cls_id] if isinstance(_names, dict) and cls_id in _names
                else _names[cls_id] if isinstance(_names, list) and cls_id < len(_names)
                else f"class_{cls_id}"
            )

            for geom_dict, val in shapes(binary, mask=binary, transform=transform):
                if val == 0:
                    continue
                poly = shape(geom_dict)

                # Min area filter
                if self.config.min_area_m2 > 0:
                    if self._approx_area_m2(poly, crs) < self.config.min_area_m2:
                        continue

                # Simplify geometry
                if self.config.simplify_tolerance > 0:
                    poly = poly.simplify(
                        self.config.simplify_tolerance,
                        preserve_topology=True,
                    )

                if poly.is_empty:
                    continue

                # Reproject to output CRS
                out_geom = poly
                if self.config.output_crs and self.config.output_crs != crs:
                    out_geom = self._reproject_geom(poly, crs, self.config.output_crs)
                    if out_geom is None:
                        continue

                features.append({
                    "type": "Feature",
                    "geometry": mapping(out_geom),
                    "properties": {
                        "class_id": int(cls_id),
                        "class_name": class_name,
                        "area_m2": round(self._approx_area_m2(poly, crs), 2),
                    },
                })

        output_crs = self.config.output_crs or crs
        geojson: Dict[str, Any] = {
            "type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": output_crs}},
            "features": features,
        }

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(geojson, indent=2))
            logger.info("Wrote %d features to %s", len(features), output_path)

        logger.info(
            "Vectorized %d polygons across %d classes.",
            len(features), len(classes_to_process),
        )
        return geojson

    def raster_to_shapefile(
        self,
        mask_path: Union[str, Path],
        output_path: Union[str, Path],
        class_names: Optional[List[str]] = None,
        class_filter: Optional[List[int]] = None,
    ) -> Path:
        """Vectorize a classified GeoTIFF mask to a Shapefile.

        Args:
            mask_path: Path to the uint8 label mask GeoTIFF.
            output_path: Destination .shp file path.
            class_names: Class ID → name mapping.
            class_filter: Only export these class IDs.

        Returns:
            Path to the created Shapefile.

        Raises:
            ImportError: If geopandas is not installed.
        """
        try:
            import geopandas as gpd
            from shapely.geometry import shape
        except ImportError as exc:
            raise ImportError("raster_to_shapefile requires geopandas. pip install geopandas") from exc

        geojson = self.raster_to_geojson(mask_path, class_names, class_filter)
        if not geojson["features"]:
            logger.warning("No features to write to Shapefile.")

        gdf = gpd.GeoDataFrame.from_features(
            geojson["features"],
            crs=self.config.output_crs or "EPSG:4326",
        )
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(str(output_path), driver="ESRI Shapefile")
        logger.info("Wrote Shapefile with %d features → %s", len(gdf), output_path)
        return output_path

    def geojson_to_mask(
        self,
        geojson: Dict[str, Any],
        reference_path: Union[str, Path],
        output_path: Union[str, Path],
        class_field: str = "class_id",
        background: int = 0,
    ) -> np.ndarray:
        """Rasterize a GeoJSON FeatureCollection back to a label mask.

        Args:
            geojson: GeoJSON FeatureCollection dict.
            reference_path: Reference GeoTIFF for spatial grid and CRS.
            output_path: Destination GeoTIFF mask.
            class_field: Feature property to use as the class value.
            background: Background pixel value.

        Returns:
            uint8 numpy mask array.
        """
        try:
            import rasterio
            from rasterio.features import rasterize
            from shapely.geometry import shape
        except ImportError as exc:
            raise ImportError("geojson_to_mask requires rasterio + shapely.") from exc

        with rasterio.open(reference_path) as src:
            profile = src.profile.copy()
            out_shape = (src.height, src.width)
            transform = src.transform

        shapes_values = []
        for feat in geojson.get("features", []):
            try:
                geom = shape(feat["geometry"])
                val = feat["properties"].get(class_field, 1)
                shapes_values.append((geom.__geo_interface__, int(val)))
            except Exception:
                continue

        if shapes_values:
            mask = rasterize(
                shapes_values,
                out_shape=out_shape,
                transform=transform,
                fill=background,
                dtype=np.uint8,
            )
        else:
            mask = np.full(out_shape, background, dtype=np.uint8)

        profile.update(dtype="uint8", count=1, compress="lzw")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(mask[np.newaxis, ...])

        logger.info("Rasterized %d features → %s", len(shapes_values), output_path)
        return mask

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _approx_area_m2(geom: Any, crs: str) -> float:
        """Approximate polygon area in m².

        Uses a simple UTM projection or the geometry's native area
        if the CRS is already projected.
        """
        try:
            from pyproj import CRS, Transformer
            from shapely.ops import transform as shp_transform

            src_crs = CRS(crs)
            if src_crs.is_geographic:
                # Project to an equal-area CRS for area computation
                ea_crs = CRS("EPSG:6933")  # WGS 84 / NSIDC EASE-Grid 2.0 Global
                transformer = Transformer.from_crs(src_crs, ea_crs, always_xy=True)
                projected = shp_transform(transformer.transform, geom)
                return float(projected.area)
            return float(geom.area)
        except Exception:
            return float(geom.area)  # fallback: native area

    @staticmethod
    def _reproject_geom(geom: Any, src_crs: str, dst_crs: str) -> Optional[Any]:
        """Reproject a shapely geometry between CRS."""
        try:
            from pyproj import Transformer
            from shapely.ops import transform as shp_transform

            transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
            return shp_transform(transformer.transform, geom)
        except Exception as exc:
            logger.warning("Reprojection failed: %s", exc)
            return None
