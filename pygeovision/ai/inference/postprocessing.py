"""
Inference postprocessing for PyGeoVision predictions.

Converts raw model output masks into clean, production-ready results:
- Morphological cleanup (fill holes, remove noise)
- Polygon vectorization (raster mask → GeoJSON / Shapefile)
- Confidence thresholding and CRF refinement
- Connected component analysis

Example:
    >>> from pygeovision.ai.inference.postprocessing import PostProcessor
    >>> pp = PostProcessor(min_area_pixels=50)
    >>> clean_mask = pp.cleanup(raw_mask)
    >>> geojson = pp.vectorize(clean_mask, transform, crs)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PostProcessorConfig:
    """Configuration for mask postprocessing.

    Attributes:
        min_area_pixels: Minimum connected component area (pixels) to keep.
        fill_holes: Fill enclosed holes in binary masks.
        morphological_closing: Apply closing to smooth boundaries.
        kernel_size: Structuring element size for morphological ops.
        simplify_tolerance: Polygon simplification tolerance (pixels).
    """
    min_area_pixels: int = 50
    fill_holes: bool = True
    morphological_closing: bool = True
    kernel_size: int = 3
    simplify_tolerance: float = 1.0


class PostProcessor:
    """Postprocessing pipeline for model prediction masks.

    Applies morphological cleanup, small-region removal, and optional
    polygon vectorization to raw binary or multi-class predictions.

    Args:
        min_area_pixels: Minimum object area in pixels.
        fill_holes: Fill holes inside foreground regions.
        morphological_closing: Apply morphological closing.
        kernel_size: Kernel size for morphological operations.
        simplify_tolerance: Douglas-Peucker simplification tolerance.

    Example:
        >>> pp = PostProcessor(min_area_pixels=100, fill_holes=True)
        >>> clean = pp.cleanup(raw_binary_mask)
        >>> geojson = pp.to_geojson(clean, transform, "EPSG:4326")
    """

    def __init__(
        self,
        min_area_pixels: int = 50,
        fill_holes: bool = True,
        morphological_closing: bool = True,
        kernel_size: int = 3,
        simplify_tolerance: float = 1.0,
    ) -> None:
        self.config = PostProcessorConfig(
            min_area_pixels=min_area_pixels,
            fill_holes=fill_holes,
            morphological_closing=morphological_closing,
            kernel_size=kernel_size,
            simplify_tolerance=simplify_tolerance,
        )

    def cleanup(self, mask: np.ndarray) -> np.ndarray:
        """Apply morphological cleanup to a binary or label mask.

        Args:
            mask: uint8 array (H, W). Binary (0/1) or multi-class label.

        Returns:
            Cleaned uint8 mask of same shape.
        """
        try:
            from scipy import ndimage
        except ImportError:
            logger.warning("scipy not installed; skipping morphological cleanup.")
            return mask

        result = mask.copy()

        if mask.max() <= 1:
            # Binary mask
            result = self._cleanup_binary(result, ndimage)
        else:
            # Multi-class: clean each class independently
            for cls_id in np.unique(mask):
                if cls_id == 0:
                    continue
                binary = (mask == cls_id).astype(np.uint8)
                cleaned = self._cleanup_binary(binary, ndimage)
                result[mask == cls_id] = 0
                result[cleaned == 1] = cls_id

        return result.astype(np.uint8)

    def _cleanup_binary(self, binary: np.ndarray, ndimage: Any) -> np.ndarray:
        """Apply cleanup to a single binary channel."""
        from scipy.ndimage import binary_fill_holes, binary_closing, label

        if self.config.morphological_closing:
            struct = np.ones((self.config.kernel_size, self.config.kernel_size))
            binary = binary_closing(binary, structure=struct).astype(np.uint8)

        if self.config.fill_holes:
            binary = binary_fill_holes(binary).astype(np.uint8)

        # Remove small connected components
        if self.config.min_area_pixels > 0:
            labeled, n = label(binary)
            for comp in range(1, n + 1):
                if (labeled == comp).sum() < self.config.min_area_pixels:
                    binary[labeled == comp] = 0

        return binary

    def to_geojson(
        self,
        mask: np.ndarray,
        transform: Any,
        crs: str,
        class_names: Optional[List[str]] = None,
        output_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Vectorize a label mask to GeoJSON FeatureCollection.

        Args:
            mask: uint8 label mask (H, W).
            transform: Affine transform from the source raster.
            crs: CRS string (e.g. 'EPSG:4326').
            class_names: List mapping class IDs to names.
            output_path: If provided, write GeoJSON to this file.

        Returns:
            GeoJSON FeatureCollection dict.

        Raises:
            ImportError: If rasterio or shapely is not installed.
        """
        try:
            from rasterio.features import shapes
            from shapely.geometry import shape, mapping
            from shapely.ops import unary_union
        except ImportError as exc:
            raise ImportError(
                "Vectorization requires rasterio and shapely. "
                "Install: pip install rasterio shapely"
            ) from exc

        features = []
        for cls_id in np.unique(mask):
            if cls_id == 0:
                continue
            binary = (mask == cls_id).astype(np.uint8)
            class_name = (
                class_names[cls_id] if class_names and cls_id < len(class_names) else str(cls_id)
            )
            for geom, val in shapes(binary, mask=binary, transform=transform):
                if val == 0:
                    continue
                poly = shape(geom)
                if poly.is_empty or poly.area == 0:
                    continue
                if self.config.simplify_tolerance > 0:
                    poly = poly.simplify(
                        self.config.simplify_tolerance * abs(transform.a),
                        preserve_topology=True,
                    )
                features.append({
                    "type": "Feature",
                    "geometry": mapping(poly),
                    "properties": {
                        "class_id": int(cls_id),
                        "class_name": class_name,
                        "area_m2": round(poly.area, 2),
                    },
                })

        geojson = {
            "type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": crs}},
            "features": features,
        }

        if output_path is not None:
            import json
            Path(output_path).write_text(json.dumps(geojson, indent=2))
            logger.info("Saved %d features to %s", len(features), output_path)

        logger.info("Vectorized %d polygon features.", len(features))
        return geojson

    def apply_confidence_threshold(
        self,
        prob_map: np.ndarray,
        threshold: float = 0.5,
        background_id: int = 0,
    ) -> np.ndarray:
        """Convert a probability map to a label mask with confidence thresholding.

        Pixels where the max class probability is below ``threshold`` are
        assigned to the background class.

        Args:
            prob_map: Float array (C, H, W) of class probabilities.
            threshold: Minimum confidence to assign a class.
            background_id: Class ID to assign uncertain pixels.

        Returns:
            uint8 label mask (H, W).
        """
        max_prob = prob_map.max(axis=0)
        pred = prob_map.argmax(axis=0).astype(np.uint8)
        pred[max_prob < threshold] = background_id
        return pred
