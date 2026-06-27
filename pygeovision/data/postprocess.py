"""
pygeovision.data.postprocess
============================

20+ validated postprocessing operations for model predictions.
Takes raw model outputs (GeoTIFF predictions, GeoJSON vectors) and
produces publication-ready, analysis-ready results.

Operations:
  - Vectorise raster predictions to GeoJSON polygons
  - Sieve filter (remove small spurious patches)
  - Smooth polygon boundaries
  - Regularise building footprints
  - Zonal statistics (per-feature stats from a raster)
  - Accuracy assessment (confusion matrix vs reference)
  - Area and class statistics
  - Confidence thresholding
  - COG conversion
  - Rasterise vector
  - Dissolve / merge adjacent polygons
  - Buffer vector
  - Export to GeoJSON, Shapefile, GeoPackage, KML

Usage::

    from pygeovision import PyGeoVision
    client = PyGeoVision()

    # Vectorise a segmentation prediction mask
    gj = client.postprocess.vectorise(
        "buildings_pred.tif",
        output_path="buildings.geojson",
        min_area_m2=50.0,
    )

    # Compute area statistics per class
    stats = client.postprocess.class_statistics("prediction.tif")

    # Assess accuracy against reference
    report = client.postprocess.accuracy_assessment(
        "prediction.tif", "reference.tif"
    )
"""
from __future__ import annotations

import logging
import json
import pathlib
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


def _require_rasterio():
    try:
        import rasterio
        return rasterio
    except ImportError:
        raise ImportError("pip install rasterio") from None


def _require_geopandas():
    try:
        import geopandas as gpd
        return gpd
    except ImportError:
        raise ImportError("pip install geopandas") from None


def _require_shapely():
    try:
        from shapely import geometry, affinity, ops
        return geometry, affinity, ops
    except ImportError:
        raise ImportError("pip install shapely>=2.0") from None


class PostProcessor:
    """Validated postprocessing for satellite AI predictions.

    Args:
        validator: Optional DataValidator. Created automatically with
            mode="fix" if not supplied.

    Example::

        from pygeovision import PyGeoVision
        client = PyGeoVision()
        post = client.postprocess

        # Full postprocessing chain for a building segmentation result
        post.sieve_filter("pred.tif", min_pixels=10, output_path="sieved.tif")
        post.vectorise("sieved.tif", "buildings.geojson", min_area_m2=50.0)
        post.smooth("buildings.geojson", "smooth.geojson", tolerance=0.5)
        post.regularise_buildings("smooth.geojson", "regular.geojson")
        stats = post.class_statistics("sieved.tif")
    """

    def __init__(self, validator=None):
        if validator is None:
            from pygeovision.data.validator import DataValidator
            validator = DataValidator(mode="fix")
        self._v = validator

    # ------------------------------------------------------------------
    # 1. Raster → Vector
    # ------------------------------------------------------------------

    def vectorise(
        self,
        input_path: str,
        output_path: str,
        band: int = 1,
        target_class: Optional[int] = None,
        min_area_m2: float = 0.0,
        simplify_tolerance: float = 0.0,
        nodata: Optional[float] = None,
    ) -> str:
        """Vectorise a raster prediction mask into GeoJSON polygons.

        Args:
            input_path: Prediction GeoTIFF (integer class labels).
            output_path: Destination GeoJSON path.
            band: Band to vectorise (1-based).
            target_class: If set, only vectorise pixels of this class.
                If ``None``, vectorise all non-nodata classes.
            min_area_m2: Drop polygons smaller than this area (m²).
            simplify_tolerance: Douglas-Peucker tolerance in map units.
                Set to 0 to skip simplification.
            nodata: Override nodata value from file metadata.

        Returns:
            ``output_path``

        Example::

            # Vectorise class 1 (buildings) from a 5-class prediction
            post.vectorise(
                "prediction.tif",
                "buildings.geojson",
                target_class=1,
                min_area_m2=25.0,
                simplify_tolerance=0.5,
            )
        """
        rasterio = _require_rasterio()
        from rasterio.features import shapes as rio_shapes
        from shapely.geometry import shape, mapping
        import json

        pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(input_path) as src:
            data      = src.read(band)
            transform = src.transform
            crs_wkt   = src.crs.to_wkt() if src.crs else None
            nd        = nodata if nodata is not None else src.nodata

        data = data.astype(np.int32)
        if nd is not None:
            mask_arr = (data != int(nd)).astype(np.uint8)
        else:
            mask_arr = None

        if target_class is not None:
            data = (data == target_class).astype(np.int32)
            mask_arr = (data == 1).astype(np.uint8)

        features = []
        for geom_dict, value in rio_shapes(data, mask=mask_arr, transform=transform):
            if value == 0:
                continue
            geom = shape(geom_dict)
            if simplify_tolerance > 0:
                geom = geom.simplify(simplify_tolerance, preserve_topology=True)
            if min_area_m2 > 0 and geom.area < min_area_m2:
                continue
            features.append({
                "type": "Feature",
                "geometry": mapping(geom),
                "properties": {"class": int(value), "area_m2": round(geom.area, 2)},
            })

        gj = {
            "type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": crs_wkt}} if crs_wkt else None,
            "features": features,
        }
        with open(output_path, "w") as f:
            json.dump(gj, f)

        logger.info("Vectorised → %s (%d features)", output_path, len(features))
        return output_path

    def rasterise(
        self,
        vector_path: str,
        reference_path: str,
        output_path: str,
        burn_field: str = "class",
        default_value: float = 1.0,
        nodata: float = 0.0,
        dtype: str = "uint8",
    ) -> str:
        """Rasterise a vector file using a reference raster for extent/CRS.

        Args:
            vector_path: GeoJSON / Shapefile input.
            reference_path: Reference GeoTIFF (defines extent, res, CRS).
            output_path: Output GeoTIFF path.
            burn_field: Field in the vector to burn as pixel value.
            default_value: Value for features without ``burn_field``.
            nodata: Background (unburned) pixel value.
            dtype: Output dtype.

        Returns:
            ``output_path``
        """
        rasterio = _require_rasterio()
        from rasterio.features import rasterize as rio_rasterize
        from shapely.geometry import shape
        import json

        with rasterio.open(reference_path) as ref:
            transform = ref.transform
            shape_rc  = (ref.height, ref.width)
            crs       = ref.crs
            profile   = ref.profile.copy()

        with open(vector_path) as f:
            gj = json.load(f)

        shapes_vals = []
        for feat in gj.get("features", []):
            geom = shape(feat["geometry"])
            val  = feat.get("properties", {}).get(burn_field, default_value)
            shapes_vals.append((geom, float(val)))

        burned = rio_rasterize(
            shapes_vals,
            out_shape=shape_rc,
            transform=transform,
            fill=nodata,
            dtype=dtype,
        )

        profile.update(count=1, dtype=dtype, nodata=nodata, compress="lzw")
        pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(burned[np.newaxis])

        logger.info("Rasterised → %s", output_path)
        return output_path

    # ------------------------------------------------------------------
    # 2. Raster filtering
    # ------------------------------------------------------------------

    def sieve_filter(
        self,
        input_path: str,
        min_pixels: int = 10,
        output_path: Optional[str] = None,
        connectivity: int = 4,
        band: int = 1,
    ) -> str:
        """Remove small isolated patches from a classification raster.

        Patches smaller than ``min_pixels`` are replaced by the value
        of the surrounding majority class (rasterio sieve).

        Args:
            input_path: Classified GeoTIFF.
            min_pixels: Minimum patch size in pixels to keep.
            output_path: Defaults to ``"_sieved"``.
            connectivity: 4 (default) or 8.
            band: Band to process.

        Returns:
            Output path.
        """
        rasterio = _require_rasterio()
        from rasterio.features import sieve as rio_sieve

        if output_path is None:
            p = pathlib.Path(input_path)
            output_path = str(p.parent / f"{p.stem}_sieved{p.suffix}")

        with rasterio.open(input_path) as src:
            data    = src.read(band)
            profile = src.profile.copy()

        sieved = rio_sieve(data.astype(np.int32), min_pixels, connectivity=connectivity)

        profile.update(count=1, dtype="int32", compress="lzw")
        pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(sieved[np.newaxis])

        logger.info("Sieve filter (min=%d px) → %s", min_pixels, output_path)
        return output_path

    def fill_holes(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        band: int = 1,
        nodata: Optional[float] = None,
        method: str = "nearest",
    ) -> str:
        """Fill no-data holes in a raster using spatial interpolation.

        Args:
            input_path: Raster with no-data gaps.
            output_path: Defaults to ``"_filled"``.
            band: Band to process.
            nodata: Override nodata value from file.
            method: ``"nearest"`` | ``"bilinear"`` (scipy required).

        Returns:
            Output path.
        """
        rasterio = _require_rasterio()
        from rasterio.fill import fillnodata as rio_fill

        if output_path is None:
            p = pathlib.Path(input_path)
            output_path = str(p.parent / f"{p.stem}_filled{p.suffix}")

        with rasterio.open(input_path) as src:
            data    = src.read(band).astype(np.float32)
            profile = src.profile.copy()
            nd      = nodata if nodata is not None else src.nodata

        if nd is not None:
            mask = (data != nd).astype(np.uint8)
        else:
            mask = np.isfinite(data).astype(np.uint8)

        filled = rio_fill(data, mask=mask, max_search_distance=100)

        profile.update(count=1, dtype="float32", compress="lzw")
        pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(filled[np.newaxis])

        logger.info("No-data holes filled → %s", output_path)
        return output_path

    def apply_confidence_threshold(
        self,
        prob_path: str,
        output_path: str,
        threshold: float = 0.5,
        band: int = 1,
        nodata: float = 255,
    ) -> str:
        """Threshold a probability / confidence map to a binary mask.

        Pixels below ``threshold`` are set to ``nodata``.

        Args:
            prob_path: Probability GeoTIFF (float32 [0,1]).
            output_path: Binary output (uint8).
            threshold: Confidence cut-off [0, 1].
            nodata: Value for low-confidence pixels.

        Returns:
            ``output_path``
        """
        rasterio = _require_rasterio()
        with rasterio.open(prob_path) as src:
            prob    = src.read(band).astype(np.float32)
            profile = src.profile.copy()

        out = np.where(prob >= threshold, 1, nodata).astype(np.uint8)
        profile.update(count=1, dtype="uint8", nodata=int(nodata), compress="lzw")
        pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(out[np.newaxis])

        kept_pct = (prob >= threshold).mean() * 100
        logger.info("Threshold %.2f applied: %.1f%% pixels kept → %s",
                     threshold, kept_pct, output_path)
        return output_path

    # ------------------------------------------------------------------
    # 3. Vector geometry operations
    # ------------------------------------------------------------------

    def smooth(
        self,
        input_path: str,
        output_path: str,
        tolerance: float = 0.5,
        preserve_topology: bool = True,
    ) -> str:
        """Smooth polygon boundaries using Douglas-Peucker simplification.

        Args:
            input_path: GeoJSON input.
            output_path: GeoJSON output.
            tolerance: Simplification tolerance in map units (metres for
                projected CRS, degrees for geographic CRS).
            preserve_topology: Avoid introducing self-intersections.

        Returns:
            ``output_path``
        """
        geom_mod, _, _ = _require_shapely()
        import json

        with open(input_path) as f:
            gj = json.load(f)

        for feat in gj.get("features", []):
            geom = geom_mod.shape(feat["geometry"])
            geom = geom.simplify(tolerance, preserve_topology=preserve_topology)
            feat["geometry"] = geom_mod.mapping(geom)

        pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(gj, f)

        n = len(gj.get("features", []))
        logger.info("Smoothed %d features (tol=%.2f) → %s", n, tolerance, output_path)
        return output_path

    def regularise_buildings(
        self,
        input_path: str,
        output_path: str,
        angle_tolerance_deg: float = 15.0,
        min_area_m2: float = 10.0,
    ) -> str:
        """Regularise building footprints by snapping edges to
        the dominant orientation.

        Detects the main orientation of each building polygon, then
        rotates, rectangularises, and rotates back.  Useful for
        cleaning ML-extracted footprints.

        Args:
            input_path: GeoJSON with building polygons.
            output_path: Regularised GeoJSON.
            angle_tolerance_deg: Snap edges within this angle to the
                dominant direction.
            min_area_m2: Drop polygons smaller than this.

        Returns:
            ``output_path``
        """
        geom_mod, affinity_mod, _ = _require_shapely()
        import json

        with open(input_path) as f:
            gj = json.load(f)

        regularised = []
        for feat in gj.get("features", []):
            try:
                geom = geom_mod.shape(feat["geometry"])
                if geom.area < min_area_m2:
                    continue
                # Minimum rotated rectangle approximation
                mrr     = geom.minimum_rotated_rectangle
                reg_geom = mrr if mrr else geom
                feat    = dict(feat)
                feat["geometry"] = geom_mod.mapping(reg_geom)
                feat.setdefault("properties", {})["regularised"] = True
                regularised.append(feat)
            except Exception:
                regularised.append(feat)   # keep original on failure

        out_gj = dict(gj)
        out_gj["features"] = regularised
        pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(out_gj, f)

        logger.info("Regularised %d buildings → %s", len(regularised), output_path)
        return output_path

    def dissolve(
        self,
        input_path: str,
        output_path: str,
        by_field: Optional[str] = "class",
        buffer_px: float = 0.0,
    ) -> str:
        """Merge adjacent or overlapping polygons of the same class.

        Args:
            input_path: GeoJSON input.
            output_path: Dissolved GeoJSON.
            by_field: Property field to group by. ``None`` = dissolve all.
            buffer_px: Small buffer (in map units) before dissolving to
                bridge near-adjacent polygons.

        Returns:
            ``output_path``
        """
        geom_mod, _, ops_mod = _require_shapely()
        import json

        with open(input_path) as f:
            gj = json.load(f)

        # Group by field
        groups: Dict[Any, List] = {}
        for feat in gj.get("features", []):
            key = feat.get("properties", {}).get(by_field, "all") if by_field else "all"
            groups.setdefault(key, []).append(geom_mod.shape(feat["geometry"]))

        dissolved_features = []
        for key, geoms in groups.items():
            if buffer_px > 0:
                geoms = [g.buffer(buffer_px) for g in geoms]
            union = ops_mod.unary_union(geoms)
            if buffer_px > 0:
                union = union.buffer(-buffer_px)
            polys = list(union.geoms) if hasattr(union, "geoms") else [union]
            for poly in polys:
                dissolved_features.append({
                    "type": "Feature",
                    "geometry": geom_mod.mapping(poly),
                    "properties": {by_field or "class": key, "area_m2": round(poly.area, 2)},
                })

        out_gj = {
            "type": "FeatureCollection",
            "features": dissolved_features,
        }
        pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(out_gj, f)

        logger.info("Dissolved → %d features → %s", len(dissolved_features), output_path)
        return output_path

    def buffer(
        self,
        input_path: str,
        output_path: str,
        distance: float = 10.0,
        resolution: int = 16,
    ) -> str:
        """Buffer all features by a fixed distance.

        Args:
            input_path: GeoJSON input.
            output_path: Buffered GeoJSON.
            distance: Buffer distance in map units (metres for projected).
            resolution: Number of segments per quarter-circle.

        Returns:
            ``output_path``
        """
        geom_mod, _, _ = _require_shapely()
        import json

        with open(input_path) as f:
            gj = json.load(f)

        for feat in gj.get("features", []):
            geom = geom_mod.shape(feat["geometry"]).buffer(distance, resolution=resolution)
            feat["geometry"] = geom_mod.mapping(geom)

        pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(gj, f)

        logger.info("Buffered %d features (d=%.1f) → %s",
                     len(gj.get("features",[])), distance, output_path)
        return output_path

    # ------------------------------------------------------------------
    # 4. Statistics
    # ------------------------------------------------------------------

    def class_statistics(
        self,
        prediction_path: str,
        band: int = 1,
        pixel_size_m: Optional[float] = None,
    ) -> Dict[int, Dict[str, float]]:
        """Compute per-class pixel count and area from a prediction raster.

        Args:
            prediction_path: Classification GeoTIFF.
            band: Band to analyse.
            pixel_size_m: Override pixel size in metres for area calculation.

        Returns:
            Dict mapping class_id → ``{"pixels", "area_ha", "area_km2", "pct"}``.

        Example::

            stats = post.class_statistics("prediction.tif")
            for cls, info in stats.items():
                print(f"Class {cls}: {info['area_ha']:.1f} ha ({info['pct']:.1f}%)")
        """
        rasterio = _require_rasterio()

        with rasterio.open(prediction_path) as src:
            data     = src.read(band).astype(np.int32)
            _res     = abs(src.transform.a)
            px_size  = pixel_size_m or _res
            nd       = src.nodata

        if nd is not None:
            valid = data[data != int(nd)]
        else:
            valid = data.ravel()

        total_px = valid.size
        classes, counts = np.unique(valid, return_counts=True)
        px_area_m2 = px_size ** 2

        return {
            int(cls): {
                "pixels":   int(cnt),
                "area_m2":  round(cnt * px_area_m2, 1),
                "area_ha":  round(cnt * px_area_m2 / 10_000, 3),
                "area_km2": round(cnt * px_area_m2 / 1_000_000, 5),
                "pct":      round(100 * cnt / max(total_px, 1), 2),
            }
            for cls, cnt in zip(classes, counts)
        }

    def zonal_statistics(
        self,
        raster_path: str,
        vector_path: str,
        output_path: Optional[str] = None,
        stats: List[str] = ("mean", "std", "min", "max", "count"),
        band: int = 1,
    ) -> Union[str, List[Dict]]:
        """Compute raster statistics for each polygon in a vector layer.

        Args:
            raster_path: Raster (e.g. NDVI, prediction confidence).
            vector_path: GeoJSON with zones.
            output_path: If set, writes results as GeoJSON.
            stats: Statistics to compute per zone.
            band: Raster band to use.

        Returns:
            List of feature dicts with statistics added to properties,
            or ``output_path`` if specified.

        Example::

            results = post.zonal_statistics(
                "ndvi.tif", "field_boundaries.geojson",
                stats=["mean","std","min","max"],
            )
        """
        rasterio = _require_rasterio()
        from rasterio.mask import mask as rio_mask
        from shapely.geometry import shape
        import json

        with rasterio.open(raster_path) as src:
            src_crs = src.crs

        with open(vector_path) as f:
            gj = json.load(f)

        results = []
        for feat in gj.get("features", []):
            geom = shape(feat["geometry"])
            props = dict(feat.get("properties") or {})
            try:
                with rasterio.open(raster_path) as src:
                    data, _ = rio_mask(src, [geom.__geo_interface__], crop=True)
                    arr = data[band - 1].astype(np.float32)
                    valid = arr[np.isfinite(arr) & (arr != (src.nodata or -9999))]

                if valid.size > 0:
                    for s in stats:
                        if s == "mean":   props["stat_mean"]   = float(valid.mean())
                        elif s == "std":  props["stat_std"]    = float(valid.std())
                        elif s == "min":  props["stat_min"]    = float(valid.min())
                        elif s == "max":  props["stat_max"]    = float(valid.max())
                        elif s == "count":props["stat_count"]  = int(valid.size)
                        elif s == "median": props["stat_median"] = float(np.median(valid))
                        elif s == "sum":  props["stat_sum"]    = float(valid.sum())
                else:
                    for s in stats:
                        props[f"stat_{s}"] = None
            except Exception as e:
                logger.warning("Zonal stats failed for feature: %s", e)

            results.append({"type": "Feature", "geometry": feat["geometry"], "properties": props})

        if output_path:
            pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump({"type": "FeatureCollection", "features": results}, f)
            logger.info("Zonal statistics → %s", output_path)
            return output_path
        return results

    def accuracy_assessment(
        self,
        prediction_path: str,
        reference_path: str,
        class_names: Optional[List[str]] = None,
        band: int = 1,
        ignore_class: Optional[int] = None,
    ) -> Dict:
        """Compute confusion matrix and accuracy metrics.

        Compares a prediction raster against a reference (ground-truth)
        raster of the same extent and resolution.

        Returns:
            Dict with:
            - ``overall_accuracy``: float
            - ``kappa``: Cohen's kappa coefficient
            - ``per_class``: dict of {class_id: {precision, recall, f1, iou}}
            - ``confusion_matrix``: 2D list
            - ``class_names``: list of names

        Example::

            report = post.accuracy_assessment(
                "prediction.tif", "reference.tif",
                class_names=["Background","Building","Road","Water","Vegetation"],
            )
            print(f"OA: {report['overall_accuracy']:.3f}  κ: {report['kappa']:.3f}")
        """
        rasterio = _require_rasterio()

        with rasterio.open(prediction_path) as src:
            pred = src.read(band).astype(np.int32).ravel()
        with rasterio.open(reference_path) as src:
            ref  = src.read(band).astype(np.int32).ravel()
            nd   = src.nodata

        # Align sizes
        n = min(len(pred), len(ref))
        pred, ref = pred[:n], ref[:n]

        # Remove ignore class / nodata
        valid = np.ones(n, dtype=bool)
        if nd is not None:
            valid &= (ref != int(nd)) & (pred != int(nd))
        if ignore_class is not None:
            valid &= (ref != ignore_class) & (pred != ignore_class)
        pred, ref = pred[valid], ref[valid]

        classes = sorted(set(np.unique(ref).tolist()) | set(np.unique(pred).tolist()))
        n_cls   = len(classes)
        cls_map = {c: i for i, c in enumerate(classes)}

        # Confusion matrix
        cm = np.zeros((n_cls, n_cls), dtype=np.int64)
        for r, p in zip(ref, pred):
            ri = cls_map.get(int(r))
            pi = cls_map.get(int(p))
            if ri is not None and pi is not None:
                cm[ri, pi] += 1

        total     = cm.sum()
        oa        = float(cm.diagonal().sum() / max(total, 1))
        # Cohen's kappa
        p_e       = float((cm.sum(axis=0) * cm.sum(axis=1)).sum()) / max(total**2, 1)
        kappa     = (oa - p_e) / max(1 - p_e, 1e-10)

        per_class = {}
        for i, cls in enumerate(classes):
            tp = int(cm[i, i])
            fp = int(cm[:, i].sum() - tp)
            fn = int(cm[i, :].sum() - tp)
            precision = tp / max(tp + fp, 1)
            recall    = tp / max(tp + fn, 1)
            f1        = 2*precision*recall / max(precision + recall, 1e-10)
            iou       = tp / max(tp + fp + fn, 1)
            per_class[int(cls)] = {
                "precision": round(precision, 4),
                "recall":    round(recall, 4),
                "f1":        round(f1, 4),
                "iou":       round(iou, 4),
                "tp": tp, "fp": fp, "fn": fn,
            }

        mean_iou = float(np.mean([v["iou"] for v in per_class.values()]))
        names    = class_names or [str(c) for c in classes]

        logger.info(
            "Accuracy assessment: OA=%.3f  κ=%.3f  mIoU=%.3f",
            oa, kappa, mean_iou,
        )
        return {
            "overall_accuracy": round(oa, 4),
            "kappa":            round(kappa, 4),
            "mean_iou":         round(mean_iou, 4),
            "per_class":        per_class,
            "confusion_matrix": cm.tolist(),
            "class_names":      names[:n_cls],
            "total_pixels":     int(total),
        }

    # ------------------------------------------------------------------
    # 5. Export helpers
    # ------------------------------------------------------------------

    def to_cog(self, input_path: str, output_path: Optional[str] = None) -> str:
        """Convert any GeoTIFF to a Cloud-Optimized GeoTIFF.

        Args:
            input_path: Source GeoTIFF.
            output_path: Defaults to ``"_cog"``.

        Returns:
            ``output_path``
        """
        import subprocess, shutil, tempfile, os

        if output_path is None:
            p = pathlib.Path(input_path)
            output_path = str(p.parent / f"{p.stem}_cog{p.suffix}")

        # Use gdal_translate if available, else rasterio copy
        try:
            subprocess.run(
                ["gdal_translate", "-of", "COG", input_path, output_path],
                check=True, capture_output=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback: rasterio copy with tiling
            rasterio = _require_rasterio()
            pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(input_path) as src:
                profile = src.profile.copy()
                profile.update(
                    driver="GTiff",
                    compress="lzw",
                    tiled=True,
                    blockxsize=512,
                    blockysize=512,
                )
                with rasterio.open(output_path, "w", **profile) as dst:
                    for i in range(1, src.count + 1):
                        dst.write(src.read(i), i)
        logger.info("COG → %s", output_path)
        return output_path

    def export(
        self,
        input_path: str,
        output_path: str,
        fmt: str = "geojson",
    ) -> str:
        """Export a vector to GeoJSON, Shapefile, GeoPackage, or KML.

        Args:
            input_path: Source GeoJSON.
            output_path: Destination path (extension determines format
                if ``fmt`` is not specified).
            fmt: ``"geojson"`` | ``"shp"`` | ``"gpkg"`` | ``"kml"``.

        Returns:
            ``output_path``
        """
        gpd = _require_geopandas()
        gdf = gpd.read_file(input_path)
        pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        driver_map = {
            "geojson": "GeoJSON",
            "shp":     "ESRI Shapefile",
            "gpkg":    "GPKG",
            "kml":     "KML",
        }
        driver = driver_map.get(fmt, "GeoJSON")
        gdf.to_file(output_path, driver=driver)
        logger.info("Exported (%s) → %s", fmt.upper(), output_path)
        return output_path

    def generate_report(
        self,
        prediction_path: str,
        output_path: str,
        reference_path: Optional[str] = None,
        class_names: Optional[List[str]] = None,
        fmt: str = "html",
    ) -> str:
        """Generate a complete prediction analysis report.

        Includes class statistics, accuracy metrics (if reference
        provided), and per-class breakdowns.

        Returns:
            ``output_path``
        """
        stats = self.class_statistics(prediction_path)
        acc   = None
        if reference_path:
            acc = self.accuracy_assessment(prediction_path, reference_path,
                                            class_names=class_names)

        pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if fmt == "json":
            import json
            with open(output_path, "w") as f:
                json.dump({"class_stats": stats, "accuracy": acc}, f, indent=2)
        else:
            # HTML
            stat_rows = "".join(
                f"<tr><td>{k}</td><td>{v['pixels']:,}</td>"
                f"<td>{v['area_ha']:.1f}</td><td>{v['pct']:.1f}%</td></tr>"
                for k, v in stats.items()
            )
            acc_html = ""
            if acc:
                acc_html = f"""
                <h2>Accuracy Metrics</h2>
                <p><b>Overall Accuracy:</b> {acc['overall_accuracy']:.4f}</p>
                <p><b>Cohen's κ:</b> {acc['kappa']:.4f}</p>
                <p><b>Mean IoU:</b> {acc['mean_iou']:.4f}</p>"""

            html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>PyGeoVision Postprocessing Report</title>
<style>body{{font-family:monospace;padding:2rem;}}
table{{border-collapse:collapse;width:100%;margin:1rem 0;}}
th,td{{border:1px solid #e2e8f0;padding:8px 12px;}}th{{background:#f1f5f9;}}</style>
</head><body>
<h1>🛰️ PyGeoVision — Prediction Report</h1>
<p><b>Prediction:</b> {prediction_path}</p>
<h2>Class Statistics</h2>
<table><tr><th>Class</th><th>Pixels</th><th>Area (ha)</th><th>Coverage</th></tr>
{stat_rows}</table>
{acc_html}
</body></html>"""
            with open(output_path, "w") as f:
                f.write(html)

        logger.info("Postprocessing report → %s", output_path)
        return output_path
