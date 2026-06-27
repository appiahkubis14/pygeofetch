"""
PyGeoVision Geospatial AI Pipelines.

Each pipeline uses PyGeoFetch (via the pgv_client) for data acquisition
and PyGeoVision AI for model inference. The pgv_client is a PyGeoVision
instance that wraps PyGeoFetch's search/download/post-process API.

Available pipelines (10):
    change_detection       building_footprints    land_cover
    crop_monitoring        disaster_assessment    deforestation
    urban_growth           water_bodies           solar_detection
    carbon_estimation

Example:
    >>> import pygeovision as pgv
    >>> client = pgv.PyGeoVision()
    >>> result = client.pipeline("building_footprints",
    ...     bbox=(-0.15, 51.47, -0.10, 51.52), date="2024-06")
    >>> print(result.output_path, result.stats)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Result from a geospatial pipeline run."""
    pipeline: str
    output_path: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: str = ""

    def __str__(self) -> str:
        if self.success:
            return f"PipelineResult({self.pipeline}, output={self.output_path})"
        return f"PipelineResult({self.pipeline}, FAILED: {self.error})"


# ---------------------------------------------------------------------------
# Base pipeline
# ---------------------------------------------------------------------------

class BasePipeline(ABC):
    """Abstract base for all PyGeoVision pipelines.

    ``pgv_client`` is a PyGeoVision instance — call
    ``pgv_client.search()`` and ``pgv_client.download()`` for data,
    which delegate to PyGeoFetch under the hood.
    """

    def __init__(self, pgv_client: Any) -> None:
        self.pgv = pgv_client  # PyGeoVision instance

    @abstractmethod
    def run(
        self,
        bbox: Tuple[float, float, float, float],
        output_dir: Union[str, Path],
        **kwargs: Any,
    ) -> PipelineResult: ...

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _search_and_download(
        self,
        bbox: Tuple[float, float, float, float],
        date: str,
        output_dir: Path,
        collections: Optional[List[str]] = None,
        providers: Optional[List[str]] = None,
        cloud_cover_max: float = 20.0,
        post_process: Optional[List[str]] = None,
        max_results: int = 5,
    ) -> Optional[Path]:
        """Search PyGeoFetch → download best scene → return local path."""
        try:
            # Build date range from a YYYY-MM string
            if len(date) == 7:  # YYYY-MM
                start = f"{date}-01"
                end = f"{date}-28"
            else:
                start = end = date

            results = self.pgv.search(
                bbox=bbox,
                date_range=(start, end),
                collections=collections or ["sentinel-2-l2a"],
                providers=providers,
                cloud_cover_max=cloud_cover_max,
                max_results=max_results,
                sort_by="cloud_cover",
                sort_order="asc",
            )

            if not results:
                return None

            # Download best (lowest cloud cover) scene
            pp = post_process or ["unzip", "reproject:EPSG:4326"]
            downloads = self.pgv.download(
                results[:1],
                output_dir=output_dir / "imagery",
                parallel=1,
                post_process=pp,
                verify_checksum=False,
            )

            if downloads and downloads[0].success and downloads[0].path:
                return downloads[0].path

        except Exception as exc:
            logger.error("_search_and_download failed: %s", exc)
        return None

    def _search_and_download_pair(
        self,
        bbox: Tuple[float, float, float, float],
        date_before: str,
        date_after: str,
        output_dir: Path,
        collections: Optional[List[str]] = None,
        cloud_cover_max: float = 25.0,
    ) -> Tuple[Optional[Path], Optional[Path]]:
        """Download a bi-temporal (before/after) image pair."""
        before_path = self._search_and_download(
            bbox, date_before, output_dir / "before",
            collections=collections, cloud_cover_max=cloud_cover_max,
        )
        after_path = self._search_and_download(
            bbox, date_after, output_dir / "after",
            collections=collections, cloud_cover_max=cloud_cover_max,
        )
        return before_path, after_path

    def _run_model(
        self,
        image_path: Path,
        output_path: Path,
        model_name: str,
        num_classes: int,
        in_channels: int = 3,
        tile_size: int = 512,
        overlap: int = 64,
    ) -> Optional[Any]:
        """Load model from hub and run tiled inference."""
        try:
            from pygeovision.ai.models.hub import ModelHub
            from pygeovision.ai.inference.tiled_inference import TiledInference

            hub = ModelHub()
            model = hub.load(model_name, num_classes=num_classes, in_channels=in_channels)
            engine = TiledInference(model, tile_size=tile_size, overlap=overlap)
            return engine.run(image_path, output_path, num_classes=num_classes)
        except Exception as exc:
            logger.warning("Model inference failed: %s", exc)
            return None


# ---------------------------------------------------------------------------
# All 10 pipeline implementations
# ---------------------------------------------------------------------------

class ChangeDetectionPipeline(BasePipeline):
    """Bi-temporal change detection using Sentinel-2 and a siamese/transformer model.

    Kwargs:
        date_before: ISO date for first (before) image.
        date_after: ISO date for second (after) image.
        model: 'siamese_unet' or 'changeformer'.
        cloud_cover_max: Max cloud cover per image.
    """
    NAME = "change_detection"

    def run(self, bbox, output_dir, date_before="2022-06", date_after="2024-06",
            model="siamese_unet", cloud_cover_max=25.0, **kwargs):
        output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("[%s] bbox=%s before=%s after=%s", self.NAME, bbox, date_before, date_after)
        try:
            before_path, after_path = self._search_and_download_pair(
                bbox, date_before, date_after, output_dir, cloud_cover_max=cloud_cover_max,
            )
            if not before_path or not after_path:
                return PipelineResult(self.NAME, success=False,
                    error="Could not acquire imagery for one or both time periods.")

            output_path = output_dir / "change_mask.tif"
            pred = self._run_model(before_path, output_path, model, num_classes=2)

            return PipelineResult(
                pipeline=self.NAME, output_path=output_path if pred is not None else None,
                metadata={"date_before": date_before, "date_after": date_after, "model": model},
                success=pred is not None,
                error="" if pred is not None else "Model inference failed",
            )
        except Exception as exc:
            return PipelineResult(self.NAME, success=False, error=str(exc))


class LandCoverPipeline(BasePipeline):
    """Land cover classification using ESA WorldCover labels or a trained model.

    Kwargs:
        date: YYYY-MM acquisition date.
        source: 'worldcover', 'dynamic_world', or model name.
        num_classes: Number of land cover classes.
    """
    NAME = "land_cover"

    def run(self, bbox, output_dir, date="2023-06", source="worldcover",
            num_classes=11, **kwargs):
        output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("[%s] source=%s date=%s", self.NAME, source, date)
        try:
            if source == "worldcover":
                img_path = self._search_and_download(bbox, date, output_dir)
                if not img_path:
                    return PipelineResult(self.NAME, success=False, error="No imagery found.")
                from pygeovision.ai.labeling.esa_worldcover import ESAWorldCoverLabeler
                from pygeovision.ai.data.dataset import TileMetadata
                import rasterio
                with rasterio.open(img_path) as src:
                    meta = TileMetadata(path=img_path, bounds=src.bounds, crs=src.crs,
                                        height=src.height, width=src.width)
                lbl = ESAWorldCoverLabeler()
                out = output_dir / "land_cover.tif"
                r = lbl.label_tile(img_path, meta, out)
                return PipelineResult(self.NAME, output_path=out if r.success else None,
                    stats=r.class_distribution or {}, success=r.success,
                    metadata={"source": source, "date": date})
            elif source == "dynamic_world":
                img_path = self._search_and_download(bbox, date, output_dir)
                if not img_path:
                    return PipelineResult(self.NAME, success=False, error="No imagery.")
                from pygeovision.ai.labeling.dynamic_world import DynamicWorldLabeler
                from pygeovision.ai.data.dataset import TileMetadata
                import rasterio
                with rasterio.open(img_path) as src:
                    meta = TileMetadata(path=img_path, bounds=src.bounds, crs=src.crs,
                                        height=src.height, width=src.width)
                lbl = DynamicWorldLabeler(start_date=f"{date}-01", end_date=f"{date}-28")
                out = output_dir / "land_cover.tif"
                r = lbl.label_tile(img_path, meta, out)
                return PipelineResult(self.NAME, output_path=out if r.success else None,
                    success=r.success, metadata={"source": source})
            else:
                img_path = self._search_and_download(bbox, date, output_dir)
                if not img_path:
                    return PipelineResult(self.NAME, success=False, error="No imagery.")
                out = output_dir / "land_cover.tif"
                pred = self._run_model(img_path, out, source, num_classes=num_classes)
                return PipelineResult(self.NAME, output_path=out if pred is not None else None,
                    success=pred is not None, metadata={"source": source, "date": date})
        except Exception as exc:
            return PipelineResult(self.NAME, success=False, error=str(exc))


class BuildingFootprintsPipeline(BasePipeline):
    """Building footprint segmentation.

    Kwargs:
        date: YYYY-MM.
        model: Segmentation model name.
        cloud_cover_max: Max cloud %.
    """
    NAME = "building_footprints"

    def run(self, bbox, output_dir, date="2024-06", model="unet_resnet50",
            cloud_cover_max=15.0, **kwargs):
        output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
        try:
            img_path = self._search_and_download(bbox, date, output_dir,
                cloud_cover_max=cloud_cover_max)
            if not img_path:
                return PipelineResult(self.NAME, success=False, error="No imagery found.")
            out = output_dir / "building_mask.tif"
            pred = self._run_model(img_path, out, model, num_classes=2)
            stats = {}
            if pred is not None:
                import numpy as np
                stats["building_coverage"] = float((pred == 1).mean())
            return PipelineResult(self.NAME, output_path=out if pred is not None else None,
                stats=stats, metadata={"model": model, "date": date},
                success=pred is not None)
        except Exception as exc:
            return PipelineResult(self.NAME, success=False, error=str(exc))


class CropMonitoringPipeline(BasePipeline):
    """Crop type mapping and agricultural monitoring."""
    NAME = "crop_monitoring"

    def run(self, bbox, output_dir, date="2023-06", crop_classes=None,
            model="segformer_b2", **kwargs):
        output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
        try:
            img_path = self._search_and_download(bbox, date, output_dir)
            if not img_path:
                return PipelineResult(self.NAME, success=False, error="No imagery.")
            num_classes = len(crop_classes) + 1 if crop_classes else 10
            out = output_dir / "crop_map.tif"
            pred = self._run_model(img_path, out, model, num_classes=num_classes)
            return PipelineResult(self.NAME, output_path=out if pred is not None else None,
                success=pred is not None, metadata={"date": date, "crop_classes": crop_classes})
        except Exception as exc:
            return PipelineResult(self.NAME, success=False, error=str(exc))


class DisasterAssessmentPipeline(BasePipeline):
    """Rapid damage assessment after natural disasters using bi-temporal imagery."""
    NAME = "disaster_assessment"

    def run(self, bbox, output_dir, pre_date="2024-01", post_date="2024-02",
            disaster_type="generic", model="siamese_unet", **kwargs):
        output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
        try:
            pre_path, post_path = self._search_and_download_pair(
                bbox, pre_date, post_date, output_dir, cloud_cover_max=30.0)
            if not pre_path or not post_path:
                return PipelineResult(self.NAME, success=False,
                    error="Imagery not found for both dates.")
            out = output_dir / "damage_assessment.tif"
            pred = self._run_model(pre_path, out, model, num_classes=4)
            return PipelineResult(self.NAME, output_path=out if pred is not None else None,
                metadata={"pre_date": pre_date, "post_date": post_date,
                          "disaster_type": disaster_type},
                success=pred is not None)
        except Exception as exc:
            return PipelineResult(self.NAME, success=False, error=str(exc))


class DeforestationPipeline(BasePipeline):
    """Forest loss and deforestation detection."""
    NAME = "deforestation"

    def run(self, bbox, output_dir, baseline_year="2020", analysis_year="2024",
            model="changeformer", **kwargs):
        output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
        try:
            bl_path, cur_path = self._search_and_download_pair(
                bbox, f"{baseline_year}-07", f"{analysis_year}-07", output_dir,
                collections=["sentinel-2-l2a"])
            if not bl_path or not cur_path:
                return PipelineResult(self.NAME, success=False, error="Imagery unavailable.")
            out = output_dir / "deforestation_map.tif"
            pred = self._run_model(bl_path, out, model, num_classes=3)
            return PipelineResult(self.NAME, output_path=out if pred is not None else None,
                metadata={"baseline_year": baseline_year, "analysis_year": analysis_year},
                success=pred is not None)
        except Exception as exc:
            return PipelineResult(self.NAME, success=False, error=str(exc))


class UrbanGrowthPipeline(BasePipeline):
    """Urban expansion and impervious surface change detection."""
    NAME = "urban_growth"

    def run(self, bbox, output_dir, start_year="2018", end_year="2024",
            model="siamese_unet", **kwargs):
        output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
        try:
            # Landsat for multi-year comparisons (better temporal consistency)
            start_path, end_path = self._search_and_download_pair(
                bbox, f"{start_year}-06", f"{end_year}-06", output_dir,
                collections=["landsat-c2-l2"])
            if not start_path or not end_path:
                return PipelineResult(self.NAME, success=False, error="Landsat imagery not found.")
            out = output_dir / "urban_growth.tif"
            pred = self._run_model(start_path, out, model, num_classes=2)
            return PipelineResult(self.NAME, output_path=out if pred is not None else None,
                metadata={"start_year": start_year, "end_year": end_year},
                success=pred is not None)
        except Exception as exc:
            return PipelineResult(self.NAME, success=False, error=str(exc))


class WaterBodiesPipeline(BasePipeline):
    """Surface water body mapping — NDWI or deep learning."""
    NAME = "water_bodies"

    def run(self, bbox, output_dir, date="2024-06", method="ndwi",
            cloud_cover_max=10.0, **kwargs):
        output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
        try:
            # Post-process includes NDWI computation when method == 'ndwi'
            pp = ["unzip", "reproject:EPSG:4326", "ndwi"] if method == "ndwi" \
                 else ["unzip", "reproject:EPSG:4326"]
            img_path = self._search_and_download(
                bbox, date, output_dir, cloud_cover_max=cloud_cover_max,
                post_process=pp)
            if not img_path:
                return PipelineResult(self.NAME, success=False, error="No clear imagery found.")

            if method == "ndwi":
                # PyGeoFetch has already computed NDWI via post-process
                out = img_path  # NDWI is the processed output
                import numpy as np, rasterio
                with rasterio.open(img_path) as src:
                    ndwi = src.read(1).astype(np.float32)
                water_mask = (ndwi > 0.3).astype(np.uint8)
                coverage = float(water_mask.mean())
                stats = {"water_coverage": coverage}
            else:
                out = output_dir / "water_mask.tif"
                pred = self._run_model(img_path, out, "unet_resnet50", num_classes=2)
                import numpy as np
                stats = {"water_coverage": float((pred == 1).mean())} if pred is not None else {}

            return PipelineResult(self.NAME, output_path=Path(out),
                stats=stats, metadata={"date": date, "method": method}, success=True)
        except Exception as exc:
            return PipelineResult(self.NAME, success=False, error=str(exc))


class SolarDetectionPipeline(BasePipeline):
    """Solar panel / photovoltaic installation detection."""
    NAME = "solar_detection"

    def run(self, bbox, output_dir, date="2024-06", cloud_cover_max=5.0,
            model="unet_efficientnet_b4", **kwargs):
        output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
        try:
            img_path = self._search_and_download(
                bbox, date, output_dir, cloud_cover_max=cloud_cover_max)
            if not img_path:
                return PipelineResult(self.NAME, success=False, error="No cloud-free imagery.")
            out = output_dir / "solar_mask.tif"
            pred = self._run_model(img_path, out, model, num_classes=2,
                                   tile_size=256, overlap=32)
            stats = {}
            if pred is not None:
                import numpy as np
                stats["panel_coverage"] = float((pred == 1).mean())
            return PipelineResult(self.NAME, output_path=out if pred is not None else None,
                stats=stats, metadata={"date": date, "model": model},
                success=pred is not None)
        except Exception as exc:
            return PipelineResult(self.NAME, success=False, error=str(exc))


class CarbonEstimationPipeline(BasePipeline):
    """Above-ground biomass and carbon stock estimation via NDVI proxy."""
    NAME = "carbon_estimation"

    def run(self, bbox, output_dir, date="2024-06", **kwargs):
        output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
        try:
            # Request NDVI post-processing from PyGeoFetch
            img_path = self._search_and_download(
                bbox, date, output_dir,
                post_process=["unzip", "reproject:EPSG:4326", "ndvi"])
            if not img_path:
                return PipelineResult(self.NAME, success=False, error="No imagery.")

            import rasterio, numpy as np
            with rasterio.open(img_path) as src:
                ndvi = src.read(1).astype(np.float32)
                profile = src.profile.copy()
                pixel_area_ha = abs(src.res[0] * src.res[1]) / 10000

            # Simple allometric AGB → carbon (placeholder; replace with trained model)
            agb = np.clip(50.0 * ndvi ** 2, 0, 500).astype(np.float32)
            carbon = agb * 0.47

            out = output_dir / "carbon_map.tif"
            profile.update(dtype="float32", count=1, compress="lzw")
            with rasterio.open(out, "w", **profile) as dst:
                dst.write(carbon[np.newaxis, ...])

            valid = carbon[carbon > 0]
            total_area = float(carbon.size * pixel_area_ha)
            stats = {
                "mean_carbon_mg_ha": float(valid.mean()) if valid.size else 0.0,
                "total_area_ha": round(total_area, 1),
                "total_carbon_mg": round(float(valid.mean() * total_area), 1) if valid.size else 0.0,
            }
            return PipelineResult(self.NAME, output_path=out, stats=stats,
                metadata={"date": date}, success=True)
        except Exception as exc:
            return PipelineResult(self.NAME, success=False, error=str(exc))


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

_PIPELINE_REGISTRY: Dict[str, type] = {
    "change_detection":     ChangeDetectionPipeline,
    "land_cover":           LandCoverPipeline,
    "building_footprints":  BuildingFootprintsPipeline,
    "crop_monitoring":      CropMonitoringPipeline,
    "disaster_assessment":  DisasterAssessmentPipeline,
    "deforestation":        DeforestationPipeline,
    "urban_growth":         UrbanGrowthPipeline,
    "water_bodies":         WaterBodiesPipeline,
    "solar_detection":      SolarDetectionPipeline,
    "carbon_estimation":    CarbonEstimationPipeline,
}


def get_pipeline(name: str, pgv_client: Any) -> BasePipeline:
    """Instantiate a pipeline by name.

    Args:
        name: Pipeline name.
        pgv_client: PyGeoVision client (provides .search() and .download()
                   which delegate to PyGeoFetch).

    Returns:
        Instantiated pipeline.
    """
    if name not in _PIPELINE_REGISTRY:
        raise ValueError(
            f"Unknown pipeline '{name}'. Available: {sorted(_PIPELINE_REGISTRY.keys())}"
        )
    return _PIPELINE_REGISTRY[name](pgv_client)


def list_pipelines() -> List[str]:
    return sorted(_PIPELINE_REGISTRY.keys())


__all__ = [
    "BasePipeline", "PipelineResult", "get_pipeline", "list_pipelines",
    "ChangeDetectionPipeline", "LandCoverPipeline", "BuildingFootprintsPipeline",
    "CropMonitoringPipeline", "DisasterAssessmentPipeline", "DeforestationPipeline",
    "UrbanGrowthPipeline", "WaterBodiesPipeline", "SolarDetectionPipeline",
    "CarbonEstimationPipeline",
]
