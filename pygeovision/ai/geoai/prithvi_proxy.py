"""Prithvi proxy for the GeoAI Engine — exposes all Prithvi-EO capabilities."""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Union
logger = logging.getLogger(__name__)


class PrithviProxy:
    """Prithvi-EO subsystem for the PyGeoVision GeoAI Engine.

    Access via: client.geoai.prithvi (lazy-loaded)

    Example::

        pgv.geoai.prithvi.load("prithvi_eo_2_0")
        lc_map = pgv.geoai.prithvi.land_cover("hls_scene.tif")
        change = pgv.geoai.prithvi.change_detection("before.tif", "after.tif")
    """

    def __init__(self) -> None:
        self._prithvi     = None
        self._tasks       = None
        self._multi_temp  = None
        self._model_name  = "prithvi_eo_2_0"

    def load(self, model_name: str = "prithvi_eo_2_0",
              weights_path: Optional[str] = None,
              device: Optional[str] = None) -> "PrithviProxy":
        """Load a Prithvi-EO model. Returns self for chaining."""
        from pygeovision.models.foundation.prithvi import Prithvi
        self._prithvi = Prithvi(model_name, device=device).load(weights_path)
        self._model_name = model_name
        logger.info("Prithvi proxy loaded: %s", model_name)
        return self

    def _ensure_tasks(self) -> Any:
        if self._tasks is None:
            from pygeovision.models.foundation.prithvi import PrithviTasks
            self._tasks = PrithviTasks(self._model_name)
        return self._tasks

    def _ensure_mt(self) -> Any:
        if self._multi_temp is None:
            from pygeovision.models.foundation.prithvi import PrithviMultiTemporal
            self._multi_temp = PrithviMultiTemporal(self._model_name)
        return self._multi_temp

    def extract_features(self, image_path: str,
                          source: str = "hls") -> Any:
        """Extract CLS token features from an HLS GeoTIFF."""
        if self._prithvi is None: self.load()
        return self._prithvi.extract_features(image_path, source=source)

    def land_cover(self, image_path: str, source: str = "hls",
                    output_path: Optional[str] = None) -> Dict:
        """Land cover classification (10 HLS-based classes)."""
        return self._ensure_tasks().land_cover(image_path, source, output_path)

    def crop_mapping(self, image_path: str, source: str = "hls",
                      output_path: Optional[str] = None) -> Dict:
        """Crop type mapping (10 major crop classes)."""
        return self._ensure_tasks().crop_mapping(image_path, source, output_path)

    def flood_detection(self, image_path: str, source: str = "sentinel2",
                         output_path: Optional[str] = None) -> Dict:
        """Binary flood detection."""
        return self._ensure_tasks().flood_detection(image_path, source, output_path)

    def biomass_estimation(self, image_path: str, source: str = "hls") -> Dict:
        """Above-ground biomass estimation (t DM/ha)."""
        return self._ensure_tasks().biomass_estimation(image_path, source)

    def deforestation_detection(self, before: str, after: str,
                                  output_path: Optional[str] = None) -> Dict:
        """Detect deforestation using Prithvi multi-temporal features."""
        return self._ensure_tasks().deforestation_detection(before, after, output_path)

    def change_detection(self, before: str, after: str,
                          source: str = "hls",
                          output_path: Optional[str] = None) -> Dict:
        """Generic change detection between two dates."""
        return self._ensure_mt().detect_change(before, after, source, output_path)

    def time_series(self, image_paths: List[str],
                     dates: Optional[List[str]] = None,
                     source: str = "hls") -> Dict:
        """Multi-temporal analysis with Prithvi temporal attention."""
        return self._ensure_mt().process_time_series(image_paths, dates, source)

    def monitor_trend(self, image_paths: List[str],
                       dates: Optional[List[str]] = None) -> Dict:
        """Analyse temporal trends in Prithvi feature space."""
        return self._ensure_mt().monitor_trend(image_paths, dates)

    def predict_seasonal(self, image_paths: List[str],
                          dates: Optional[List[str]] = None) -> Dict:
        """Fit seasonal cycle model to time series."""
        return self._ensure_mt().predict_seasonal(image_paths, dates)

    def finetune(self, task: str = "land_cover", num_classes: int = 10,
                  model_name: Optional[str] = None, **kwargs) -> Dict:
        """Fine-tune Prithvi for a geospatial task."""
        from pygeovision.models.foundation.prithvi import finetune_prithvi
        return finetune_prithvi(model_name=model_name or self._model_name,
                                 task=task, num_classes=num_classes, **kwargs)

    def list_models(self) -> List[str]:
        from pygeovision.models.foundation.prithvi import list_prithvi_models
        return list_prithvi_models()

    def get_info(self, model_name: Optional[str] = None) -> Dict:
        from pygeovision.models.foundation.prithvi import get_prithvi_info
        return get_prithvi_info(model_name or self._model_name)

    def __repr__(self) -> str:
        return f"PrithviProxy(model={self._model_name!r}, EO-1.0+EO-2.0)"
