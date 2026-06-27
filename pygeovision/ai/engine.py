"""
PyGeoVision AI Engine — central hub for all AI subsystems.

Lazy-loads all AI components on first access so importing pygeovision
without AI extras doesn't fail. All data operations use PyGeoFetch
(via the parent PyGeoVision client).
"""

from __future__ import annotations

import logging
from typing import Any, Optional, TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pygeovision.ai.models.hub import ModelHub
    from pygeovision.ai.models.registry import ModelRegistry
    from pygeovision.ai.monitoring import DriftDetector, PerformanceTracker
    from pygeovision.ai.experiments import ExperimentTracker


class AIEngine:
    """Lazy-loaded AI engine. Access via ``client.ai``.

    All heavy imports (torch, transformers, geoai, etc.) happen on first
    access. All data retrieval delegates to PyGeoFetch via the parent
    PyGeoVision client.

    Example:
        >>> model = client.ai.models.load("segformer_b2", num_classes=10)
        >>> tracker = client.ai.experiments.new("my_run", seed=42)
        >>> drift = client.ai.monitor.detect(reference_feats, prod_feats)
    """

    def __init__(self, pgv_client: Any) -> None:
        self._pgv = pgv_client          # PyGeoVision instance
        self._models_hub: Optional[Any] = None
        self._registry: Optional[Any] = None

    @property
    def models(self) -> "ModelHub":
        """ModelHub — download, cache, and load pretrained models."""
        if self._models_hub is None:
            from pygeovision.ai.models.hub import ModelHub
            self._models_hub = ModelHub()
        return self._models_hub  # type: ignore[return-value]

    @property
    def registry(self) -> "ModelRegistry":
        """ModelRegistry — list and inspect all registered architectures."""
        from pygeovision.ai.models.registry import registry
        return registry  # type: ignore[return-value]

    def label(self, tiles: Any, labeler: str, output_dir: str = "./labels", **kwargs: Any) -> Any:
        """Auto-label tiles using one of PyGeoVision's labelers.

        Labelers: 'osm', 'microsoft_buildings', 'google_buildings',
                  'esa_worldcover', 'dynamic_world', 'sam', 'foundation'.
        """
        from pygeovision.ai.labeling import (
            OSMLabeler, ESAWorldCoverLabeler, MicrosoftBuildingsLabeler,
            GoogleBuildingsLabeler, DynamicWorldLabeler,
        )
        _LABELERS = {
            "osm": OSMLabeler,
            "microsoft_buildings": MicrosoftBuildingsLabeler,
            "google_buildings": GoogleBuildingsLabeler,
            "esa_worldcover": ESAWorldCoverLabeler,
            "dynamic_world": DynamicWorldLabeler,
        }
        if labeler == "sam":
            from pygeovision.ai.labeling.sam_labeler import SAMLabeler
            lbl = SAMLabeler(**kwargs)
        elif labeler == "foundation":
            from pygeovision.ai.labeling.foundation_labeler import FoundationModelLabeler
            lbl = FoundationModelLabeler(**kwargs)
        elif labeler == "label_studio":
            from pygeovision.ai.labeling.label_studio import LabelStudioLabeler
            lbl = LabelStudioLabeler(**kwargs)
        elif labeler in _LABELERS:
            lbl = _LABELERS[labeler](**kwargs)
        else:
            raise ValueError(
                f"Unknown labeler '{labeler}'. "
                f"Available: {list(_LABELERS) + ['sam', 'foundation', 'label_studio']}"
            )
        return lbl.label_tiles(tiles, output_dir=output_dir)

    def train(self, model: Any, train_dataset: Any, val_dataset: Any = None, **kwargs: Any) -> Any:
        """Train a model with GeoTrainer."""
        from pygeovision.ai.training.trainer import GeoTrainer
        trainer = GeoTrainer(model=model, train_dataset=train_dataset, val_dataset=val_dataset, **kwargs)
        return trainer.fit()

    def infer(self, model: Any, input_path: str, output_path: str, num_classes: int = 2, **kwargs: Any) -> Any:
        """Run tiled inference on a large GeoTIFF scene."""
        from pygeovision.ai.inference.tiled_inference import TiledInference
        engine = TiledInference(model, **kwargs)
        return engine.run(input_path, output_path, num_classes=num_classes)

    def pipeline(self, name: str, bbox: Any, **kwargs: Any) -> Any:
        """Run an AI pipeline (delegates to client.pipeline)."""
        return self._pgv.pipeline(name, bbox=bbox, **kwargs)

    @property
    def monitor(self) -> "_MonitorProxy":
        return _MonitorProxy()

    @property
    def experiments(self) -> "_ExperimentsProxy":
        return _ExperimentsProxy()


class _MonitorProxy:
    def drift_detector(self, **kwargs: Any) -> "DriftDetector":
        from pygeovision.ai.monitoring import DriftDetector
        return DriftDetector(**kwargs)

    def performance_tracker(self, **kwargs: Any) -> "PerformanceTracker":
        from pygeovision.ai.monitoring import PerformanceTracker
        return PerformanceTracker(**kwargs)

    def detect(self, reference_features: Any, production_features: Any,
               threshold: float = 0.1, method: str = "ks") -> Any:
        from pygeovision.ai.monitoring import DriftDetector
        import numpy as np
        d = DriftDetector(threshold=threshold, method=method)
        d.fit_reference(np.array(reference_features))
        return d.check(np.array(production_features))


class _ExperimentsProxy:
    def new(self, name: str, **kwargs: Any) -> "ExperimentTracker":
        from pygeovision.ai.experiments import ExperimentTracker
        return ExperimentTracker(name, **kwargs)

    def load(self, run_id: str, **kwargs: Any) -> "ExperimentTracker":
        from pygeovision.ai.experiments import ExperimentTracker
        return ExperimentTracker.load(run_id, **kwargs)
