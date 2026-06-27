"""
PyGeoVision Training Infrastructure (Phase 4).

Distributed training, HPO, experiment tracking, model optimisation, and serving.
"""
from pygeovision.training.trainer   import GeoTrainer, TrainingConfig
from pygeovision.training.optimizer import build_optimizer, build_scheduler
from pygeovision.training.metrics   import SegmentationMetrics, DetectionMetrics, ChangeDetectionMetrics
from pygeovision.training.experiment import ExperimentTracker
from pygeovision.training.hpo       import OptunaHPO, ModelOptimizer

__all__ = [
    "GeoTrainer", "TrainingConfig",
    "build_optimizer", "build_scheduler",
    "SegmentationMetrics", "DetectionMetrics", "ChangeDetectionMetrics",
    "ExperimentTracker",
    "OptunaHPO", "ModelOptimizer",
]
