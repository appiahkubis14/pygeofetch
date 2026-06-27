"""
PyGeoVision-specific exception hierarchy.

PyGeoVision inherits all PyGeoFetch exceptions for data operations. These
exceptions cover the AI engine and PyGeoVision orchestration layer only.
"""

from __future__ import annotations


class PyGeoVisionError(Exception):
    """Base exception for all PyGeoVision errors."""

    def __init__(self, message: str, details: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} — {self.details}"
        return self.message


class PyGeoVisionConfigError(PyGeoVisionError):
    """Raised for invalid or missing configuration."""


class PyGeoVisionAuthError(PyGeoVisionError):
    """Raised when authentication with a provider fails."""


# ------------------------------------------------------------------
# AI Engine Errors
# ------------------------------------------------------------------


class AIEngineError(PyGeoVisionError):
    """Base class for all GeoAI engine errors."""


class AINotAvailableError(AIEngineError):
    """Raised when AI dependencies are not installed."""

    def __init__(self) -> None:
        super().__init__(
            "GeoAI engine requires additional dependencies.",
            details="Install them with: pip install 'pygeovision[ai]'",
        )


class GPUError(AIEngineError):
    """Raised for GPU/CUDA-related errors."""


class OutOfMemoryError(GPUError):
    """Raised when GPU or CPU runs out of memory during model operations."""


# ------------------------------------------------------------------
# Data / Dataset Errors
# ------------------------------------------------------------------


class DatasetError(AIEngineError):
    """Raised for dataset preparation or loading errors."""


class TilingError(DatasetError):
    """Raised when the tiling engine fails."""


class AugmentationError(DatasetError):
    """Raised for augmentation pipeline errors."""


class PreprocessingError(DatasetError):
    """Raised for image preprocessing errors."""


# ------------------------------------------------------------------
# Labeling Errors
# ------------------------------------------------------------------


class LabelingError(AIEngineError):
    """Base class for labeling errors."""


class OSMLabelingError(LabelingError):
    """Raised for OpenStreetMap labeling failures."""


class BuildingFootprintError(LabelingError):
    """Raised when building footprint data cannot be retrieved."""


class SAMLabelingError(LabelingError):
    """Raised for SAM (Segment Anything Model) labeling errors."""


class FoundationModelLabelingError(LabelingError):
    """Raised for foundation model pseudo-label generation errors."""


# ------------------------------------------------------------------
# Model Errors
# ------------------------------------------------------------------


class ModelError(AIEngineError):
    """Base class for model-related errors."""


class ModelNotFoundError(ModelError):
    """Raised when a requested model is not in the registry."""

    def __init__(self, model_id: str) -> None:
        super().__init__(
            f"Model '{model_id}' not found in the registry.",
            details="Use `pygeovision ai models list` to see available models.",
        )
        self.model_id = model_id


class ModelDownloadError(ModelError):
    """Raised when a model weight download fails."""


class ModelLoadError(ModelError):
    """Raised when a model cannot be loaded from disk."""


class ModelExportError(ModelError):
    """Raised during model export (ONNX, TorchScript, TensorRT)."""


class ArchitectureError(ModelError):
    """Raised for unsupported or misconfigured architecture specifications."""


# ------------------------------------------------------------------
# Training Errors
# ------------------------------------------------------------------


class TrainingError(AIEngineError):
    """Base class for training errors."""


class CheckpointError(TrainingError):
    """Raised for checkpoint save/load failures."""


class DistributedTrainingError(TrainingError):
    """Raised for multi-GPU / distributed training errors."""


class LossError(TrainingError):
    """Raised for loss function errors (NaN, Inf)."""


# ------------------------------------------------------------------
# Inference Errors
# ------------------------------------------------------------------


class InferenceError(AIEngineError):
    """Base class for inference errors."""


class TiledInferenceError(InferenceError):
    """Raised for tiled inference failures."""


class PostProcessingError(InferenceError):
    """Raised for prediction post-processing failures."""


class VectorizationError(InferenceError):
    """Raised when raster-to-vector conversion fails."""


# ------------------------------------------------------------------
# Pipeline Errors
# ------------------------------------------------------------------


class PipelineError(AIEngineError):
    """Base class for end-to-end pipeline errors."""


class PipelineConfigError(PipelineError):
    """Raised for invalid pipeline configuration."""


class PipelineStepError(PipelineError):
    """Raised when a specific pipeline step fails."""

    def __init__(self, step: str, message: str) -> None:
        super().__init__(f"Pipeline step '{step}' failed: {message}")
        self.step = step


# ------------------------------------------------------------------
# Monitoring Errors
# ------------------------------------------------------------------


class MonitoringError(AIEngineError):
    """Raised for model monitoring errors."""


class DriftDetectionError(MonitoringError):
    """Raised for data drift detection failures."""


# ------------------------------------------------------------------
# Experiment Errors
# ------------------------------------------------------------------


class ExperimentError(AIEngineError):
    """Raised for experiment tracking errors."""
