"""PyGeoVision Core — configuration, exceptions, and engine."""

from pygeovision.core.config import PyGeoVisionConfig
from pygeovision.core.exceptions import (  # noqa: F401
    PyGeoVisionError,
    PyGeoVisionConfigError,
    PyGeoVisionAuthError,
    AIEngineError,
    AINotAvailableError,
    ModelNotFoundError,
    TrainingError,
    InferenceError,
    PipelineError,
    LabelingError,
)


def _get_engine():
    """Lazy import PyGeoVisionEngine (requires pygeofetch dependencies)."""
    from pygeovision.core.engine import PyGeoVisionEngine
    return PyGeoVisionEngine
