"""PyGeoVision AI inference package."""

try:
    from pygeovision.ai.inference.tiled_inference import TiledInference
except (ImportError, AttributeError):
    TiledInference = None  # type: ignore[assignment,misc]

from pygeovision.ai.inference.postprocessing import PostProcessor

try:
    from pygeovision.ai.inference.ensemble import EnsembleInference
except (ImportError, AttributeError):
    EnsembleInference = None  # type: ignore[assignment,misc]

__all__ = ["TiledInference", "PostProcessor", "EnsembleInference"]
