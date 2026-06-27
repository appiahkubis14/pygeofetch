"""densenet — classification architecture. Delegates to registry.get_model() for construction."""
from pygeovision.models.registry import get_model, model_registry
__all__ = ["get_model", "model_registry"]
