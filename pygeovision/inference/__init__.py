"""
PyGeoVision Advanced Inference Engine (B1-B6).
Tiled inference with Gaussian blending, batch processing, memory-efficient streaming.
No GeoAI dependency.
"""
from pygeovision.inference.tiled   import TiledInference, GaussianBlend
from pygeovision.inference.batch   import BatchInferenceEngine
from pygeovision.inference.stream  import StreamingInference
from pygeovision.inference.stream  import EnsembleInference

__all__ = [
    "TiledInference", "GaussianBlend",
    "BatchInferenceEngine", "StreamingInference", "EnsembleInference",
]
