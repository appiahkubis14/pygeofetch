"""
PyGeoFetch Processing Engine — complete geospatial preprocessing,
spectral index computation, and post-processing.
"""
from pygeofetch.processing.base import ProcessingResult
from pygeofetch.processing.preprocessor import Preprocessor
from pygeofetch.processing.indices import SpectralIndices
from pygeofetch.processing.postprocessor import PostProcessor
from pygeofetch.processing.sar import SARProcessor
from pygeofetch.processing.pipeline import ProcessingPipeline

__all__ = [
    "ProcessingResult", "Preprocessor", "SpectralIndices",
    "PostProcessor", "SARProcessor", "ProcessingPipeline",
]
