"""
PyGeoVision GeoAI Engine.

The AI engine is the new intelligence layer built on top of PyGeoFetch data.
All data retrieval within the AI engine uses PyGeoFetch — no data pipeline code
is duplicated here.

The AI engine is lazy-loaded: import this module only when AI functionality is
needed. The core ``pygeovision`` package installs without PyTorch or heavy ML deps.

Requires: ``pip install 'pygeovision[ai]'``
"""

from pygeovision.ai.engine import AIEngine

__all__ = ["AIEngine"]
