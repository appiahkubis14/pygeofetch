"""
PyGeoVision Core Engine — wraps PyGeoFetch for all satellite data operations.

PyGeoVision NEVER reimplements data fetching. This module is the single
integration point between PyGeoVision's AI layer and PyGeoFetch's data layer.

Uses pygeofetch CLI as primary backend + pystac_client as Python fallback.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pygeovision.core.config import PyGeoVisionConfig
from pygeovision.core.exceptions import (
    PyGeoVisionError,
    PyGeoVisionConfigError,
    PyGeoVisionAuthError,
)

logger = logging.getLogger(__name__)


class PyGeoVisionEngine:
    """Core engine: thin wrapper around SatelliteFetcher (pygeofetch).

    All satellite data operations delegate to the SatelliteFetcher
    which uses pygeofetch CLI as its primary backend.
    """

    def __init__(
        self,
        config: Optional[PyGeoVisionConfig] = None,
        api_key: Optional[str] = None,
    ) -> None:
        from pygeovision.data.fetch import SatelliteFetcher
        self.config = config or PyGeoVisionConfig()
        self._fetcher = SatelliteFetcher(config_path=None)
        self._api_key = api_key

    def search(self, *args, **kwargs):
        return self._fetcher.search(*args, **kwargs)

    def download(self, *args, **kwargs):
        return self._fetcher.download(*args, **kwargs)

    def add_credentials(self, *args, **kwargs):
        return self._fetcher.add_credentials(*args, **kwargs)

    def run_pipeline(self, *args, **kwargs):
        return self._fetcher.run_pipeline(*args, **kwargs)

    def status(self):
        return self._fetcher.status()
