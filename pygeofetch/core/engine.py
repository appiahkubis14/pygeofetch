"""
PyGeoFetch main engine.

The ``PyGeoFetch`` class is the single entry point for the Python API.
It wires together authentication, search, and download subsystems and
exposes a clean high-level interface.

Example::

    from pathlib import Path
    from pygeofetch import PyGeoFetch
    from pygeofetch.models import SearchQuery

    sb = PyGeoFetch()
    sb.auth.add("usgs", username="user", password="pass")

    results = sb.search(
        SearchQuery(
            bbox=(-74.1, 40.6, -73.7, 40.9),
            start_date="2024-01-01",
            end_date="2024-06-01",
            cloud_cover_max=20,
        ),
        providers=["usgs", "copernicus", "aws_earth"],
    )

    sb.download(results[:5], destination=Path("./data/"))
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from pygeofetch.core.authenticator import AuthManager
from pygeofetch.core.downloader import AdaptiveDownloader
from pygeofetch.core.searcher import FederatedSearcher
from pygeofetch.models.download_task import DownloadOptions, DownloadResult
from pygeofetch.models.satellite_data import SatelliteData
from pygeofetch.models.search_query import SearchQuery
from pygeofetch.utils.logging_setup import get_logger, setup_logging
from pygeofetch.processing.preprocessor import Preprocessor
from pygeofetch.processing.indices import SpectralIndices
from pygeofetch.processing.postprocessor import PostProcessor
from pygeofetch.processing.sar import SARProcessor
from pygeofetch.processing.batch import BatchProcessor


logger = get_logger(__name__)


class PyGeoFetch:
    """
    Universal satellite data pipeline.

    ``PyGeoFetch`` is the top-level API object that provides access
    to all subsystems:

    - ``sb.auth``     – :class:`~pygeofetch.core.authenticator.AuthManager`
    - ``sb.searcher`` – :class:`~pygeofetch.core.searcher.FederatedSearcher`
    - ``sb.downloader`` – :class:`~pygeofetch.core.downloader.AdaptiveDownloader`

    Attributes:
        auth: Authentication manager for all providers.
        searcher: Federated search engine.
        downloader: Adaptive parallel downloader.
        config: Runtime configuration dict.

    Example::

        sb = PyGeoFetch(log_level="INFO")
        results = sb.search(SearchQuery(bbox=(-74, 40, -73, 41)))
        sb.download(results, Path("./data/"))
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        log_level: str = "INFO",
        log_json: bool = False,
        cache_ttl: int = 3600,
        max_search_workers: int = 8,
        progress_callback: Optional[Callable] = None,
        auth_backend: str = "file",
    ) -> None:
        """
        Initialize PyGeoFetch.

        Args:
            config_path: Optional path to a custom config YAML file.
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
            log_json: Emit JSON-formatted log lines if True.
            cache_ttl: Search result cache TTL in seconds.
            max_search_workers: Maximum parallel provider searches.
            progress_callback: Optional callable(DownloadProgress) for progress events.
        """
        setup_logging(level=log_level, use_rich=not log_json)
        logger.debug("Initializing PyGeoFetch")

        self.auth = AuthManager(storage_backend=auth_backend)
        self.searcher = FederatedSearcher(
            auth_manager=self.auth,
            cache_ttl=cache_ttl,
            max_workers=max_search_workers,
        )
        self.downloader = AdaptiveDownloader(
            auth_manager=self.auth,
            progress_callback=progress_callback,
        )
        self.config: Dict[str, Any] = {}
        if config_path:
            self._load_config(config_path)


        # Processing subsystems (lazy-free — always available)
        self.preprocess = Preprocessor()
        self.indices    = SpectralIndices()
        self.post       = PostProcessor()
        self.sar        = SARProcessor()
        self.batch      = BatchProcessor(engine=self)
        logger.info("PyGeoFetch ready")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: SearchQuery,
        providers: Optional[List[str]] = None,
        use_cache: bool = True,
    ) -> List[SatelliteData]:
        """
        Search for satellite data across one or more providers.

        Args:
            query: Search parameters (bbox, dates, cloud cover, etc.).
            providers: Provider IDs to query. Queries all authenticated/free
                providers when omitted.
            use_cache: Use in-memory result cache (default True).

        Returns:
            List of :class:`~pygeofetch.models.satellite_data.SatelliteData`
            sorted by relevance score descending.

        Example::

            results = sb.search(
                SearchQuery(
                    bbox=(-74.1, 40.6, -73.7, 40.9),
                    start_date="2024-01-01",
                    cloud_cover_max=10,
                ),
                providers=["copernicus", "aws_earth"],
            )
        """
        return self.searcher.search(query, providers=providers, use_cache=use_cache)

    def search_and_save(
        self,
        query: SearchQuery,
        output: Path,
        providers: Optional[List[str]] = None,
    ) -> List[SatelliteData]:
        """
        Search and save results to a GeoJSON file.

        Args:
            query: Search parameters.
            output: Output GeoJSON file path.
            providers: Provider IDs to query.

        Returns:
            List of results (also written to *output*).
        """
        results = self.search(query, providers=providers)
        self.searcher.save_results(results, output)
        return results

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download(
        self,
        data: Union[SatelliteData, List[SatelliteData]],
        destination: Path,
        options: Optional[DownloadOptions] = None,
        item_done_callback=None,
    ) -> List[DownloadResult]:
        """
        Download one or more satellite data products.

        Args:
            data: Single item or list of SatelliteData to download.
            destination: Target directory (created if it does not exist).
            options: Download configuration (parallel, retry, checksum, etc.).

        Returns:
            List of :class:`~pygeofetch.models.download_task.DownloadResult`.

        Example::

            results = sb.download(
                search_results[:10],
                destination=Path("./data/"),
                options=DownloadOptions(parallel=4, verify_checksum=True),
            )
        """
        if isinstance(data, SatelliteData):
            data = [data]
        return self.downloader.download_many(data, Path(destination), options, item_done_callback=item_done_callback)

    def download_from_file(
        self,
        search_results_path: Path,
        destination: Path,
        options: Optional[DownloadOptions] = None,
    ) -> List[DownloadResult]:
        """
        Load search results from a GeoJSON file and download them.

        Args:
            search_results_path: Path to GeoJSON file from :meth:`search_and_save`.
            destination: Target directory.
            options: Download configuration.

        Returns:
            List of DownloadResults.
        """
        data_list = FederatedSearcher.load_results(search_results_path)
        logger.info(f"Loaded {len(data_list)} items from {search_results_path}")
        return self.download(data_list, destination, options)

    # ------------------------------------------------------------------
    # Auth convenience methods
    # ------------------------------------------------------------------

    def add_credentials(
        self,
        provider: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Store credentials for a provider.

        Args:
            provider: Provider identifier (e.g. ``"usgs"``, ``"copernicus"``).
            username: Username or email.
            password: Password or secret.
            api_key: API key (alternative to username/password).
            **kwargs: Extra credentials passed through to the provider.

        Example::

            sb.add_credentials("usgs", username="user", password="s3cr3t")
            sb.add_credentials("planet", api_key="PL_KEY")
        """
        creds: Dict[str, str] = {}
        if username:
            creds["username"] = username
        if password:
            creds["password"] = password
        if api_key:
            creds["api_key"] = api_key
        creds.update({k: str(v) for k, v in kwargs.items()})
        self.auth.add_credentials(provider, creds)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """
        Return a summary of the current system status.

        Returns:
            Dict with keys: ``providers_authenticated``, ``providers_free``,
            ``cache_entries``, ``version``.
        """
        from pygeofetch.providers import list_provider_info, get_free_providers
        from pygeofetch import __version__

        authed = [item["provider"] for item in self.auth.list()]
        free = list(get_free_providers())
        return {
            "version": __version__,
            "providers_authenticated": authed,
            "providers_free": free,
            "cache_entries": len(self.searcher.cache._cache),
        }

    def clear_cache(self) -> int:
        """
        Clear the in-memory search result cache.

        Returns:
            Number of cache entries removed.
        """
        count = self.searcher.cache.clear()
        logger.info(f"Cleared {count} cache entries")
        return count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_config(self, path: Path) -> None:
        """Load additional configuration from a YAML file."""
        import yaml

        with open(path) as f:
            extra = yaml.safe_load(f) or {}
        self.config.update(extra)
        logger.debug(f"Loaded extra config from {path}")


    def pipeline(self, name: str):
        """Create a chainable processing pipeline."""
        from pygeofetch.processing.pipeline import ProcessingPipeline
        return ProcessingPipeline(name=name, engine=self)

    def batch_process(self, inputs, chain, output_dir=".", parallel=2):
        """Batch process multiple files through a processing chain."""
        return self.batch.process(inputs, chain, output_dir=output_dir, parallel=parallel)

    def __repr__(self) -> str:
        authed = [item["provider"] for item in self.auth.list()]
        return (
            f"PyGeoFetch("
            f"authenticated={authed}, "
            f"cache_entries={len(self.searcher.cache._cache)})"
        )
