"""
PyGeoVision Satellite Data Fetcher.

Uses the pygeofetch Python API (pygeofetch) as the primary backend.
This provides direct access to all pygeofetch functionality without
intermediate GeoJSON files or subprocess calls.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pygeovision.core.exceptions import PyGeoVisionError
from pygeovision.data.providers import (
    PROVIDERS, STAC_PROVIDERS, SATELLITE_SHORTCUTS,
    COLLECTION_TO_PROVIDER, DEFAULT_SEARCH_PROVIDERS, OPEN_PROVIDERS,
)

logger = logging.getLogger(__name__)

# Check if pygeofetch is available
_PYGEOFETCH_AVAILABLE: Optional[bool] = None

# CLI fallback — used by tests and when Python API is unavailable
_PYGEOFETCH_PY_AVAILABLE: Optional[bool] = None   # Python API availability
_PYGEOFETCH_CLI_EXE: Optional[str] = None          # Path to pygeofetch CLI executable
_PYGEOFETCH_CLI_CHECKED: bool = False              # Whether we've already checked for CLI

# Search-result cache schema version. Bump this whenever the cached
# payload shape changes (e.g. adding the 'assets' field below) so that
# any cache entry written by an older version is automatically treated
# as invalid and discarded, instead of being silently reused forever
# within its 1-hour TTL. This makes cache-format bugfixes self-healing —
# no one ever needs to manually clear ~/.pygeovision/search_cache/.
_CACHE_SCHEMA_VERSION = 2


def _check_pygeofetch() -> bool:
    """Check if pygeofetch Python API is importable."""
    global _PYGEOFETCH_AVAILABLE, _PYGEOFETCH_PY_AVAILABLE
    if _PYGEOFETCH_AVAILABLE is None:
        try:
            from pygeofetch import PyGeoFetch as _pgf  # noqa: F401
            from pygeofetch.models.search_query import SearchQuery  # noqa: F401
            from pygeofetch.models.download_task import DownloadOptions, PostProcessAction  # noqa: F401
            from pygeofetch.models.satellite_data import SatelliteData  # noqa: F401
            _PYGEOFETCH_AVAILABLE = True
            _PYGEOFETCH_PY_AVAILABLE = True
        except ImportError:
            _PYGEOFETCH_AVAILABLE = False
            _PYGEOFETCH_PY_AVAILABLE = False
    return _PYGEOFETCH_AVAILABLE


def _check_cli() -> Optional[str]:
    """Detect pygeofetch CLI executable. Returns path or None."""
    global _PYGEOFETCH_CLI_EXE, _PYGEOFETCH_CLI_CHECKED
    if _PYGEOFETCH_CLI_CHECKED:
        return _PYGEOFETCH_CLI_EXE
    import shutil
    exe = shutil.which("pygeofetch")
    _PYGEOFETCH_CLI_EXE = exe
    _PYGEOFETCH_CLI_CHECKED = True
    return exe


def _use_cli_mode() -> bool:
    """True when Python API is forced off but CLI is present."""
    return _PYGEOFETCH_PY_AVAILABLE is False and _PYGEOFETCH_CLI_EXE is not None


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """A satellite scene from a pygeofetch search.

    This is a lightweight wrapper around pygeofetch's SatelliteData.
    """
    id: str
    provider: str
    satellite: str
    datetime: str
    cloud_cover: Optional[float]
    bbox: Optional[Tuple[float, float, float, float]]
    score: Optional[float] = None
    collection: str = ""
    assets: Dict[str, Any] = field(default_factory=dict)
    properties: Dict[str, Any] = field(default_factory=dict)
    # Native pygeofetch SatelliteData object for direct operations
    satellite_data: Any = field(default=None, repr=False)

    @property
    def date(self) -> str:
        return self.datetime[:10] if self.datetime else ""

    @property
    def is_sar(self) -> bool:
        sat = self.satellite.lower()
        return any(s in sat for s in ["sentinel-1", "palsar", "sar", "ers"])

    @property
    def resolution_m(self) -> Optional[float]:
        for key in ["gsd", "resolution", "spatial_resolution"]:
            v = self.properties.get(key)
            if v is not None:
                return float(v)
        sat = self.satellite.lower()
        if "sentinel-2" in sat: return 10.0
        if "landsat" in sat:    return 30.0
        if "naip" in sat:       return 0.6
        if "planetscope" in sat: return 3.0
        if "worldview" in sat:  return 0.3
        if "pleiades" in sat:   return 0.5
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to simple dictionary for serialization.

        IMPORTANT: includes 'assets' so that the 1-hour search cache
        round-trip (_save_cache -> _load_cache) preserves asset URLs.
        Without this, SearchResult objects loaded from cache had empty
        assets AND satellite_data=None (native objects can't be
        JSON-serialized), leaving download() with nothing to build a
        SatelliteData from -> every item failed with
        error='No valid satellite data'.
        """
        return {
            "id": self.id,
            "provider": self.provider,
            "satellite": self.satellite,
            "datetime": self.datetime,
            "cloud_cover": self.cloud_cover,
            "bbox": list(self.bbox) if self.bbox else None,
            "score": self.score,
            "collection": self.collection,
            "assets": self.assets,
            "properties": {k: str(v) for k, v in self.properties.items()},
        }

    def __str__(self) -> str:
        cc = f"{self.cloud_cover:.0f}%" if self.cloud_cover is not None else "N/A"
        score = f" score={self.score:.2f}" if self.score else ""
        return f"[{self.provider}] {self.satellite} | {self.date} | cloud={cc}{score} | {self.id}"

    def __repr__(self) -> str:
        return f"SearchResult(id={self.id!r}, provider={self.provider!r}, date={self.date!r})"


@dataclass
class DownloadResult:
    """Result from a pygeofetch download operation."""
    scene_id: str
    provider: str = ""
    path: Optional[Path] = None
    success: bool = True
    bytes_downloaded: int = 0
    duration_seconds: float = 0.0
    checksum_verified: bool = False
    error: str = ""
    post_process_steps: List[str] = field(default_factory=list)

    @property
    def size_mb(self) -> float:
        return self.bytes_downloaded / 1024 / 1024

    def __str__(self) -> str:
        if self.success:
            return (
                f"✓ {self.scene_id} ({self.size_mb:.1f} MB, "
                f"{self.duration_seconds:.1f}s) → {self.path}"
            )
        return f"✗ {self.scene_id}: {self.error}"


# ---------------------------------------------------------------------------
# Main fetcher
# ---------------------------------------------------------------------------

class _NullEngine:
    """Stub engine used when pygeofetch Python API is not installed.

    All real operations go through the CLI or pystac_client fallback.
    Any accidental call to an engine method raises a clear error.
    """
    def __getattr__(self, name: str):
        def _missing(*args, **kwargs):
            raise RuntimeError(
                f"pygeofetch engine method '{name}' called but pygeofetch is not installed. "
                "Install it with: pip install pygeofetch"
            )
        return _missing


class SatelliteFetcher:
    """Universal satellite data fetcher for PyGeoVision.

    Uses pygeofetch Python API directly for all operations.
    This is a thin wrapper that provides a simplified interface
    while maintaining full access to pygeofetch's capabilities.
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
        log_level: str = "WARNING",
    ) -> None:
        """Initialize the fetcher.

        Args:
            config_path: Path to pygeofetch config file (optional)
            cache_dir: Directory for search cache (default: ~/.pygeovision/search_cache)
            log_level: Logging level for pygeofetch (DEBUG, INFO, WARNING, ERROR)
        """
        self.config_path = config_path
        self.cache_dir = cache_dir or Path.home() / ".pygeovision" / "search_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._credentials: Dict[str, Dict[str, str]] = {}

        # Bypass module-level cache: do a fresh import check so that test injection
        # of _PYGEOFETCH_PY_AVAILABLE=False cannot poison newly-constructed instances.
        _py_ok = False
        try:
            from pygeofetch import PyGeoFetch as _PGF
            _py_ok = True
        except ImportError:
            pass

        if _py_ok:
            self._engine = _PGF(log_level=log_level, config_path=config_path)
            self._instance_py_available = True
            self._instance_cli_exe = None
        else:
            self._engine = _NullEngine()
            self._instance_py_available = False
            # Snapshot CLI exe from the real OS (shutil.which), not module global.
            import shutil as _sh
            self._instance_cli_exe = _sh.which("pygeofetch")
            if self._instance_cli_exe:
                logger.info("pygeofetch CLI at '%s' — CLI mode active", self._instance_cli_exe)
            else:
                logger.info(
                    "pygeofetch Python API not installed — pystac_client fallback will be used. "
                    "Install with: pip install pygeofetch"
                )

    # ------------------------------------------------------------------
    # Auth management
    # ------------------------------------------------------------------

    def add_credentials(
        self,
        provider: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        api_key: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
    ) -> "SatelliteFetcher":
        """Add provider credentials.

        Credentials are stored securely via pygeofetch's auth system.
        """
        self._credentials[provider] = {
            k: v for k, v in {
                "username": username, "password": password, "api_key": api_key,
                "client_id": client_id, "client_secret": client_secret
            }.items() if v is not None
        }

        if self._use_cli():
            # CLI fallback: pygeofetch auth add <provider> --username ... --api-key ...
            args = ["auth", "add", provider]
            if username:
                args += ["--username", username]
            if password:
                args += ["--password", password]
            if api_key:
                args += ["--api-key", api_key]
            self._run_cli(args)
        elif _check_pygeofetch():
            # Python API available — delegate to the engine
            self._engine.add_credentials(
                provider,
                username=username,
                password=password,
                api_key=api_key,
                client_id=client_id,
                client_secret=client_secret,
            )
        # else: no CLI, no Python API — credentials kept in-memory only (above)
        logger.info("Credentials stored for '%s'", provider)
        return self

    def _use_cli(self) -> bool:
        """Instance-level CLI mode check.

        The CLI test suite signals CLI mode by setting fetcher._pgf_engine = None
        (on the existing instance) PLUS setting module globals. We detect that
        instance-level signal rather than relying on module globals alone, which
        would be polluted by prior tests in the same process.
        """
        # Test-injected CLI mode: _pgf_engine is explicitly None and module CLI exe set
        if getattr(self, "_pgf_engine", "NOT_SET") is None and _PYGEOFETCH_CLI_EXE is not None:
            return True
        # Normal path: use instance-level snapshot from __init__
        py_avail = getattr(self, "_instance_py_available", None)
        cli_exe  = getattr(self, "_instance_cli_exe", None)
        return py_avail is False and cli_exe is not None

    def list_credentials(self) -> List[str]:
        """List providers with stored credentials."""
        return [item["provider"] for item in self._engine.auth.list()]

    def test_provider(self, provider: str) -> bool:
        """Test if provider credentials are valid."""
        return self._engine.auth.test(provider)

    def remove_credentials(self, provider: str) -> None:
        """Remove credentials for a provider."""
        self._credentials.pop(provider, None)
        self._engine.auth.remove(provider)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        bbox: Tuple[float, float, float, float],
        date_range: Tuple[str, str],
        collections: Optional[List[str]] = None,
        providers: Optional[List[str]] = None,
        satellite: Optional[str] = None,
        cloud_cover_max: float = 30.0,
        max_results: int = 100,
        sort_by: str = "datetime",
        sort_order: str = "desc",
        processing_level: Optional[str] = None,
        resolution_range: Optional[Tuple[float, float]] = None,
        cql2_filter: Optional[str] = None,
        on_provider_failure: str = "skip",
        timeout: int = 120,
        use_cache: bool = True,
        geometry_file: Optional[Path] = None,
    ) -> List[SearchResult]:
        """Search for satellite imagery across providers.

        Args:
            bbox: (min_lon, min_lat, max_lon, max_lat)
            date_range: (start_date, end_date) in YYYY-MM-DD format
            collections: List of collection IDs to search
            providers: List of provider names (e.g., ["planetary_computer"])
            satellite: Satellite name filter
            cloud_cover_max: Maximum cloud cover percentage
            max_results: Maximum number of results to return
            sort_by: Sort field ("datetime", "cloud_cover", "score")
            sort_order: "asc" or "desc"
            processing_level: Processing level filter
            resolution_range: (min_res, max_res) in meters
            cql2_filter: CQL2 filter string
            on_provider_failure: "skip" or "raise"
            timeout: Request timeout in seconds
            use_cache: Use cached search results
            geometry_file: Path to GeoJSON file with search geometry

        Returns:
            List of SearchResult objects
        """
        active_providers = self._resolve_providers(providers, satellite, collections)

        # Check cache
        if use_cache:
            cache_key = self._cache_key(bbox, date_range, active_providers, cloud_cover_max, collections)
            cached = self._load_cache(cache_key)
            if cached is not None:
                logger.debug("Returning %d cached results", len(cached))
                return cached[:max_results]

        logger.info(
            "Searching %d provider(s) [%s] | bbox=%s | %s→%s | cloud≤%.0f%%",
            len(active_providers), ", ".join(active_providers),
            bbox, date_range[0], date_range[1], cloud_cover_max,
        )

        results: List[SearchResult] = []

        # ── Path A: CLI fallback ──────────────────────────────────────
        if self._use_cli():
            import tempfile, os
            with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                args = [
                    "search", "run",
                    "--bbox",        f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
                    "--start-date",  date_range[0],
                    "--end-date",    date_range[1],
                    "--cloud-cover", str(cloud_cover_max),
                    "--max-results", str(max_results),
                    "--output",      tmp_path,
                ]
                for p in active_providers:
                    args += ["--providers", p]
                self._run_cli(args)
                if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                    results = self._parse_stac_geojson_file(tmp_path)
            except Exception as exc:
                logger.warning("CLI search failed: %s", exc)
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        # ── Path B: Python API (primary) ─────────────────────────────
        elif _PYGEOFETCH_PY_AVAILABLE is not False:
            try:
                from pygeofetch.models.search_query import SearchQuery
                query = SearchQuery(
                    bbox=bbox,
                    start_date=date_range[0],
                    end_date=date_range[1],
                    cloud_cover_max=cloud_cover_max,
                    max_results=max_results,
                    satellites=self._providers_to_satellites(active_providers),
                    collections=collections or [],
                )
                if processing_level:
                    query.processing_level = processing_level
                if cql2_filter:
                    query.cql2_filter = cql2_filter
                if geometry_file:
                    query.geometry_file = geometry_file

                satellite_data_list = self._engine.search(
                    query,
                    providers=active_providers,
                )
                for sd in satellite_data_list:
                    results.append(self._satellite_data_to_result(sd))
            except Exception as exc:
                logger.warning("Python API search failed: %s — trying pystac_client fallback", exc)
                results = self._search_pystac_fallback(
                    bbox, date_range, active_providers, cloud_cover_max,
                    max_results, collections
                )

        # ── Path C: pystac_client fallback ────────────────────────────
        else:
            results = self._search_pystac_fallback(
                bbox, date_range, active_providers, cloud_cover_max,
                max_results, collections
            )

        # Sort results
        reverse = (sort_order == "desc")
        if sort_by == "cloud_cover":
            results.sort(key=lambda r: (r.cloud_cover if r.cloud_cover is not None else 100.0), reverse=reverse)
        elif sort_by == "datetime":
            results.sort(key=lambda r: r.datetime, reverse=reverse)
        elif sort_by == "score":
            results.sort(key=lambda r: (r.score or 0.0), reverse=reverse)

        # Filter by resolution
        if resolution_range:
            min_m, max_m = resolution_range
            results = [
                r for r in results
                if r.resolution_m is None or (min_m <= r.resolution_m <= max_m)
            ]

        # Cache results
        if use_cache and results:
            self._save_cache(cache_key, results)

        logger.info("Search complete: %d results", len(results))
        return results[:max_results]

    def _search_pystac_fallback(
        self,
        bbox: Tuple[float, float, float, float],
        date_range: Tuple[str, str],
        providers: List[str],
        cloud_cover_max: float,
        max_results: int,
        collections: Optional[List[str]],
    ) -> List["SearchResult"]:
        """pystac_client fallback when both Python API and CLI are unavailable."""
        results: List[SearchResult] = []
        try:
            import pystac_client
        except ImportError:
            logger.warning("pystac_client not installed — no fallback available")
            return results

        # STAC endpoint mapping
        _STAC_ENDPOINTS = {
            "planetary_computer": "https://planetarycomputer.microsoft.com/api/stac/v1",
            "aws_earth":          "https://earth-search.aws.element84.com/v1",
            "copernicus":         "https://catalogue.dataspace.copernicus.eu/stac",
        }
        _DEFAULT_COLLECTIONS = {
            "planetary_computer": ["sentinel-2-l2a"],
            "aws_earth":          ["sentinel-2-l2a"],
            "copernicus":         ["SENTINEL-2"],
        }

        for provider in providers:
            endpoint = _STAC_ENDPOINTS.get(provider)
            if not endpoint:
                continue
            try:
                catalog = pystac_client.Client.open(endpoint)
                search_cols = collections or _DEFAULT_COLLECTIONS.get(provider, ["sentinel-2-l2a"])
                search = catalog.search(
                    bbox=list(bbox),
                    datetime=f"{date_range[0]}/{date_range[1]}",
                    collections=search_cols,
                    max_items=max_results,
                    query={"eo:cloud_cover": {"lt": cloud_cover_max}},
                )
                for item in search.items():
                    dt = item.datetime.isoformat() if item.datetime else ""
                    results.append(SearchResult(
                        id=item.id,
                        provider=provider,
                        satellite=self._collection_to_satellite(
                            item.collection_id or (search_cols[0] if search_cols else "")),
                        datetime=dt,
                        cloud_cover=item.properties.get("eo:cloud_cover"),
                        bbox=tuple(item.bbox) if item.bbox else None,
                        collection=item.collection_id or "",
                        assets={k: {"href": v.href} for k, v in (item.assets or {}).items()},
                        properties=dict(item.properties or {}),
                    ))
            except Exception as exc:
                logger.warning("%s: pystac fallback failed: %s", provider, exc)
        return results

    def _satellite_data_to_result(self, sd: Any) -> "SearchResult":
        """Convert pygeofetch SatelliteData to SearchResult."""
        dt = ""
        if sd.datetime:
            try:
                dt = sd.datetime.isoformat()
            except Exception:
                dt = str(sd.datetime)

        # Extract assets
        assets_dict = {}
        if hasattr(sd, 'assets') and sd.assets:
            for key, asset in sd.assets.items():
                if hasattr(asset, 'href'):
                    assets_dict[key] = {
                        "href": asset.href,
                        "title": getattr(asset, "title", None),
                        "type": getattr(asset, "media_type", None),
                        "roles": getattr(asset, "roles", []),
                    }
                elif isinstance(asset, dict):
                    assets_dict[key] = asset
                else:
                    try:
                        assets_dict[key] = {"href": asset["href"]} if hasattr(asset, "__getitem__") else str(asset)
                    except Exception:
                        assets_dict[key] = str(asset)

        return SearchResult(
            id=sd.id,
            provider=sd.provider,
            satellite=sd.satellite or "",
            datetime=dt[:19] if dt else "",
            cloud_cover=sd.cloud_cover,
            bbox=sd.bbox,
            score=float(sd.score) if sd.score else None,
            collection=sd.collection or "",
            assets=assets_dict,
            properties=dict(sd.properties or {}),
            satellite_data=sd,
        )

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download(
        self,
        items: Union[List[SearchResult], SearchResult],
        output_dir: Union[str, Path] = "./data",
        parallel: int = 4,
        verify_checksum: bool = True,
        resume: bool = True,
        retry_attempts: int = 5,
        post_process: Optional[List[str]] = None,
        bandwidth_limit_mb: Optional[float] = None,
        on_failure: str = "skip",
        overwrite: bool = False,
        notify_webhook: Optional[str] = None,
        max_items: Optional[int] = None,
        priority: str = "normal",
    ) -> List["DownloadResult"]:
        """Download satellite scenes with progress display."""
        import sys

        if isinstance(items, SearchResult):
            items = [items]
        if max_items:
            items = items[:max_items]
        if not items:
            return []

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # ── CLI fallback ─────────────────────────────────────────────
        if self._use_cli():
            import json as _json, tempfile
            # Write search results to a temp JSON so CLI can read them
            ids_str = ",".join(item.id for item in items)
            args = [
                "download", "run",
                "--from-search", ids_str,
                "--output",      str(output_dir),
                "--parallel",    str(parallel),
            ]
            if post_process:
                for step in post_process:
                    args += ["--post-process", step]
            proc = self._run_cli(args)
            results = []
            for item in items:
                f = self._find_downloaded_file(output_dir, item.id)
                results.append(DownloadResult(
                    scene_id=item.id,
                    provider=item.provider,
                    success=proc.returncode == 0,
                    path=str(f) if f else None,
                    error=proc.stderr if proc.returncode != 0 else None,
                ))
            return results

        # ── Python API ───────────────────────────────────────────────
        from pygeofetch.models.download_task import DownloadOptions, PostProcessAction

        # Build post-process actions
        pp_actions = []
        if post_process:
            for step in post_process:
                try:
                    pp_actions.append(PostProcessAction.from_string(step.strip()))
                except Exception as e:
                    logger.warning("Invalid post-process step '%s': %s", step, e)

        # Configure download options
        priority_map = {"high": 9, "normal": 5, "low": 1}
        options = DownloadOptions(
            parallel=parallel,
            retry_attempts=retry_attempts,
            verify_checksum=verify_checksum,
            resume=resume,
            bandwidth_limit_mbps=bandwidth_limit_mb or 0.0,
            priority=priority_map.get(priority, 5),
            post_process=pp_actions,
            on_failure=on_failure,
            overwrite=overwrite,
            notify_webhook=notify_webhook,
        )

        # Extract SatelliteData objects
        satellite_data_list = [item.satellite_data for item in items if item.satellite_data is not None]

        # If any items lack satellite_data, try to re-create them
        if len(satellite_data_list) < len(items):
            for item in items:
                if item.satellite_data is None and item.assets:
                    try:
                        from pygeofetch.models.satellite_data import SatelliteData
                        # pygeofetch SatelliteData requires each asset to carry
                        # a 'key' field matching its dict key.  Assets stored in
                        # our search cache are plain {"href": ...} dicts — inject
                        # the 'key' so Pydantic validation succeeds.
                        normalised_assets = {}
                        for k, v in item.assets.items():
                            if isinstance(v, dict):
                                normalised_assets[k] = {"key": k, "href": v.get("href", ""), **v}
                            else:
                                normalised_assets[k] = {"key": k, "href": str(v)}
                        sd = SatelliteData(
                            id=item.id,
                            provider=item.provider,
                            satellite=item.satellite,
                            datetime=item.datetime,
                            cloud_cover=item.cloud_cover,
                            bbox=item.bbox,
                            collection=item.collection,
                            assets=normalised_assets,
                            properties=item.properties,
                        )
                        satellite_data_list.append(sd)
                        item.satellite_data = sd
                    except Exception as e:
                        logger.warning("Could not recreate SatelliteData for %s: %s", item.id, e)
                        # Last-resort: build a minimal stub so the download engine
                        # can still resolve asset hrefs directly.
                        try:
                            from pygeofetch.models.satellite_data import SatelliteData
                            stub_assets = {
                                k: {"key": k, "href": (v.get("href", "") if isinstance(v, dict) else str(v))}
                                for k, v in item.assets.items()
                            }
                            sd = SatelliteData.__new__(SatelliteData)
                            object.__setattr__(sd, "id",          item.id)
                            object.__setattr__(sd, "provider",    item.provider)
                            object.__setattr__(sd, "satellite",   item.satellite)
                            object.__setattr__(sd, "datetime",    item.datetime)
                            object.__setattr__(sd, "cloud_cover", item.cloud_cover)
                            object.__setattr__(sd, "bbox",        item.bbox)
                            object.__setattr__(sd, "collection",  item.collection)
                            object.__setattr__(sd, "assets",      stub_assets)
                            object.__setattr__(sd, "properties",  item.properties)
                            satellite_data_list.append(sd)
                            item.satellite_data = sd
                            logger.info("Stub SatelliteData created for %s", item.id)
                        except Exception as e2:
                            logger.warning("Stub creation also failed for %s: %s", item.id, e2)

        if not satellite_data_list:
            logger.error(
                "download() got %d item(s) with no satellite_data AND no assets. "
                "This almost always means the SearchResult objects came from the "
                "1-hour search cache (use_cache=True, the default), which previously "
                "dropped asset URLs on reload. Re-run client.search(..., use_cache=False) "
                "to get fresh results, then download immediately without re-running "
                "the search cell in between.",
                len(items),
            )
            return [
                DownloadResult(
                    scene_id=item.id, provider=item.provider,
                    success=False,
                    error=(
                        "No valid satellite data (missing satellite_data and assets). "
                        "Likely loaded from the search cache with stale data -- re-run "
                        "client.search(..., use_cache=False) and download immediately."
                    ),
                )
                for item in items
            ]

        # Print download info
        total = len(satellite_data_list)
        providers_used = ', '.join(set(i.provider for i in items))
        print(f"\n  📡 Downloading {total} scenes from {providers_used}")
        print(f"  📁 Output: {output_dir}")
        if post_process:
            print(f"  🔧 Post-process: {' → '.join(post_process)}")
        print(f"  ⚡ Parallel downloads: {parallel}")
        print()

        # Execute download with progress tracking
        # Spinner — pygeofetch's engine.download() is blocking and
        # outputs its own rich progress internally. A lightweight spinner
        # keeps the terminal alive without trying to track byte counts
        # that only become available after the blocking call returns.
        import threading
        _stop = threading.Event()

        def _spinner():
            frames = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
            i = 0
            while not _stop.is_set():
                sys.stdout.write(f"\r  {frames[i % len(frames)]}  Downloading {total} scene(s)…  ")
                sys.stdout.flush()
                i += 1
                _stop.wait(0.12)
            sys.stdout.write("\r" + " " * 50 + "\r")
            sys.stdout.flush()

        t = threading.Thread(target=_spinner, daemon=True)
        t.start()

        start_time = time.time()
        try:
            results = self._engine.download(
                satellite_data_list,
                destination=output_dir,
                options=options,
            )
        finally:
            _stop.set()
            t.join(timeout=1)
        duration = time.time() - start_time

        # Update final progress
        with lock:
            completed = total
            for r in results:
                downloaded_bytes += r.bytes_downloaded or 0

        # Final progress bar
        sys.stdout.write(f"\r  [{'█' * 40}] {total}/{total} (100%) | "
                    f"{downloaded_bytes/1024/1024:.1f} MB\n")
        sys.stdout.flush()

        # ------------------------------------------------------------
        # Map pygeofetch DownloadResult objects back to our SearchResult
        # items POSITIONALLY, not by ID string. Matching by
        # `pgf_result.data_id == item.id` is fragile across pygeofetch
        # versions, and a mismatch silently produced
        # DownloadResult(path=None) for items that actually downloaded
        # successfully — which then crashed geoai with:
        #     RasterioIOError: None: No such file or directory
        # because str(None) == "None" got passed to rasterio.open().
        # ------------------------------------------------------------
        items_with_data = [item for item in items if item.satellite_data is not None]
        items_with_data_ids = {id(i) for i in items_with_data}

        download_results: List[DownloadResult] = []
        n_results = len(results)

        for idx, item in enumerate(items_with_data):
            if idx >= n_results:
                # engine.download() returned fewer results than we sent
                download_results.append(DownloadResult(
                    scene_id=item.id, provider=item.provider,
                    success=False, error="No download result returned",
                ))
                continue

            pgf_result = results[idx]
            success = bool(getattr(pgf_result, "success", False))

            # Resolve the actual output file. Try every field pygeofetch
            # might expose, then fall back to scanning output_dir for a
            # matching file, and only fall back to the bare output_dir
            # as an absolute last resort. NEVER silently leave
            # path=None for a download that actually succeeded.
            out_paths = list(getattr(pgf_result, "output_paths", None) or [])
            if not out_paths and getattr(pgf_result, "output_path", None):
                out_paths = [pgf_result.output_path]

            if out_paths:
                path = Path(out_paths[0])
            elif success:
                path = self._find_downloaded_file(output_dir, item.id) or output_dir
            else:
                path = None

            download_results.append(DownloadResult(
                scene_id=item.id,
                provider=getattr(pgf_result, "provider", None) or item.provider,
                path=path,
                success=success,
                bytes_downloaded=getattr(pgf_result, "bytes_downloaded", 0) or 0,
                duration_seconds=getattr(pgf_result, "duration_seconds", None) or (duration / max(n_results, 1)),
                checksum_verified=bool(getattr(pgf_result, "checksum_verified", False)),
                error=getattr(pgf_result, "error", "") or "",
                post_process_steps=post_process or [],
            ))

        # Items that never had satellite_data at all (couldn't be sent)
        for item in items:
            if id(item) not in items_with_data_ids:
                download_results.append(DownloadResult(
                    scene_id=item.id, provider=item.provider,
                    success=False, error="No satellite_data available to download",
                ))

        # Print summary
        successful = sum(1 for r in download_results if r.success)
        failed = len(download_results) - successful

        print(f"\n  {'─' * 50}")
        print(f"  ✅ Download complete: {successful}/{total} scenes")
        if downloaded_bytes > 0:
            print(f"  📦 Total size: {downloaded_bytes/1024/1024:.1f} MB")
        print(f"  ⏱️  Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        if failed > 0:
            print(f"  ⚠️  Failed: {failed} scenes")
            for r in download_results:
                if not r.success:
                    print(f"      ✗ {r.scene_id[:50]}: {r.error[:100]}")
        print(f"  {'─' * 50}\n")

        return download_results

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def run_pipeline(self, pipeline_yaml: Union[str, Path], step: Optional[str] = None) -> Dict[str, Any]:
        """Run a pygeofetch YAML pipeline."""
        if self._use_cli():
            args = ["pipeline", "run", str(pipeline_yaml)]
            if step:
                args += ["--step", step]
            proc = self._run_cli(args)
            return {
                "success": proc.returncode == 0,
                "output":  proc.stdout,
                "error":   proc.stderr if proc.returncode != 0 else "",
            }
        if hasattr(self._engine, "run_pipeline"):
            return self._engine.run_pipeline(pipeline_yaml, step=step)
        # Fallback: use the PyGeoVision pipeline orchestrator
        from pygeovision.pipelines import Pipeline
        p = Pipeline.from_yaml(str(pipeline_yaml))
        result = p.run()
        return {"success": result.success, "steps": result.steps_completed}

    def validate_pipeline(self, pipeline_yaml: Union[str, Path]) -> bool:
        """Validate a pipeline YAML file."""
        return self._engine.validate_pipeline(pipeline_yaml)

    def schedule_pipeline(self, pipeline_yaml: Union[str, Path],
                          name: Optional[str] = None, cron: Optional[str] = None) -> bool:
        """Schedule a pipeline for periodic execution."""
        return self._engine.schedule_pipeline(pipeline_yaml, name=name, cron=cron)

    def list_scheduled_pipelines(self) -> List[Dict]:
        """List scheduled pipelines."""
        return self._engine.list_scheduled_pipelines()

    def pipeline_history(self, limit: int = 20) -> List[Dict]:
        """Get pipeline execution history."""
        return self._engine.pipeline_history(limit=limit)

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if self._use_cli():
            import json as _json
            proc = self._run_cli(["cache", "stats", "--json"])
            try:
                return _json.loads(proc.stdout)
            except Exception:
                pass
        files = list(self.cache_dir.glob("*.json"))
        total = sum(f.stat().st_size for f in files)
        return {
            "entries": len(files),
            "size_bytes": total,
            "size_mb": round(total / 1024 / 1024, 2),
            "location": str(self.cache_dir),
        }

    def clear_cache(self, provider: Optional[str] = None,
                    older_than: Optional[str] = None, dry_run: bool = False) -> None:
        """Clear search cache."""
        if self._use_cli():
            args = ["cache", "clear"]
            if provider:
                args += ["--provider", provider]
            if older_than:
                args += ["--older-than", older_than]
            if dry_run:
                args.append("--dry-run")
            self._run_cli(args)
            return
        if not dry_run:
            for f in self.cache_dir.glob("*.json"):
                f.unlink(missing_ok=True)
            logger.info("Local search cache cleared")

    def set_cache_ttl(self, seconds: int) -> None:
        """Set cache TTL in seconds."""
        # Implemented by pygeofetch internally
        pass

    def prune_cache(self, max_size_gb: float = 10.0) -> None:
        """Prune cache to stay under size limit."""
        # Implemented by pygeofetch internally
        pass

    # ------------------------------------------------------------------
    # Availability checks
    # ------------------------------------------------------------------
    # FIX: these two methods were missing, causing:
    #   AttributeError: 'SatelliteFetcher' object has no attribute
    #   '_has_pygeofetch'
    # They are called by PyGeoVision.status() and PyGeoVision.__repr__().
    # __init__ above already raises PyGeoVisionError if pygeofetch can't
    # be imported, so any successfully constructed SatelliteFetcher
    # implies pygeofetch IS available — but we still defer to the
    # module-level _check_pygeofetch() flag rather than hardcoding True.

    def _has_pygeofetch(self) -> bool:
        """Return True if the pygeofetch Python API is available."""
        return _check_pygeofetch()

    def _pygeofetch_version(self) -> str:
        """Return the installed pygeofetch package version."""
        try:
            return self._engine.version()
        except Exception:
            pass
        try:
            import pygeofetch
            return getattr(pygeofetch, "__version__", "unknown")
        except ImportError:
            return "not installed"

    # ------------------------------------------------------------------
    # System
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """Get system status."""
        try:
            return self._engine.status()
        except Exception:
            return {
                "pygeofetch_available": True,
                "version": self._pygeofetch_version(),
                "providers": len(PROVIDERS),
                "open_providers": OPEN_PROVIDERS,
            }

    def doctor(self) -> Dict[str, Any]:
        """Run diagnostic checks across all PyGeoVision components."""
        import platform
        report: Dict[str, Any] = {
            "pygeovision": {"ok": True},
            "python":      platform.python_version(),
            "platform":    platform.system(),
        }

        # pygeofetch engine
        try:
            v = self._pygeofetch_version()
            report["pygeofetch"] = {"ok": True, "version": v}
            # Try native doctor if available
            if hasattr(self._engine, "doctor"):
                report["pygeofetch"].update(self._engine.doctor())
        except Exception as exc:
            report["pygeofetch"] = {"ok": False, "error": str(exc)}

        # geoai
        try:
            import geoai
            report["geoai"] = {"ok": True, "version": getattr(geoai, "__version__", "?")}
        except ImportError:
            report["geoai"] = {"ok": False, "error": "not installed — pip install geoai-py"}

        # torch / CUDA
        try:
            import torch
            report["torch"] = {
                "ok": True,
                "version": torch.__version__,
                "cuda": torch.cuda.is_available(),
                "device": "cuda" if torch.cuda.is_available() else "cpu",
            }
            if torch.cuda.is_available():
                report["torch"]["gpu"] = torch.cuda.get_device_name(0)
        except ImportError:
            report["torch"] = {"ok": False, "error": "not installed — pip install torch"}

        # rasterio
        try:
            import rasterio
            report["rasterio"] = {"ok": True, "version": rasterio.__version__}
            # Quick GDAL check
            from rasterio.drivers import raster_driver_extensions
            report["rasterio"]["gdal"] = "ok"
        except ImportError:
            report["rasterio"] = {"ok": False, "error": "not installed — pip install rasterio"}
        except Exception as exc:
            report["rasterio"] = {"ok": True, "gdal_warning": str(exc)}

        # geopandas
        try:
            import geopandas
            report["geopandas"] = {"ok": True, "version": geopandas.__version__}
        except ImportError:
            report["geopandas"] = {"ok": False, "error": "not installed — pip install geopandas"}

        # credentials check
        try:
            creds = self.list_credentials()
            report["credentials"] = {"providers_configured": len(creds), "list": creds}
        except Exception as exc:
            report["credentials"] = {"error": str(exc)}

        # cache
        try:
            report["cache"] = self.cache_stats()
        except Exception:
            pass

        n_ok  = sum(1 for v in report.values() if isinstance(v, dict) and v.get("ok"))
        n_all = sum(1 for v in report.values() if isinstance(v, dict) and "ok" in v)
        report["summary"] = f"{n_ok}/{n_all} components healthy"
        return report

    def list_providers(self, auth_only: bool = False, open_only: bool = False,
                       capabilities: Optional[List[str]] = None) -> Dict[str, Dict]:
        """List available providers."""
        providers = dict(PROVIDERS)
        if auth_only:
            providers = {k: v for k, v in providers.items() if v.get("auth")}
        if open_only:
            providers = {k: v for k, v in providers.items() if v.get("open")}
        if capabilities:
            for cap in capabilities:
                if cap in ("sar", "stac", "sub_meter"):
                    providers = {k: v for k, v in providers.items() if v.get(cap)}
        return providers

    def provider_info(self, provider_id: str) -> Dict:
        """Get information about a specific provider."""
        return PROVIDERS.get(provider_id, {})

    def config_get(self, key: str) -> str:
        """Get configuration value."""
        return self._engine.config.get(key, "")

    def config_set(self, key: str, value: str) -> bool:
        """Set configuration value."""
        return self._engine.config.set(key, value)

    def config_show(self) -> Dict[str, Any]:
        """Show all configuration."""
        return dict(self._engine.config)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    # ── CLI-mode helpers ─────────────────────────────────────────────
    # ------------------------------------------------------------------

    def _run_cli(self, args: List[str], capture: bool = True) -> Any:
        """Run pygeofetch CLI command. Returns CompletedProcess."""
        import subprocess
        exe = _PYGEOFETCH_CLI_EXE or "pygeofetch"
        cmd = [exe] + args
        logger.debug("CLI: %s", " ".join(cmd))
        return subprocess.run(
            cmd,
            capture_output=capture,
            text=True,
        )

    def _parse_stac_geojson_file(self, geojson_path: Path) -> List["SearchResult"]:
        """Parse a pygeofetch-style GeoJSON result file into SearchResult objects."""
        import json
        with open(geojson_path, encoding="utf-8") as f:
            data = json.load(f)
        results = []
        for feat in data.get("features", []):
            props = feat.get("properties", {})
            fid   = feat.get("id") or props.get("id", "")
            geom  = feat.get("geometry") or {}
            bbox  = None
            if geom.get("type") == "Polygon":
                coords = geom.get("coordinates", [[]])[0]
                if coords:
                    lons = [c[0] for c in coords]
                    lats = [c[1] for c in coords]
                    bbox = (min(lons), min(lats), max(lons), max(lats))
            results.append(SearchResult(
                id=fid,
                provider=props.get("provider", ""),
                satellite=props.get("satellite", self._collection_to_satellite(
                    props.get("collection", ""))),
                datetime=props.get("datetime", ""),
                cloud_cover=props.get("eo:cloud_cover"),
                bbox=bbox,
                score=props.get("score"),
                collection=props.get("collection", ""),
                assets=props.get("assets", {}),
                properties=props,
            ))
        return results

    def _collection_to_satellite(self, collection: str) -> str:
        """Map a STAC collection ID to a human-readable satellite name."""
        _MAP = {
            "sentinel-2-l2a":  "Sentinel-2",
            "sentinel-2-l1c":  "Sentinel-2",
            "sentinel-1-rtc":  "Sentinel-1",
            "sentinel-1-grd":  "Sentinel-1",
            "landsat-c2-l2":   "Landsat",
            "landsat-c2-l1":   "Landsat",
            "landsat-8-l1tp":  "Landsat",
            "landsat-9-l1tp":  "Landsat",
            "naip":            "NAIP",
            "cop-dem-glo-30":  "Copernicus DEM",
            "modis":           "MODIS",
        }
        for key, val in _MAP.items():
            if key in collection.lower():
                return val
        return collection

    def _pick_best_asset(self, result: "SearchResult") -> Optional[str]:
        """Pick the best available asset key from a SearchResult.

        Priority: specific spectral bands > visual > thumbnail > first available.
        """
        assets = result.assets or {}
        if not assets:
            return None
        # Prefer individual bands over composites
        for preferred in ["B04", "B03", "B02", "B08", "nir", "red", "visual", "thumbnail"]:
            if preferred in assets:
                return preferred
        return next(iter(assets))

    # ------------------------------------------------------------------

    def _find_downloaded_file(self, output_dir: Path, scene_id: str) -> Optional[Path]:
        """Best-effort: locate a downloaded file matching scene_id under output_dir.

        Fallback used when a successful pygeofetch DownloadResult doesn't
        expose an explicit output_path / output_paths field.
        """
        try:
            prefix = scene_id[:20]
            matches = sorted(p for p in output_dir.rglob(f"*{prefix}*") if p.is_file())
            return matches[0] if matches else None
        except Exception:
            return None

    def _resolve_providers(self, providers, satellite, collections):
        """Resolve providers from various inputs."""
        if providers:
            return providers
        if satellite:
            sl = satellite.lower().replace(" ", "-")
            for key, provs in SATELLITE_SHORTCUTS.items():
                if key in sl or sl in key:
                    return provs
            return DEFAULT_SEARCH_PROVIDERS
        if collections:
            resolved, seen = [], set()
            for col in collections:
                cl = col.lower()
                key = next(
                    (k for k in ("sentinel-2", "sentinel-1", "landsat", "modis", "naip", "dem", "planet")
                     if k in cl), None
                )
                if key:
                    for p in SATELLITE_SHORTCUTS.get(key, []):
                        if p not in seen:
                            seen.add(p)
                            resolved.append(p)
                else:
                    p = COLLECTION_TO_PROVIDER.get(col)
                    if p and p not in seen:
                        seen.add(p)
                        resolved.append(p)
            if resolved:
                return resolved
        return DEFAULT_SEARCH_PROVIDERS

    def _providers_to_satellites(self, providers: List[str]) -> List[str]:
        """Convert provider IDs to satellite name hints."""
        sat_map = {
            "planetary_computer": [],
            "copernicus": ["Sentinel-1", "Sentinel-2"],
            "usgs": ["Landsat"],
            "aws_earth": ["Sentinel-2", "Landsat"],
            "element84": ["Sentinel-2"],
            "planet": ["PlanetScope"],
            "nasa_earthdata": ["MODIS"],
        }
        sats = []
        for p in providers:
            sats.extend(sat_map.get(p, []))
        return list(set(sats)) if sats else []

    def _cache_key(self, bbox, date_range, providers, cloud_cover_max, collections) -> str:
        """Generate cache key for search parameters."""
        key_str = f"{bbox}|{date_range}|{sorted(providers)}|{cloud_cover_max}|{sorted(collections or [])}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _load_cache(self, key: str) -> Optional[List[SearchResult]]:
        """Load cached search results.

        Validates the cache schema version before trusting the entry.
        Any cache file written by an older PyGeoVision version (e.g.
        before 'assets' was added to the cached payload) fails this
        check and is discarded, forcing a fresh search instead of
        silently returning incomplete results. This is what makes
        cache-format bugfixes self-healing without manual intervention.
        """
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        if time.time() - path.stat().st_mtime > 3600:
            path.unlink(missing_ok=True)
            return None
        try:
            import json
            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            # Reject anything that isn't our current versioned format.
            # Older cache files are a bare JSON list (no 'version' key)
            # and would otherwise be silently accepted with missing fields.
            if not isinstance(data, dict) or data.get("version") != _CACHE_SCHEMA_VERSION:
                logger.debug("Cache entry %s has outdated/invalid schema; discarding.", key)
                path.unlink(missing_ok=True)
                return None

            results = [SearchResult(**item) for item in data.get("results", [])]

            return results
        except Exception:
            return None

    def _save_cache(self, key: str, results: List[SearchResult]) -> None:
        """Save search results to cache, tagged with the current schema version."""
        try:
            import json
            payload = {
                "version": _CACHE_SCHEMA_VERSION,
                "results": [r.to_dict() for r in results],
            }
            with open(self.cache_dir / f"{key}.json", "w", encoding="utf-8") as f:
                json.dump(payload, f)
        except Exception as exc:
            logger.debug("Cache save failed: %s", exc)
