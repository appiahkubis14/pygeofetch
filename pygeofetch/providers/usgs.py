"""
USGS Earth Explorer (EarthExplorer) M2M API provider.

Provides access to Landsat, MODIS, and other USGS datasets via the
Machine-to-Machine (M2M) API v1.5.

Authentication:
    Uses username + password (EarthExplorer account).
    Tokens are cached and refreshed automatically.

Supported datasets include:
    - Landsat Collection 2 (Level-1 and Level-2)
    - MODIS Terra/Aqua
    - EO-1 ALI/Hyperion
    - Many more via dataset search

Example::

    from pygeofetch.providers.usgs import USGSProvider
    from pygeofetch.models.user_auth import Credentials, AuthType

    provider = USGSProvider()
    creds = Credentials(
        provider="usgs",
        auth_type=AuthType.USERNAME_PASSWORD,
        username="myuser",
        password="mypassword",
    )
    session = provider.authenticate(creds)

    from pygeofetch.models.search_query import SearchQuery
    results = provider.search(SearchQuery(
        bbox=(-74.1, 40.6, -73.7, 40.9),
        start_date="2024-01-01",
        end_date="2024-06-01",
    ))
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from pygeofetch.models.download_task import DownloadOptions, DownloadResult, DownloadStatus
from pygeofetch.models.satellite_data import (
    DataFormat,
    ProcessingLevel,
    ProviderCapabilities,
    QuotaInfo,
    SatelliteAsset,
    SatelliteData,
)
from pygeofetch.models.search_query import SearchQuery
from pygeofetch.models.user_auth import AuthSession, Credentials
from pygeofetch.providers.base import (
    AbstractBaseProvider,
    AuthenticationError,
    DownloadError,
    ProviderError,
    SearchError,
)
from pygeofetch.utils.retry_handler import RetryConfig, retry_on_failure


class USGSProvider(AbstractBaseProvider):
    """
    USGS Earth Explorer M2M API provider.

    Provides search and download access to the full USGS EarthExplorer
    catalog including all Landsat generations, MODIS, and specialty datasets.

    Attributes:
        PROVIDER_ID: 'usgs'
        DISPLAY_NAME: 'USGS Earth Explorer'
        REQUIRES_AUTH: True
        BASE_URL: M2M API endpoint
    """

    PROVIDER_ID = "usgs"
    DISPLAY_NAME = "USGS Earth Explorer"
    REQUIRES_AUTH = True
    DESCRIPTION = (
        "Access to Landsat, MODIS, and 500+ other datasets via "
        "the USGS Machine-to-Machine (M2M) API."
    )
    DATA_TYPES = ["Landsat", "MODIS", "EO-1", "ASTER", "SRTM", "Sentinel-2"]
    BASE_URL = "https://m2m.cr.usgs.gov/api/api/json/stable"

    # Default dataset aliases
    DEFAULT_DATASETS = {
        "landsat8": "landsat_ot_c2_l2",
        "landsat9": "landsat_ot_c2_l2",
        "landsat7": "landsat_etm_c2_l2",
        "modis": "modis_09a1_v6",
    }

    def authenticate(self, credentials: Credentials) -> AuthSession:
        """
        Authenticate with the USGS M2M API.

        Args:
            credentials: Must contain username and password.

        Returns:
            AuthSession with API token.

        Raises:
            AuthenticationError: If credentials are invalid.
        """
        if not credentials.username or not credentials.get_password():
            raise AuthenticationError("USGS requires username and password")

        try:
            import httpx
            payload = {
                "username": credentials.username,
                "password": credentials.get_password(),
            }
            response = httpx.post(
                f"{self.BASE_URL}/login",
                json=payload,
                timeout=30,
                headers={"User-Agent": "PyGeoFetch/0.1.0"},
            )
            if response.status_code != 200:
                self._handle_http_error(response)

            data = response.json()
            if data.get("errorCode"):
                raise AuthenticationError(
                    f"USGS login failed: {data.get('errorMessage', 'Unknown error')}"
                )

            token = data.get("data")
            if not token:
                raise AuthenticationError("USGS returned no API token")

            session = AuthSession(
                provider=self.PROVIDER_ID,
                access_token=token,
                expires_at=datetime.utcnow() + timedelta(hours=2),
            )
            self._session = session
            self._logger.info(f"Authenticated with USGS M2M API as {credentials.username!r}")
            return session

        except AuthenticationError:
            raise
        except Exception as exc:
            raise AuthenticationError(f"USGS authentication error: {exc}") from exc

    def validate_credentials(self, credentials: Credentials) -> bool:
        """Check if credentials are non-empty (does not make a network call)."""
        return bool(credentials.username and credentials.get_password())

    def search(self, query: SearchQuery) -> List[SatelliteData]:
        """
        Search the USGS EarthExplorer catalog.

        Args:
            query: Unified search query.

        Returns:
            List of matching SatelliteData records.

        Raises:
            SearchError: If search fails.
        """
        self.require_auth()

        datasets = self._resolve_datasets(query)
        results: List[SatelliteData] = []

        for dataset_name in datasets:
            try:
                scene_results = self._search_dataset(dataset_name, query)
                results.extend(scene_results)
                if len(results) >= query.max_results:
                    break
            except Exception as exc:
                self._logger.warning(f"Search failed for dataset {dataset_name!r}: {exc}")

        return results[: query.max_results]

    def _resolve_datasets(self, query: SearchQuery) -> List[str]:
        """Map query satellite names to USGS dataset codes."""
        if query.collections:
            return query.collections

        if not query.satellites:
            return ["landsat_ot_c2_l2", "landsat_etm_c2_l2"]

        datasets = set()
        for sat in query.satellites:
            sat_lower = sat.lower().replace("-", "").replace(" ", "")
            for key, code in self.DEFAULT_DATASETS.items():
                if key in sat_lower:
                    datasets.add(code)
                    break
            else:
                datasets.add(sat_lower)

        return list(datasets)

    def _search_dataset(self, dataset_name: str, query: SearchQuery) -> List[SatelliteData]:
        """Search a specific USGS dataset."""
        import httpx

        payload: Dict[str, Any] = {
            "datasetName": dataset_name,
            "maxResults": min(query.max_results, 100),
            "startingNumber": 1,
            "metadataType": "full",
        }

        if query.start_date or query.end_date:
            payload["temporalFilter"] = {}
            if query.start_date:
                payload["temporalFilter"]["start"] = str(query.start_date)
            if query.end_date:
                payload["temporalFilter"]["end"] = str(query.end_date)

        if query.bbox:
            bbox = query.bbox
            payload["spatialFilter"] = {
                "filterType": "mbr",
                "lowerLeft": {"latitude": bbox.min_lat, "longitude": bbox.min_lon},
                "upperRight": {"latitude": bbox.max_lat, "longitude": bbox.max_lon},
            }

        if query.cloud_cover_max < 100:
            payload["cloudCoverFilter"] = {
                "min": int(query.cloud_cover_min),
                "max": int(query.cloud_cover_max),
                "includeUnknown": False,
            }

        response = httpx.post(
            f"{self.BASE_URL}/scene-search",
            json={"datasetName": dataset_name, **payload},
            headers={"X-Auth-Token": self._session.access_token},  # type: ignore
            timeout=60,
        )
        self._handle_http_error(response)

        data = response.json()
        if data.get("errorCode"):
            raise SearchError(f"USGS search error: {data.get('errorMessage')}")

        scenes = (data.get("data") or {}).get("results", [])
        return [self._scene_to_satellite_data(s, dataset_name) for s in scenes]

    def _scene_to_satellite_data(self, scene: Dict, dataset_name: str) -> SatelliteData:
        """Convert a USGS scene dict to SatelliteData."""
        scene_id = scene.get("entityId", scene.get("displayId", "unknown"))
        display_id = scene.get("displayId", scene_id)

        # Parse spatial bounds
        bbox = None
        spatial = scene.get("spatialBounds") or scene.get("spatialCoverage", {})
        if spatial and spatial.get("type") == "Polygon":
            coords = spatial.get("coordinates", [[]])[0]
            if coords:
                lons = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                bbox = (min(lons), min(lats), max(lons), max(lats))

        # Parse acquisition date
        dt = None
        date_str = scene.get("acquisitionDate") or scene.get("startingDate")
        if date_str:
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                pass

        # Identify satellite from dataset name
        satellite = "Landsat"
        if "ot_c2" in dataset_name:
            satellite = "Landsat 8/9"
        elif "etm" in dataset_name:
            satellite = "Landsat 7"
        elif "modis" in dataset_name:
            satellite = "MODIS"

        cloud_cover = scene.get("cloudCover")
        if cloud_cover is not None:
            try:
                cloud_cover = float(cloud_cover)
            except (ValueError, TypeError):
                cloud_cover = None

        # Build assets from browse/download links
        assets: Dict[str, SatelliteAsset] = {}
        browse_list = scene.get("browse", [])
        if browse_list:
            for i, b in enumerate(browse_list):
                assets[f"thumbnail_{i}"] = SatelliteAsset(
                    key=f"thumbnail_{i}",
                    href=b.get("browseUrl", ""),
                    title=b.get("browseName", "Thumbnail"),
                    roles=["thumbnail"],
                )

        return SatelliteData(
            id=scene_id,
            provider=self.PROVIDER_ID,
            collection=dataset_name,
            satellite=satellite,
            datetime=dt,
            bbox=bbox,
            cloud_cover=cloud_cover,
            processing_level=ProcessingLevel.L2,
            assets=assets,
            properties={
                "display_id": display_id,
                "dataset": dataset_name,
                **{k: v for k, v in scene.items() if k not in ("browse",)},
            },
        )

    def download(
        self,
        data: SatelliteData,
        destination: Path,
        options: DownloadOptions,
    ) -> DownloadResult:
        """
        Download a USGS scene using the M2M download API.

        Args:
            data: SatelliteData scene to download.
            destination: Output directory.
            options: Download configuration.

        Returns:
            DownloadResult.
        """
        self.require_auth()
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)

        import httpx

        try:
            # Request download URLs
            dataset = data.properties.get("dataset", "landsat_ot_c2_l2")
            payload = {
                "downloads": [{"entityId": data.id, "productId": "5e83d14fb9436d88"}],
                "downloadApplication": "EE",
            }
            response = httpx.post(
                f"{self.BASE_URL}/download-request",
                json=payload,
                headers={"X-Auth-Token": self._session.access_token},  # type: ignore
                timeout=60,
            )
            self._handle_http_error(response)
            download_data = response.json().get("data", {})
            downloads = download_data.get("availableDownloads", [])

            if not downloads:
                return DownloadResult(
                    status=DownloadStatus.FAILED,
                    data_id=data.id,
                    provider=self.PROVIDER_ID,
                    error="No download URLs available for this scene",
                )

            start_time = time.time()
            output_paths = []

            for dl in downloads[:1]:  # Download first available
                url = dl.get("url")
                if not url:
                    continue
                filename = url.split("/")[-1].split("?")[0] or f"{data.id}.tar"
                output_file = destination / filename

                self._logger.info(f"Downloading {filename} from USGS...")
                with httpx.stream("GET", url, timeout=options.timeout_seconds, follow_redirects=True) as resp:
                    self._handle_http_error(resp)
                    with open(output_file, "wb") as f:
                        for chunk in resp.iter_bytes(chunk_size=int(options.chunk_size_mb * 1024 * 1024)):
                            f.write(chunk)
                output_paths.append(output_file)

            duration = time.time() - start_time
            total_bytes = sum(p.stat().st_size for p in output_paths if p.exists())

            return DownloadResult(
                status=DownloadStatus.COMPLETED,
                data_id=data.id,
                provider=self.PROVIDER_ID,
                output_path=output_paths[0] if output_paths else None,
                output_paths=output_paths,
                bytes_downloaded=total_bytes,
                duration_seconds=duration,
            )

        except Exception as exc:
            return DownloadResult(
                status=DownloadStatus.FAILED,
                data_id=data.id,
                provider=self.PROVIDER_ID,
                error=str(exc),
                error_type=type(exc).__name__,
            )

    def get_capabilities(self) -> ProviderCapabilities:
        """Return USGS provider capabilities."""
        return ProviderCapabilities(
            provider_id="usgs",
            name="USGS Earth Explorer",
            search=True,
            download=True,
            streaming=False,
            stac=False,
            requires_auth=True,
            supported_satellites=["Landsat 4", "Landsat 5", "Landsat 7", "Landsat 8", "Landsat 9", "MODIS", "EO-1"],
            supported_formats=[DataFormat.GEOTIFF, DataFormat.HDF4, DataFormat.TAR],
            supports_cloud_filter=True,
            supports_date_filter=True,
            supports_aoi_filter=True,
        )

    def get_quota_info(self) -> QuotaInfo:
        """Return USGS quota information (no hard quota for most users)."""
        self.require_auth()
        return QuotaInfo(
            provider=self.PROVIDER_ID,
            extra_info={"note": "USGS does not impose download quotas for most datasets"},
        )

    def list_datasets(self) -> List[Dict[str, Any]]:
        """
        List available USGS datasets.

        Returns:
            List of dataset info dicts.
        """
        self.require_auth()
        import httpx
        response = httpx.post(
            f"{self.BASE_URL}/dataset-search",
            json={},
            headers={"X-Auth-Token": self._session.access_token},  # type: ignore
            timeout=30,
        )
        self._handle_http_error(response)
        return response.json().get("data", [])
