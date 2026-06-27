"""
Copernicus Data Space Ecosystem (CDSE) provider.

Provides access to Sentinel-1, Sentinel-2, Sentinel-3, Sentinel-5P, and
other Copernicus missions via the ESA CDSE OData API and STAC catalog.

Authentication:
    Uses Copernicus Data Space account credentials with OAuth2 token exchange.
    Free registration at https://dataspace.copernicus.eu

Supported datasets:
    - Sentinel-1 (SAR C-band)
    - Sentinel-2 (Multispectral, 10-60m)
    - Sentinel-3 (Ocean/Land colour, OLCI/SLSTR)
    - Sentinel-5P (Atmospheric composition)
    - Sentinel-6 (Sea level)

Example::

    from pygeofetch.providers.copernicus import CopernicusProvider
    from pygeofetch.models.user_auth import Credentials, AuthType

    provider = CopernicusProvider()
    session = provider.authenticate(Credentials(
        provider="copernicus",
        auth_type=AuthType.USERNAME_PASSWORD,
        username="user@example.com",
        password="password",
    ))

    results = provider.search(SearchQuery(
        bbox=(-74.1, 40.6, -73.7, 40.9),
        start_date="2024-01-01",
        end_date="2024-03-01",
        satellites=["Sentinel-2"],
        cloud_cover_max=15,
    ))
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

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
    SearchError,
)


class CopernicusProvider(AbstractBaseProvider):
    """
    Copernicus Data Space Ecosystem provider.

    Supports search across all Sentinel missions and download via
    S3 presigned URLs or direct HTTPS.

    Attributes:
        PROVIDER_ID: 'copernicus'
        DISPLAY_NAME: 'Copernicus Data Space'
        REQUIRES_AUTH: True
    """

    PROVIDER_ID = "copernicus"
    DISPLAY_NAME = "Copernicus Data Space"
    REQUIRES_AUTH = True
    DESCRIPTION = (
        "Access to all Sentinel satellite missions (Sentinel-1 through 6) "
        "via ESA's Copernicus Data Space Ecosystem."
    )
    DATA_TYPES = ["Sentinel-1", "Sentinel-2", "Sentinel-3", "Sentinel-5P", "Sentinel-6"]
    BASE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1"
    AUTH_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    S3_ENDPOINT = "https://eodata.dataspace.copernicus.eu"

    # Map user-friendly satellite names to CDSE collection names
    SATELLITE_COLLECTION_MAP = {
        "sentinel-1": "SENTINEL-1",
        "sentinel-2": "SENTINEL-2",
        "sentinel-3": "SENTINEL-3",
        "sentinel-5": "SENTINEL-5P",
        "sentinel-5p": "SENTINEL-5P",
        "sentinel-6": "SENTINEL-6",
    }

    def authenticate(self, credentials: Credentials) -> AuthSession:
        """
        Obtain an OAuth2 access token from Copernicus Identity Service.

        Args:
            credentials: Must contain username and password.

        Returns:
            AuthSession with access and refresh tokens.

        Raises:
            AuthenticationError: If login fails.
        """
        if not credentials.username or not credentials.get_password():
            raise AuthenticationError("Copernicus requires username (email) and password")

        try:
            import httpx
            response = httpx.post(
                self.AUTH_URL,
                data={
                    "grant_type": "password",
                    "username": credentials.username,
                    "password": credentials.get_password(),
                    "client_id": "cdse-public",
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=30,
            )
            if response.status_code != 200:
                try:
                    err = response.json().get("error_description", response.text[:200])
                except Exception:
                    err = response.text[:200]
                raise AuthenticationError(f"Copernicus login failed: {err}")

            token_data = response.json()
            session = AuthSession(
                provider=self.PROVIDER_ID,
                access_token=token_data["access_token"],
                refresh_token=token_data.get("refresh_token"),
                expires_at=datetime.utcnow() + timedelta(seconds=token_data.get("expires_in", 600)),
                session_data={"username": credentials.username},
            )
            self._session = session
            self._logger.info(f"Authenticated with Copernicus Data Space as {credentials.username!r}")
            return session

        except AuthenticationError:
            raise
        except Exception as exc:
            raise AuthenticationError(f"Copernicus auth error: {exc}") from exc

    def _refresh_token_if_needed(self) -> None:
        """Refresh the access token if it expires soon."""
        if not self._session:
            return
        mins = self._session.minutes_until_expiry
        if mins is not None and mins < 5 and self._session.refresh_token:
            try:
                import httpx
                response = httpx.post(
                    self.AUTH_URL,
                    data={
                        "grant_type": "refresh_token",
                        "refresh_token": self._session.refresh_token,
                        "client_id": "cdse-public",
                    },
                    timeout=30,
                )
                if response.status_code == 200:
                    token_data = response.json()
                    self._session.access_token = token_data["access_token"]
                    self._session.refresh_token = token_data.get("refresh_token", self._session.refresh_token)
                    self._session.expires_at = datetime.utcnow() + timedelta(
                        seconds=token_data.get("expires_in", 600)
                    )
                    self._logger.debug("Copernicus token refreshed")
            except Exception as exc:
                self._logger.warning(f"Token refresh failed: {exc}")

    def validate_credentials(self, credentials: Credentials) -> bool:
        return bool(credentials.username and credentials.get_password())

    def search(self, query: SearchQuery) -> List[SatelliteData]:
        """
        Search the Copernicus OData catalog.

        Args:
            query: Search parameters.

        Returns:
            List of matching SatelliteData records.
        """
        self.require_auth()
        self._refresh_token_if_needed()

        collections = self._resolve_collections(query)
        results: List[SatelliteData] = []

        for collection in collections:
            try:
                items = self._search_collection(collection, query)
                results.extend(items)
                if len(results) >= query.max_results:
                    break
            except Exception as exc:
                self._logger.warning(f"Search failed for collection {collection!r}: {exc}")

        return results[: query.max_results]

    def _resolve_collections(self, query: SearchQuery) -> List[str]:
        """Map query satellite names to Copernicus collection codes."""
        if query.collections:
            return [c.upper() for c in query.collections]

        if not query.satellites:
            return ["SENTINEL-2"]  # sensible default

        collections = set()
        for sat in query.satellites:
            sat_lower = sat.lower()
            for key, col in self.SATELLITE_COLLECTION_MAP.items():
                if key in sat_lower:
                    collections.add(col)
                    break
            else:
                collections.add(sat.upper())

        return list(collections)

    def _build_odata_filter(self, collection: str, query: SearchQuery) -> str:
        """Build OData $filter string for the given collection and query."""
        filters = [f"Collection/Name eq '{collection}'"]

        if query.start_date:
            filters.append(f"ContentDate/Start ge {query.start_date}T00:00:00.000Z")
        if query.end_date:
            filters.append(f"ContentDate/Start le {query.end_date}T23:59:59.000Z")

        if query.bbox:
            bbox = query.bbox
            polygon = (
                f"POLYGON(("
                f"{bbox.min_lon} {bbox.min_lat},"
                f"{bbox.max_lon} {bbox.min_lat},"
                f"{bbox.max_lon} {bbox.max_lat},"
                f"{bbox.min_lon} {bbox.max_lat},"
                f"{bbox.min_lon} {bbox.min_lat}))"
            )
            filters.append(f"OData.CSC.Intersects(area=geography'SRID=4326;{polygon}')")

        if query.cloud_cover_max < 100:
            filters.append(f"Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value le {query.cloud_cover_max})")
        if query.cloud_cover_min > 0:
            filters.append(f"Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value ge {query.cloud_cover_min})")

        # Processing level filter
        if query.processing_levels:
            level_filter = " or ".join(
                f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'processingLevel' and att/OData.CSC.StringAttribute/Value eq '{level}')"
                for level in query.processing_levels
            )
            filters.append(f"({level_filter})")

        return " and ".join(filters)

    def _search_collection(self, collection: str, query: SearchQuery) -> List[SatelliteData]:
        """Perform OData search for a specific collection."""
        import httpx

        params: Dict[str, Any] = {
            "$filter": self._build_odata_filter(collection, query),
            "$orderby": "ContentDate/Start desc",
            "$top": min(query.max_results, 1000),
            "$expand": "Attributes",
        }

        response = httpx.get(
            f"{self.BASE_URL}/Products",
            params=params,
            headers={"Authorization": f"Bearer {self._session.access_token}"},  # type: ignore
            timeout=60,
        )
        self._handle_http_error(response)
        data = response.json()
        return [self._product_to_satellite_data(p) for p in data.get("value", [])]

    def _product_to_satellite_data(self, product: Dict) -> SatelliteData:
        """Convert a Copernicus OData product dict to SatelliteData."""
        product_id = product.get("Id", "")
        name = product.get("Name", "")
        collection = product.get("S3Path", "").split("/")[2] if product.get("S3Path") else ""

        # Parse footprint
        bbox = None
        footprint = product.get("Footprint") or ""
        if "POLYGON" in footprint:
            try:
                coords_str = footprint.split("((")[1].split("))")[0]
                coords = [
                    (float(p.split()[0]), float(p.split()[1]))
                    for p in coords_str.split(",")
                ]
                lons = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                bbox = (min(lons), min(lats), max(lons), max(lats))
            except Exception:
                pass

        # Parse datetime
        dt = None
        date_str = product.get("ContentDate", {}).get("Start")
        if date_str:
            try:
                dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            except Exception:
                pass

        # Extract attributes
        attributes = {
            a.get("Name"): a.get("Value")
            for a in (product.get("Attributes") or [])
        }
        cloud_cover = attributes.get("cloudCover")
        if cloud_cover is not None:
            try:
                cloud_cover = float(cloud_cover)
            except (ValueError, TypeError):
                cloud_cover = None

        # Determine satellite from name
        satellite = "Sentinel"
        if name.startswith("S1"):
            satellite = "Sentinel-1"
        elif name.startswith("S2"):
            satellite = "Sentinel-2"
        elif name.startswith("S3"):
            satellite = "Sentinel-3"
        elif name.startswith("S5P"):
            satellite = "Sentinel-5P"

        # Build assets
        assets: Dict[str, SatelliteAsset] = {}
        s3_path = product.get("S3Path")
        if s3_path:
            assets["data"] = SatelliteAsset(
                key="data",
                href=f"{self.S3_ENDPOINT}{s3_path}",
                title=name,
                roles=["data"],
                size_bytes=product.get("ContentLength"),
            )

        download_href = f"{self.BASE_URL}/Products({product_id})/$value"
        assets["download"] = SatelliteAsset(
            key="download",
            href=download_href,
            title=f"Download {name}",
            roles=["data"],
            size_bytes=product.get("ContentLength"),
        )

        proc_level_str = attributes.get("processingLevel", "UNKNOWN")
        proc_level = ProcessingLevel.UNKNOWN
        for pl in ProcessingLevel:
            if pl.value == proc_level_str:
                proc_level = pl
                break

        return SatelliteData(
            id=product_id,
            provider=self.PROVIDER_ID,
            collection=collection or satellite.upper().replace("-", ""),
            satellite=satellite,
            datetime=dt,
            bbox=bbox,
            cloud_cover=cloud_cover,
            processing_level=proc_level,
            assets=assets,
            properties={
                "name": name,
                "s3_path": s3_path,
                **attributes,
            },
        )

    def download(
        self,
        data: SatelliteData,
        destination: Path,
        options: DownloadOptions,
    ) -> DownloadResult:
        """
        Download a Copernicus product via authenticated HTTPS.

        Args:
            data: SatelliteData to download.
            destination: Output directory.
            options: Download configuration.

        Returns:
            DownloadResult.
        """
        self.require_auth()
        self._refresh_token_if_needed()

        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)

        download_asset = data.assets.get("download") or data.assets.get("data")
        if not download_asset:
            return DownloadResult(
                status=DownloadStatus.FAILED,
                data_id=data.id,
                provider=self.PROVIDER_ID,
                error="No download URL available for this product",
            )

        name = data.properties.get("name", data.id)
        output_file = destination / f"{name}.zip"

        try:
            import httpx
            start_time = time.time()

            with httpx.stream(
                "GET",
                download_asset.href,
                headers={"Authorization": f"Bearer {self._session.access_token}"},  # type: ignore
                timeout=options.timeout_seconds,
                follow_redirects=True,
            ) as resp:
                self._handle_http_error(resp)
                with open(output_file, "wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=int(options.chunk_size_mb * 1024 * 1024)):
                        f.write(chunk)

            duration = time.time() - start_time
            file_size = output_file.stat().st_size

            return DownloadResult(
                status=DownloadStatus.COMPLETED,
                data_id=data.id,
                provider=self.PROVIDER_ID,
                output_path=output_file,
                output_paths=[output_file],
                bytes_downloaded=file_size,
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
        return ProviderCapabilities(
            provider_id="copernicus",
            name="Copernicus CDSE",
            search=True,
            download=True,
            streaming=True,
            stac=False,
            requires_auth=True,
            supported_satellites=["Sentinel-1", "Sentinel-2", "Sentinel-3", "Sentinel-5P", "Sentinel-6"],
            supported_formats=[DataFormat.SAFE, DataFormat.GEOTIFF, DataFormat.ZIP],
            supports_cloud_filter=True,
            supports_date_filter=True,
            supports_aoi_filter=True,
            has_quota=True,
        )

    def get_quota_info(self) -> QuotaInfo:
        self.require_auth()
        return QuotaInfo(
            provider=self.PROVIDER_ID,
            extra_info={
                "note": "Copernicus provides free access; rate limits may apply",
                "docs": "https://dataspace.copernicus.eu/terms-and-conditions",
            },
        )
