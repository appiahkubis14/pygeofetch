"""
NASA Earthdata Cloud provider for PyGeoFetch.

Provides direct S3 access to cloud-hosted NASA datasets. Uses Earthdata
Login (OAuth2) to obtain temporary AWS credentials, then accesses data
directly from S3 without downloading through NASA servers.

Supports cloud-hosted MODIS, VIIRS, ICESat-2, GEDI, ECCO, and many more.

Example::

    from pygeofetch.providers.nasa_earthdata_cloud import NASAEarthdataCloudProvider
    from pygeofetch.models.user_auth import Credentials

    provider = NASAEarthdataCloudProvider()
    creds = Credentials(provider="nasa_earthdata_cloud",
                        username="myuser", password="mypass")
    session = provider.authenticate(creds)
    results = provider.search(query)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from pygeofetch.models.download_task import DownloadOptions, DownloadResult, DownloadStatus
from pygeofetch.models.satellite_data import (
    DataFormat, ProviderCapabilities, QuotaInfo, SatelliteData,
)
from pygeofetch.models.search_query import SearchQuery
from pygeofetch.models.user_auth import AuthSession, Credentials
from pygeofetch.providers.base import (
    AbstractBaseProvider, AuthenticationError, SearchError, DownloadError,
)


class NASAEarthdataCloudProvider(AbstractBaseProvider):
    """
    NASA Earthdata Cloud provider — direct S3 access to NASA datasets.

    Uses the EDL (Earthdata Login) OAuth2 flow to obtain temporary AWS
    credentials. Data is then read directly from S3, avoiding bandwidth
    charges and latency vs. downloading through NASA servers.

    Attributes:
        PROVIDER_ID: 'nasa_earthdata_cloud'
        REQUIRES_AUTH: True
    """

    PROVIDER_ID = "nasa_earthdata_cloud"
    DISPLAY_NAME = "NASA Earthdata Cloud"
    REQUIRES_AUTH = True
    DESCRIPTION = (
        "Direct cloud access to petabyte-scale NASA Earth science data on AWS S3 "
        "via Earthdata Login. Supports MODIS, VIIRS, ICESat-2, GEDI, and more."
    )
    DATA_TYPES = ["MODIS", "VIIRS", "ICESat-2", "GEDI", "ECCO", "GPM", "MERRA-2"]
    SATELLITES = ["Terra", "Aqua", "CALIPSO", "ICESat-2", "GEDI"]
    BASE_URL = "https://cmr.earthdata.nasa.gov/search"
    EDL_URL = "https://urs.earthdata.nasa.gov"
    S3_CREDS_URL = "https://data.earthaccess.nasa.gov/s3credentials"

    def authenticate(self, credentials: Credentials) -> AuthSession:
        """
        Authenticate with NASA Earthdata Login and obtain S3 credentials.

        Args:
            credentials: Username/password for Earthdata Login.

        Returns:
            AuthSession containing S3 credentials in session_data.

        Raises:
            AuthenticationError: If Earthdata login fails.
        """
        import httpx

        if not credentials.username or not credentials.password:
            raise AuthenticationError(
                "NASA Earthdata Cloud requires username and password. "
                "Register at: https://urs.earthdata.nasa.gov"
            )

        # Step 1: EDL login to get bearer token
        try:
            resp = httpx.post(
                f"{self.EDL_URL}/api/users/tokens",
                auth=(credentials.username, credentials.password),
                timeout=30,
            )
            if resp.status_code == 401:
                raise AuthenticationError("Invalid Earthdata Login credentials.")
            if resp.status_code not in (200, 201):
                raise AuthenticationError(f"EDL auth failed: HTTP {resp.status_code}")

            token_data = resp.json()
            edl_token = token_data[0].get("access_token") if isinstance(token_data, list) else token_data.get("access_token")

        except AuthenticationError:
            raise
        except Exception as exc:
            raise AuthenticationError(f"EDL authentication error: {exc}") from exc

        # Step 2: Get temporary AWS credentials for S3 direct access
        s3_creds: Dict[str, str] = {}
        try:
            s3_resp = httpx.get(
                self.S3_CREDS_URL,
                headers={"Authorization": f"Bearer {edl_token}"},
                timeout=30,
            )
            if s3_resp.status_code == 200:
                creds_data = s3_resp.json()
                s3_creds = {
                    "aws_access_key_id": creds_data.get("accessKeyId", ""),
                    "aws_secret_access_key": creds_data.get("secretAccessKey", ""),
                    "aws_session_token": creds_data.get("sessionToken", ""),
                    "expiration": creds_data.get("expiration", ""),
                }
                self._logger.info("Obtained temporary S3 credentials for NASA Earthdata Cloud")
        except Exception as exc:
            self._logger.warning(f"Could not obtain S3 credentials (will fall back to HTTPS): {exc}")

        from datetime import datetime, timezone, timedelta
        session = AuthSession(
            provider=self.PROVIDER_ID,
            access_token=edl_token,
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            session_data={"s3_credentials": s3_creds, "username": credentials.username},
        )
        self._session = session
        self._logger.info(f"NASA Earthdata Cloud: authenticated as {credentials.username}")
        return session

    def validate_credentials(self, credentials: Credentials) -> bool:
        return bool(credentials.username and credentials.password)

    def search(self, query: SearchQuery) -> List[SatelliteData]:
        """
        Search NASA CMR for cloud-hosted granules.

        Args:
            query: Search parameters.

        Returns:
            List of SatelliteData with S3 URLs.
        """
        self.require_auth()
        import httpx

        params: Dict[str, Any] = {
            "cloud_hosted": "true",
            "page_size": min(query.max_results, 2000),
        }

        # Spatial filter
        if query.bbox:
            bb = query.bbox
            params["bounding_box"] = f"{bb.min_lon},{bb.min_lat},{bb.max_lon},{bb.max_lat}"

        # Temporal filter
        if query.start_date or query.end_date:
            start = str(query.start_date) + "T00:00:00Z" if query.start_date else ""
            end = str(query.end_date) + "T23:59:59Z" if query.end_date else ""
            params["temporal"] = f"{start},{end}"

        # Collection shortnames
        if query.satellites:
            params["short_name[]"] = query.satellites

        try:
            headers = {"Authorization": f"Bearer {self._session.access_token}"}
            resp = httpx.get(
                f"{self.BASE_URL}/granules.json",
                params=params,
                headers=headers,
                timeout=self.config.get("timeout", 60),
            )
            self._handle_http_error(resp)
            data = resp.json()
            entries = data.get("feed", {}).get("entry", [])
            results = [self._parse_granule(e) for e in entries]
            self._logger.info(f"NASA Earthdata Cloud: {len(results)} granules found")
            return results

        except Exception as exc:
            raise SearchError(f"NASA Earthdata Cloud search failed: {exc}") from exc

    def _parse_granule(self, entry: Dict[str, Any]) -> SatelliteData:
        """Convert a CMR granule entry to SatelliteData."""
        from pygeofetch.models.satellite_data import SatelliteAsset

        granule_id = entry.get("id", "")
        bbox_raw = entry.get("boxes", [""])[0].split() if entry.get("boxes") else []
        bbox = None
        if len(bbox_raw) == 4:
            try:
                bbox = (float(bbox_raw[1]), float(bbox_raw[0]),
                        float(bbox_raw[3]), float(bbox_raw[2]))
            except ValueError:
                pass

        assets: Dict[str, SatelliteAsset] = {}
        for link in entry.get("links", []):
            href = link.get("href", "")
            rel = link.get("rel", "")
            if "s3://" in href or href.endswith((".nc", ".h5", ".hdf", ".tif", ".zarr")):
                key = f"data_{len(assets)}"
                assets[key] = SatelliteAsset(
                    key=key, href=href,
                    roles=["data"],
                    media_type=link.get("type"),
                )

        return SatelliteData(
            id=granule_id,
            provider=self.PROVIDER_ID,
            collection=entry.get("collection_concept_id"),
            satellite=entry.get("data_center", "NASA"),
            bbox=bbox,
            assets=assets,
            properties={
                "title": entry.get("title", ""),
                "producer_granule_id": entry.get("producer_granule_id", ""),
                "cloud_hosted": True,
            },
        )

    def download(
        self,
        data: SatelliteData,
        destination: Path,
        options: DownloadOptions,
    ) -> DownloadResult:
        """
        Download or directly access NASA cloud data.

        Prefers direct S3 access if AWS credentials are available,
        falls back to HTTPS download.

        Args:
            data: SatelliteData to download.
            destination: Output directory.
            options: Download configuration.

        Returns:
            DownloadResult.
        """
        self.require_auth()
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)

        s3_creds = self._session.session_data.get("s3_credentials", {}) if self._session.session_data else {}
        start_time = time.time()
        output_paths = []
        total_bytes = 0

        for key, asset in (data.data_assets or data.assets).items():
            href = asset.href
            if not href:
                continue

            filename = href.split("/")[-1] or f"{data.id}_{key}"
            out_file = destination / filename

            if out_file.exists() and not getattr(options, "overwrite", False):
                output_paths.append(out_file)
                continue

            try:
                if href.startswith("s3://") and s3_creds.get("aws_access_key_id"):
                    self._download_s3(href, out_file, s3_creds)
                else:
                    self._download_https(href, out_file, options)
                output_paths.append(out_file)
                total_bytes += out_file.stat().st_size
            except Exception as exc:
                self._logger.warning(f"Failed to download {key}: {exc}")

        duration = time.time() - start_time
        if not output_paths:
            return DownloadResult(
                status=DownloadStatus.FAILED, data_id=data.id,
                provider=self.PROVIDER_ID, error="No assets downloaded",
            )
        return DownloadResult(
            status=DownloadStatus.COMPLETED, data_id=data.id, provider=self.PROVIDER_ID,
            output_path=output_paths[0], output_paths=output_paths,
            bytes_downloaded=total_bytes, duration_seconds=duration,
        )

    def _download_s3(self, s3_uri: str, out_file: Path, creds: Dict[str, str]) -> None:
        """Download from S3 using temporary credentials."""
        try:
            import boto3
            s3 = boto3.client(
                "s3",
                aws_access_key_id=creds["aws_access_key_id"],
                aws_secret_access_key=creds["aws_secret_access_key"],
                aws_session_token=creds.get("aws_session_token"),
            )
            # Parse s3://bucket/key
            parts = s3_uri.replace("s3://", "").split("/", 1)
            bucket, key = parts[0], parts[1] if len(parts) > 1 else ""
            s3.download_file(bucket, key, str(out_file))
            self._logger.debug(f"S3 download: {s3_uri} → {out_file}")
        except ImportError:
            raise DownloadError("boto3 not installed. Install with: pip install boto3")

    def _download_https(self, url: str, out_file: Path, options: DownloadOptions) -> None:
        """HTTPS fallback download with Bearer token auth."""
        import httpx
        headers = {}
        if self._session and self._session.access_token:
            headers["Authorization"] = f"Bearer {self._session.access_token}"
        with httpx.stream("GET", url, headers=headers, timeout=options.timeout_seconds,
                          follow_redirects=True) as resp:
            self._handle_http_error(resp)
            with open(out_file, "wb") as f:
                for chunk in resp.iter_bytes(chunk_size=int(options.chunk_size_mb * 1024 * 1024)):
                    f.write(chunk)

    def get_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            provider_id=self.PROVIDER_ID,
            name=self.DISPLAY_NAME,
            description=self.DESCRIPTION,
            auth_type="oauth2",
            satellites=["Terra", "Aqua", "CALIPSO", "ICESat-2", "GEDI"],
            search=True, download=True, streaming=True,
            stac=True, supports_direct_s3=True,
            supports_aoi_filter=True, supports_date_filter=True,
            requires_auth=True, has_quota=False,
            regions=["global"],
            endpoint_url=self.BASE_URL,
            docs_url="https://www.earthdata.nasa.gov/esds/cloud-evolution",
        )

    def get_quota_info(self) -> QuotaInfo:
        return QuotaInfo(
            provider=self.PROVIDER_ID,
            extra_info={"note": "No download quota; data accessed directly from AWS S3."},
        )
