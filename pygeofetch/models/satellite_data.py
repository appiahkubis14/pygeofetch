"""
Satellite data models for PyGeoFetch.

This module defines the core data structures used throughout the package
to represent satellite imagery metadata, assets, and collection information.
All models are STAC 1.0 compliant.

Example::

    from pygeofetch.models.satellite_data import SatelliteData, SatelliteAsset

    data = SatelliteData(
        id="LC08_L2SP_013032_20240115",
        provider="usgs",
        satellite="Landsat-8",
        sensor="OLI-TIRS",
        datetime=datetime(2024, 1, 15),
        bbox=(-75.0, 40.0, -74.0, 41.0),
        cloud_cover=5.2,
        processing_level="L2SP",
    )
"""

from __future__ import annotations

from datetime import datetime as DateTime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator


class ProcessingLevel(str, Enum):
    """Standard satellite data processing levels."""

    RAW = "RAW"
    L0 = "L0"
    L1 = "L1"
    L1A = "L1A"
    L1B = "L1B"
    L1C = "L1C"
    L1T = "L1T"
    L2 = "L2"
    L2A = "L2A"
    L2SP = "L2SP"
    L3 = "L3"
    L4 = "L4"
    ANALYSIS_READY = "ARD"
    UNKNOWN = "UNKNOWN"


class DataFormat(str, Enum):
    """Supported satellite data file formats."""

    GEOTIFF = "GeoTIFF"
    COG = "COG"  # Cloud Optimized GeoTIFF
    NETCDF = "NetCDF"
    HDF4 = "HDF4"
    HDF5 = "HDF5"
    ZARR = "Zarr"
    JP2 = "JPEG2000"
    PNG = "PNG"
    SAFE = "SAFE"  # ESA Sentinel format
    ZIP = "ZIP"
    TAR = "TAR"
    UNKNOWN = "UNKNOWN"


class SatelliteAsset(BaseModel):
    """
    Represents a single downloadable asset within a satellite data record.

    Attributes:
        key: Unique identifier for this asset (e.g., 'B4', 'thumbnail', 'metadata').
        href: URL or path to the asset.
        title: Human-readable title.
        media_type: MIME type of the asset.
        roles: STAC asset roles (e.g., ['data', 'overview', 'thumbnail']).
        size_bytes: File size in bytes if known.
        checksum_md5: MD5 checksum for integrity verification.
        checksum_sha256: SHA-256 checksum for integrity verification.
        extra_fields: Additional provider-specific metadata.
    """

    key: str
    href: str
    title: Optional[str] = None
    media_type: Optional[str] = None
    roles: List[str] = Field(default_factory=list)
    size_bytes: Optional[int] = None
    checksum_md5: Optional[str] = None
    checksum_sha256: Optional[str] = None
    extra_fields: Dict[str, Any] = Field(default_factory=dict)

    @property
    def size_mb(self) -> Optional[float]:
        """Return size in megabytes."""
        return self.size_bytes / (1024 * 1024) if self.size_bytes else None

    def is_data_asset(self) -> bool:
        """Return True if this is a downloadable raster data asset."""
        # Must have 'data' role — 'overview' alone is often JSON/tilejson, not raster
        if "data" not in self.roles:
            return False
        # Exclude non-raster types explicitly
        non_raster = ("application/json", "text/", "application/xml")
        if self.media_type and any(self.media_type.startswith(t) for t in non_raster):
            return False
        return True


class SatelliteData(BaseModel):
    """
    Core model representing a satellite data product/scene.

    STAC 1.0 compliant. Represents a single acquisition or derived product
    from any supported satellite data provider.

    Attributes:
        id: Unique identifier for this scene/product.
        provider: Provider name (e.g., 'usgs', 'copernicus').
        collection: Collection or dataset name.
        satellite: Satellite platform name (e.g., 'Landsat-8', 'Sentinel-2A').
        sensor: Sensor instrument (e.g., 'OLI', 'MSI').
        datetime: Primary acquisition datetime (UTC).
        start_datetime: Start of acquisition window.
        end_datetime: End of acquisition window.
        bbox: Bounding box as (min_lon, min_lat, max_lon, max_lat).
        geometry: GeoJSON geometry dict.
        cloud_cover: Cloud cover percentage (0-100).
        processing_level: Data processing level.
        data_format: Primary file format.
        assets: Dictionary of downloadable assets.
        stac_extensions: STAC extension URLs used.
        properties: Additional STAC-compliant properties.
        links: STAC links (self, root, collection, etc.).
        score: Relevance/quality score for search result ranking (0-1).

    Example::

        data = SatelliteData(
            id="S2A_MSIL2A_20240115T154911_N0510_R054_T18TWL",
            provider="copernicus",
            collection="SENTINEL-2",
            satellite="Sentinel-2A",
            sensor="MSI",
            datetime=datetime(2024, 1, 15, 15, 49, 11),
            bbox=(-74.1, 40.6, -73.7, 40.9),
            cloud_cover=3.5,
            processing_level=ProcessingLevel.L2A,
        )
    """

    id: str
    provider: str
    collection: Optional[str] = None
    satellite: Optional[str] = None
    sensor: Optional[str] = None
    datetime: Optional[DateTime] = None
    start_datetime: Optional[DateTime] = None
    end_datetime: Optional[DateTime] = None
    bbox: Optional[Tuple[float, float, float, float]] = None
    geometry: Optional[Dict[str, Any]] = None
    cloud_cover: Optional[float] = Field(None, ge=0, le=100)
    processing_level: ProcessingLevel = ProcessingLevel.UNKNOWN
    data_format: DataFormat = DataFormat.UNKNOWN
    assets: Dict[str, SatelliteAsset] = Field(default_factory=dict)
    stac_extensions: List[str] = Field(default_factory=list)
    properties: Dict[str, Any] = Field(default_factory=dict)
    links: List[Dict[str, str]] = Field(default_factory=list)
    score: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, v: Optional[Tuple]) -> Optional[Tuple]:
        """Validate bounding box coordinates."""
        if v is None:
            return v
        min_lon, min_lat, max_lon, max_lat = v
        if not (-180 <= min_lon <= 180 and -180 <= max_lon <= 180):
            raise ValueError(f"Longitude values must be between -180 and 180, got {min_lon}, {max_lon}")
        if not (-90 <= min_lat <= 90 and -90 <= max_lat <= 90):
            raise ValueError(f"Latitude values must be between -90 and 90, got {min_lat}, {max_lat}")
        if min_lon >= max_lon:
            raise ValueError(f"min_lon ({min_lon}) must be less than max_lon ({max_lon})")
        if min_lat >= max_lat:
            raise ValueError(f"min_lat ({min_lat}) must be less than max_lat ({max_lat})")
        return v

    @model_validator(mode="after")
    def validate_datetime_range(self) -> "SatelliteData":
        """Validate that start_datetime is before end_datetime."""
        if self.start_datetime and self.end_datetime:
            if self.start_datetime >= self.end_datetime:
                raise ValueError("start_datetime must be before end_datetime")
        return self

    @property
    def total_size_bytes(self) -> int:
        """Total size of all assets in bytes."""
        return sum(a.size_bytes or 0 for a in self.assets.values())

    @property
    def total_size_mb(self) -> float:
        """Total size of all data assets in megabytes."""
        return self.total_size_bytes / (1024 * 1024)

    @property
    def data_assets(self) -> Dict[str, SatelliteAsset]:
        """Return only primary data assets (not thumbnails/metadata)."""
        return {k: v for k, v in self.assets.items() if v.is_data_asset()}

    def to_stac_item(self) -> Dict[str, Any]:
        """
        Export as a STAC 1.0 Item dictionary.

        Returns:
            STAC-compliant dictionary ready for serialization.
        """
        item: Dict[str, Any] = {
            "type": "Feature",
            "stac_version": "1.0.0",
            "stac_extensions": self.stac_extensions,
            "id": self.id,
            "geometry": self.geometry or self._bbox_to_geometry(),
            "bbox": list(self.bbox) if self.bbox else None,
            "properties": {
                **self.properties,
                "datetime": self.datetime.isoformat() if self.datetime else None,
                "platform": self.satellite,
                "instruments": [self.sensor] if self.sensor else [],
                "eo:cloud_cover": self.cloud_cover,
                "processing:level": self.processing_level.value,
                "providers": [{"name": self.provider, "roles": ["producer"]}],
            },
            "assets": {
                k: {
                    "href": v.href,
                    "title": v.title,
                    "type": v.media_type,
                    "roles": v.roles,
                    **({"size": v.size_bytes} if v.size_bytes else {}),
                }
                for k, v in self.assets.items()
            },
            "links": self.links,
        }
        if self.collection:
            item["collection"] = self.collection
        return item

    def _bbox_to_geometry(self) -> Optional[Dict[str, Any]]:
        """Convert bbox to GeoJSON Polygon geometry."""
        if not self.bbox:
            return None
        min_lon, min_lat, max_lon, max_lat = self.bbox
        return {
            "type": "Polygon",
            "coordinates": [[
                [min_lon, min_lat], [max_lon, min_lat],
                [max_lon, max_lat], [min_lon, max_lat],
                [min_lon, min_lat],
            ]],
        }

    @classmethod
    def from_stac_item(cls, item: Dict[str, Any], provider: str) -> "SatelliteData":
        """
        Create a SatelliteData instance from a STAC Item dictionary.

        Args:
            item: STAC-compliant item dictionary.
            provider: Provider name string.

        Returns:
            SatelliteData instance.
        """
        props = item.get("properties", {})
        assets_raw = item.get("assets", {})
        assets = {}
        for key, asset_data in assets_raw.items():
            assets[key] = SatelliteAsset(
                key=key,
                href=asset_data.get("href", ""),
                title=asset_data.get("title"),
                media_type=asset_data.get("type"),
                roles=asset_data.get("roles", []),
                size_bytes=asset_data.get("size"),
            )

        bbox_raw = item.get("bbox")
        bbox = tuple(bbox_raw) if bbox_raw and len(bbox_raw) == 4 else None

        dt_str = props.get("datetime")
        dt = None
        if dt_str and dt_str != "null":
            from dateutil.parser import parse
            dt = parse(dt_str)

        return cls(
            id=item["id"],
            provider=provider,
            collection=item.get("collection"),
            satellite=props.get("platform"),
            sensor=(props.get("instruments") or [None])[0],
            datetime=dt,
            bbox=bbox,
            geometry=item.get("geometry"),
            cloud_cover=props.get("eo:cloud_cover"),
            properties=props,
            assets=assets,
            stac_extensions=item.get("stac_extensions", []),
            links=item.get("links", []),
        )


class ProviderCapabilities(BaseModel):
    """
    Describes what a provider supports.

    Attributes:
        search: Whether the provider supports search/discovery.
        download: Whether files can be downloaded.
        streaming: Whether streaming access is supported.
        stac: Whether the provider exposes a STAC API.
        max_results: Maximum search results returnable.
        supported_satellites: List of satellite platforms available.
        supported_formats: List of data formats available.
        requires_auth: Whether authentication is required.
        has_quota: Whether the provider enforces download quotas.
        supports_aoi_filter: Whether spatial filtering is supported.
        supports_cloud_filter: Whether cloud cover filtering is supported.
        supports_date_filter: Whether date range filtering is supported.
        extra_info: Additional provider-specific capability metadata.
    """

    provider_id: str = ""
    name: str = ""
    description: str = ""
    auth_type: str = "username_password"
    satellites: List[str] = Field(default_factory=list)
    # Capability flags
    search: bool = True
    download: bool = True
    streaming: bool = False
    stac: bool = False
    supports_sar: bool = False
    supports_sub_meter: bool = False  # <1m resolution imagery
    supports_tasking: bool = False    # New satellite tasking orders
    supports_direct_s3: bool = False  # Direct cloud storage access
    supports_cql2: bool = False       # CQL2 filter expressions
    # Filters
    supports_aoi_filter: bool = True
    supports_cloud_filter: bool = False
    supports_date_filter: bool = True
    supports_resolution_filter: bool = False
    supports_processing_level_filter: bool = False
    # Metadata
    max_results: Optional[int] = None
    supported_satellites: List[str] = Field(default_factory=list)
    supported_formats: List[DataFormat] = Field(default_factory=list)
    requires_auth: bool = True
    has_quota: bool = False
    regions: List[str] = Field(default_factory=list)  # e.g. ["global", "europe"]
    resolution_min_m: Optional[float] = None   # Best resolution in metres
    resolution_max_m: Optional[float] = None   # Coarsest resolution in metres
    endpoint_url: str = ""
    docs_url: str = ""
    extra_info: Dict[str, Any] = Field(default_factory=dict)


class QuotaInfo(BaseModel):
    """
    Represents a provider's quota/usage information.

    Attributes:
        provider: Provider name.
        total_bytes: Total allowed download quota in bytes.
        used_bytes: Bytes already consumed.
        remaining_bytes: Bytes remaining.
        reset_datetime: When the quota resets.
        requests_per_minute: API rate limit.
        requests_used_today: API calls made today.
        extra_info: Additional quota details.
    """

    provider: str
    total_bytes: Optional[int] = None
    used_bytes: Optional[int] = None
    remaining_bytes: Optional[int] = None
    reset_datetime: Optional[DateTime] = None
    requests_per_minute: Optional[int] = None
    requests_used_today: Optional[int] = None
    extra_info: Dict[str, Any] = Field(default_factory=dict)

    @property
    def usage_percent(self) -> Optional[float]:
        """Return percentage of quota used (0-100)."""
        if self.total_bytes and self.used_bytes:
            return (self.used_bytes / self.total_bytes) * 100
        return None
