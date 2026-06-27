"""
Download task and result models for PyGeoFetch.

Defines all data structures related to download management including
tasks, options, results, and progress tracking.

Example::

    from pygeofetch.models.download_task import DownloadTask, DownloadOptions

    options = DownloadOptions(
        parallel=4,
        retry_attempts=3,
        verify_checksum=True,
        post_process=["reproject:EPSG:4326"],
    )
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class DownloadStatus(str, Enum):
    """Status values for a download task."""

    PENDING = "pending"
    QUEUED = "queued"
    DOWNLOADING = "downloading"
    PAUSED = "paused"
    VERIFYING = "verifying"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class RetryStrategy(str, Enum):
    """Retry backoff strategies."""

    FIXED = "fixed"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    EXPONENTIAL_JITTER = "exponential_jitter"


class ChecksumAlgorithm(str, Enum):
    """Checksum algorithms for file verification."""

    MD5 = "md5"
    SHA256 = "sha256"
    SHA512 = "sha512"


class PostProcessAction(BaseModel):
    """
    A single post-download processing action.

    Attributes:
        action: Action name (e.g., 'reproject', 'compress', 'unzip').
        params: Action-specific parameters.
    """

    action: str
    params: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_string(cls, s: str) -> "PostProcessAction":
        """
        Parse from string like 'reproject:EPSG:4326' or 'compress:lzw'.

        Args:
            s: String in format 'action' or 'action:param1:param2'.
        """
        parts = s.split(":", 1)
        action = parts[0].strip()
        params: Dict[str, Any] = {}
        if len(parts) > 1:
            params["value"] = parts[1].strip()
        return cls(action=action, params=params)


class DownloadOptions(BaseModel):
    """
    Configuration options for download operations.

    Attributes:
        parallel: Number of concurrent downloads.
        chunk_size_mb: Download chunk size in megabytes.
        retry_attempts: Number of retry attempts on failure.
        retry_strategy: Backoff strategy for retries.
        retry_delay_seconds: Base delay between retries.
        verify_checksum: Whether to verify file checksums after download.
        checksum_algorithm: Algorithm for checksum verification.
        bandwidth_limit_mbps: Maximum download bandwidth in Mbps (0=unlimited).
        timeout_seconds: Request timeout in seconds.
        output_format: Convert to this format after download (optional).
        post_process: List of post-download actions.
        overwrite: Whether to overwrite existing files.
        keep_original: Whether to keep original file after post-processing.
        notify_webhook: Webhook URL for completion notifications.
        priority: Download priority (higher = sooner, 1-10).
        metadata_only: Download metadata but not actual data files.
    """

    parallel: int = Field(default=2, ge=1, le=32)
    chunk_size_mb: float = Field(default=8.0, gt=0)
    retry_attempts: int = Field(default=3, ge=0, le=10)
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_JITTER
    retry_delay_seconds: float = Field(default=1.0, gt=0)
    verify_checksum: bool = True
    checksum_algorithm: ChecksumAlgorithm = ChecksumAlgorithm.MD5
    bandwidth_limit_mbps: float = Field(default=0, ge=0)
    timeout_seconds: int = Field(default=300, ge=10)
    output_format: Optional[str] = None
    post_process: List[PostProcessAction] = Field(default_factory=list)
    overwrite: bool = False
    keep_original: bool = False
    notify_webhook: Optional[str] = None
    notify_email: Optional[str] = None
    priority: int = Field(default=5, ge=1, le=10)
    metadata_only: bool = False
    resume: bool = True          # Resume interrupted downloads
    on_failure: str = "skip"     # "skip", "abort", "retry"
    bands: List[str] = Field(default_factory=list)  # e.g. ["B02","B03","B04"] — empty = all data assets

    @classmethod
    def with_post_process_strings(
        cls, post_process_strings: List[str], **kwargs: Any
    ) -> "DownloadOptions":
        """
        Convenience constructor accepting post-process as strings.

        Args:
            post_process_strings: List like ['reproject:EPSG:4326', 'compress:lzw'].
            **kwargs: Other DownloadOptions fields.
        """
        actions = [PostProcessAction.from_string(s) for s in post_process_strings]
        return cls(post_process=actions, **kwargs)


class DownloadProgress(BaseModel):
    """Real-time progress information for an active download."""

    task_id: str
    status: DownloadStatus
    bytes_downloaded: int = 0
    bytes_total: Optional[int] = None
    speed_bps: float = 0.0  # bytes per second
    eta_seconds: Optional[float] = None
    percent_complete: float = 0.0
    current_file: Optional[str] = None
    files_completed: int = 0
    files_total: int = 0
    started_at: Optional[datetime] = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def speed_mbps(self) -> float:
        """Return download speed in Mbps."""
        return (self.speed_bps * 8) / (1024 * 1024)


class DownloadResult(BaseModel):
    """
    Result of a completed download task.

    Attributes:
        task_id: Unique task identifier.
        status: Final status of the download.
        data_id: ID of the SatelliteData that was downloaded.
        provider: Provider used for download.
        output_path: Path to the downloaded/processed file.
        output_paths: All output paths if multiple files were created.
        bytes_downloaded: Total bytes downloaded.
        duration_seconds: Time taken for the complete download.
        checksum_verified: Whether checksum verification passed.
        post_process_completed: Whether all post-processing succeeded.
        error: Error message if download failed.
        error_type: Error category for diagnostic purposes.
        retries_used: Number of retry attempts used.
        metadata: Additional download metadata.
    """

    task_id: str = Field(default_factory=lambda: str(uuid4()))
    status: DownloadStatus = DownloadStatus.PENDING
    data_id: str = ""
    provider: str = ""
    output_path: Optional[Path] = None
    output_paths: List[Path] = Field(default_factory=list)
    bytes_downloaded: int = 0
    duration_seconds: float = 0.0
    checksum_verified: Optional[bool] = None
    post_process_completed: bool = False
    error: Optional[str] = None
    error_type: Optional[str] = None
    retries_used: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Return True if download completed successfully."""
        return self.status == DownloadStatus.COMPLETED

    @property
    def total_size_mb(self) -> float:
        """Return total downloaded size in megabytes."""
        return self.bytes_downloaded / (1024 * 1024) if self.bytes_downloaded else 0.0

    @property
    def speed_mbps(self) -> float:
        """Return average download speed in Mbps."""
        if self.duration_seconds > 0 and self.bytes_downloaded > 0:
            return (self.bytes_downloaded * 8) / (self.duration_seconds * 1024 * 1024)
        return 0.0


class DownloadTask(BaseModel):
    """
    Represents a unit of work in the download queue.

    Attributes:
        id: Unique task identifier.
        data_id: SatelliteData ID to download.
        asset_key: Specific asset key to download (None = all data assets).
        asset_href: Direct URL to download from.
        provider: Provider handling this download.
        destination: Target directory or file path.
        options: Download configuration.
        status: Current task status.
        priority: Queue priority (higher = sooner).
        created_at: When the task was created.
        scheduled_at: When the task should start.
        result: Download result (set on completion).
        depends_on: IDs of tasks that must complete first.
        tags: Arbitrary string tags for grouping.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    data_id: str
    asset_key: Optional[str] = None
    asset_href: str
    provider: str
    destination: Path
    options: DownloadOptions = Field(default_factory=DownloadOptions)
    status: DownloadStatus = DownloadStatus.PENDING
    priority: int = Field(default=5, ge=1, le=10)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    scheduled_at: Optional[datetime] = None
    result: Optional[DownloadResult] = None
    depends_on: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

    def mark_started(self) -> None:
        """Mark task as actively downloading."""
        self.status = DownloadStatus.DOWNLOADING

    def mark_completed(self, result: DownloadResult) -> None:
        """Mark task as successfully completed."""
        self.status = DownloadStatus.COMPLETED
        self.result = result

    def mark_failed(self, error: str, error_type: str = "unknown") -> None:
        """Mark task as failed with error details."""
        self.status = DownloadStatus.FAILED
        if self.result is None:
            self.result = DownloadResult(
                data_id=self.data_id,
                provider=self.provider,
            )
        self.result.status = DownloadStatus.FAILED
        self.result.error = error
        self.result.error_type = error_type
