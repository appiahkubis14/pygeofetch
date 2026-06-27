"""
Adaptive download manager for PyGeoFetch.

Provides resilient parallel downloads with chunked transfer, checksum
verification, retry logic, progress reporting, and post-download actions.

Example::

    from pygeofetch.core.downloader import AdaptiveDownloader
    from pygeofetch.models.download_task import DownloadOptions

    downloader = AdaptiveDownloader()
    options = DownloadOptions(parallel=4, verify_checksum=True, retry_attempts=3)

    results = downloader.download_many(
        data_list=search_results,
        destination=Path("./data/"),
        options=options,
    )
"""

from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from pygeofetch.models.download_task import (
    DownloadOptions,
    DownloadProgress,
    DownloadResult,
    DownloadStatus,
    DownloadTask,
)
from pygeofetch.models.satellite_data import SatelliteData
from pygeofetch.utils.logging_setup import get_logger

logger = get_logger(__name__)


class AdaptiveDownloader:
    """
    Resilient parallel download manager.

    Features:
    - Configurable parallel workers
    - Chunked streaming downloads
    - Automatic retry with exponential backoff
    - Checksum verification (MD5/SHA256)
    - Real-time progress callbacks
    - Post-download action pipeline (extract, reproject)
    - Bandwidth throttling

    Attributes:
        auth_manager: AuthManager for provider sessions.
        progress_callback: Optional callable(DownloadProgress) for progress updates.
    """

    def __init__(
        self,
        auth_manager: Optional[Any] = None,
        progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
    ) -> None:
        self.auth_manager = auth_manager
        self.progress_callback = progress_callback
        self._active_tasks: Dict[str, DownloadTask] = {}
        self._provider_cache: Dict[str, Any] = {}

    def download(
        self,
        data: SatelliteData,
        destination: Path,
        options: Optional[DownloadOptions] = None,
    ) -> DownloadResult:
        """
        Download a single satellite data product.

        Args:
            data: SatelliteData to download.
            destination: Target directory.
            options: Download configuration (uses defaults if None).

        Returns:
            DownloadResult with status and output paths.
        """
        options = options or DownloadOptions()
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)

        provider = self._get_provider(data.provider)
        logger.info(f"Downloading {data.id!r} from {data.provider!r} → {destination}")

        for attempt in range(options.retry_attempts + 1):
            try:
                t0 = time.time()
                result = provider.download(data, destination, options)
                elapsed = time.time() - t0
                # Patch duration if provider didn't set it
                if result.duration_seconds == 0.0 and elapsed > 0:
                    result = result.model_copy(update={"duration_seconds": elapsed})
                if result.success:
                    if options.verify_checksum:
                        result = self._verify_checksums(result, data, options)
                    if options.post_process:
                        result = self._run_post_process(result, options)
                    size_mb = result.bytes_downloaded / (1024 * 1024) if result.bytes_downloaded else 0
                    speed = (size_mb / result.duration_seconds) if result.duration_seconds > 0 else 0
                    logger.info(
                        f"Downloaded {data.id!r}: {size_mb:.1f} MB "
                        f"in {result.duration_seconds:.1f}s"
                        + (f" ({speed:.1f} MB/s)" if speed > 0 else "")
                    )
                return result
            except Exception as exc:
                if attempt < options.retry_attempts:
                    delay = self._backoff(attempt, options)
                    logger.warning(
                        f"Download attempt {attempt+1}/{options.retry_attempts+1} failed "
                        f"for {data.id!r}: {exc}. Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"Download failed after {options.retry_attempts+1} attempts: {exc}")
                    return DownloadResult(
                        status=DownloadStatus.FAILED,
                        data_id=data.id,
                        provider=data.provider,
                        error=str(exc),
                        error_type=type(exc).__name__,
                        retries_used=attempt,
                    )

        return DownloadResult(
            status=DownloadStatus.FAILED,
            data_id=data.id,
            provider=data.provider,
            error="Exhausted retries",
        )

    def download_many(
        self,
        data_list: List[SatelliteData],
        destination: Path,
        options: Optional[DownloadOptions] = None,
        item_done_callback: Optional[Callable[[int, int, DownloadResult], None]] = None,
    ) -> List[DownloadResult]:
        """
        Download multiple satellite data products in parallel.

        Args:
            data_list: List of SatelliteData to download.
            destination: Target directory.
            options: Download configuration.
            item_done_callback: Optional callable(completed, total, result) called
                after each item finishes. Used to drive progress bars.

        Returns:
            List of DownloadResults in same order as input.
        """
        options = options or DownloadOptions()
        results: List[Optional[DownloadResult]] = [None] * len(data_list)
        completed_count = 0
        total = len(data_list)

        logger.info(
            f"Downloading {total} items with {options.parallel} parallel workers..."
        )

        with ThreadPoolExecutor(max_workers=options.parallel) as executor:
            future_to_idx = {
                executor.submit(self.download, data, destination / data.provider, options): i
                for i, data in enumerate(data_list)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results[idx] = result
                except Exception as exc:
                    result = DownloadResult(
                        status=DownloadStatus.FAILED,
                        data_id=data_list[idx].id,
                        provider=data_list[idx].provider,
                        error=str(exc),
                    )
                    results[idx] = result

                completed_count += 1
                self._emit_progress(data_list, results, options)
                if item_done_callback:
                    item_done_callback(completed_count, total, result)

        completed = sum(1 for r in results if r and r.success)
        failed = sum(1 for r in results if r and not r.success)
        logger.info(f"Downloads complete: {completed} succeeded, {failed} failed")

        return results  # type: ignore

    def _get_provider(self, provider_id: str) -> Any:
        """Get a provider instance with session from auth manager."""
        if provider_id not in self._provider_cache:
            from pygeofetch.providers import get_provider
            prov = get_provider(provider_id)
            if self.auth_manager and prov.REQUIRES_AUTH:
                try:
                    session = self.auth_manager.authenticate(provider_id)
                    prov.set_session(session)
                except Exception as exc:
                    logger.warning(f"Auth for {provider_id} failed: {exc}")
            self._provider_cache[provider_id] = prov
        return self._provider_cache[provider_id]

    def _verify_checksums(
        self, result: DownloadResult, data: SatelliteData, options: DownloadOptions
    ) -> DownloadResult:
        """Verify checksums of downloaded files against provider-provided values."""
        verified = True
        for path in result.output_paths:
            if not path.exists():
                continue
            # Find expected checksum from assets
            for asset in data.assets.values():
                filename = path.name
                if filename in asset.href:
                    if asset.checksum_md5:
                        actual = self._compute_checksum(path, "md5")
                        if actual.lower() != asset.checksum_md5.lower():
                            logger.warning(
                                f"MD5 mismatch for {path.name}: "
                                f"expected {asset.checksum_md5}, got {actual}"
                            )
                            verified = False
                    elif asset.checksum_sha256:
                        actual = self._compute_checksum(path, "sha256")
                        if actual.lower() != asset.checksum_sha256.lower():
                            logger.warning(f"SHA256 mismatch for {path.name}")
                            verified = False
        result.checksum_verified = verified
        return result

    def _compute_checksum(self, path: Path, algorithm: str) -> str:
        hasher = hashlib.new(algorithm)
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _run_post_process(
        self, result: DownloadResult, options: DownloadOptions
    ) -> DownloadResult:
        """Run configured post-download processing actions."""
        for action in options.post_process:
            try:
                result = self._apply_action(result, action)
            except Exception as exc:
                logger.warning(f"Post-process action {action.action!r} failed: {exc}")
        result.post_process_completed = True
        return result

    def _apply_action(self, result: DownloadResult, action: Any) -> DownloadResult:
        """Apply a single post-process action to download results."""
        if action.action == "unzip":
            from pygeofetch.utils.file_utils import safe_extract
            new_paths = []
            for path in result.output_paths:
                if path.suffix in (".zip", ".tar", ".gz"):
                    extracted = safe_extract(path, path.parent)
                    new_paths.extend(extracted)
                else:
                    new_paths.append(path)
            result.output_paths = new_paths
            if new_paths:
                result.output_path = new_paths[0]

        elif action.action == "reproject":
            # Requires rasterio - skip if not available
            target_crs = action.params.get("value", "EPSG:4326")
            try:
                import rasterio
                from rasterio.warp import calculate_default_transform, reproject, Resampling
                new_paths = []
                for path in result.output_paths:
                    if path.suffix.lower() in (".tif", ".tiff"):
                        out_path = path.with_stem(f"{path.stem}_{target_crs.replace(':', '_')}")
                        with rasterio.open(path) as src:
                            transform, width, height = calculate_default_transform(
                                src.crs, target_crs, src.width, src.height, *src.bounds
                            )
                            meta = src.meta.copy()
                            meta.update(crs=target_crs, transform=transform, width=width, height=height)
                            with rasterio.open(out_path, "w", **meta) as dst:
                                for i in range(1, src.count + 1):
                                    reproject(
                                        source=rasterio.band(src, i),
                                        destination=rasterio.band(dst, i),
                                        src_transform=src.transform,
                                        src_crs=src.crs,
                                        dst_transform=transform,
                                        dst_crs=target_crs,
                                        resampling=Resampling.nearest,
                                    )
                        new_paths.append(out_path)
                    else:
                        new_paths.append(path)
                result.output_paths = new_paths
            except ImportError:
                logger.warning("rasterio not installed; skipping reproject action")

        elif action.action == "compress":
            logger.debug(f"Compress action {action.params} noted (requires rasterio)")

        return result

    def _backoff(self, attempt: int, options: DownloadOptions) -> float:
        """Calculate retry delay based on strategy."""
        import random
        delay = options.retry_delay_seconds * (2 ** attempt)
        delay = min(delay, 60.0)
        if "jitter" in options.retry_strategy:
            delay *= (0.5 + random.random() * 0.5)
        return delay

    def _emit_progress(
        self,
        data_list: List[SatelliteData],
        results: List[Optional[DownloadResult]],
        options: DownloadOptions,
    ) -> None:
        """Emit progress update via callback if configured."""
        if not self.progress_callback:
            return
        completed = sum(1 for r in results if r is not None)
        succeeded = sum(1 for r in results if r and r.success)
        total_bytes = sum(r.bytes_downloaded for r in results if r)
        progress = DownloadProgress(
            task_id="batch",
            status=DownloadStatus.DOWNLOADING,
            bytes_downloaded=total_bytes,
            files_completed=completed,
            files_total=len(data_list),
            percent_complete=(completed / len(data_list)) * 100,
        )
        try:
            self.progress_callback(progress)
        except Exception:
            pass

    @property
    def _result_total_size_mb(self) -> float:
        """Property used by result objects."""
        return 0.0
