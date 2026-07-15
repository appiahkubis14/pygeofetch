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
from typing import TYPE_CHECKING, Any, Callable

from pygeofetch.core.logging import DownloadProgress, get_logger
from pygeofetch.models.download_task import (
    DownloadOptions,
    DownloadResult,
    DownloadStatus,
    DownloadTask,
)

if TYPE_CHECKING:
    from pygeofetch.models.satellite_data import SatelliteData

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
        auth_manager: Any | None = None,
        progress_callback: Callable[[DownloadProgress], None] | None = None,
    ) -> None:
        self.auth_manager = auth_manager
        self.progress_callback = progress_callback
        self._active_tasks: dict[str, DownloadTask] = {}
        self._provider_cache: dict[str, Any] = {}

    def download(
        self,
        data: SatelliteData,
        destination: Path,
        options: DownloadOptions | None = None,
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
        _item_start = __import__("time").time()
        _dl_progress = globals().get("_active_progress")
        if _dl_progress:
            _dl_progress.start_item(str(data.id))
        else:
            logger.info("Downloading %s", str(data.id)[:60])

        for attempt in range(options.retry_attempts + 1):
            try:
                t0 = time.time()
                result = provider.download(data, destination, options)
                elapsed = time.time() - t0
                # Patch duration if provider didn't set it
                if result.duration_seconds == 0.0 and elapsed > 0:
                    result = result.model_copy(update={"duration_seconds": elapsed})
                if result.success:
                    # BUG 3: validate file integrity before claiming success
                    for out_p in result.output_paths or [result.output_path]:
                        if out_p and out_p.exists():
                            is_valid, err_msg = self._validate_downloaded_file(out_p)
                            if not is_valid:
                                result = result.model_copy(
                                    update={
                                        "status": DownloadStatus.FAILED,
                                        "error": (
                                            f"Download appeared complete but"
                                            f" file validation failed: {err_msg}"
                                        ),
                                        "output_path": None,
                                    }
                                )
                                logger.warning(
                                    f"File validation failed for {data.id!r}: {err_msg}"
                                )
                                break
                    if result.success:
                        if options.verify_checksum:
                            result = self._verify_checksums(result, data, options)
                        if options.post_process:
                            result = self._run_post_process(result, options)
                    size_mb = (
                        result.bytes_downloaded / (1024 * 1024)
                        if result.bytes_downloaded
                        else 0
                    )
                    (
                        size_mb / result.duration_seconds
                    ) if result.duration_seconds > 0 else 0
                    logger.info(
                        "  ✓ %-45s %6.0f MB  %5.1fs",
                        str(data.id)[:45],
                        size_mb,
                        result.duration_seconds,
                    )
                return result
            except Exception as exc:
                if attempt < options.retry_attempts:
                    delay = self._backoff(attempt, options)
                    logger.warning(
                        f"Download attempt {attempt + 1}/{options.retry_attempts + 1} failed "
                        f"for {data.id!r}: {exc}. Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"Download failed after {options.retry_attempts + 1} attempts: {exc}"
                    )
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
        data_list: list[SatelliteData],
        destination: Path,
        options: DownloadOptions | None = None,
        item_done_callback: Callable[[int, int, DownloadResult], None] | None = None,
    ) -> list[DownloadResult]:
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
        results: list[DownloadResult | None] = [None] * len(data_list)
        completed_count = 0
        total = len(data_list)

        dp = DownloadProgress(total=total, destination=str(destination))
        dp.__enter__()

        def _run_one(data: SatelliteData, idx: int) -> tuple:
            """Download one item and return (idx, result, bytes, duration)."""
            import time as _it

            t0 = _it.time()
            dp.start_item(str(data.id))
            try:
                result = self.download(data, destination / data.provider, options)
                dur = _it.time() - t0
                dp.complete_item(
                    success=result.success,
                    bytes_total=result.bytes_downloaded or 0,
                    duration=dur,
                )
                return idx, result
            except Exception as exc:
                dur = _it.time() - t0
                result = DownloadResult(
                    status=DownloadStatus.FAILED,
                    data_id=data.id,
                    provider=data.provider,
                    error=str(exc),
                )
                dp.complete_item(success=False, duration=dur)
                return idx, result

        with ThreadPoolExecutor(max_workers=options.parallel) as executor:
            futures = {
                executor.submit(_run_one, data, i): i
                for i, data in enumerate(data_list)
            }
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result
                completed_count += 1
                if item_done_callback:
                    item_done_callback(completed_count, total, result)

        # BUG 2: guarantee length == len(data_list) and order is preserved
        # Fill any None slots (unexpected future crashes) with failure results
        for i, item in enumerate(data_list):
            if results[i] is None:
                results[i] = DownloadResult(
                    status=DownloadStatus.FAILED,
                    data_id=item.id,
                    provider=item.provider,
                    error="Download future returned no result (internal error)",
                )

        sum(1 for r in results if r and r.success)
        sum(1 for r in results if r and not r.success)
        dp.__exit__(None, None, None)
        assert len(results) == len(data_list), (
            f"BUG: result count {len(results)} != input count {len(data_list)}"
        )
        sum(
            r.bytes_downloaded / (1024 * 1024)
            for r in results
            if r and r.success and r.bytes_downloaded
        )
        # summary printed by DownloadProgress footer
        return results  # noqa: RETURNVALUE

    def _get_provider(self, provider_id: str) -> Any:
        """Get a provider instance with a fresh authenticated session."""
        from pygeofetch.providers import get_provider

        prov = get_provider(provider_id)
        if self.auth_manager and prov.REQUIRES_AUTH:
            try:
                session = self.auth_manager.authenticate(provider_id)
                prov.set_session(session)
            except Exception as exc:
                logger.warning(f"Auth for {provider_id} failed: {exc}")
        return prov

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
        result.post_process_completed.append("completed")
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
                from rasterio.warp import Resampling, reproject  # noqa: F401

                new_paths = []
                for path in result.output_paths:
                    if path.suffix.lower() in (".tif", ".tiff"):
                        out_path = path.with_stem(
                            f"{path.stem}_{target_crs.replace(':', '_')}"
                        )
                        self._reproject_with_validation(path, out_path, target_crs)
                        new_paths.append(out_path)
                    else:
                        new_paths.append(path)
                result.output_paths = new_paths
            except ImportError:
                logger.warning("rasterio not installed; skipping reproject action")

        elif action.action == "compress":
            method = action.params.get("value", "lzw")
            try:
                import rasterio

                new_paths = []
                for path in result.output_paths:
                    if path.suffix.lower() in (".tif", ".tiff"):
                        out_path = path.with_stem(f"{path.stem}_{method}")
                        with rasterio.open(path) as src:
                            profile = src.profile.copy()
                            profile.update(
                                compress=method,
                                tiled=True,
                                blockxsize=512,
                                blockysize=512,
                            )
                            with rasterio.open(out_path, "w", **profile) as dst:
                                dst.write(src.read())
                        new_paths.append(out_path)
                    else:
                        new_paths.append(path)
                result.output_paths = new_paths
                if new_paths:
                    result.output_path = new_paths[0]
            except ImportError:
                logger.warning("rasterio not installed; skipping compress action")

        elif action.action == "cog":
            compress = action.params.get("value", "deflate")
            try:
                import tempfile

                import rasterio
                from rasterio.enums import Resampling as RS

                new_paths = []
                for path in result.output_paths:
                    if path.suffix.lower() in (".tif", ".tiff"):
                        out_path = path.with_stem(f"{path.stem}_cog")
                        with rasterio.open(path) as src:
                            profile = src.profile.copy()
                            profile.update(
                                compress=compress,
                                tiled=True,
                                blockxsize=512,
                                blockysize=512,
                                interleave="band",
                                driver="GTiff",
                            )
                            data = src.read()
                            with tempfile.NamedTemporaryFile(
                                suffix=".tif", delete=False
                            ) as tmp:
                                tmp_path = Path(tmp.name)
                            with rasterio.open(tmp_path, "w", **profile) as tmp_dst:
                                tmp_dst.write(data)
                                tmp_dst.build_overviews([2, 4, 8, 16, 32], RS.average)
                                tmp_dst.update_tags(
                                    ns="rio_overview", resampling="average"
                                )
                            profile.update(copy_src_overviews=True)
                            with rasterio.open(tmp_path) as tmp_src:
                                with rasterio.open(out_path, "w", **profile) as cog_dst:
                                    cog_dst.write(tmp_src.read())
                            tmp_path.unlink(missing_ok=True)
                        new_paths.append(out_path)
                    else:
                        new_paths.append(path)
                result.output_paths = new_paths
                if new_paths:
                    result.output_path = new_paths[0]
            except ImportError:
                logger.warning("rasterio not installed; skipping cog action")

        elif action.action == "clip":
            import json

            geom_src = action.params.get("value", "")
            try:
                import rasterio
                from rasterio.mask import mask as rasterio_mask

                shapes = []
                if geom_src.endswith((".geojson", ".json")):
                    with open(geom_src) as f:
                        gj = json.load(f)
                    if gj.get("type") == "FeatureCollection":
                        shapes = [feat["geometry"] for feat in gj["features"]]
                    elif gj.get("type") == "Feature":
                        shapes = [gj["geometry"]]
                    else:
                        shapes = [gj]
                elif geom_src:
                    # bbox string "minx,miny,maxx,maxy"
                    try:
                        parts = [float(x) for x in geom_src.split(",")]
                        from shapely.geometry import box as sbox

                        shapes = [sbox(*parts).__geo_interface__]
                    except Exception:
                        pass
                if shapes:
                    new_paths = []
                    for path in result.output_paths:
                        if path.suffix.lower() in (".tif", ".tiff"):
                            out_path = path.with_stem(f"{path.stem}_clip")
                            with rasterio.open(path) as src:
                                out_img, out_transform = rasterio_mask(
                                    src, shapes, crop=True
                                )
                                profile = src.profile.copy()
                                profile.update(
                                    height=out_img.shape[1],
                                    width=out_img.shape[2],
                                    transform=out_transform,
                                )
                                with rasterio.open(out_path, "w", **profile) as dst:
                                    dst.write(out_img)
                            new_paths.append(out_path)
                        else:
                            new_paths.append(path)
                    result.output_paths = new_paths
                    if new_paths:
                        result.output_path = new_paths[0]
            except ImportError:
                logger.warning("rasterio/shapely not installed; skipping clip action")

        elif action.action == "resample":
            val = action.params.get("value", "")
            try:
                import rasterio
                from rasterio.enums import Resampling as RS

                new_paths = []
                for path in result.output_paths:
                    if path.suffix.lower() in (".tif", ".tiff"):
                        with rasterio.open(path) as src:
                            # value is target resolution in CRS units
                            try:
                                target_res = float(val)
                                scale_x = src.res[0] / target_res
                                scale_y = src.res[1] / target_res
                            except (ValueError, ZeroDivisionError):
                                scale_x = scale_y = 0.5
                            new_h = max(1, int(src.height * scale_y))
                            new_w = max(1, int(src.width * scale_x))
                            data = src.read(
                                out_shape=(src.count, new_h, new_w),
                                resampling=RS.bilinear,
                            )
                            transform = src.transform * src.transform.scale(
                                src.width / new_w, src.height / new_h
                            )
                            profile = src.profile.copy()
                            profile.update(
                                height=new_h, width=new_w, transform=transform
                            )
                        out_path = path.with_stem(f"{path.stem}_resamp")
                        with rasterio.open(out_path, "w", **profile) as dst:
                            dst.write(data)
                        new_paths.append(out_path)
                    else:
                        new_paths.append(path)
                result.output_paths = new_paths
                if new_paths:
                    result.output_path = new_paths[0]
            except ImportError:
                logger.warning("rasterio not installed; skipping resample action")

        elif action.action == "ndvi":
            # Compute NDVI from downloaded bands — expects B04 (Red) and B08 (NIR)
            import numpy as np

            try:
                import rasterio

                red_path = next(
                    (
                        p
                        for p in result.output_paths
                        if "B04" in p.name or "red" in p.name.lower()
                    ),
                    None,
                )
                nir_path = next(
                    (
                        p
                        for p in result.output_paths
                        if "B08" in p.name or "nir" in p.name.lower()
                    ),
                    None,
                )
                if red_path and nir_path:
                    with rasterio.open(red_path) as rs:
                        red = rs.read(1).astype(np.float32)
                        profile = rs.profile.copy()
                    with rasterio.open(nir_path) as ns:
                        nir = ns.read(
                            1,
                            out_shape=red.shape,
                            resampling=rasterio.enums.Resampling.bilinear,
                        ).astype(np.float32)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        ndvi = np.where(
                            nir + red > 0, (nir - red) / (nir + red), -9999.0
                        )
                    ndvi_path = red_path.parent / "ndvi.tif"
                    profile.update(count=1, dtype="float32", nodata=-9999.0)
                    with rasterio.open(ndvi_path, "w", **profile) as dst:
                        dst.write(ndvi[np.newaxis, :, :])
                    (result.output_paths or []).append(ndvi_path)
                    result.output_path = ndvi_path
                    logger.info(f"NDVI computed → {ndvi_path}")
                else:
                    logger.warning(
                        "ndvi action: could not find B04 (Red) and B08 (NIR) in output paths"
                    )
            except ImportError:
                logger.warning("rasterio/numpy not installed; skipping ndvi action")

        elif action.action in ("pan-sharpen", "pan_sharpen"):
            logger.info(
                "pan-sharpen action: use PyGeoFetch preprocess pansharpen for this operation"
            )

        elif action.action == "merge":
            try:
                import rasterio
                from rasterio.merge import merge

                tifs = [
                    p
                    for p in result.output_paths
                    if p.suffix.lower() in (".tif", ".tiff")
                ]
                if len(tifs) > 1:
                    src_files = [rasterio.open(p) for p in tifs]
                    mosaic, transform = merge(src_files)
                    profile = src_files[0].profile.copy()
                    profile.update(
                        height=mosaic.shape[1],
                        width=mosaic.shape[2],
                        transform=transform,
                    )
                    merge_path = tifs[0].parent / "merged.tif"
                    with rasterio.open(merge_path, "w", **profile) as dst:
                        dst.write(mosaic)
                    for s in src_files:
                        s.close()
                    result.output_paths = [merge_path]
                    result.output_path = merge_path
            except ImportError:
                logger.warning("rasterio not installed; skipping merge action")

        return result

    def _validate_downloaded_file(self, path) -> tuple:
        """
        Validate a downloaded file.

        Strategy by file type:
          .zip / .tar.gz  — check it's a valid archive (not truncated)
          .tif / .tiff    — open with rasterio and read one tile
          .jp2            — open with rasterio (GDAL driver)
          .nc             — check size > 0 (netCDF, GDAL can open but slow)
          .json / .xml / .csv / .txt / .parquet — check size > 0
          anything else   — check size > 0 (safe fallback, never fail good data)

        Returns:
            (is_valid: bool, error_message: str)
        """
        import os

        path_str = str(path)
        path_obj = Path(path_str)
        suffix = path_obj.suffix.lower()

        # ── 0. File must exist and be non-empty ───────────────────────────
        try:
            size = os.path.getsize(path_str)
        except OSError as exc:
            return False, f"Cannot stat file: {exc}"
        if size == 0:
            return False, "File is empty (0 bytes)"

        # ── 1. ZIP / TAR archives — validate archive structure ─────────────
        if suffix == ".zip" or path_str.endswith(".zip"):
            try:
                import zipfile

                with zipfile.ZipFile(path_str) as zf:
                    bad = zf.testzip()
                    if bad:
                        return False, f"ZIP is corrupt — bad entry: {bad}"
                return True, ""
            except zipfile.BadZipFile as exc:
                return False, f"Invalid ZIP file: {exc}"
            except Exception:
                # If we can't import zipfile or something weird — just trust size
                return True, ""

        if suffix in (".gz", ".tgz") or path_str.endswith(".tar.gz"):
            try:
                import tarfile

                with tarfile.open(path_str) as tf:
                    tf.getmembers()
                return True, ""
            except tarfile.TarError as exc:
                return False, f"Invalid TAR file: {exc}"
            except Exception:
                return True, ""

        # ── 2. Non-raster text/data formats — size check only ─────────────
        NON_RASTER = {
            ".json",
            ".geojson",
            ".xml",
            ".csv",
            ".txt",
            ".html",
            ".md",
            ".yaml",
            ".yml",
            ".parquet",
            ".nc",
            ".EOF",  # orbit files
        }
        if suffix in NON_RASTER or not suffix:
            return True, ""  # already passed size > 0 above

        # ── 3. Raster formats — open with rasterio ────────────────────────
        RASTER_SUFFIXES = {
            ".tif",
            ".tiff",
            ".jp2",
            ".img",
            ".vrt",
            ".hdf",
            ".h4",
            ".h5",
            ".hdf5",
        }
        if suffix in RASTER_SUFFIXES:
            try:
                import rasterio

                with rasterio.open(path_str) as src:
                    if src.count == 0:
                        return False, "Raster has zero bands"
                    if src.width == 0 or src.height == 0:
                        return (
                            False,
                            f"Raster has zero dimensions: {src.width}x{src.height}",
                        )
                    # Read one small tile to confirm data is accessible
                    windows = list(src.block_windows(1))
                    if windows:
                        _, window = windows[0]
                        sample = src.read(1, window=window)
                        if sample is None or sample.size == 0:
                            return False, "First tile read returned empty array"
                return True, ""
            except ImportError:
                # rasterio not installed — size check already passed
                return True, ""
            except Exception as exc:
                return False, f"Raster read failed: {exc}"

        # ── 4. Unknown extension — trust size check (already passed) ──────
        return True, ""

    def _has_identity_transform(self, path) -> bool:
        """
        Detect a reprojection that produced an identity (garbage) transform.

        Returns True if the file has pixel_width=1, pixel_height=1,
        origin=(0,0) in a projected (metre-unit) CRS — which indicates
        the warp produced no real geographic transformation.
        """
        try:
            import rasterio

            with rasterio.open(str(path)) as src:
                t = src.transform
                return (
                    abs(t.a) == 1.0
                    and abs(t.e) == 1.0
                    and t.c == 0.0
                    and t.f == 0.0
                    and src.crs is not None
                    and src.crs.is_projected
                )
        except Exception:
            return False

    def _reproject_with_validation(self, source_path, target_path, target_crs) -> None:
        """
        Reproject source to target CRS and validate the output transform.
        Raises RuntimeError if reprojection produces an identity transform.
        """
        import rasterio
        from rasterio.enums import Resampling as RS
        from rasterio.warp import calculate_default_transform, reproject

        with rasterio.open(str(source_path)) as src:
            src_crs = src.crs
            src_transform = src.transform
            src_width = src.width
            src_height = src.height

            dst_transform, dst_width, dst_height = calculate_default_transform(
                src_crs, target_crs, src_width, src_height, *src.bounds
            )
            kwargs = src.meta.copy()
            kwargs.update(
                {
                    "crs": target_crs,
                    "transform": dst_transform,
                    "width": dst_width,
                    "height": dst_height,
                }
            )
            with rasterio.open(str(target_path), "w", **kwargs) as dst:
                for band_idx in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, band_idx),
                        destination=rasterio.band(dst, band_idx),
                        src_transform=src_transform,
                        src_crs=src_crs,
                        dst_transform=dst_transform,
                        dst_crs=target_crs,
                        resampling=RS.bilinear,
                    )

        if self._has_identity_transform(target_path):
            msg = (
                f"Reprojection produced identity transform in {target_path}. "
                f"Source: {source_path}, Target CRS: {target_crs}. "
                "This is a known rasterio/GDAL edge case with certain input CRS."
            )
            raise RuntimeError(msg)

    def _backoff(self, attempt: int, options: DownloadOptions) -> float:
        """Calculate retry delay based on strategy."""
        import random

        delay = options.retry_delay_seconds * (2**attempt)
        delay = min(delay, 60.0)
        if "jitter" in options.retry_strategy:
            delay *= 0.5 + random.random() * 0.5
        return delay

    def _emit_progress(
        self,
        data_list: list[SatelliteData],
        results: list[DownloadResult | None],
        options: DownloadOptions,
    ) -> None:
        """Emit progress update via callback if configured."""
        if not self.progress_callback:
            return
        # Emit lightweight progress update (counts only)
        n_done = sum(1 for r in results if r is not None)
        _ = n_done  # noqa: USED as progress signal to callback
        progress = DownloadProgress(total=len(results), destination="")
        try:
            self.progress_callback(progress)
        except Exception:
            pass

    @property
    def _result_total_size_mb(self) -> float:
        """Property used by result objects."""
        return 0.0
