"""
PyGeoVision — World-Class Geospatial AI Platform.

PyGeoVision unifies two world-class open-source packages:

  🛰️  pygeofetch  — Universal satellite data pipeline (22+ providers)
                    CLI: pygeofetch search/download/pipeline/auth/cache
                    Python: SatelliteFetcher (wraps pygeofetch CLI + pystac_client)

  🤖  geoai       — AI for geospatial data (PyTorch, transformers, SMP)
                    Segmentation, detection, classification, change detection,
                    embeddings, SAM, Prithvi, cloud masking, ONNX, and more.

Architecture:
  PyGeoVision = pygeofetch (data) + geoai (AI) + integration layer

  client.data.*   → pygeofetch: search, download, pipeline, auth, cache
  client.geoai.*  → geoai: segment, detect, classify, change, train, infer, ...
  client.pipeline() → end-to-end: data (pygeofetch) → AI (geoai) → output

Quick start:
    >>> import pygeovision as pgv
    >>>
    >>> client = pgv.PyGeoVision()
    >>>
    >>> # Add credentials (stored in system keyring via pygeofetch)
    >>> client.data.add_credentials("usgs", username="user", password="pass")
    >>> client.data.add_credentials("planet", api_key="PL_KEY")
    >>>
    >>> # Search 22+ satellite providers
    >>> results = client.search(
    ...     bbox=(-74.1, 40.6, -73.7, 40.9),
    ...     date_range=("2024-01-01", "2024-06-01"),
    ...     providers=["planetary_computer", "copernicus", "usgs"],
    ...     cloud_cover_max=15,
    ... )
    >>> print(f"Found {len(results)} scenes")
    >>>
    >>> # Download with post-processing
    >>> downloads = client.download(
    ...     results[:5],
    ...     output_dir="./data/",
    ...     post_process=["unzip", "reproject:EPSG:4326", "compress:lzw", "cog"],
    ... )
    >>>
    >>> # AI: segment buildings using geoai
    >>> masks = client.geoai.segment.buildings(
    ...     downloads[0].path,
    ...     output_vector="buildings.geojson",
    ... )
    >>>
    >>> # End-to-end pipeline: search → download → AI
    >>> result = client.pipeline(
    ...     "building_footprints",
    ...     bbox=(-74.1, 40.6, -73.7, 40.9),
    ...     date="2024-06",
    ... )
"""

from __future__ import annotations

import logging
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pygeovision._version import __version__
from pygeovision.core.exceptions import (  # noqa: F401
    PyGeoVisionError,
    PyGeoVisionConfigError,
    PyGeoVisionAuthError,
    AIEngineError,
    AINotAvailableError,
    ModelNotFoundError,
    TrainingError,
    InferenceError,
    PipelineError,
    LabelingError,
)
from pygeovision.core.config import PyGeoVisionConfig
from pygeovision.data.fetch import SatelliteFetcher, SearchResult, DownloadResult
from pygeovision.data.pipeline import DataPipeline
from pygeovision.datasets.registry import dataset_registry, DatasetRegistry, DatasetInfo
from pygeovision.ai.models.zoo import model_zoo, ModelZoo, ModelSpec
from pygeovision.ai.pipelines.domains import list_pipelines as list_all_pipelines

logger = logging.getLogger(__name__)

__all__ = [
    "PyGeoVision",
    "__version__",
    "PyGeoVisionError",
    "SatelliteFetcher",
    "SearchResult",
    "DownloadResult",
    "DataPipeline",
    "dataset_registry",
    "DatasetRegistry",
    "DatasetInfo",
    "model_zoo",
    "ModelZoo",
    "ModelSpec",
    "list_all_pipelines",
]


class PyGeoVision:
    """PyGeoVision — World-Class Geospatial AI Platform.

    Unified interface combining pygeofetch (satellite data) and
    geoai (geospatial AI) into one production-ready platform.

    Args:
        config_path: Path to PyGeoVision or pygeofetch config YAML.
        cache_dir: Override local cache directory.
        pygeofetch_cmd: Override the pygeofetch CLI command
            (e.g. 'PyGeoFetch' on some systems).
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR').

    Attributes:
        data: SatelliteFetcher — full pygeofetch Python API.
        geoai: GeoAIEngine — full geoai integration layer.
        config: PyGeoVisionConfig.

    Example — complete end-to-end workflow::

        import pygeovision as pgv

        client = pgv.PyGeoVision()

        # Authenticate with providers
        client.data.add_credentials("usgs", username="user", password="pass")
        client.data.add_credentials("copernicus",
            client_id="my-id", client_secret="my-secret")
        client.data.add_credentials("planet", api_key="PL_KEY")

        # Search satellite data across 22+ providers
        results = client.search(
            bbox=(-0.15, 51.47, -0.10, 51.52),
            date_range=("2024-06-01", "2024-06-30"),
            providers=["planetary_computer", "copernicus"],
            cloud_cover_max=10,
        )

        # Download with post-processing (delegates to PyGeoFetch)
        downloads = client.download(
            results[:3],
            output_dir="./sentinel2/",
            parallel=4,
            post_process=["unzip", "reproject:EPSG:4326", "cog"],
        )

        # GeoAI: segment buildings (delegates to geoai)
        masks = client.geoai.segment.buildings(
            downloads[0].path,
            output_path="buildings.tif",
            output_vector="buildings.geojson",
        )

        # End-to-end pipeline
        result = client.pipeline("building_footprints", bbox=..., date="2024-06")

        # YAML pipeline (delegates to PyGeoFetch pipeline run)
        client.data.run_pipeline("weekly-sentinel2.yaml")
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        cache_dir: Optional[Path] = None,
        pygeofetch_cmd: str = "pygeofetch",   # kept for backward compat, unused
        log_level: str = "INFO",
    ) -> None:
        logging.basicConfig(
            level=getattr(logging, log_level.upper(), logging.INFO),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

        self.config = PyGeoVisionConfig.load(config_path) if config_path else PyGeoVisionConfig()

        # Core data layer — wraps pygeofetch
        self.data = SatelliteFetcher(
            config_path=Path(config_path) if config_path else None,
            cache_dir=cache_dir,
        )

        self._ai_engine: Optional[Any] = None

        # ── Phase 2+ Independent layers ─────────────────────────────────
        # All accessible directly on the client object.

        # Auto-labeling — 7+ sources (OSM, MS Buildings, Google, ESA, SAM …)
        from pygeovision.labeling import (
            OSMLabeler, MicrosoftBuildingsLabeler, GoogleBuildingsLabeler,
            ESAWorldCoverLabeler, DynamicWorldLabeler,
            SAMAutoLabeler, FoundationModelLabeler,
            ActiveLearner, LabelQualityAssessor, AutoLabelPipeline,
        )
        self.labeling = _LabelingClientProxy()

        # Geospatial losses — Dice, Focal, Tversky, Boundary, Lovász, OHEM
        from pygeovision.losses import (
            DiceLoss, FocalLoss, TverskyLoss, ComboLoss,
            BoundaryAwareLoss, LovaszLoss, OhemCrossEntropy,
            GeospatialMixedLoss, ClassBalancedCrossEntropy,
        )
        self.losses = _LossesClientProxy()

        # Advanced inference — Gaussian tiling, batch, streaming, ensemble
        from pygeovision.inference import TiledInference, BatchInferenceEngine
        self.inference = _InferenceClientProxy()

        # Explainability — GradCAM, uncertainty, SHAP, attention maps
        self.xai = _XAIClientProxy()

        # Monitoring — drift detection, performance tracking, alerts
        self.monitoring = _MonitoringClientProxy()

        # Edge deployment — NVIDIA Jetson, ONNX Runtime
        self.edge = _EdgeClientProxy()

        # Cloud deployment — AWS SageMaker, Azure ML, GCP Vertex AI
        self.cloud = _CloudClientProxy()

        # Advanced AI — few-shot, multi-task, AutoML, VLM, time series, 3D
        self.few_shot   = _FewShotClientProxy()
        self.multitask  = _MultiTaskClientProxy()
        self.automl     = _AutoMLClientProxy()
        self.vlm        = _VLMClientProxy()
        self.timeseries = _TimeSeriesClientProxy()
        self.pointcloud = _PointCloudClientProxy()

        # ── NEW: Data Validation + Full Preprocessing Stack ─────────────
        # DataValidator — mandatory before every model run
        from pygeovision.data.validator import DataValidator
        self.validator = DataValidator(mode="fix")

        # Preprocessor — 100+ spatial preprocessing operations
        from pygeovision.preprocess import Preprocessor
        self.preprocess = Preprocessor(validator=self.validator)

        # SpectralIndices — 22 validated indices (NDVI, EVI, NBR, TCT, PCA…)
        from pygeovision.data.indices import SpectralIndices
        self.indices = SpectralIndices(validator=self.validator)

        # PostProcessor — 20+ prediction postprocessing operations
        from pygeovision.data.postprocess import PostProcessor
        self.postprocess = PostProcessor(validator=self.validator)

    # ------------------------------------------------------------------
    # Authentication (delegates to PyGeoFetch via self.data)
    # ------------------------------------------------------------------

    def add_credentials(
        self,
        provider: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        api_key: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
    ) -> "PyGeoVision":
        """Add satellite provider credentials (stored via pygeofetch keyring).

        Delegates to ``pygeofetch auth add PROVIDER ...`` for secure storage.

        Auth modes by provider:
            usgs, nasa_earthdata                   → username + password
            planet, opentopography, airbus_oneatlas → api_key
            copernicus, sentinel_hub, maxar_gbdx    → client_id + client_secret
            aws_earth, planetary_computer, element84 → no auth needed

        Args:
            provider: pygeofetch provider ID (22+ supported).
            username: For user/pass auth providers.
            password: For user/pass auth providers.
            api_key: For API key providers.
            client_id: For OAuth2 providers.
            client_secret: For OAuth2 providers.

        Returns:
            Self (for method chaining).

        Example:
            >>> client \\
            ...   .add_credentials("usgs", username="user", password="pass") \\
            ...   .add_credentials("planet", api_key="PL_KEY") \\
            ...   .add_credentials("copernicus",
            ...       client_id="my-id", client_secret="my-secret")
        """
        self.data.add_credentials(
            provider, username=username, password=password,
            api_key=api_key, client_id=client_id, client_secret=client_secret,
        )
        return self

    # ------------------------------------------------------------------
    # Search (delegates to PyGeoFetch via self.data)
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
        limit: Optional[int] = None,          # alias for max_results
        sort_by: str = "datetime",
        sort_order: str = "desc",
        processing_level: Optional[str] = None,
        resolution_range: Optional[Tuple[float, float]] = None,
        cql2_filter: Optional[str] = None,
        on_provider_failure: str = "skip",
        timeout: int = 120,
        use_cache: bool = True,
    ) -> List[SearchResult]:
        """Search for satellite imagery across 22+ pygeofetch providers.

        Delegates to ``pygeofetch search run`` (CLI) with pystac_client
        as fallback for STAC providers.

        All 22 providers supported:
            Open access (no credentials):
                planetary_computer, aws_earth, element84, noaa_big_data,
                esa_scihub, jaxa_earth, isro_bhuvan, inpe_cbers,
                digitalglobe, geoserver_generic

            Requires credentials:
                usgs, copernicus, nasa_earthdata, nasa_earthdata_cloud,
                opentopography, planet, sentinel_hub, maxar_gbdx,
                airbus_oneatlas, alaska_satellite_facility,
                google_earth_engine, terrabotics

        Args:
            bbox: (min_lon, min_lat, max_lon, max_lat) in WGS84.
            date_range: (start_date, end_date) as 'YYYY-MM-DD'.
            collections: STAC collection IDs e.g. ['sentinel-2-l2a'].
            providers: pygeofetch provider IDs. Auto-selected from
                       collections/satellite if not specified.
            satellite: Shortcut name ('sentinel-2', 'landsat', 'planet',
                       'worldview', 'pleiades', 'dem', 'modis', etc.)
            cloud_cover_max: Max cloud cover % (0–100).
            max_results: Maximum scenes to return.
            sort_by: 'datetime', 'cloud_cover', 'score', 'satellite'.
            sort_order: 'asc' or 'desc'.
            processing_level: 'L2A', 'L1C', 'L1TP', etc.
            resolution_range: (min_m, max_m) spatial resolution filter.
            cql2_filter: CQL2 expression for advanced filtering.
            on_provider_failure: 'skip', 'abort', or 'retry'.
            timeout: HTTP timeout in seconds.
            use_cache: Use 1-hour result cache.

        Returns:
            List of SearchResult objects.

        Example:
            >>> # Open access — no credentials needed
            >>> results = client.search(
            ...     bbox=(-0.15, 51.47, -0.10, 51.52),
            ...     date_range=("2024-06-01", "2024-06-30"),
            ...     collections=["sentinel-2-l2a"],
            ...     cloud_cover_max=10,
            ... )
            >>> for r in results[:5]:
            ...     print(r)
        """
        # `limit` is a convenience alias for max_results (matches STAC convention)
        if limit is not None:
            max_results = limit
        return self.data.search(
            bbox=bbox,
            date_range=date_range,
            collections=collections,
            providers=providers,
            satellite=satellite,
            cloud_cover_max=cloud_cover_max,
            max_results=max_results,
            sort_by=sort_by,
            sort_order=sort_order,
            processing_level=processing_level,
            resolution_range=resolution_range,
            cql2_filter=cql2_filter,
            on_provider_failure=on_provider_failure,
            timeout=timeout,
            use_cache=use_cache,
        )

    # ------------------------------------------------------------------
    # Download (delegates to PyGeoFetch via self.data)
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
        bands: Optional[List[str]] = None,    # filter assets by band names
    ) -> List[DownloadResult]:
        """Download satellite scenes via pygeofetch.

        Delegates to ``pygeofetch download run`` for resilient, parallel
        downloading with full post-processing support.

        Post-processing actions (chained in order):
            unzip                    Extract ZIP/TAR archives
            reproject:EPSG:4326      Reproject to target CRS
            compress:lzw             Apply compression
            ndvi                     Compute NDVI
            ndwi                     Compute NDWI
            composite                Temporal composite
            atmospheric:sen2cor      Atmospheric correction
            clip:area.geojson        Clip to geometry
            resample:10              Resample to N metres
            cog                      Cloud Optimized GeoTIFF
            merge                    Merge overlapping scenes
            pan-sharpen              Pan-sharpen multispectral

        Args:
            items: SearchResult(s) from search().
            output_dir: Local download directory.
            parallel: Concurrent downloads (default 4).
            verify_checksum: SHA256 verify each file post-download.
            resume: Auto-resume interrupted downloads.
            retry_attempts: Max retries per file (exponential backoff).
            post_process: Processing chain (list of action strings).
            bandwidth_limit_mb: Download throttle in MB/s.
            on_failure: 'skip', 'abort', or 'retry'.
            overwrite: Overwrite existing files.
            notify_webhook: Slack/Teams webhook for completion notification.

        Returns:
            List of DownloadResult objects.

        Example:
            >>> results = client.search(bbox=..., date_range=...)
            >>> downloads = client.download(
            ...     results[:5],
            ...     output_dir="./sentinel2/",
            ...     parallel=4,
            ...     post_process=["unzip", "reproject:EPSG:4326", "compress:lzw", "cog"],
            ...     verify_checksum=True,
            ... )
            >>> for d in downloads:
            ...     print(d)
        """
        # If bands requested, filter assets on each SearchResult so that
        # only those band assets are passed to pygeofetch's download engine.
        if bands:
            for item in (items if isinstance(items, list) else [items]):
                if item.assets:
                    item.assets = {
                        k: v for k, v in item.assets.items()
                        if k in bands or any(b.lower() in k.lower() for b in bands)
                    }
        return self.data.download(
            items=items,
            output_dir=output_dir,
            parallel=parallel,
            verify_checksum=verify_checksum,
            resume=resume,
            retry_attempts=retry_attempts,
            post_process=post_process,
            bandwidth_limit_mb=bandwidth_limit_mb,
            on_failure=on_failure,
            overwrite=overwrite,
            notify_webhook=notify_webhook,
        )

    # ------------------------------------------------------------------
    # GeoAI (delegates to geoai via GeoAIEngine)
    # ------------------------------------------------------------------

    @property
    def geoai(self) -> Any:
        """Access the full GeoAI integration layer.

        All geoai capabilities exposed as organised subsystems.
        Requires: pip install geoai-py

        Subsystems:
            .segment    Buildings, solar, water, agriculture, SAM, custom
            .detect     Cars, ships, parking, grounded SAM, RF-DETR
            .classify   Scene, land cover, CLIP zero-shot, batch
            .change     ChangeSTAR bi-temporal change detection
            .train      Segmentation, detection, classification, chips
            .infer      Tiled GeoTIFF inference with blend modes
            .embed      DINOv3, Tessera, patch/pixel embeddings
            .sam        Segment Anything Model
            .prithvi    NASA Prithvi foundation model
            .cloud      Cloud masking and statistics
            .sr         ESRGAN super-resolution
            .onnx       ONNX export and inference
            .download   NAIP, Overture Maps, Planetary Computer
            .utils      Raster/vector/metrics utilities
            .pipeline   GeoAI pipeline orchestration
            .map        Leafmap interactive visualization
            .caption    Moondream VLM captioning
            .water      Water body segmentation
            .rfdetr     RF-DETR real-time detection
            .timm       timm-based segmentation/regression
            .landcover  Land cover training
            .canopy     Canopy height estimation
            .dinov3     DINOv3 analysis and fine-tuning
            .tessera    Tessera satellite embeddings

        Example:
            >>> # Segment buildings (geoai.BuildingFootprintExtractor)
            >>> client.geoai.segment.buildings(
            ...     "sentinel2.tif",
            ...     output_vector="buildings.geojson",
            ... )
            >>> # Change detection (geoai.changestar_detect)
            >>> client.geoai.change.detect("2020.tif", "2024.tif")
            >>> # Train segmentation model (geoai.train_segmentation_model)
            >>> client.geoai.train.segmentation(
            ...     "./chips/", "model.pth", num_classes=5
            ... )
        """
        if self._ai_engine is None:
            from pygeovision.ai.geoai import GeoAIEngine
            self._ai_engine = GeoAIEngine(pgv_client=self)
        return self._ai_engine

    # ------------------------------------------------------------------
    # Pipelines (data + AI end-to-end)
    # ------------------------------------------------------------------

    def pipeline(
        self,
        pipeline_name: str,
        bbox: Tuple[float, float, float, float],
        output_dir: Union[str, Path] = "./pipeline_output",
        **kwargs: Any,
    ) -> Any:
        """Run an end-to-end geospatial pipeline (data + AI).

        Downloads imagery via pygeofetch then runs geoai AI model.

        Available pipelines:
            change_detection     Bi-temporal change detection
            land_cover           Global land cover (ESA WorldCover / geoai)
            building_footprints  Building segmentation (geoai)
            crop_monitoring      Crop type mapping (geoai)
            disaster_assessment  Rapid damage assessment (geoai)
            deforestation        Forest loss detection (geoai)
            urban_growth         Urban expansion monitoring (geoai)
            water_bodies         Surface water mapping (geoai/NDWI)
            solar_detection      Solar panel detection (geoai)
            carbon_estimation    Biomass/carbon via NDVI (pygeofetch+geoai)

        Args:
            pipeline_name: Pipeline name from the list above.
            bbox: (min_lon, min_lat, max_lon, max_lat).
            output_dir: Output directory for results.
            **kwargs: Pipeline-specific arguments (date, date_before,
                      date_after, model, source, method, etc.).

        Returns:
            PipelineResult with output_path and stats.

        Example:
            >>> result = client.pipeline(
            ...     "building_footprints",
            ...     bbox=(-0.15, 51.47, -0.10, 51.52),
            ...     date="2024-06",
            ... )
            >>> print(f"Coverage: {result.stats['building_coverage']:.1%}")

            >>> result = client.pipeline(
            ...     "change_detection",
            ...     bbox=(-74.1, 40.6, -73.7, 40.9),
            ...     date_before="2020-01",
            ...     date_after="2024-01",
            ... )
        """
        from pygeovision.ai.pipelines import get_pipeline
        p = get_pipeline(pipeline_name, pgv_client=self)
        return p.run(bbox=bbox, output_dir=output_dir, **kwargs)

    def create_pipeline(
        self,
        name: str,
        description: str = "",
        schedule: Optional[str] = None,
    ) -> DataPipeline:
        """Create a new pygeofetch YAML data pipeline programmatically.

        Example::

            pipeline = client.create_pipeline("weekly-sentinel2")
            pipeline.search(
                providers=["planetary_computer", "copernicus"],
                bbox=(-74.1, 40.6, -73.7, 40.9),
                date_range="last_7_days",
                cloud_cover="0-10",
            ).filter(
                "data.cloud_cover < 5"
            ).download(
                parallel=4,
                output="./raw/",
                post_process=["unzip", "reproject:EPSG:4326", "cog"],
            ).export(
                format="cloud_optimized_geotiff",
                destination="s3://my-bucket/",
            ).schedule("0 6 * * 1")
            pipeline.run()
        """
        return DataPipeline(name=name, description=description, schedule=schedule)

    def run_pipeline_yaml(
        self,
        pipeline_yaml: Union[str, Path],
        step: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run a pygeofetch YAML pipeline file.

        Delegates to ``pygeofetch pipeline run FILE``.

        Example pipeline YAML::

            name: weekly-sentinel2-ndvi
            schedule: "0 6 * * 1"
            steps:
              - search:
                  providers: [copernicus, aws_earth, planetary_computer]
                  date_range: last_7_days
                  cloud_cover: 0-10
                  bbox: "-74.1,40.6,-73.7,40.9"
              - filter:
                  expression: "data.cloud_cover < 5"
              - download:
                  parallel: 4
                  output: ./raw/
                  verify_checksum: true
              - export:
                  format: cloud_optimized_geotiff
                  destination: s3://my-bucket/ndvi/

        Args:
            pipeline_yaml: Path to YAML pipeline file.
            step: Run only a specific named step.

        Returns:
            Dict with run summary.
        """
        return self.data.run_pipeline(pipeline_yaml, step=step)

    # ------------------------------------------------------------------
    # Provider management (delegates to PyGeoFetch)
    # ------------------------------------------------------------------

    def list_providers(
        self,
        auth_only: bool = False,
        open_only: bool = False,
        capabilities: Optional[List[str]] = None,
    ) -> Dict[str, Dict]:
        """List all 22 pygeofetch satellite data providers.

        Args:
            auth_only: Only providers requiring authentication.
            open_only: Only open-access (no credentials needed) providers.
            capabilities: Filter by: 'sar', 'optical', 'stac', 'sub_meter'.

        Returns:
            Dict of provider_id → metadata (name, satellites, auth, etc.).
        """
        return self.data.list_providers(
            auth_only=auth_only, open_only=open_only, capabilities=capabilities
        )

    def test_provider(self, provider: str) -> bool:
        """Test connectivity to a pygeofetch provider."""
        return self.data.test_provider(provider)

    # ------------------------------------------------------------------
    # Cache management (delegates to PyGeoFetch)
    # ------------------------------------------------------------------

    def clear_cache(
        self,
        provider: Optional[str] = None,
        older_than: Optional[str] = None,
    ) -> None:
        """Clear pygeofetch search result cache.

        Args:
            provider: Clear only this provider's cache (None = all).
            older_than: Clear entries older than duration (e.g. '7d', '1h').
        """
        self.data.clear_cache(provider=provider, older_than=older_than)

    def cache_stats(self) -> Dict[str, Any]:
        """Get pygeofetch cache statistics."""
        return self.data.cache_stats()

    # ------------------------------------------------------------------
    # System status
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """Return full PyGeoVision system status.

        Includes: pygeofetch version, geoai availability, torch,
        rasterio, registered AI models, and provider count.
        """
        info: Dict[str, Any] = {
            "pygeovision_version": __version__,
            "python": platform.python_version(),
            "platform": platform.system(),
        }

        # pygeofetch status
        pf_status = self.data.status()
        info["pygeofetch"] = {
            "available": self.data._has_pygeofetch(),
            "version": self.data._pygeofetch_version(),
            "providers": 22,
            "open_providers": len([p for p in pf_status.get("open_providers", [])]),
        }

        # geoai status
        try:
            import geoai
            info["geoai"] = {
                "available": True,
                "version": getattr(geoai, "__version__", "unknown"),
            }
        except ImportError:
            info["geoai"] = {"available": False}

        # torch status
        try:
            import torch
            info["torch"] = {
                "version": torch.__version__,
                "cuda": torch.cuda.is_available(),
                "device": "cuda" if torch.cuda.is_available() else (
                    "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                    else "cpu"
                ),
            }
            if torch.cuda.is_available():
                info["torch"]["gpu"] = torch.cuda.get_device_name(0)
        except ImportError:
            info["torch"] = {"available": False}

        # rasterio status
        try:
            import rasterio
            info["rasterio"] = rasterio.__version__
        except ImportError:
            info["rasterio"] = None

        # geopandas status
        try:
            import geopandas
            info["geopandas"] = geopandas.__version__
        except ImportError:
            info["geopandas"] = None

        # AI model registry
        try:
            from pygeovision.ai.models.registry import registry
            info["registered_ai_models"] = len(registry)
        except Exception:
            info["registered_ai_models"] = 0

        return info

    def doctor(self) -> Dict[str, Any]:
        """Run comprehensive diagnostics on pygeofetch + geoai installation."""
        return self.data.doctor()

    def __repr__(self) -> str:
        pgf = "✓" if self.data._has_pygeofetch() else "✗"
        try:
            import geoai
            ga = "✓+independent"
        except ImportError:
            ga = "independent"
        from pygeovision.datasets.registry import dataset_registry
        from pygeovision.ai.models.zoo import model_zoo
        from pygeovision.ai.pipelines.domains import list_pipelines
        return (
            f"PyGeoVision(v{__version__} | "
            f"pygeofetch={pgf} | geoai={ga} | "
            f"datasets={len(dataset_registry)} | models={len(model_zoo)} | "
            f"pipelines={len(list_pipelines())} | "
            f"labeling=7src | losses=10 | inference=4 | "
            f"xai=4 | monitoring=3 | cloud=3·aws·azure·gcp | "
            f"edge=ONNX+Jetson | vlm=CLIP+Moon | few_shot | timeseries | 3D)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2+ client proxy classes — thin facades on the new independent layers
# ─────────────────────────────────────────────────────────────────────────────

class _LabelingClientProxy:
    """client.labeling — 7+ auto-labeling sources."""
    def osm(self, bbox, categories=None, output_path="./labels/osm.tif", **kw):
        from pygeovision.labeling.osm import OSMLabeler
        return OSMLabeler().label(bbox, categories=categories, output_path=output_path, **kw)
    def microsoft_buildings(self, bbox, output_path="./labels/ms_buildings.tif", **kw):
        from pygeovision.labeling.buildings import MicrosoftBuildingsLabeler
        return MicrosoftBuildingsLabeler().label(bbox, output_path=output_path, **kw)
    def google_buildings(self, bbox, output_path="./labels/google_buildings.tif", **kw):
        from pygeovision.labeling.buildings import GoogleBuildingsLabeler
        return GoogleBuildingsLabeler().label(bbox, output_path=output_path, **kw)
    def esa_worldcover(self, bbox, output_path="./labels/esa_worldcover.tif", **kw):
        from pygeovision.labeling.landcover import ESAWorldCoverLabeler
        return ESAWorldCoverLabeler().label(bbox, output_path=output_path, **kw)
    def dynamic_world(self, bbox, date_range=None, output_path="./labels/dynamic_world.tif", **kw):
        from pygeovision.labeling.landcover import DynamicWorldLabeler
        return DynamicWorldLabeler().label(bbox, date_range=date_range or ("2024-01-01","2024-12-31"),
                                            output_path=output_path, **kw)
    def sam_auto(self, image_path, output_path="./labels/sam.tif", **kw):
        from pygeovision.labeling.sam_auto import SAMAutoLabeler
        return SAMAutoLabeler().auto_label(image_path, output_path=output_path, **kw)
    def foundation(self, image_path, output_path="./labels/foundation.tif", **kw):
        from pygeovision.labeling.foundation import FoundationModelLabeler
        return FoundationModelLabeler().pseudo_label(image_path, output_path, **kw)
    def pipeline(self, bbox, sources=None, output_dir="./labels/", **kw):
        from pygeovision.labeling.pipeline import AutoLabelPipeline
        return AutoLabelPipeline(sources=sources).run(bbox, output_dir=output_dir, **kw)
    def quality(self, label_path):
        from pygeovision.labeling.quality import LabelQualityAssessor
        return LabelQualityAssessor().assess(label_path)
    def active_learner(self, strategy="entropy", budget=100):
        from pygeovision.labeling.active import ActiveLearner
        return ActiveLearner(strategy=strategy, budget=budget)
    def __repr__(self): return "LabelingLayer(osm|ms_buildings|google_buildings|esa_worldcover|dynamic_world|sam_auto|foundation|pipeline|quality|active)"


class _LossesClientProxy:
    """client.losses — geospatial-specific loss functions."""
    @property
    def dice(self): from pygeovision.losses.segmentation import DiceLoss; return DiceLoss()
    @property
    def focal(self): from pygeovision.losses.segmentation import FocalLoss; return FocalLoss()
    @property
    def tversky(self): from pygeovision.losses.segmentation import TverskyLoss; return TverskyLoss()
    @property
    def combo(self): from pygeovision.losses.segmentation import ComboLoss; return ComboLoss()
    @property
    def boundary(self): from pygeovision.losses.segmentation import BoundaryAwareLoss; return BoundaryAwareLoss()
    @property
    def lovasz(self): from pygeovision.losses.segmentation import LovaszLoss; return LovaszLoss()
    @property
    def ohem(self): from pygeovision.losses.segmentation import OhemCrossEntropy; return OhemCrossEntropy()
    @property
    def mixed(self): from pygeovision.losses.segmentation import GeospatialMixedLoss; return GeospatialMixedLoss()
    @property
    def ciou(self): from pygeovision.losses.detection import CIoULoss; return CIoULoss()
    @property
    def class_balanced(self): from pygeovision.losses.class_balance import ClassBalancedCrossEntropy; return ClassBalancedCrossEntropy()
    def get(self, name, **kw):
        MAP = {"dice": "DiceLoss", "focal": "FocalLoss", "tversky": "TverskyLoss",
               "combo": "ComboLoss", "boundary": "BoundaryAwareLoss", "lovasz": "LovaszLoss",
               "ohem": "OhemCrossEntropy", "mixed": "GeospatialMixedLoss"}
        if name not in MAP:
            raise ValueError(f"Loss '{name}' not found. Available: {list(MAP)}")
        mod = __import__("pygeovision.losses.segmentation", fromlist=[MAP[name]])
        return getattr(mod, MAP[name])(**kw)
    def __repr__(self): return "LossesLayer(dice|focal|tversky|combo|boundary|lovasz|ohem|mixed|ciou|class_balanced)"


class _InferenceClientProxy:
    """client.inference — advanced tiled/batch/streaming inference."""
    def tiled(self, model, chip_size=512, overlap=128, blend_mode="gaussian", **kw):
        from pygeovision.inference.tiled import TiledInference
        return TiledInference(model=model, chip_size=chip_size, overlap=overlap,
                               blend_mode=blend_mode, **kw)
    def batch(self, model, n_workers=4, **kw):
        from pygeovision.inference.batch import BatchInferenceEngine
        return BatchInferenceEngine(model=model, n_workers=n_workers, **kw)
    def streaming(self, model, chip_size=1024, **kw):
        from pygeovision.inference.stream import StreamingInference
        return StreamingInference(model=model, chip_size=chip_size, **kw)
    def ensemble(self, models, weights=None, fusion="mean", **kw):
        from pygeovision.inference.stream import EnsembleInference
        return EnsembleInference(models=models, weights=weights, fusion=fusion, **kw)
    def gaussian_blend(self, size=512, sigma_ratio=0.25):
        from pygeovision.inference.tiled import GaussianBlend
        return GaussianBlend.window(size, sigma_ratio)
    def __repr__(self): return "InferenceLayer(tiled|batch|streaming|ensemble|gaussian_blend)"


class _XAIClientProxy:
    """client.xai — explainability for geospatial models."""
    def gradcam(self, model, target_layer=None):
        from pygeovision.explainability.gradcam import GradCAM
        return GradCAM(model, target_layer)
    def gradcam_pp(self, model, target_layer=None):
        from pygeovision.explainability.gradcam import GradCAMPlusPlus
        return GradCAMPlusPlus(model, target_layer)
    def uncertainty(self, model, n_passes=20):
        from pygeovision.explainability.uncertainty import UncertaintyEstimator
        return UncertaintyEstimator(model, n_passes)
    def attention(self, model):
        from pygeovision.explainability.attention import AttentionMapExtractor
        return AttentionMapExtractor(model)
    def shap(self, model):
        from pygeovision.explainability.shap_geo import GeospatialSHAP
        return GeospatialSHAP(model)
    def __repr__(self): return "XAILayer(gradcam|gradcam_pp|uncertainty|attention|shap)"


class _MonitoringClientProxy:
    """client.monitoring — drift detection and performance tracking."""
    def drift_detector(self, model=None):
        from pygeovision.monitoring.drift import DriftDetector
        return DriftDetector(model=model)
    def performance_tracker(self, model_name="model"):
        from pygeovision.monitoring.tracker import ModelPerformanceTracker
        return ModelPerformanceTracker(model_name=model_name)
    def alert_manager(self, channels=None):
        from pygeovision.monitoring.alerts import AlertManager
        return AlertManager(channels=channels)
    def __repr__(self): return "MonitoringLayer(drift_detector|performance_tracker|alert_manager)"


class _EdgeClientProxy:
    """client.edge — edge deployment (Jetson, ONNX Runtime)."""
    def onnx_runtime(self, onnx_path, device="cpu"):
        from pygeovision.edge.onnx_rt import ONNXRuntimeInference
        return ONNXRuntimeInference(onnx_path, device=device)
    def export_onnx(self, model, output_path, input_shape=(1,4,512,512), **kw):
        from pygeovision.edge.onnx_rt import ONNXRuntimeInference
        return ONNXRuntimeInference.from_pytorch(model, output_path, input_shape, **kw)
    def jetson(self):
        from pygeovision.edge.jetson import JetsonDeployer
        return JetsonDeployer()
    def __repr__(self): return "EdgeLayer(onnx_runtime|export_onnx|jetson)"


class _CloudClientProxy:
    """client.cloud — cloud deployment (AWS, Azure, GCP)."""
    def aws(self, region="us-east-1", **kw):
        from pygeovision.cloud.deploy import AWSDeployer
        return AWSDeployer(region=region, **kw)
    def azure(self, **kw):
        from pygeovision.cloud.deploy import AzureDeployer
        return AzureDeployer(**kw)
    def gcp(self, project_id=None, region="us-central1"):
        from pygeovision.cloud.deploy import GCPDeployer
        return GCPDeployer(project_id=project_id, region=region)
    def deploy(self, provider, model_path, endpoint_name, **kw):
        from pygeovision.cloud.deploy import CloudDeployer
        return CloudDeployer.from_provider(provider, **kw).deploy(model_path, endpoint_name)
    def __repr__(self): return "CloudLayer(aws|azure|gcp|deploy)"


class _FewShotClientProxy:
    """client.few_shot — few-shot learning for geospatial classification."""
    def learner(self, backbone="dinov2-base", method="prototypical"):
        from pygeovision.advanced.few_shot import FewShotLearner
        return FewShotLearner(backbone=backbone, method=method)
    def __repr__(self): return "FewShotLayer(learner)"


class _MultiTaskClientProxy:
    """client.multitask — multi-task model training."""
    def model(self, backbone="resnet50", tasks=None, n_classes=None):
        from pygeovision.advanced.multitask import MultiTaskLearner
        return MultiTaskLearner(backbone=backbone, tasks=tasks, n_classes=n_classes)
    def __repr__(self): return "MultiTaskLayer(model)"


class _AutoMLClientProxy:
    """client.automl — automated hyperparameter optimisation."""
    def optimizer(self, metric="val_iou", n_trials=50, backend="optuna"):
        from pygeovision.advanced.automl import GeoAutoML
        return GeoAutoML(metric=metric, n_trials=n_trials, backend=backend)
    def __repr__(self): return "AutoMLLayer(optimizer)"


class _VLMClientProxy:
    """client.vlm — vision-language models (CLIP, Moondream)."""
    def clip(self, model="remoteclip-b32"):
        from pygeovision.advanced.vlm.clip_geo import CLIPGeo
        return CLIPGeo(model=model)
    def moondream(self):
        from pygeovision.advanced.vlm.moondream_geo import MoondreamGeo
        return MoondreamGeo()
    def retrieval(self, model="openclip-b32"):
        from pygeovision.advanced.vlm.retrieval import GeoImageRetrieval
        return GeoImageRetrieval(model=model)
    def zero_shot(self, image_path, categories, model="remoteclip-b32"):
        return self.clip(model).zero_shot(image_path, categories)
    def caption(self, image_path):
        return self.moondream().caption(image_path)
    def vqa(self, image_path, question):
        return self.moondream().vqa(image_path, question)
    def __repr__(self): return "VLMLayer(clip|moondream|retrieval|zero_shot|caption|vqa)"


class _TimeSeriesClientProxy:
    """client.timeseries — temporal analysis of satellite image stacks."""
    def analyzer(self, sensor="sentinel2"):
        from pygeovision.advanced.timeseries import GeoTimeSeries
        return GeoTimeSeries(sensor=sensor)
    def ndvi_series(self, image_paths, date_strings=None, sensor="sentinel2"):
        from pygeovision.advanced.timeseries import GeoTimeSeries
        return GeoTimeSeries(sensor).compute_index_series(image_paths, "ndvi", date_strings)
    def index_series(self, image_paths, index="ndvi", date_strings=None, sensor="sentinel2"):
        from pygeovision.advanced.timeseries import GeoTimeSeries
        return GeoTimeSeries(sensor).compute_index_series(image_paths, index, date_strings)
    def __repr__(self): return "TimeSeriesLayer(analyzer|ndvi_series|index_series)"


class _PointCloudClientProxy:
    """client.pointcloud — 3D LiDAR and point cloud processing."""
    def processor(self):
        from pygeovision.advanced.pointcloud import PointCloudProcessor
        return PointCloudProcessor()
    def canopy_height_model(self, las_path, output_path, resolution=1.0):
        return self.processor().canopy_height_model(las_path, output_path, resolution)
    def __repr__(self): return "PointCloudLayer(processor|canopy_height_model)"
