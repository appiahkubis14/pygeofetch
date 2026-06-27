"""
PyGeoVision Data Pipeline — YAML-based satellite data workflows.

Wraps pygeofetch pipeline functionality with a Python API for
creating, running, validating and scheduling data pipelines.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

logger = logging.getLogger(__name__)


@dataclass
class PipelineStep:
    """A single step in a PyGeoVision data pipeline."""
    type: str  # 'search', 'filter', 'download', 'export', 'ai'
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {self.type: self.config}


class DataPipeline:
    """Build and run pygeofetch data pipelines programmatically.

    Creates YAML pipeline files compatible with ``pygeofetch pipeline run``
    and executes them, with optional GeoAI processing steps.

    Example:
        >>> pipeline = DataPipeline("weekly-sentinel2")
        >>> pipeline.search(
        ...     providers=["planetary_computer", "copernicus"],
        ...     bbox=(-74.1, 40.6, -73.7, 40.9),
        ...     date_range="last_7_days",
        ...     cloud_cover="0-10",
        ... )
        >>> pipeline.filter("data.cloud_cover < 5")
        >>> pipeline.download(parallel=4, output="./raw/", verify_checksum=True)
        >>> pipeline.export(format="cloud_optimized_geotiff", destination="s3://bucket/")
        >>> pipeline.schedule("0 6 * * 1")  # Every Monday 6am UTC
        >>> pipeline.run()
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        schedule: Optional[str] = None,
    ) -> None:
        self.name = name
        self.description = description
        self.schedule = schedule
        self.steps: List[PipelineStep] = []
        self._fetcher: Optional[Any] = None

    def search(
        self,
        providers: Optional[List[str]] = None,
        bbox: Optional[Union[str, tuple]] = None,
        date_range: Union[str, tuple] = "last_7_days",
        cloud_cover: str = "0-20",
        max_results: int = 50,
        satellites: Optional[List[str]] = None,
        collections: Optional[List[str]] = None,
        processing_level: Optional[str] = None,
        sort_by: str = "cloud_cover",
        cql2: Optional[str] = None,
    ) -> "DataPipeline":
        """Add a search step to the pipeline.

        Args:
            providers: pygeofetch provider IDs.
            bbox: Bounding box as 'minx,miny,maxx,maxy' string or tuple.
            date_range: Date range string ('last_7_days', 'last_30_days',
                        'this_month') or (start, end) tuple.
            cloud_cover: Cloud cover range as 'MIN-MAX' (e.g. '0-15').
            max_results: Maximum scenes to find.
            satellites: Satellite names filter.
            collections: STAC collection IDs.
            processing_level: e.g. 'L2A', 'L1C'.
            sort_by: 'cloud_cover', 'datetime', 'score'.
            cql2: CQL2 filter expression.

        Returns:
            Self (for chaining).
        """
        config: Dict[str, Any] = {
            "cloud_cover": cloud_cover,
            "max_results": max_results,
            "sort_by": sort_by,
        }
        if providers:
            config["providers"] = providers
        if bbox:
            config["bbox"] = (
                f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
                if isinstance(bbox, (tuple, list))
                else bbox
            )
        if isinstance(date_range, tuple):
            config["start_date"] = date_range[0]
            config["end_date"] = date_range[1]
        else:
            config["date_range"] = date_range
        if satellites:
            config["satellites"] = satellites
        if collections:
            config["collections"] = collections
        if processing_level:
            config["processing_level"] = processing_level
        if cql2:
            config["cql2"] = cql2

        self.steps.append(PipelineStep("search", config))
        return self

    def filter(self, expression: str) -> "DataPipeline":
        """Add a filter step to the pipeline.

        Args:
            expression: Python-like filter expression on result properties.
                Examples:
                - ``"data.cloud_cover < 5"``
                - ``"data.satellite in ['Sentinel-2A', 'Sentinel-2B']"``
                - ``"data.score > 0.9"``

        Returns:
            Self (for chaining).
        """
        self.steps.append(PipelineStep("filter", {"expression": expression}))
        return self

    def download(
        self,
        output: str = "./raw/",
        parallel: int = 4,
        verify_checksum: bool = True,
        resume: bool = True,
        retry: int = 5,
        post_process: Optional[List[str]] = None,
        bandwidth_limit: Optional[str] = None,
        on_failure: str = "skip",
    ) -> "DataPipeline":
        """Add a download step to the pipeline.

        Args:
            output: Local output directory.
            parallel: Concurrent download workers.
            verify_checksum: SHA256 verify each file.
            resume: Auto-resume interrupted downloads.
            retry: Max retry attempts.
            post_process: Processing chain (e.g. ['unzip', 'reproject:EPSG:4326', 'cog']).
            bandwidth_limit: Throttle (e.g. '10MB', '500KB').
            on_failure: 'skip', 'abort', or 'retry'.

        Returns:
            Self (for chaining).
        """
        config: Dict[str, Any] = {
            "output": output,
            "parallel": parallel,
            "verify_checksum": verify_checksum,
            "resume": resume,
            "retry": retry,
            "on_failure": on_failure,
        }
        if post_process:
            config["post_process"] = ",".join(post_process)
        if bandwidth_limit:
            config["bandwidth_limit"] = bandwidth_limit
        self.steps.append(PipelineStep("download", config))
        return self

    def export(
        self,
        format: str = "cloud_optimized_geotiff",
        destination: str = "./output/",
        compress: bool = True,
    ) -> "DataPipeline":
        """Add an export step to the pipeline.

        Args:
            format: 'cloud_optimized_geotiff', 'geotiff', 'netcdf', 'zarr'.
            destination: Local path or S3/GCS URI.
            compress: Apply LZW compression.

        Returns:
            Self (for chaining).
        """
        self.steps.append(PipelineStep("export", {
            "format": format,
            "destination": destination,
            "compress": compress,
        }))
        return self

    def ai_process(
        self,
        model: str,
        task: str = "segmentation",
        output: str = "./predictions/",
        num_classes: int = 2,
    ) -> "DataPipeline":
        """Add a GeoAI processing step to the pipeline.

        Args:
            model: GeoAI model name or HuggingFace Hub ID.
            task: 'segmentation', 'detection', 'classification'.
            output: Output directory for AI predictions.
            num_classes: Number of output classes.

        Returns:
            Self (for chaining).
        """
        self.steps.append(PipelineStep("ai", {
            "model": model,
            "task": task,
            "output": output,
            "num_classes": num_classes,
        }))
        return self

    def set_schedule(self, cron: str) -> "DataPipeline":
        """Set a cron schedule for recurring execution.

        Args:
            cron: Cron expression (e.g. '0 6 * * 1' = every Monday at 6am UTC).

        Returns:
            Self (for chaining).
        """
        self.schedule = cron
        return self

    def to_yaml(self) -> str:
        """Convert the pipeline to a pygeofetch-compatible YAML string."""
        doc: Dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "steps": [step.to_dict() for step in self.steps],
        }
        if self.schedule:
            doc["schedule"] = self.schedule
        return yaml.dump(doc, default_flow_style=False, sort_keys=False)

    def save(self, path: Union[str, Path]) -> Path:
        """Save the pipeline YAML to disk.

        Args:
            path: Output YAML file path.

        Returns:
            Path to the saved file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_yaml())
        logger.info("Pipeline saved to %s", path)
        return path

    def validate(self, fetcher: Optional[Any] = None) -> bool:
        """Validate the pipeline without running it.

        Args:
            fetcher: SatelliteFetcher instance (optional).

        Returns:
            True if valid.
        """
        if not self.steps:
            logger.warning("Pipeline '%s' has no steps.", self.name)
            return False

        if fetcher and fetcher._has_pygeofetch():
            with __import__("tempfile").NamedTemporaryFile(
                suffix=".yaml", delete=False, mode="w"
            ) as f:
                f.write(self.to_yaml())
                tmp = Path(f.name)
            try:
                return fetcher.validate_pipeline(tmp)
            finally:
                tmp.unlink(missing_ok=True)
        return True

    def run(
        self,
        fetcher: Optional[Any] = None,
        output_dir: Optional[Path] = None,
        step: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute the pipeline.

        Args:
            fetcher: SatelliteFetcher instance. If None, creates one.
            output_dir: Override output directory.
            step: Run only this specific step name.

        Returns:
            Dict with execution summary.
        """
        if fetcher is None:
            from pygeovision.data.fetch import SatelliteFetcher
            fetcher = SatelliteFetcher()

        # Save to temp YAML and delegate to pygeofetch
        with __import__("tempfile").NamedTemporaryFile(
            suffix=".yaml", delete=False, mode="w"
        ) as f:
            f.write(self.to_yaml())
            tmp = Path(f.name)

        try:
            result = fetcher.run_pipeline(tmp, step=step)
            logger.info(
                "Pipeline '%s' %s",
                self.name,
                "succeeded" if result.get("success") else "failed",
            )
            return result
        finally:
            tmp.unlink(missing_ok=True)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "DataPipeline":
        """Load a pipeline from an existing YAML file.

        Args:
            path: Path to the YAML file.

        Returns:
            DataPipeline instance.
        """
        with open(path) as f:
            data = yaml.safe_load(f)

        pipeline = cls(
            name=data.get("name", "unnamed"),
            description=data.get("description", ""),
            schedule=data.get("schedule"),
        )
        for step in data.get("steps", []):
            for step_type, config in step.items():
                pipeline.steps.append(PipelineStep(step_type, config or {}))

        return pipeline

    def __repr__(self) -> str:
        return (
            f"DataPipeline(name={self.name!r}, steps={len(self.steps)}, "
            f"schedule={self.schedule!r})"
        )
