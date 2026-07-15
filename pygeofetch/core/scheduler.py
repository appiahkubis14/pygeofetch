"""
Pipeline scheduler for PyGeoFetch.

Supports one-shot and recurring (cron-style) pipeline execution loaded from
YAML definitions.  Uses the standard library ``sched`` module for lightweight
in-process scheduling; for production use, integrate with a proper job
scheduler (APScheduler, Celery Beat, cron, etc.).

Example YAML pipeline::

    name: weekly-sentinel2
    schedule: "0 0 * * 0"   # weekly on Sunday
    steps:
      - search:
          providers: [copernicus, aws_earth]
          date_range: last_7_days
          cloud_cover: 0-10
      - download:
          parallel: 4
          output: ./raw/
      - export:
          format: cloud_optimized_geotiff
          destination: s3://my-bucket/results/

Example usage::

    from pygeofetch.core.scheduler import PipelineScheduler

    scheduler = PipelineScheduler()
    scheduler.load_pipeline("pipeline.yaml")
    scheduler.run_once("weekly-sentinel2")
    scheduler.start()   # blocking; runs on schedule
"""

from __future__ import annotations

import sched
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from pygeofetch.core.logging import get_logger

logger = get_logger(__name__)


class PipelineStep:
    """Single step in a pipeline definition."""

    def __init__(self, step_type: str, config: dict[str, Any]) -> None:
        self.step_type = step_type
        self.config = config

    def __repr__(self) -> str:
        return f"PipelineStep(type={self.step_type!r}, config={self.config})"


class Pipeline:
    """
    A named pipeline consisting of ordered steps.

    Attributes:
        name: Unique pipeline identifier.
        steps: Ordered list of PipelineStep objects.
        schedule: Optional cron expression (``"0 0 * * 0"`` for weekly).
        description: Human-readable description.
    """

    def __init__(
        self,
        name: str,
        steps: list[PipelineStep],
        schedule: str | None = None,
        description: str = "",
    ) -> None:
        self.name = name
        self.steps = steps
        self.schedule = schedule
        self.description = description
        self.last_run: datetime | None = None
        self.run_count: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Pipeline:
        """
        Parse a pipeline from a dictionary (e.g. loaded from YAML).

        Args:
            data: Pipeline definition dict.

        Returns:
            Pipeline instance.

        Raises:
            ValueError: If required keys are missing.
        """
        name = data.get("name")
        if not name:
            msg = "Pipeline definition missing 'name'"
            raise ValueError(msg)

        raw_steps = data.get("steps", [])
        steps: list[PipelineStep] = []
        for raw in raw_steps:
            if not isinstance(raw, dict) or len(raw) != 1:
                msg = f"Invalid step definition: {raw!r}"
                raise ValueError(msg)
            step_type, cfg = next(iter(raw.items()))
            steps.append(PipelineStep(step_type=step_type, config=cfg or {}))

        return cls(
            name=name,
            steps=steps,
            schedule=data.get("schedule"),
            description=data.get("description", ""),
        )

    def __repr__(self) -> str:
        return f"Pipeline(name={self.name!r}, steps={len(self.steps)}, schedule={self.schedule!r})"


class PipelineRunner:
    """
    Executes a Pipeline against a PyGeoFetch engine instance.

    Attributes:
        engine: The PyGeoFetch instance to run pipeline steps against.
    """

    def __init__(self, engine: Any) -> None:
        self.engine = engine

    def run(self, pipeline: Pipeline) -> dict[str, Any]:
        """
        Execute all steps in a pipeline sequentially.

        Args:
            pipeline: Pipeline to execute.

        Returns:
            Dict with execution summary including step results and timing.
        """
        logger.info(
            f"Starting pipeline: {pipeline.name!r} ({len(pipeline.steps)} steps)"
        )
        start_time = datetime.utcnow()
        step_results: list[dict[str, Any]] = []
        context: dict[str, Any] = {}  # shared state passed between steps

        for i, step in enumerate(pipeline.steps):
            logger.info(f"  Step {i + 1}/{len(pipeline.steps)}: {step.step_type}")
            try:
                result = self._run_step(step, context)
                step_results.append(
                    {"step": step.step_type, "status": "ok", "result": result}
                )
                context[step.step_type] = result
            except Exception as exc:
                logger.error(f"  Step {step.step_type!r} failed: {exc}")
                step_results.append(
                    {"step": step.step_type, "status": "error", "error": str(exc)}
                )
                break  # halt pipeline on step failure

        duration = (datetime.utcnow() - start_time).total_seconds()
        pipeline.last_run = start_time
        pipeline.run_count += 1

        summary = {
            "pipeline": pipeline.name,
            "started_at": start_time.isoformat(),
            "duration_seconds": duration,
            "steps": step_results,
            "success": all(s["status"] == "ok" for s in step_results),
        }
        logger.info(
            f"Pipeline {pipeline.name!r} finished in {duration:.1f}s "
            f"({'OK' if summary['success'] else 'FAILED'})"
        )
        return summary

    def _run_step(self, step: PipelineStep, context: dict[str, Any]) -> Any:
        """Dispatch a single step to the appropriate handler."""
        handler = getattr(self, f"_step_{step.step_type}", None)
        if handler is None:
            msg = f"Unknown step type: {step.step_type!r}"
            raise NotImplementedError(msg)
        return handler(step.config, context)

    # ------------------------------------------------------------------
    # Step handlers
    # ------------------------------------------------------------------

    def _step_search(self, config: dict[str, Any], context: dict[str, Any]) -> Any:
        """Execute a search step."""
        from pygeofetch.models.search_query import BoundingBox, SearchQuery

        providers = config.get("providers")
        cloud_str = config.get("cloud_cover", "0-100")
        cloud_min, cloud_max = self._parse_cloud(cloud_str)

        date_range = config.get("date_range", "")
        start_date, end_date = self._parse_date_range(date_range, config)

        bbox_raw = config.get("bbox")
        bbox = BoundingBox.from_string(bbox_raw) if bbox_raw else None

        query = SearchQuery(
            bbox=bbox,
            start_date=start_date,
            end_date=end_date,
            cloud_cover_min=cloud_min,
            cloud_cover_max=cloud_max,
            max_results=config.get("max_results", 500),
            providers=providers or [],
        )
        results = self.engine.search(query, providers=providers)
        logger.info(f"    search → {len(results)} results")
        return results

    def _step_filter(self, config: dict[str, Any], context: dict[str, Any]) -> Any:
        """Filter search results by an expression."""
        results = context.get("search", [])
        expression = config.get("expression", "")
        if not expression or not results:
            return results

        filtered = []
        for item in results:
            try:
                # Simple safe eval: expose `data` object
                if eval(expression, {"data": item, "__builtins__": {}}):  # noqa: S307
                    filtered.append(item)
            except Exception:
                filtered.append(item)  # keep on eval error
        logger.info(f"    filter: {len(results)} → {len(filtered)} items")
        return filtered

    def _step_download(self, config: dict[str, Any], context: dict[str, Any]) -> Any:
        """Download items from a previous search step."""
        from pygeofetch.models.download_task import DownloadOptions

        items = context.get("search") or context.get("filter") or []
        if not items:
            logger.warning("    download: no items to download")
            return []

        output = Path(config.get("output", "./pipeline_output"))
        options = DownloadOptions(
            parallel=config.get("parallel", 2),
            retry_attempts=config.get("retry", 3),
            verify_checksum=config.get("verify_checksum", False),
        )
        results = self.engine.download(items, output, options)
        succeeded = sum(1 for r in results if r.success)
        logger.info(f"    download: {succeeded}/{len(results)} succeeded")
        return results

    def _step_process(self, config: dict[str, Any], context: dict[str, Any]) -> Any:
        """Placeholder for processing actions (atmospheric correction, NDVI, etc.)."""
        actions = config if isinstance(config, list) else config.get("actions", [])
        logger.info(
            f"    process: {len(actions)} actions (stub — integrate custom processors)"
        )
        return {"actions": actions, "status": "stub"}

    def _step_export(self, config: dict[str, Any], context: dict[str, Any]) -> Any:
        """Placeholder for export step."""
        destination = config.get("destination", "./export")
        fmt = config.get("format", "geotiff")
        logger.info(f"    export: format={fmt!r} → {destination} (stub)")
        return {"format": fmt, "destination": destination, "status": "stub"}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_cloud(cloud_str: str):
        """Parse '0-20' into (0, 20)."""
        try:
            parts = str(cloud_str).split("-")
            return float(parts[0]), float(parts[1]) if len(parts) > 1 else 100.0
        except Exception:
            return 0.0, 100.0

    @staticmethod
    def _parse_date_range(date_range: str, config: dict[str, Any]):
        """Resolve date_range keywords or explicit start/end."""
        from datetime import date

        today = date.today()
        if date_range == "last_7_days":
            return today - timedelta(days=7), today
        if date_range == "last_30_days":
            return today - timedelta(days=30), today
        if date_range == "last_year":
            return today - timedelta(days=365), today

        start = config.get("start_date") or config.get("start")
        end = config.get("end_date") or config.get("end") or str(today)
        return start, end


class PipelineScheduler:
    """
    Loads, stores, and schedules pipelines for recurring execution.

    Example::

        scheduler = PipelineScheduler(engine=sb)
        scheduler.load_pipeline("pipeline.yaml")
        scheduler.run_once("my-pipeline")
        scheduler.start()   # blocking loop
    """

    def __init__(self, engine: Any | None = None) -> None:
        self._pipelines: dict[str, Pipeline] = {}
        self._engine = engine
        self._runner: PipelineRunner | None = None
        self._scheduler = sched.scheduler(time.monotonic, time.sleep)
        self._running = False
        self._thread: threading.Thread | None = None

    def set_engine(self, engine: Any) -> None:
        """Attach a PyGeoFetch engine instance."""
        self._engine = engine
        self._runner = PipelineRunner(engine)

    def load_pipeline(self, path: str | Path) -> Pipeline:
        """
        Load a pipeline from a YAML file and register it.

        Args:
            path: Path to the YAML pipeline definition.

        Returns:
            Parsed Pipeline instance.
        """
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)
        pipeline = Pipeline.from_dict(data)
        self._pipelines[pipeline.name] = pipeline
        logger.info(f"Loaded pipeline {pipeline.name!r} from {path}")
        return pipeline

    def add_pipeline(self, pipeline: Pipeline) -> None:
        """Register a Pipeline object directly."""
        self._pipelines[pipeline.name] = pipeline

    def list_pipelines(self) -> list[dict[str, Any]]:
        """Return a summary of all registered pipelines."""
        return [
            {
                "name": p.name,
                "steps": len(p.steps),
                "schedule": p.schedule,
                "last_run": p.last_run.isoformat() if p.last_run else None,
                "run_count": p.run_count,
            }
            for p in self._pipelines.values()
        ]

    def run_once(self, name: str) -> dict[str, Any]:
        """
        Execute a pipeline immediately, once.

        Args:
            name: Pipeline name as registered.

        Returns:
            Execution summary dict from PipelineRunner.run().
        """
        if name not in self._pipelines:
            msg = f"Unknown pipeline: {name!r}"
            raise KeyError(msg)
        runner = self._get_runner()
        return runner.run(self._pipelines[name])

    def start(self, blocking: bool = True) -> None:
        """
        Start the scheduler loop.

        Schedules all pipelines that have a cron expression.  Runs blocking
        or in a background thread.

        Args:
            blocking: If True (default), blocks the calling thread.
        """
        self._running = True
        self._schedule_all()
        if blocking:
            self._run_loop()
        else:
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """Stop the scheduler loop."""
        self._running = False
        logger.info("Scheduler stopped")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_runner(self) -> PipelineRunner:
        if self._runner is None:
            if self._engine is None:
                msg = "No engine attached — call set_engine() first"
                raise RuntimeError(msg)
            self._runner = PipelineRunner(self._engine)
        return self._runner

    def _schedule_all(self) -> None:
        """Schedule all pipelines that have a cron expression."""
        for pipeline in self._pipelines.values():
            if pipeline.schedule:
                delay = self._next_run_delay(pipeline.schedule)
                self._scheduler.enter(
                    delay, 1, self._run_pipeline_and_reschedule, (pipeline,)
                )
                logger.info(
                    f"Scheduled {pipeline.name!r} to run in {delay:.0f}s "
                    f"(cron: {pipeline.schedule!r})"
                )

    def _run_pipeline_and_reschedule(self, pipeline: Pipeline) -> None:
        """Execute a pipeline and re-queue it for the next run."""
        if not self._running:
            return
        try:
            self._get_runner().run(pipeline)
        except Exception as exc:
            logger.error(f"Pipeline {pipeline.name!r} error: {exc}")
        finally:
            if self._running and pipeline.schedule:
                delay = self._next_run_delay(pipeline.schedule)
                self._scheduler.enter(
                    delay, 1, self._run_pipeline_and_reschedule, (pipeline,)
                )

    def _run_loop(self) -> None:
        """Blocking scheduler event loop."""
        logger.info("Scheduler event loop started")
        while self._running:
            self._scheduler.run(blocking=False)
            time.sleep(1)
        logger.info("Scheduler event loop exited")

    @staticmethod
    def _next_run_delay(cron_expr: str) -> float:
        """
        Compute seconds until the next cron trigger.

        This is a *simplified* implementation supporting only the five-field
        POSIX cron format (minute hour dom month dow).  For production use,
        replace with ``croniter`` or ``APScheduler``.

        Args:
            cron_expr: Five-field cron expression.

        Returns:
            Seconds until next scheduled run (minimum 60 s).
        """
        try:
            import croniter  # type: ignore

            itr = croniter.croniter(cron_expr, datetime.utcnow())
            next_dt = itr.get_next(datetime)
            delay = (next_dt - datetime.utcnow()).total_seconds()
            return max(delay, 1.0)
        except ImportError:
            # croniter not installed → fall back to a 1-hour default
            logger.debug(
                "croniter not installed; defaulting to 3600 s schedule interval"
            )
            return 3600.0
        except Exception as exc:
            logger.warning(
                f"Could not parse cron {cron_expr!r}: {exc}; defaulting to 3600 s"
            )
            return 3600.0


# Type alias to avoid circular import confusion
