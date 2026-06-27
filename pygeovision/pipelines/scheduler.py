"""Cron-based pipeline scheduling."""
from __future__ import annotations
import logging, threading, time
from typing import Any, Callable, Dict, List, Optional
logger = logging.getLogger(__name__)


class PipelineScheduler:
    """Schedule pipelines with cron expressions.

    Example::

        scheduler = PipelineScheduler()
        scheduler.add_job("building_extract", fn=my_pipeline_fn, cron="0 6 * * *")
        scheduler.start()   # background thread
        # ...
        scheduler.stop()
    """

    def __init__(self) -> None:
        self._jobs: List[Dict] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def add_job(self, name: str, fn: Callable, cron: str,
                 args: tuple = (), kwargs: dict = None) -> "PipelineScheduler":
        """Add a scheduled job.

        Args:
            name: Job identifier
            fn: Callable to execute
            cron: Cron expression ("min hour day month weekday")
        """
        self._jobs.append({
            "name": name, "fn": fn, "cron": cron,
            "args": args, "kwargs": kwargs or {},
            "last_run": None, "n_runs": 0, "errors": 0,
        })
        logger.info("Scheduled: '%s' @ '%s'", name, cron)
        return self

    def _parse_cron(self, cron: str) -> bool:
        """Check if cron expression matches current time."""
        try:
            from croniter import croniter
            return croniter.match(cron, time.time())
        except ImportError:
            # Fallback: simple check without croniter
            import datetime
            now = datetime.datetime.now()
            parts = cron.split()
            if len(parts) != 5:
                return False
            # Only check minute wildcard for simplicity
            return parts[0] == "*" or int(parts[0]) == now.minute

    def _run_loop(self) -> None:
        while self._running:
            for job in self._jobs:
                if self._parse_cron(job["cron"]):
                    try:
                        job["fn"](*job["args"], **job["kwargs"])
                        job["n_runs"] += 1
                        job["last_run"] = time.time()
                    except Exception as exc:
                        job["errors"] += 1
                        logger.error("Job '%s' failed: %s", job["name"], exc)
            time.sleep(60)  # Check every minute

    def start(self) -> "PipelineScheduler":
        """Start the scheduler in a background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("Scheduler started with %d jobs", len(self._jobs))
        return self

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Scheduler stopped")

    def status(self) -> List[Dict]:
        return [{"name": j["name"], "cron": j["cron"],
                 "n_runs": j["n_runs"], "errors": j["errors"],
                 "last_run": j["last_run"]} for j in self._jobs]

    def run_now(self, name: str) -> Any:
        """Run a scheduled job immediately."""
        for job in self._jobs:
            if job["name"] == name:
                return job["fn"](*job["args"], **job["kwargs"])
        raise KeyError(f"Job '{name}' not found")
