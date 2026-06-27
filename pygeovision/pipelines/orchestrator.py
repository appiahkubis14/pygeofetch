"""Pipeline execution engine with dependency management and retry logic."""
from __future__ import annotations
import logging, time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    pipeline_name: str
    success: bool
    steps_completed: List[str]
    steps_failed: List[str]
    duration_s: float
    outputs: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class Pipeline:
    """A composable, executable pipeline of steps.

    Example::

        from pygeovision.pipelines import Pipeline, SearchStep, DownloadStep

        p = Pipeline("building_extraction")
        p.add(SearchStep(name="search", params={...}))
        p.add(DownloadStep(name="download", depends_on=["search"]))
        result = p.run(client=pgv_client)
    """

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self._steps: List[Any] = []
        self._hooks: Dict[str, List[Callable]] = {"before_step": [], "after_step": [], "on_error": []}

    def add(self, step: Any) -> "Pipeline":
        self._steps.append(step)
        return self

    def on(self, event: str, fn: Callable) -> "Pipeline":
        self._hooks.setdefault(event, []).append(fn)
        return self

    def run(self, context: Optional[Dict] = None, dry_run: bool = False) -> PipelineResult:
        """Execute the pipeline."""
        ctx = context or {}
        completed, failed = [], []
        t_start = time.time()

        # Build execution order (topological sort)
        ordered = self._topo_sort()

        for step in ordered:
            step_name = step.name if hasattr(step, "name") else str(step)

            # Check dependencies
            deps = step.depends_on if hasattr(step, "depends_on") else []
            if any(d in failed for d in deps):
                logger.warning("Skipping '%s' — dependency failed", step_name)
                failed.append(step_name)
                continue

            logger.info("Running step: %s", step_name)
            for hook in self._hooks.get("before_step", []):
                hook(step, ctx)

            if dry_run:
                completed.append(step_name)
                continue

            # Execute with retry
            last_exc = None
            max_retries = getattr(step, "retry_on_fail", 1)
            for attempt in range(max_retries):
                try:
                    result = step.run(ctx) if hasattr(step, "run") else {}
                    ctx[step_name] = result
                    # Also set canonical names for common steps
                    action = getattr(step, "action", "")
                    if action == "search":
                        ctx["search_results"] = result
                    elif action == "download":
                        ctx["downloads"] = result
                    elif action == "infer":
                        ctx["predictions"] = result
                    completed.append(step_name)
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                    logger.warning("Step '%s' attempt %d failed: %s", step_name, attempt+1, exc)

            if last_exc:
                failed.append(step_name)
                for hook in self._hooks.get("on_error", []):
                    hook(step, ctx, last_exc)
                logger.error("Step '%s' failed after %d attempts: %s", step_name, max_retries, last_exc)

            for hook in self._hooks.get("after_step", []):
                hook(step, ctx)

        duration = time.time() - t_start
        return PipelineResult(
            pipeline_name=self.name,
            success=len(failed) == 0,
            steps_completed=completed,
            steps_failed=failed,
            duration_s=round(duration, 2),
            outputs=ctx,
        )

    def _topo_sort(self) -> List[Any]:
        """Topological sort by dependency order."""
        name_to_step = {(s.name if hasattr(s, "name") else str(i)): s
                        for i, s in enumerate(self._steps)}
        visited, order = set(), []

        def _visit(name: str) -> None:
            if name in visited: return
            visited.add(name)
            step = name_to_step.get(name)
            if step is None: return
            for dep in (step.depends_on if hasattr(step, "depends_on") else []):
                _visit(dep)
            order.append(step)

        for name in name_to_step:
            _visit(name)
        return order

    @classmethod
    def from_config(cls, config: Dict) -> "Pipeline":
        """Build a Pipeline from a parsed YAML config dict."""
        from pygeovision.pipelines.steps import (Step, SearchStep, DownloadStep,
                                                   InferStep, ExportStep)
        ACTION_MAP = {
            "search": SearchStep, "download": DownloadStep,
            "infer": InferStep, "export": ExportStep,
        }
        p = cls(config["name"], config.get("description", ""))
        for s_cfg in config.get("steps", []):
            action = s_cfg.get("action", "custom")
            cls_ = ACTION_MAP.get(action, Step)
            step = cls_(
                name=s_cfg["name"],
                action=action,
                params=s_cfg.get("params", {}),
                depends_on=s_cfg.get("depends_on", []),
                retry_on_fail=s_cfg.get("retry_on_fail", 1),
            )
            p.add(step)
        return p

    @classmethod
    def from_yaml(cls, path: str) -> "Pipeline":
        from pygeovision.pipelines.yaml_parser import PipelineYAMLParser
        config = PipelineYAMLParser().load(path)
        return cls.from_config(config)

    def __repr__(self) -> str:
        return f"Pipeline(name={self.name!r}, steps={len(self._steps)})"


class PipelineOrchestrator:
    """Manage and run multiple named pipelines.

    Example::

        orch = PipelineOrchestrator(client=pgv_client)
        orch.register("buildings", Pipeline.from_yaml("buildings.yaml"))
        result = orch.run("buildings")
    """

    def __init__(self, client: Any = None) -> None:
        self.client = client
        self._pipelines: Dict[str, Pipeline] = {}

    def register(self, name: str, pipeline: Pipeline) -> "PipelineOrchestrator":
        self._pipelines[name] = pipeline
        return self

    def register_yaml(self, path: str) -> "PipelineOrchestrator":
        p = Pipeline.from_yaml(path)
        return self.register(p.name, p)

    def run(self, name: str, context: Optional[Dict] = None,
             dry_run: bool = False) -> PipelineResult:
        if name not in self._pipelines:
            raise KeyError(f"Pipeline '{name}' not registered. Available: {list(self._pipelines)}")
        ctx = {"client": self.client, **(context or {})}
        return self._pipelines[name].run(ctx, dry_run=dry_run)

    def list(self) -> List[str]:
        return list(self._pipelines.keys())

    def __repr__(self) -> str:
        return f"PipelineOrchestrator({len(self._pipelines)} pipelines: {self.list()})"
