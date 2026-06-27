"""
PyGeoVision experiment tracking and reproducibility utilities.

Provides lightweight experiment management that works alongside (or
instead of) MLflow/W&B for tracking hyperparameters, metrics, and
artifact paths.

Example:
    >>> from pygeovision.ai.experiments import ExperimentTracker
    >>> exp = ExperimentTracker(name="building_seg_v2")
    >>> exp.log_params({"lr": 1e-4, "batch_size": 16, "model": "segformer_b2"})
    >>> for epoch in range(100):
    ...     exp.log_metrics({"miou": 0.82, "loss": 0.15}, step=epoch)
    >>> exp.save()
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExperimentRecord:
    """Stored record of a single experiment run.

    Attributes:
        name: Experiment name.
        run_id: Unique run identifier.
        start_time: ISO start timestamp.
        end_time: ISO end timestamp (empty until saved).
        params: Hyperparameter dict.
        metrics: Step-indexed metric history.
        tags: Arbitrary key-value tags.
        artifacts: Paths to saved artifacts.
        environment: Captured environment info.
    """
    name: str
    run_id: str
    start_time: str
    end_time: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    environment: Dict[str, str] = field(default_factory=dict)


class ExperimentTracker:
    """Lightweight experiment tracker for PyGeoVision training runs.

    Tracks hyperparameters, metrics per step, artifact paths, and
    environment info. Saves to a JSON file for easy inspection and
    integrates with the PyGeoVision model hub.

    Args:
        name: Experiment name.
        save_dir: Directory for experiment records.
        tags: Initial key-value tags.
        seed: Random seed for reproducibility (set globally if provided).

    Example:
        >>> tracker = ExperimentTracker("land_cover_v3", seed=42)
        >>> tracker.log_params({"model": "segformer_b5", "lr": 5e-5})
        >>> tracker.log_metrics({"val_miou": 0.81}, step=10)
        >>> tracker.log_artifact("/path/to/best.pth")
        >>> tracker.save()
    """

    def __init__(
        self,
        name: str,
        save_dir: Optional[Path] = None,
        tags: Optional[Dict[str, str]] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.save_dir = Path(save_dir or Path.home() / ".pygeovision" / "experiments")
        self.save_dir.mkdir(parents=True, exist_ok=True)

        run_id = self._generate_run_id(name)
        self.record = ExperimentRecord(
            name=name,
            run_id=run_id,
            start_time=datetime.utcnow().isoformat(),
            tags=tags or {},
            environment=self._capture_env(),
        )

        if seed is not None:
            self.set_seed(seed)
            self.record.tags["seed"] = str(seed)

        logger.info("Experiment '%s' started (run_id=%s).", name, run_id)

    # ------------------------------------------------------------------
    # Logging API
    # ------------------------------------------------------------------

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters for this run.

        Args:
            params: Dict of parameter names to values.

        Example:
            >>> tracker.log_params({"lr": 1e-4, "batch_size": 16})
        """
        self.record.params.update(params)
        logger.debug("Logged params: %s", list(params.keys()))

    def log_metrics(self, metrics: Dict[str, float], step: int = 0) -> None:
        """Log metrics at a training step.

        Args:
            metrics: Dict of metric names to float values.
            step: Training step or epoch number.

        Example:
            >>> tracker.log_metrics({"val_miou": 0.82, "val_loss": 0.23}, step=50)
        """
        ts = time.time()
        for key, value in metrics.items():
            self.record.metrics.setdefault(key, []).append(
                {"step": step, "value": float(value), "ts": ts}
            )

    def log_artifact(self, path: str | Path) -> None:
        """Record the path to a saved artifact (checkpoint, GeoJSON, etc.).

        Args:
            path: Path to the artifact file.
        """
        self.record.artifacts.append(str(path))

    def set_tag(self, key: str, value: str) -> None:
        """Set an arbitrary key-value tag.

        Args:
            key: Tag name.
            value: Tag value.
        """
        self.record.tags[key] = value

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> Path:
        """Finalize and save the experiment record to disk.

        Returns:
            Path to the saved JSON record.
        """
        self.record.end_time = datetime.utcnow().isoformat()
        out_path = self.save_dir / f"{self.record.run_id}.json"
        out_path.write_text(json.dumps(asdict(self.record), indent=2))
        logger.info("Experiment saved → %s", out_path)
        return out_path

    @classmethod
    def load(cls, run_id: str, save_dir: Optional[Path] = None) -> "ExperimentTracker":
        """Load an existing experiment record by run_id.

        Args:
            run_id: Run identifier.
            save_dir: Directory where records are stored.

        Returns:
            ExperimentTracker with loaded record.
        """
        save_dir = Path(save_dir or Path.home() / ".pygeovision" / "experiments")
        path = save_dir / f"{run_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"No experiment record found: {path}")
        data = json.loads(path.read_text())
        tracker = cls.__new__(cls)
        tracker.save_dir = save_dir
        tracker.record = ExperimentRecord(**data)
        return tracker

    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all saved experiment records in save_dir.

        Returns:
            List of dicts with run_id, name, start_time, and tags.
        """
        records = []
        for path in sorted(self.save_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text())
                records.append({
                    "run_id": data.get("run_id", path.stem),
                    "name": data.get("name", ""),
                    "start_time": data.get("start_time", ""),
                    "tags": data.get("tags", {}),
                })
            except Exception:
                pass
        return records

    def get_best_metric(self, metric: str, mode: str = "max") -> Optional[float]:
        """Return the best recorded value for a metric.

        Args:
            metric: Metric name.
            mode: 'max' or 'min'.

        Returns:
            Best float value, or None if metric not recorded.
        """
        values = [e["value"] for e in self.record.metrics.get(metric, [])]
        if not values:
            return None
        return max(values) if mode == "max" else min(values)

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------

    @staticmethod
    def set_seed(seed: int) -> None:
        """Set global random seeds for reproducible training.

        Args:
            seed: Integer seed value.
        """
        import random
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except ImportError:
            pass
        logger.info("Random seed set to %d.", seed)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_run_id(name: str) -> str:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        short_hash = hashlib.md5(f"{name}{time.time()}".encode()).hexdigest()[:6]
        safe_name = "".join(c if c.isalnum() else "_" for c in name)[:20]
        return f"{safe_name}_{ts}_{short_hash}"

    @staticmethod
    def _capture_env() -> Dict[str, str]:
        env = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "hostname": platform.node(),
        }
        try:
            import torch
            env["torch"] = torch.__version__
            env["cuda"] = str(torch.cuda.is_available())
            if torch.cuda.is_available():
                env["cuda_device"] = torch.cuda.get_device_name(0)
        except ImportError:
            pass
        return env
