"""
Experiment tracking — MLflow / W&B / TensorBoard (Phase 4.1).
"""
from __future__ import annotations
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Unified experiment tracker for MLflow, W&B, and TensorBoard.

    Example::

        tracker = ExperimentTracker("building_seg", backends=["mlflow", "wandb"])
        tracker.start_run(config=cfg.to_dict())
        for epoch in range(100):
            tracker.log({"train_loss": 0.5, "val_iou": 0.8}, step=epoch)
        tracker.log_model("model.pth", "best_model")
        tracker.end_run()
    """

    def __init__(
        self,
        experiment_name: str,
        run_name: str = "",
        backends: Optional[List[str]] = None,
        mlflow_uri: str = "mlruns",
        wandb_project: str = "pygeovision",
        tb_log_dir: str = "./tb_logs",
    ) -> None:
        self.experiment_name = experiment_name
        self.run_name = run_name or f"run_{int(time.time())}"
        self.backends = backends or ["tensorboard"]
        self._mlflow_run = None
        self._wandb_run = None
        self._tb_writer = None
        self._mlflow_uri = mlflow_uri
        self._wandb_project = wandb_project
        self._tb_log_dir = tb_log_dir
        self._history: Dict[str, List] = {}
        self._start_time = time.time()

    def start_run(self, config: Optional[Dict] = None) -> "ExperimentTracker":
        """Start all configured tracking backends."""
        if "mlflow" in self.backends:
            try:
                import mlflow
                mlflow.set_tracking_uri(self._mlflow_uri)
                mlflow.set_experiment(self.experiment_name)
                self._mlflow_run = mlflow.start_run(run_name=self.run_name)
                if config:
                    mlflow.log_params({k: str(v)[:250] for k, v in config.items()})
                logger.info("MLflow run started: %s", self._mlflow_run.info.run_id)
            except ImportError:
                logger.warning("mlflow not installed")
            except Exception as exc:
                logger.warning("MLflow start_run failed: %s", exc)

        if "wandb" in self.backends:
            try:
                import wandb
                self._wandb_run = wandb.init(
                    project=self._wandb_project,
                    name=self.run_name,
                    config=config or {},
                )
                logger.info("W&B run started: %s", self.run_name)
            except ImportError:
                logger.warning("wandb not installed")
            except Exception as exc:
                logger.warning("W&B start_run failed: %s", exc)

        if "tensorboard" in self.backends:
            try:
                from torch.utils.tensorboard import SummaryWriter
                log_dir = Path(self._tb_log_dir) / self.experiment_name / self.run_name
                self._tb_writer = SummaryWriter(log_dir=str(log_dir))
                logger.info("TensorBoard writer: %s", log_dir)
            except ImportError:
                logger.warning("tensorboard not installed (pip install tensorboard)")

        return self

    def log(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to all active backends."""
        for k, v in metrics.items():
            self._history.setdefault(k, []).append(v)

        if self._mlflow_run:
            try:
                import mlflow
                mlflow.log_metrics(metrics, step=step)
            except Exception:
                pass

        if self._wandb_run:
            try:
                import wandb
                log_data = dict(metrics)
                if step is not None:
                    log_data["_step"] = step
                wandb.log(log_data)
            except Exception:
                pass

        if self._tb_writer:
            try:
                s = step or len(next(iter(self._history.values()), []))
                for k, v in metrics.items():
                    self._tb_writer.add_scalar(k, v, global_step=s)
            except Exception:
                pass

    def log_model(self, model_path: Union[str, Path], artifact_name: str = "model") -> None:
        """Log a model artifact."""
        if self._mlflow_run:
            try:
                import mlflow
                mlflow.log_artifact(str(model_path), artifact_path=artifact_name)
            except Exception:
                pass
        if self._wandb_run:
            try:
                import wandb
                artifact = wandb.Artifact(artifact_name, type="model")
                artifact.add_file(str(model_path))
                wandb.log_artifact(artifact)
            except Exception:
                pass

    def log_image(self, key: str, image: Any, step: Optional[int] = None) -> None:
        """Log an image (numpy array or PIL) to tracking backends."""
        if self._tb_writer:
            try:
                import numpy as np
                if hasattr(image, "numpy"):
                    image = image.numpy()
                self._tb_writer.add_image(key, image, global_step=step)
            except Exception:
                pass
        if self._wandb_run:
            try:
                import wandb
                wandb.log({key: wandb.Image(image)}, step=step)
            except Exception:
                pass

    def end_run(self) -> Dict[str, Any]:
        """End all tracking runs and return final summary."""
        duration = time.time() - self._start_time
        summary = {
            "experiment": self.experiment_name,
            "run_name": self.run_name,
            "duration_seconds": duration,
            "metrics": {k: v[-1] if v else None for k, v in self._history.items()},
        }
        if self._mlflow_run:
            try:
                import mlflow
                mlflow.log_metric("duration_seconds", duration)
                mlflow.end_run()
            except Exception:
                pass
        if self._wandb_run:
            try:
                self._wandb_run.finish()
            except Exception:
                pass
        if self._tb_writer:
            try:
                self._tb_writer.close()
            except Exception:
                pass
        logger.info("Experiment '%s/%s' completed in %.1fs",
                    self.experiment_name, self.run_name, duration)
        return summary

    @property
    def history(self) -> Dict[str, List]:
        return self._history
