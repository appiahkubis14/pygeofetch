"""
Training callbacks for PyGeoVision trainer.

Modular callbacks for: early stopping, model checkpointing,
learning rate scheduling, MLflow/W&B logging, and custom hooks.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class Callback(ABC):
    """Base class for all training callbacks."""

    def on_train_begin(self, trainer: Any) -> None: ...
    def on_train_end(self, trainer: Any) -> None: ...
    def on_epoch_begin(self, trainer: Any, epoch: int) -> None: ...
    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]) -> None: ...
    def on_batch_begin(self, trainer: Any, batch_idx: int) -> None: ...
    def on_batch_end(self, trainer: Any, batch_idx: int, loss: float) -> None: ...


class EarlyStopping(Callback):
    """Stop training when a monitored metric stops improving.

    Args:
        monitor: Metric name to monitor (e.g. 'val_loss', 'val_miou').
        patience: Epochs to wait without improvement before stopping.
        min_delta: Minimum change to qualify as improvement.
        mode: 'min' for loss metrics, 'max' for score metrics.
        restore_best_weights: Restore best model weights on stop.

    Example:
        >>> cb = EarlyStopping(monitor="val_miou", patience=10, mode="max")
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "min",
        restore_best_weights: bool = True,
    ) -> None:
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self._best: float = float("inf") if mode == "min" else float("-inf")
        self._wait = 0
        self._best_weights: Optional[Any] = None

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]) -> None:
        import copy
        value = metrics.get(self.monitor)
        if value is None:
            return

        improved = (
            value < self._best - self.min_delta
            if self.mode == "min"
            else value > self._best + self.min_delta
        )
        if improved:
            self._best = value
            self._wait = 0
            if self.restore_best_weights:
                self._best_weights = copy.deepcopy(trainer.model.state_dict())
            logger.debug("EarlyStopping: improvement to %.5f at epoch %d", value, epoch)
        else:
            self._wait += 1
            if self._wait >= self.patience:
                logger.info(
                    "EarlyStopping: no improvement for %d epochs. Stopping.", self.patience
                )
                trainer.should_stop = True
                if self.restore_best_weights and self._best_weights:
                    trainer.model.load_state_dict(self._best_weights)
                    logger.info("Restored best model weights (%.5f).", self._best)


class ModelCheckpoint(Callback):
    """Save model checkpoints during training.

    Args:
        dirpath: Directory for saving checkpoints.
        filename: Checkpoint filename template (use {epoch}, {val_loss} etc).
        monitor: Metric to use for selecting best checkpoint.
        save_best_only: If True, only save when metric improves.
        mode: 'min' or 'max'.
        save_every_n_epochs: Save every N epochs regardless of improvement.

    Example:
        >>> cb = ModelCheckpoint(
        ...     dirpath="./checkpoints",
        ...     monitor="val_miou",
        ...     mode="max",
        ...     save_best_only=True,
        ... )
    """

    def __init__(
        self,
        dirpath: str = "./checkpoints",
        filename: str = "model_epoch{epoch:03d}_val{val_loss:.4f}",
        monitor: str = "val_loss",
        save_best_only: bool = True,
        mode: str = "min",
        save_every_n_epochs: int = 0,
    ) -> None:
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.save_every_n_epochs = save_every_n_epochs
        self._best: float = float("inf") if mode == "min" else float("-inf")
        self.best_model_path: Optional[Path] = None

    def on_train_begin(self, trainer: Any) -> None:
        self.dirpath.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]) -> None:
        import torch

        value = metrics.get(self.monitor, float("inf"))
        improved = (
            value < self._best if self.mode == "min" else value > self._best
        )

        should_save = (
            not self.save_best_only
            or improved
            or (self.save_every_n_epochs > 0 and (epoch + 1) % self.save_every_n_epochs == 0)
        )

        if not should_save:
            return

        name = self.filename.format(epoch=epoch, **metrics) + ".pth"
        path = self.dirpath / name
        torch.save({
            "epoch": epoch,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "metrics": metrics,
        }, path)

        if improved:
            self._best = value
            self.best_model_path = path
            logger.info(
                "ModelCheckpoint: saved best model (%.5f) → %s", value, path
            )
        else:
            logger.debug("ModelCheckpoint: saved checkpoint → %s", path)


class MLflowLogger(Callback):
    """Log training metrics to MLflow.

    Args:
        experiment_name: MLflow experiment name.
        run_name: Name for this training run.
        tracking_uri: MLflow tracking server URI.
        log_model: If True, log the best model artifact.

    Example:
        >>> cb = MLflowLogger(experiment_name="building_segmentation")
    """

    def __init__(
        self,
        experiment_name: str = "pygeovision",
        run_name: Optional[str] = None,
        tracking_uri: str = "mlruns",
        log_model: bool = False,
    ) -> None:
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tracking_uri = tracking_uri
        self.log_model = log_model
        self._run = None

    def on_train_begin(self, trainer: Any) -> None:
        try:
            import mlflow
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            self._run = mlflow.start_run(run_name=self.run_name)
            logger.info("MLflow run started: %s", self._run.info.run_id)
        except ImportError:
            logger.warning("mlflow not installed; logging disabled. pip install mlflow")

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]) -> None:
        if self._run is None:
            return
        try:
            import mlflow
            mlflow.log_metrics(metrics, step=epoch)
        except Exception as exc:
            logger.warning("MLflow log_metrics failed: %s", exc)

    def on_train_end(self, trainer: Any) -> None:
        if self._run is None:
            return
        try:
            import mlflow
            if self.log_model:
                mlflow.pytorch.log_model(trainer.model, "model")
            mlflow.end_run()
        except Exception as exc:
            logger.warning("MLflow end_run failed: %s", exc)


class LRSchedulerCallback(Callback):
    """Wrap any PyTorch LR scheduler as a callback.

    Args:
        scheduler: Instantiated PyTorch LR scheduler.
        monitor: Metric to pass to ReduceLROnPlateau (if applicable).
        call_on: 'epoch' or 'batch'.

    Example:
        >>> from torch.optim.lr_scheduler import CosineAnnealingLR
        >>> cb = LRSchedulerCallback(CosineAnnealingLR(optimizer, T_max=50))
    """

    def __init__(self, scheduler: Any, monitor: str = "val_loss", call_on: str = "epoch") -> None:
        self.scheduler = scheduler
        self.monitor = monitor
        self.call_on = call_on

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]) -> None:
        if self.call_on != "epoch":
            return
        sched_type = type(self.scheduler).__name__
        if "ReduceLROnPlateau" in sched_type:
            value = metrics.get(self.monitor)
            if value is not None:
                self.scheduler.step(value)
        else:
            self.scheduler.step()

    def on_batch_end(self, trainer: Any, batch_idx: int, loss: float) -> None:
        if self.call_on == "batch":
            self.scheduler.step()


class ProgressCallback(Callback):
    """Log training progress to stdout/logger.

    Args:
        log_every_n_batches: Frequency of batch-level logging.
    """

    def __init__(self, log_every_n_batches: int = 50) -> None:
        self.log_every_n_batches = log_every_n_batches
        self._epoch_start: float = 0.0

    def on_epoch_begin(self, trainer: Any, epoch: int) -> None:
        self._epoch_start = time.time()

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]) -> None:
        elapsed = time.time() - self._epoch_start
        metric_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        logger.info("Epoch %03d [%.1fs] %s", epoch, elapsed, metric_str)

    def on_batch_end(self, trainer: Any, batch_idx: int, loss: float) -> None:
        if batch_idx % self.log_every_n_batches == 0:
            logger.debug("  Batch %d | loss: %.5f", batch_idx, loss)


class CallbackList:
    """Manages and dispatches events to a list of callbacks."""

    def __init__(self, callbacks: List[Callback]) -> None:
        self.callbacks = callbacks

    def on_train_begin(self, trainer: Any) -> None:
        for cb in self.callbacks: cb.on_train_begin(trainer)

    def on_train_end(self, trainer: Any) -> None:
        for cb in self.callbacks: cb.on_train_end(trainer)

    def on_epoch_begin(self, trainer: Any, epoch: int) -> None:
        for cb in self.callbacks: cb.on_epoch_begin(trainer, epoch)

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]) -> None:
        for cb in self.callbacks: cb.on_epoch_end(trainer, epoch, metrics)

    def on_batch_begin(self, trainer: Any, batch_idx: int) -> None:
        for cb in self.callbacks: cb.on_batch_begin(trainer, batch_idx)

    def on_batch_end(self, trainer: Any, batch_idx: int, loss: float) -> None:
        for cb in self.callbacks: cb.on_batch_end(trainer, batch_idx, loss)
