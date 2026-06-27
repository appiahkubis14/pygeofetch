"""
PyGeoVision GeoTrainer — core training loop for geospatial AI models.

Handles training, validation, mixed-precision, gradient accumulation,
and distributed training setup. Integrates with callbacks and metrics.

Example:
    >>> from pygeovision.ai.training.trainer import GeoTrainer
    >>> trainer = GeoTrainer(
    ...     model=model,
    ...     train_dataset=train_ds,
    ...     val_dataset=val_ds,
    ...     num_classes=10,
    ...     task="segmentation",
    ... )
    >>> results = trainer.fit(epochs=50)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from pygeovision.ai.training.callbacks import (
    Callback, CallbackList, EarlyStopping, ModelCheckpoint, ProgressCallback,
)
from pygeovision.ai.training.losses import get_loss
from pygeovision.ai.training.metrics import ConfusionMatrix, BinaryMetrics, AverageMeter

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Results from a completed training run.

    Attributes:
        best_epoch: Epoch with best validation metric.
        best_metric: Best validation metric value.
        best_model_path: Path to the saved best checkpoint.
        history: Per-epoch metric history.
        total_time_seconds: Total training duration.
    """
    best_epoch: int = 0
    best_metric: float = 0.0
    best_model_path: Optional[Path] = None
    history: Dict[str, List[float]] = field(default_factory=dict)
    total_time_seconds: float = 0.0


class GeoTrainer:
    """Training engine for PyGeoVision geospatial AI models.

    Provides a clean training loop with support for:
    - Semantic segmentation, binary segmentation, change detection, classification
    - Mixed-precision training (BF16/FP16/FP32)
    - Gradient accumulation
    - Distributed Data Parallel (DDP)
    - Configurable loss functions and optimizers
    - Callback system for logging, checkpointing, early stopping

    Args:
        model: PyTorch model to train.
        train_dataset: Training dataset (PyTorch Dataset).
        val_dataset: Validation dataset (optional).
        num_classes: Number of output classes.
        task: Task type: 'segmentation', 'binary_segmentation',
              'classification', 'change_detection', 'super_resolution'.
        loss: Loss function name or nn.Module instance.
        optimizer: Optimizer name ('adam', 'adamw', 'sgd') or optimizer instance.
        learning_rate: Initial learning rate.
        weight_decay: L2 regularization weight.
        batch_size: Training batch size.
        num_workers: DataLoader worker processes.
        device: Compute device (auto-detected if None).
        mixed_precision: Enable automatic mixed precision.
        gradient_accumulation_steps: Steps before optimizer update.
        max_grad_norm: Gradient clipping norm (0 = disabled).
        output_dir: Directory for checkpoints and logs.
        callbacks: Additional callbacks.

    Example:
        >>> trainer = GeoTrainer(
        ...     model=unet,
        ...     train_dataset=train_ds,
        ...     val_dataset=val_ds,
        ...     num_classes=5,
        ...     task="segmentation",
        ...     learning_rate=1e-4,
        ... )
        >>> result = trainer.fit(epochs=100)
        >>> print(f"Best mIoU: {result.best_metric:.4f}")
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        num_classes: int = 2,
        task: str = "segmentation",
        loss: Union[str, nn.Module] = "auto",
        optimizer: Union[str, Any] = "adamw",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        batch_size: int = 16,
        num_workers: int = 4,
        device: Optional[str] = None,
        mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        output_dir: str = "./training_output",
        callbacks: Optional[List[Callback]] = None,
    ) -> None:
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.num_classes = num_classes
        self.task = task
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.should_stop = False  # Set by EarlyStopping callback

        # Device
        self.device = self._resolve_device(device)
        self.model = self.model.to(self.device)
        logger.info("Training on device: %s", self.device)

        # Mixed precision
        self.use_amp = mixed_precision and self.device.type != "cpu"
        self.amp_dtype = self._resolve_amp_dtype()
        self.scaler = torch.cuda.amp.GradScaler() if (self.use_amp and "cuda" in str(self.device)) else None
        if self.use_amp:
            logger.info("Mixed precision enabled (%s)", self.amp_dtype)

        # Loss
        self.criterion = self._build_loss(loss, num_classes, task)

        # Optimizer
        self.optimizer = self._build_optimizer(optimizer)

        # Callbacks
        default_callbacks: List[Callback] = [
            ProgressCallback(),
            ModelCheckpoint(
                dirpath=str(self.output_dir / "checkpoints"),
                monitor="val_miou" if "segmentation" in task else "val_loss",
                mode="max" if "segmentation" in task else "min",
            ),
        ]
        self.callbacks = CallbackList(default_callbacks + (callbacks or []))

        # History
        self.history: Dict[str, List[float]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, epochs: int = 50) -> TrainingResult:
        """Run the training loop.

        Args:
            epochs: Number of training epochs.

        Returns:
            TrainingResult with best checkpoint and metric history.
        """
        start = time.time()
        self.callbacks.on_train_begin(self)
        best_metric = float("-inf")
        best_epoch = 0
        best_model_path: Optional[Path] = None

        train_loader = self._make_loader(self.train_dataset, shuffle=True)
        val_loader = (
            self._make_loader(self.val_dataset, shuffle=False)
            if self.val_dataset
            else None
        )

        for epoch in range(epochs):
            if self.should_stop:
                logger.info("Early stopping triggered at epoch %d.", epoch)
                break

            self.callbacks.on_epoch_begin(self, epoch)

            # Training phase
            train_metrics = self._train_epoch(train_loader, epoch)

            # Validation phase
            val_metrics: Dict[str, float] = {}
            if val_loader:
                val_metrics = self._validate_epoch(val_loader)

            all_metrics = {**train_metrics, **val_metrics}
            self.callbacks.on_epoch_end(self, epoch, all_metrics)

            # Track history
            for k, v in all_metrics.items():
                self.history.setdefault(k, []).append(v)

            # Track best
            monitor_key = "val_miou" if "val_miou" in all_metrics else "val_loss"
            monitor_val = all_metrics.get(monitor_key, float("-inf"))
            if monitor_val > best_metric:
                best_metric = monitor_val
                best_epoch = epoch

        self.callbacks.on_train_end(self)

        # Get best checkpoint from ModelCheckpoint callback
        for cb in self.callbacks.callbacks:
            if isinstance(cb, ModelCheckpoint) and cb.best_model_path:
                best_model_path = cb.best_model_path

        total_time = time.time() - start
        logger.info(
            "Training complete. Best epoch: %d | Best metric: %.4f | Time: %.1fs",
            best_epoch, best_metric, total_time,
        )

        return TrainingResult(
            best_epoch=best_epoch,
            best_metric=best_metric,
            best_model_path=best_model_path,
            history=self.history,
            total_time_seconds=total_time,
        )

    def evaluate(self, dataset: Optional[Dataset] = None) -> Dict[str, float]:
        """Run evaluation on a dataset.

        Args:
            dataset: Dataset to evaluate on. Defaults to val_dataset.

        Returns:
            Dict of metric names to values.
        """
        eval_ds = dataset or self.val_dataset
        if eval_ds is None:
            raise ValueError("No dataset provided for evaluation.")
        loader = self._make_loader(eval_ds, shuffle=False)
        return self._validate_epoch(loader)

    # ------------------------------------------------------------------
    # Training loop internals
    # ------------------------------------------------------------------

    def _train_epoch(
        self, loader: DataLoader, epoch: int
    ) -> Dict[str, float]:
        self.model.train()
        loss_meter = AverageMeter("train_loss")
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(loader):
            self.callbacks.on_batch_begin(self, batch_idx)

            images, targets = self._unpack_batch(batch)

            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                loss = loss / self.gradient_accumulation_steps

            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.max_grad_norm > 0:
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

            loss_meter.update(loss.item() * self.gradient_accumulation_steps, n=images.shape[0])
            self.callbacks.on_batch_end(self, batch_idx, loss_meter.val)

        return {"train_loss": loss_meter.avg}

    @torch.no_grad()
    def _validate_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        loss_meter = AverageMeter("val_loss")

        if "segmentation" in self.task and self.num_classes > 2:
            cm = ConfusionMatrix(self.num_classes)
        else:
            cm = None
        binary_m = BinaryMetrics() if self.task == "binary_segmentation" else None

        for batch in loader:
            images, targets = self._unpack_batch(batch)
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                outputs = self.model(images)
                if hasattr(outputs, "logits"):
                    outputs = outputs.logits
                import torch.nn.functional as F
                if outputs.shape[-2:] != targets.shape[-2:]:
                    outputs = F.interpolate(
                        outputs, size=targets.shape[-2:], mode="bilinear", align_corners=False
                    )
                loss = self.criterion(outputs, targets)

            loss_meter.update(loss.item(), n=images.shape[0])
            if cm is not None:
                cm.update(outputs, targets)
            if binary_m is not None:
                binary_m.update(outputs, targets)

        metrics = {"val_loss": loss_meter.avg}
        if cm is not None:
            seg_metrics = cm.compute()
            metrics["val_miou"] = seg_metrics.mean_iou
            metrics["val_accuracy"] = seg_metrics.accuracy
        if binary_m is not None:
            bm = binary_m.compute()
            metrics.update({f"val_{k}": v for k, v in bm.items()})

        return metrics

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _unpack_batch(self, batch: Any):
        """Unpack a DataLoader batch to (images, targets)."""
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            images, targets = batch[0], batch[1]
        elif isinstance(batch, dict):
            images = batch.get("image", batch.get("pixel_values"))
            targets = batch.get("label", batch.get("mask", batch.get("target")))
        else:
            raise ValueError(f"Unexpected batch format: {type(batch)}")
        return images.to(self.device), targets.to(self.device)

    def _make_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
            drop_last=shuffle,
        )

    @staticmethod
    def _resolve_device(device: Optional[str]) -> torch.device:
        if device:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _resolve_amp_dtype(self) -> torch.dtype:
        if self.device.type == "cpu":
            return torch.float32
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            return torch.bfloat16 if cap[0] >= 8 else torch.float16
        return torch.float16

    @staticmethod
    def _build_loss(loss: Union[str, nn.Module], num_classes: int, task: str) -> nn.Module:
        if isinstance(loss, nn.Module):
            return loss
        if loss == "auto":
            if task == "binary_segmentation":
                return get_loss("dice_focal")
            if "segmentation" in task:
                return get_loss("cross_entropy")
            if task == "classification":
                return nn.CrossEntropyLoss()
            if task == "change_detection":
                return get_loss("change_detection")
            if task == "super_resolution":
                return get_loss("l1")
            return get_loss("cross_entropy")
        return get_loss(loss)

    def _build_optimizer(self, optimizer: Union[str, Any]) -> Any:
        if not isinstance(optimizer, str):
            return optimizer
        params = self.model.parameters()
        if optimizer == "adamw":
            return torch.optim.AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        if optimizer == "adam":
            return torch.optim.Adam(params, lr=self.learning_rate)
        if optimizer == "sgd":
            return torch.optim.SGD(params, lr=self.learning_rate, momentum=0.9, weight_decay=self.weight_decay)
        raise ValueError(f"Unknown optimizer '{optimizer}'. Use 'adamw', 'adam', or 'sgd'.")
