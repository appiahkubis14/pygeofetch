"""Training callbacks — EarlyStopping, ModelCheckpoint, RichProgress, LRMonitor."""
from __future__ import annotations
import json, logging, time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
logger = logging.getLogger(__name__)


class Callback:
    """Base callback class."""
    def on_train_start(self, trainer: Any) -> None: pass
    def on_epoch_start(self, trainer: Any, epoch: int) -> None: pass
    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict) -> None: pass
    def on_batch_end(self, trainer: Any, batch: int, loss: float) -> None: pass
    def on_train_end(self, trainer: Any) -> None: pass


class EarlyStopping(Callback):
    """Stop training when a metric stops improving.

    Example::

        cb = EarlyStopping(monitor="val_iou", patience=10, mode="max", min_delta=0.001)
    """

    def __init__(self, monitor: str = "val_iou", patience: int = 10,
                 mode: str = "max", min_delta: float = 0.0,
                 restore_best_weights: bool = True) -> None:
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.best_weights = None
        self.wait = 0
        self.stopped_epoch = 0
        self.stop_training = False

    def _improved(self, value: float) -> bool:
        if self.mode == "max":
            return value > self.best_value + self.min_delta
        return value < self.best_value - self.min_delta

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict) -> None:
        value = metrics.get(self.monitor)
        if value is None:
            logger.warning("EarlyStopping: metric '%s' not in metrics", self.monitor)
            return
        if self._improved(value):
            self.best_value = value
            self.wait = 0
            if self.restore_best_weights and hasattr(trainer, "model"):
                try:
                    import copy
                    self.best_weights = copy.deepcopy(trainer.model.state_dict())
                except Exception:
                    pass
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.stop_training = True
                logger.info("EarlyStopping triggered at epoch %d (best %s=%.4f)",
                            epoch, self.monitor, self.best_value)
                if self.restore_best_weights and self.best_weights and hasattr(trainer, "model"):
                    trainer.model.load_state_dict(self.best_weights)
                    logger.info("Best weights restored")


class ModelCheckpoint(Callback):
    """Save the best model checkpoint during training.

    Example::

        cb = ModelCheckpoint("./checkpoints/", monitor="val_iou", mode="max", top_k=3)
    """

    def __init__(self, dirpath: str = "./checkpoints/", filename: str = "epoch{epoch:03d}",
                 monitor: str = "val_iou", mode: str = "max", save_top_k: int = 3,
                 save_last: bool = True) -> None:
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.save_last = save_last
        self._best_k: List[tuple] = []
        self.best_model_path: Optional[str] = None
        self.dirpath.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict) -> None:
        import torch
        value = metrics.get(self.monitor, 0.0)
        ckpt_name = self.filename.format(epoch=epoch, **metrics) + ".pth"
        ckpt_path = self.dirpath / ckpt_name

        if hasattr(trainer, "model"):
            state = {
                "epoch": epoch,
                "model_state": trainer.model.state_dict(),
                "metrics": metrics,
            }
            if hasattr(trainer, "optimizer"):
                state["optimizer_state"] = trainer.optimizer.state_dict()

            torch.save(state, str(ckpt_path))

            # Track top-k
            self._best_k.append((value, str(ckpt_path)))
            reverse = (self.mode == "max")
            self._best_k.sort(key=lambda x: x[0], reverse=reverse)

            # Remove excess checkpoints
            while len(self._best_k) > self.save_top_k:
                _, old_path = self._best_k.pop()
                try: Path(old_path).unlink()
                except Exception: pass

            self.best_model_path = self._best_k[0][1]

            if self.save_last:
                import shutil
                shutil.copy(str(ckpt_path), str(self.dirpath / "last.pth"))


class LearningRateMonitor(Callback):
    """Log learning rate changes during training."""

    def __init__(self) -> None:
        self.lr_history: List[Dict] = []

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict) -> None:
        if hasattr(trainer, "optimizer"):
            lrs = [pg["lr"] for pg in trainer.optimizer.param_groups]
            entry = {"epoch": epoch, "lrs": lrs}
            self.lr_history.append(entry)
            metrics["lr"] = lrs[0]


class ProgressBar(Callback):
    """Simple progress bar that works without rich."""

    def __init__(self, total_epochs: int) -> None:
        self.total = total_epochs
        self.t_start = None

    def on_train_start(self, trainer: Any) -> None:
        self.t_start = time.time()
        print(f"\n{'='*60}")
        print(f"  PyGeoVision Training — {self.total} epochs")
        print(f"{'='*60}")

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict) -> None:
        elapsed = time.time() - (self.t_start or time.time())
        eta = elapsed / max(epoch, 1) * (self.total - epoch)
        bar_len = 30
        filled = int(bar_len * epoch / self.total)
        bar = "█" * filled + "░" * (bar_len - filled)
        metric_str = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()
                                 if isinstance(v, (int, float)))
        print(f"\r  [{bar}] {epoch}/{self.total} | {metric_str} | ETA {eta:.0f}s", end="", flush=True)
        if epoch == self.total:
            print()

    def on_train_end(self, trainer: Any) -> None:
        total = time.time() - (self.t_start or time.time())
        print(f"\n  Training complete in {total:.1f}s")
        print(f"{'='*60}\n")
