"""Checkpoint management — save, load, resume training."""
from __future__ import annotations
import json, logging
from pathlib import Path
from typing import Any, Dict, List, Optional
logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manage training checkpoints with versioning and cleanup.

    Example::

        cm = CheckpointManager("./checkpoints/", keep_top_k=3)
        cm.save(epoch=10, model=model, optimizer=opt, metrics={"val_iou": 0.85})
        cm.load_best(model, optimizer)
    """

    def __init__(self, dirpath: str = "./checkpoints/", keep_top_k: int = 3,
                 monitor: str = "val_iou", mode: str = "max") -> None:
        self.dirpath = Path(dirpath)
        self.keep_top_k = keep_top_k
        self.monitor = monitor
        self.mode = mode
        self.dirpath.mkdir(parents=True, exist_ok=True)
        self._index_path = self.dirpath / "index.json"
        self._index = self._load_index()

    def _load_index(self) -> Dict:
        if self._index_path.exists():
            return json.loads(self._index_path.read_text())
        return {"checkpoints": [], "best": None}

    def _save_index(self) -> None:
        self._index_path.write_text(json.dumps(self._index, indent=2))

    def save(self, epoch: int, model: Any, optimizer: Any = None,
              scheduler: Any = None, metrics: Optional[Dict] = None,
              extra: Optional[Dict] = None) -> str:
        """Save a training checkpoint."""
        try:
            import torch
        except ImportError:
            raise ImportError("torch required")

        ckpt_path = self.dirpath / f"epoch_{epoch:04d}.pth"
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": metrics or {},
            "extra": extra or {},
        }
        if optimizer:
            state["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler:
            state["scheduler_state_dict"] = scheduler.state_dict()

        torch.save(state, str(ckpt_path))

        # Update index
        value = (metrics or {}).get(self.monitor, 0.0)
        entry = {"path": str(ckpt_path), "epoch": epoch,
                 "metrics": metrics or {}, "monitor_value": value}
        self._index["checkpoints"].append(entry)

        reverse = (self.mode == "max")
        self._index["checkpoints"].sort(key=lambda x: x["monitor_value"], reverse=reverse)

        if self._index["checkpoints"]:
            self._index["best"] = self._index["checkpoints"][0]["path"]

        # Remove old checkpoints
        while len(self._index["checkpoints"]) > self.keep_top_k:
            old = self._index["checkpoints"].pop()
            try:
                Path(old["path"]).unlink()
                logger.debug("Removed checkpoint: %s", old["path"])
            except FileNotFoundError:
                pass

        # Save last checkpoint link
        import shutil
        shutil.copy(str(ckpt_path), str(self.dirpath / "last.pth"))
        self._save_index()
        logger.info("Checkpoint saved: epoch=%d %s=%.4f", epoch, self.monitor, value)
        return str(ckpt_path)

    def load(self, path: str, model: Any, optimizer: Any = None,
              scheduler: Any = None, strict: bool = True) -> Dict:
        """Load a checkpoint into model and optionally optimizer/scheduler."""
        try:
            import torch
        except ImportError:
            raise ImportError("torch required")

        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state["model_state_dict"], strict=strict)
        if optimizer and "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        if scheduler and "scheduler_state_dict" in state:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        logger.info("Checkpoint loaded: %s (epoch=%d)", path, state.get("epoch", "?"))
        return state

    def load_best(self, model: Any, optimizer: Any = None,
                   scheduler: Any = None) -> Optional[Dict]:
        """Load the best checkpoint."""
        best = self._index.get("best")
        if not best or not Path(best).exists():
            logger.warning("No best checkpoint found in %s", self.dirpath)
            return None
        return self.load(best, model, optimizer, scheduler)

    def load_last(self, model: Any, optimizer: Any = None,
                   scheduler: Any = None) -> Optional[Dict]:
        """Load the last checkpoint (resume training)."""
        last = self.dirpath / "last.pth"
        if not last.exists():
            return None
        return self.load(str(last), model, optimizer, scheduler)

    @property
    def best_metrics(self) -> Optional[Dict]:
        ckpts = self._index.get("checkpoints", [])
        return ckpts[0]["metrics"] if ckpts else None
