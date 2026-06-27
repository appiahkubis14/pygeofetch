"""PyGeoVision AI training package."""

try:
    from pygeovision.ai.training.trainer import GeoTrainer, TrainingResult
    from pygeovision.ai.training.losses import get_loss, DiceLoss, FocalLoss, DiceFocalLoss
    from pygeovision.ai.training.metrics import ConfusionMatrix, BinaryMetrics, AverageMeter
    from pygeovision.ai.training.callbacks import (
        Callback, EarlyStopping, ModelCheckpoint, MLflowLogger, LRSchedulerCallback
    )
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    GeoTrainer = None  # type: ignore[assignment,misc]
    TrainingResult = None  # type: ignore[assignment,misc]

    # Non-torch components — import unconditionally
    try:
        from pygeovision.ai.training.losses import get_loss, DiceLoss, FocalLoss, DiceFocalLoss
        from pygeovision.ai.training.metrics import ConfusionMatrix, BinaryMetrics, AverageMeter
        from pygeovision.ai.training.callbacks import (
            Callback, EarlyStopping, ModelCheckpoint, MLflowLogger, LRSchedulerCallback
        )
    except Exception:
        pass

from pygeovision.ai.training.distributed import (
    setup_ddp, cleanup_ddp, wrap_ddp, get_rank, get_world_size, is_main_process
)

__all__ = [
    "GeoTrainer", "TrainingResult",
    "get_loss", "DiceLoss", "FocalLoss", "DiceFocalLoss",
    "ConfusionMatrix", "BinaryMetrics", "AverageMeter",
    "Callback", "EarlyStopping", "ModelCheckpoint", "MLflowLogger", "LRSchedulerCallback",
    "setup_ddp", "cleanup_ddp", "wrap_ddp", "get_rank", "get_world_size", "is_main_process",
]
