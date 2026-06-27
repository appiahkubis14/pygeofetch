"""
Metrics for evaluating geospatial AI models.

Provides pixel-wise segmentation metrics, object detection metrics,
and change detection metrics — all computed efficiently on GPU tensors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


@dataclass
class SegmentationMetrics:
    """Accumulated segmentation metrics over a dataset.

    Attributes:
        iou_per_class: IoU for each class.
        accuracy: Overall pixel accuracy.
        mean_iou: Mean IoU across all classes.
        frequency_weighted_iou: Frequency-weighted IoU.
        precision_per_class: Precision per class.
        recall_per_class: Recall per class.
        f1_per_class: F1 score per class.
    """
    iou_per_class: List[float] = field(default_factory=list)
    accuracy: float = 0.0
    mean_iou: float = 0.0
    frequency_weighted_iou: float = 0.0
    precision_per_class: List[float] = field(default_factory=list)
    recall_per_class: List[float] = field(default_factory=list)
    f1_per_class: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, float]:
        return {
            "mean_iou": self.mean_iou,
            "accuracy": self.accuracy,
            "fw_iou": self.frequency_weighted_iou,
            **{f"iou_class_{i}": v for i, v in enumerate(self.iou_per_class)},
        }


class ConfusionMatrix:
    """Running confusion matrix for semantic segmentation.

    Accumulates predictions and targets across batches and computes
    all derived metrics at once for efficiency.

    Args:
        num_classes: Number of semantic classes.
        ignore_index: Class index to exclude from metrics.

    Example:
        >>> cm = ConfusionMatrix(num_classes=10)
        >>> for preds, targets in dataloader:
        ...     cm.update(preds, targets)
        >>> metrics = cm.compute()
        >>> print(f"mIoU: {metrics.mean_iou:.4f}")
    """

    def __init__(self, num_classes: int, ignore_index: int = -1) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate predictions into the confusion matrix.

        Args:
            preds: Predicted class logits (B, C, H, W) or class IDs (B, H, W).
            targets: Ground truth class IDs (B, H, W).
        """
        if preds.ndim == 4:
            preds = preds.argmax(dim=1)

        mask = targets != self.ignore_index
        preds = preds[mask].cpu()
        targets = targets[mask].cpu()

        # Compute histogram
        indices = self.num_classes * targets.long() + preds.long()
        self.matrix += torch.bincount(
            indices, minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)

    def reset(self) -> None:
        """Reset the confusion matrix."""
        self.matrix.zero_()

    def compute(self) -> SegmentationMetrics:
        """Compute all metrics from the accumulated confusion matrix.

        Returns:
            SegmentationMetrics with mIoU, accuracy, F1, precision, recall.
        """
        m = self.matrix.float()
        tp = m.diag()
        fp = m.sum(0) - tp
        fn = m.sum(1) - tp

        eps = 1e-7
        iou = (tp / (tp + fp + fn + eps)).tolist()
        precision = (tp / (tp + fp + eps)).tolist()
        recall = (tp / (tp + fn + eps)).tolist()
        f1 = [2 * p * r / (p + r + eps) for p, r in zip(precision, recall)]

        total = m.sum()
        accuracy = (tp.sum() / (total + eps)).item()
        freq = m.sum(1) / (total + eps)
        fw_iou = (freq * torch.tensor(iou)).sum().item()
        miou = float(np.nanmean([v for v in iou if not np.isnan(v)]))

        return SegmentationMetrics(
            iou_per_class=iou,
            accuracy=accuracy,
            mean_iou=miou,
            frequency_weighted_iou=fw_iou,
            precision_per_class=precision,
            recall_per_class=recall,
            f1_per_class=f1,
        )


class BinaryMetrics:
    """Metrics for binary segmentation (building footprints, water bodies, etc.).

    Tracks TP/FP/FN/TN for computing precision, recall, F1, and IoU.

    Example:
        >>> metrics = BinaryMetrics(threshold=0.5)
        >>> metrics.update(pred_logits, targets)
        >>> print(metrics.compute())
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self._tp = self._fp = self._fn = self._tn = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        preds = (torch.sigmoid(logits) >= self.threshold).long()
        targets = targets.long()
        self._tp += int((preds * targets).sum())
        self._fp += int((preds * (1 - targets)).sum())
        self._fn += int(((1 - preds) * targets).sum())
        self._tn += int(((1 - preds) * (1 - targets)).sum())

    def reset(self) -> None:
        self._tp = self._fp = self._fn = self._tn = 0

    def compute(self) -> Dict[str, float]:
        eps = 1e-7
        tp, fp, fn = self._tp, self._fp, self._fn
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        iou = tp / (tp + fp + fn + eps)
        accuracy = (tp + self._tn) / (tp + fp + fn + self._tn + eps)
        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "iou": float(iou),
            "accuracy": float(accuracy),
        }


class AverageMeter:
    """Tracks a running average of a scalar metric across batches.

    Example:
        >>> loss_meter = AverageMeter("loss")
        >>> for batch in loader:
        ...     loss = criterion(...)
        ...     loss_meter.update(loss.item(), n=len(batch))
        >>> print(loss_meter)
    """

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.val = self.avg = self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0.0

    def __repr__(self) -> str:
        return f"{self.name}: {self.avg:.4f} (last={self.val:.4f}, n={self.count})"
