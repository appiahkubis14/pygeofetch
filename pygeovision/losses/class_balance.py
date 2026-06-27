"""Class-balancing losses for imbalanced geospatial datasets (D2)."""
from __future__ import annotations
from typing import Any, List, Optional


class ClassBalancedCrossEntropy:
    """Cross-entropy with class-frequency-based weight balancing."""
    def __init__(self, class_counts: Optional[List[int]] = None, beta: float = 0.9999,
                 ignore_index: int = 255) -> None:
        self.class_counts = class_counts
        self.beta = beta
        self.ignore_index = ignore_index

    def _effective_weights(self) -> Any:
        try:
            import torch
            if self.class_counts is None:
                return None
            effective_num = [1.0 - self.beta**n for n in self.class_counts]
            weights = [(1.0 - self.beta) / max(e, 1e-10) for e in effective_num]
            total = sum(weights)
            weights = [w / total * len(weights) for w in weights]
            return torch.tensor(weights, dtype=torch.float32)
        except ImportError:
            return None

    def __call__(self, pred: Any, targets: Any) -> Any:
        import torch.nn.functional as F
        weights = self._effective_weights()
        return F.cross_entropy(pred, targets.long(), weight=weights,
                                ignore_index=self.ignore_index)


class LabelSmoothingCrossEntropy:
    """Cross-entropy with label smoothing — reduces overconfidence."""
    def __init__(self, smoothing: float = 0.1, ignore_index: int = 255) -> None:
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def __call__(self, pred: Any, targets: Any) -> Any:
        try:
            import torch, torch.nn.functional as F
            C = pred.size(1)
            ce = F.cross_entropy(pred, targets.long(), ignore_index=self.ignore_index)
            smooth_loss = -pred.log_softmax(dim=1).mean(dim=1)
            mask = targets != self.ignore_index
            smooth_loss = smooth_loss[mask].mean() if mask.sum() > 0 else torch.tensor(0.0)
            return (1 - self.smoothing) * ce + self.smoothing / C * smooth_loss
        except ImportError:
            raise ImportError("torch required")


class FocalCrossEntropy:
    """Focal loss implemented as a cross-entropy variant."""
    def __init__(self, gamma: float = 2.0, ignore_index: int = 255) -> None:
        self.gamma = gamma
        self.ignore_index = ignore_index
    def __call__(self, pred: Any, targets: Any) -> Any:
        from pygeovision.losses.segmentation import FocalLoss
        return FocalLoss(gamma=self.gamma, ignore_index=self.ignore_index)(pred, targets)
