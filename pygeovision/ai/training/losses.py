"""
Loss functions for geospatial AI training.

Provides segmentation, detection, and change-detection losses
optimized for satellite imagery, including class-imbalance-aware losses.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
from typing import Optional


class DiceLoss(nn.Module):
    """Soft Dice loss for binary and multi-class segmentation.

    Handles class imbalance well for sparse foreground classes (buildings, roads).

    Args:
        smooth: Smoothing constant to avoid division by zero.
        reduction: 'mean' or 'sum'.
    """

    def __init__(self, smooth: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits) if logits.shape[1] == 1 else F.softmax(logits, dim=1)
        if logits.shape[1] == 1:
            probs = probs.squeeze(1)
            targets = targets.float()
            intersection = (probs * targets).sum(dim=(-2, -1))
            union = probs.sum(dim=(-2, -1)) + targets.sum(dim=(-2, -1))
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        else:
            B, C, H, W = probs.shape
            t_onehot = F.one_hot(targets.long(), C).permute(0, 3, 1, 2).float()
            intersection = (probs * t_onehot).sum(dim=(-2, -1))
            union = probs.sum(dim=(-2, -1)) + t_onehot.sum(dim=(-2, -1))
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        loss = 1.0 - dice
        return loss.mean() if self.reduction == "mean" else loss.sum()


class FocalLoss(nn.Module):
    """Focal loss for addressing extreme class imbalance.

    Particularly useful for rare object detection in satellite imagery
    (e.g., vehicles, solar panels in large scenes).

    Args:
        alpha: Class balancing weight.
        gamma: Focusing parameter (0 = cross entropy, 2 = default).
        reduction: 'mean' or 'sum'.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Handle shape mismatch: targets (B,H,W) → (B,1,H,W)
        t = targets.float()
        if logits.dim() == 4 and t.dim() == 3:
            t = t.unsqueeze(1)
        bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean() if self.reduction == "mean" else focal.sum()


class DiceFocalLoss(nn.Module):
    """Combined Dice + Focal loss — strong default for geo-segmentation.

    Args:
        dice_weight: Weight for the Dice component.
        focal_weight: Weight for the Focal component.
        alpha: Focal loss alpha.
        gamma: Focal loss gamma.
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (
            self.dice_weight * self.dice(logits, targets)
            + self.focal_weight * self.focal(logits, targets)
        )


class TverskyLoss(nn.Module):
    """Tversky loss — asymmetric Dice that penalizes false negatives more.

    Useful for detecting rare, small objects (e.g. buildings in rural areas).

    Args:
        alpha: False negative weight (>0.5 = penalise FN more).
        beta: False positive weight.
        smooth: Smoothing constant.
    """

    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).squeeze(1)
        targets = targets.float()
        tp = (probs * targets).sum(dim=(-2, -1))
        fp = (probs * (1 - targets)).sum(dim=(-2, -1))
        fn = ((1 - probs) * targets).sum(dim=(-2, -1))
        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        return (1 - tversky).mean()


class WeightedCrossEntropyLoss(nn.Module):
    """Cross-entropy with per-class weighting for imbalanced land cover.

    Args:
        class_weights: Tensor of shape (num_classes,) with class weights.
        ignore_index: Class index to ignore in loss computation.
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits,
            targets.long(),
            weight=self.class_weights,
            ignore_index=self.ignore_index,
        )


class ChangeDetectionLoss(nn.Module):
    """Combined loss for bi-temporal change detection.

    Combines binary cross entropy for changed/unchanged with
    optional semantic segmentation losses for change class labels.

    Args:
        bce_weight: Weight for the binary change detection BCE loss.
        semantic_weight: Weight for the semantic class loss.
        focal_gamma: Focal loss gamma for the BCE component.
    """

    def __init__(
        self,
        bce_weight: float = 0.6,
        semantic_weight: float = 0.4,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.semantic_weight = semantic_weight
        self.focal = FocalLoss(gamma=focal_gamma)
        self.dice = DiceLoss()

    def forward(
        self,
        change_logits: torch.Tensor,
        change_targets: torch.Tensor,
        semantic_logits: Optional[torch.Tensor] = None,
        semantic_targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss = self.bce_weight * (
            0.5 * self.focal(change_logits, change_targets)
            + 0.5 * self.dice(change_logits, change_targets)
        )
        if semantic_logits is not None and semantic_targets is not None:
            sem_loss = F.cross_entropy(semantic_logits, semantic_targets.long())
            loss = loss + self.semantic_weight * sem_loss
        return loss


def get_loss(name: str, **kwargs) -> nn.Module:
    """Factory for loss functions by name.

    Args:
        name: Loss function name. One of:
            'dice', 'focal', 'dice_focal', 'tversky',
            'cross_entropy', 'weighted_ce', 'change_detection'.
        **kwargs: Additional arguments for the loss constructor.

    Returns:
        Instantiated loss module.

    Raises:
        ValueError: If the loss name is not recognized.
    """
    _LOSSES = {
        "dice": DiceLoss,
        "focal": FocalLoss,
        "dice_focal": DiceFocalLoss,
        "tversky": TverskyLoss,
        "cross_entropy": nn.CrossEntropyLoss,
        "weighted_ce": WeightedCrossEntropyLoss,
        "change_detection": ChangeDetectionLoss,
        "mse": nn.MSELoss,
        "l1": nn.L1Loss,
    }
    if name not in _LOSSES:
        raise ValueError(
            f"Unknown loss '{name}'. Available: {list(_LOSSES.keys())}"
        )
    return _LOSSES[name](**kwargs)
