"""
Geospatial segmentation losses (D1) — pure PyTorch, no GeoAI dependency.

Includes:
    DiceLoss          — overlap-based, class-imbalance tolerant
    FocalLoss         — down-weights easy examples
    TverskyLoss       — generalised Dice with FP/FN weighting
    ComboLoss         — Dice + CE weighted combination
    BoundaryAwareLoss — penalises errors at object boundaries
    LovaszLoss        — convex surrogate for IoU
    OhemCrossEntropy  — online hard example mining
    GeospatialMixedLoss — production-ready composite
"""
from __future__ import annotations
import logging
from typing import Optional
logger = logging.getLogger(__name__)


def _check_torch():
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError("torch required: pip install torch")


class DiceLoss:
    """Dice loss for semantic segmentation.

    Directly optimises the Dice coefficient (overlap ratio).
    Highly robust to class imbalance — ideal for geospatial where
    buildings/roads occupy <5% of pixels.

    Example::

        loss_fn = DiceLoss(smooth=1.0, per_class=True)
        loss = loss_fn(predictions, targets)   # predictions: (B, C, H, W), targets: (B, H, W)
    """

    def __init__(
        self,
        smooth: float = 1.0,
        per_class: bool = True,
        reduction: str = "mean",
        ignore_index: int = 255,
    ) -> None:
        self.smooth = smooth
        self.per_class = per_class
        self.reduction = reduction
        self.ignore_index = ignore_index

    def __call__(self, predictions: Any, targets: Any) -> Any:
        torch = _check_torch()
        import torch.nn.functional as F

        probs = torch.softmax(predictions, dim=1)   # (B, C, H, W)
        B, C, H, W = probs.shape

        mask = (targets != self.ignore_index)
        targets_masked = targets.clone()
        targets_masked[~mask] = 0

        targets_one_hot = F.one_hot(targets_masked.long(), num_classes=C)  # (B, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()     # (B, C, H, W)
        # Zero out ignored pixels
        mask_exp = mask.unsqueeze(1).expand_as(probs)
        probs = probs * mask_exp
        targets_one_hot = targets_one_hot * mask_exp

        dims = (0, 2, 3)   # average over batch, H, W
        intersection = (probs * targets_one_hot).sum(dims)
        cardinality   = probs.sum(dims) + targets_one_hot.sum(dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        if not self.per_class:
            dice = dice.mean()

        loss = 1.0 - dice
        if self.per_class:
            loss = loss.mean() if self.reduction == "mean" else loss.sum()
        return loss


class FocalLoss:
    """Focal loss — down-weights easy/well-classified examples.

    Forces the model to focus on hard, misclassified pixels.
    Particularly effective for:
        - Dense building detection (hard positives at boundaries)
        - Small objects (ships, cars)

    Example::

        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        loss = loss_fn(predictions, targets)
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        ignore_index: int = 255,
        reduction: str = "mean",
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def __call__(self, predictions: Any, targets: Any) -> Any:
        torch = _check_torch()
        import torch.nn.functional as F

        ce = F.cross_entropy(predictions, targets.long(),
                              ignore_index=self.ignore_index, reduction="none")
        pt = torch.exp(-ce)
        focal = self.alpha * (1 - pt) ** self.gamma * ce

        if self.reduction == "mean":
            valid = (targets != self.ignore_index).sum().clamp(min=1)
            return focal.sum() / valid
        return focal.sum()


class TverskyLoss:
    """Tversky loss — generalised Dice with separate FP/FN penalties.

    Setting alpha=0.3, beta=0.7 penalises false negatives more heavily,
    useful for detecting rare classes (e.g. damaged buildings in disaster maps).

    alpha: weight for false positives
    beta:  weight for false negatives (set > 0.5 to reduce misses)
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1.0,
        ignore_index: int = 255,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.ignore_index = ignore_index

    def __call__(self, predictions: Any, targets: Any) -> Any:
        torch = _check_torch()
        import torch.nn.functional as F

        probs = torch.softmax(predictions, dim=1)
        B, C, H, W = probs.shape
        mask = (targets != self.ignore_index)
        targets_c = targets.clone(); targets_c[~mask] = 0
        targets_oh = F.one_hot(targets_c.long(), C).permute(0,3,1,2).float()
        mask_exp = mask.unsqueeze(1).expand_as(probs)
        p, t = probs * mask_exp, targets_oh * mask_exp

        tp = (p * t).sum(dim=(0,2,3))
        fp = (p * (1 - t)).sum(dim=(0,2,3))
        fn = ((1 - p) * t).sum(dim=(0,2,3))
        tversky = (tp + self.smooth) / (tp + self.alpha*fp + self.beta*fn + self.smooth)
        return (1 - tversky).mean()


class ComboLoss:
    """Combo loss = weighted Dice + weighted CE.

    Default weights: 0.5 Dice + 0.5 CE, optimal for most segmentation.
    Use dice_weight=0.7 for highly imbalanced datasets.
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        smooth: float = 1.0,
        ignore_index: int = 255,
    ) -> None:
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self._dice = DiceLoss(smooth=smooth, ignore_index=ignore_index)
        self.ignore_index = ignore_index

    def __call__(self, predictions: Any, targets: Any) -> Any:
        torch = _check_torch()
        import torch.nn.functional as F
        dice = self._dice(predictions, targets)
        ce = F.cross_entropy(predictions, targets.long(), ignore_index=self.ignore_index)
        return self.dice_weight * dice + self.ce_weight * ce


class BoundaryAwareLoss:
    """Boundary-aware loss — adds extra penalty at object edges.

    Extracts boundaries using morphological erosion and adds
    focal loss specifically at boundary pixels. This sharpens
    edges in building footprints, road networks, and coastlines.
    """

    def __init__(
        self,
        boundary_weight: float = 5.0,
        base_loss: str = "combo",
        kernel_size: int = 3,
        ignore_index: int = 255,
    ) -> None:
        self.boundary_weight = boundary_weight
        self.base_loss_name = base_loss
        self.kernel_size = kernel_size
        self.ignore_index = ignore_index
        self._base_loss = ComboLoss(ignore_index=ignore_index) if base_loss == "combo" else DiceLoss(ignore_index=ignore_index)

    def _extract_boundaries(self, targets: Any) -> Any:
        """Extract boundary pixels via morphological erosion."""
        torch = _check_torch()
        import torch.nn.functional as F
        import math

        pad = self.kernel_size // 2
        # Erode: min-pool
        t = targets.unsqueeze(1).float()
        eroded = -F.max_pool2d(-t, self.kernel_size, stride=1, padding=pad)
        boundary = (targets.float().unsqueeze(1) != eroded).squeeze(1)
        return boundary & (targets != self.ignore_index)

    def __call__(self, predictions: Any, targets: Any) -> Any:
        torch = _check_torch()
        import torch.nn.functional as F

        base = self._base_loss(predictions, targets)
        boundaries = self._extract_boundaries(targets)

        if boundaries.sum() > 0:
            boundary_ce = F.cross_entropy(
                predictions, targets.long(),
                ignore_index=self.ignore_index, reduction="none"
            )
            boundary_loss = (boundary_ce * boundaries.float()).sum() / boundaries.sum().clamp(min=1)
            return base + self.boundary_weight * boundary_loss
        return base


class OhemCrossEntropy:
    """Online Hard Example Mining Cross-Entropy (OHEM).

    Selects the hardest (highest loss) pixels for backpropagation,
    ignoring easy/well-classified pixels. Particularly effective
    for scenes with large homogeneous regions (farmland, water).
    """

    def __init__(
        self,
        thresh: float = 0.7,
        min_kept: int = 100000,
        ignore_index: int = 255,
    ) -> None:
        self.thresh = thresh
        self.min_kept = min_kept
        self.ignore_index = ignore_index

    def __call__(self, predictions: Any, targets: Any) -> Any:
        torch = _check_torch()
        import torch.nn.functional as F

        ce = F.cross_entropy(predictions, targets.long(),
                              ignore_index=self.ignore_index, reduction="none")
        ce_flat = ce.view(-1)
        valid = ce_flat[ce_flat > 0]
        if valid.numel() == 0:
            return ce.mean()
        # Keep hardest examples above threshold
        thresh_value = max(self.thresh, valid.topk(min(self.min_kept, valid.numel()))[0].min())
        hard_mask = ce_flat >= thresh_value
        return ce_flat[hard_mask].mean()


class LovaszLoss:
    """Lovász-Softmax loss — convex surrogate for the IoU metric.

    Directly optimises the mean Intersection-over-Union, making it
    ideal for evaluation-aligned training of segmentation models.
    """

    def __init__(self, ignore_index: int = 255, per_image: bool = True) -> None:
        self.ignore_index = ignore_index
        self.per_image = per_image

    def __call__(self, predictions: Any, targets: Any) -> Any:
        torch = _check_torch()
        import torch.nn.functional as F

        # Lovász extension for multiclass via one-vs-rest
        B, C = predictions.shape[:2]
        probs = torch.softmax(predictions, dim=1)
        losses = []
        for c in range(C):
            fg = (targets == c).float()
            mask = (targets != self.ignore_index)
            if fg.sum() == 0:
                continue
            errors = (fg - probs[:, c]).abs()
            errors_flat = errors[mask]
            fg_flat = fg[mask]
            # Sort by error descending
            sorted_idx = torch.argsort(errors_flat, descending=True)
            sorted_errors = errors_flat[sorted_idx]
            sorted_fg = fg_flat[sorted_idx]
            # Lovász gradient
            n_pos = sorted_fg.sum()
            inter = n_pos - sorted_fg.cumsum(0)
            union = n_pos + (1 - sorted_fg).cumsum(0)
            jaccard = 1 - inter / union.clamp(min=1)
            grad = torch.cat([jaccard[:1], jaccard[1:] - jaccard[:-1]])
            losses.append((sorted_errors * grad).sum())

        return torch.stack(losses).mean() if losses else predictions.sum() * 0


class GeospatialMixedLoss:
    """Production-ready composite loss for geospatial segmentation.

    Combines multiple loss components with tunable weights:
        combo_loss + boundary_loss + ohem_loss

    This is the recommended loss for production geospatial training.

    Example::

        loss_fn = GeospatialMixedLoss(
            weights={"combo": 0.5, "boundary": 0.3, "ohem": 0.2},
            num_classes=7,
        )
        loss = loss_fn(predictions, targets)
    """

    def __init__(
        self,
        weights: Optional[dict] = None,
        num_classes: int = 2,
        ignore_index: int = 255,
    ) -> None:
        self.weights = weights or {"combo": 0.5, "boundary": 0.3, "ohem": 0.2}
        self._combo    = ComboLoss(ignore_index=ignore_index)
        self._boundary = BoundaryAwareLoss(ignore_index=ignore_index)
        self._ohem     = OhemCrossEntropy(ignore_index=ignore_index)

    def __call__(self, predictions: Any, targets: Any) -> Any:
        torch = _check_torch()
        total = None
        for name, weight in self.weights.items():
            if name == "combo":
                l = self._combo(predictions, targets)
            elif name == "boundary":
                l = self._boundary(predictions, targets)
            elif name == "ohem":
                l = self._ohem(predictions, targets)
            elif name == "dice":
                l = DiceLoss()(predictions, targets)
            elif name == "focal":
                l = FocalLoss()(predictions, targets)
            elif name == "tversky":
                l = TverskyLoss()(predictions, targets)
            else:
                continue
            total = l * weight if total is None else total + l * weight
        return total if total is not None else predictions.sum() * 0


from typing import Any
