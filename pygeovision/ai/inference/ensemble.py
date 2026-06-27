"""
Ensemble inference for PyGeoVision models.

Combines predictions from multiple models using voting, averaging,
or weighted averaging strategies for improved accuracy.

Example:
    >>> from pygeovision.ai.inference.ensemble import EnsembleInference
    >>> ensemble = EnsembleInference([model_a, model_b, model_c], strategy="mean")
    >>> predictions = ensemble.predict(image_tensor)
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_STRATEGIES = ("mean", "max", "vote", "weighted_mean")


class EnsembleInference:
    """Combine predictions from multiple models for improved accuracy.

    Supports test-time augmentation (TTA) on top of multi-model ensembling.

    Args:
        models: List of PyTorch models.
        strategy: Combination strategy: 'mean', 'max', 'vote', or 'weighted_mean'.
        weights: Per-model weights for 'weighted_mean' (must sum to 1).
        device: Compute device.
        tta: Enable test-time augmentation (horizontal + vertical flips).

    Example:
        >>> ensemble = EnsembleInference(
        ...     [unet, segformer, deeplab],
        ...     strategy="weighted_mean",
        ...     weights=[0.5, 0.3, 0.2],
        ...     tta=True,
        ... )
        >>> logits = ensemble.predict(batch)
    """

    def __init__(
        self,
        models: List[nn.Module],
        strategy: str = "mean",
        weights: Optional[Sequence[float]] = None,
        device: str = "cpu",
        tta: bool = False,
    ) -> None:
        if strategy not in _STRATEGIES:
            raise ValueError(f"strategy must be one of {_STRATEGIES}, got {strategy!r}")

        if strategy == "weighted_mean":
            if weights is None or len(weights) != len(models):
                raise ValueError(
                    "weighted_mean requires weights of the same length as models."
                )
            total = sum(weights)
            self.weights = [w / total for w in weights]
        else:
            self.weights = [1.0 / len(models)] * len(models)

        self.models = [m.to(device).eval() for m in models]
        self.strategy = strategy
        self.device = device
        self.tta = tta
        logger.info(
            "EnsembleInference: %d models, strategy=%s, tta=%s",
            len(models), strategy, tta,
        )

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Run ensemble inference on an input tensor.

        Args:
            x: Input tensor (B, C, H, W).

        Returns:
            Aggregated prediction tensor (B, C, H, W) — softmax probabilities
            for 'mean'/'weighted_mean', hard labels for 'vote'.
        """
        x = x.to(self.device)
        all_preds: List[torch.Tensor] = []

        for model, weight in zip(self.models, self.weights):
            probs = self._predict_single(model, x)
            all_preds.append(probs * weight)

        stacked = torch.stack(all_preds, dim=0)  # (N_models, B, C, H, W)

        if self.strategy == "mean":
            return stacked.mean(dim=0)
        elif self.strategy == "weighted_mean":
            return stacked.sum(dim=0)  # already weighted
        elif self.strategy == "max":
            return stacked.max(dim=0).values
        elif self.strategy == "vote":
            # Hard majority vote
            votes = stacked.argmax(dim=2)  # (N_models, B, H, W)
            # Mode across models
            mode = votes.mode(dim=0).values  # (B, H, W)
            return mode
        return stacked.mean(dim=0)

    def _predict_single(self, model, x):
        """Predict with a single model, optionally with TTA."""
        logits = model(x)
        if hasattr(logits, "logits"):
            logits = logits.logits
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        probs = F.softmax(logits, dim=1)

        if not self.tta:
            return probs

        # Test-time augmentation: hflip, vflip, hflip+vflip
        tta_preds = [probs]
        for flip_dims in ([3], [2], [2, 3]):
            aug = torch.flip(x, dims=flip_dims)
            aug_logits = model(aug)
            if hasattr(aug_logits, "logits"):
                aug_logits = aug_logits.logits
            if aug_logits.shape[-2:] != x.shape[-2:]:
                aug_logits = F.interpolate(aug_logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
            # Undo the flip on predictions
            aug_probs = torch.flip(F.softmax(aug_logits, dim=1), dims=flip_dims)
            tta_preds.append(aug_probs)

        return torch.stack(tta_preds, dim=0).mean(dim=0)
