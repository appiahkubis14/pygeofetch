"""
Optimizer configurations and builder for PyGeoVision training.

Provides ready-made optimizer setups tuned for common geospatial training
patterns — including layer-wise learning-rate decay for transformer encoders
and discriminative fine-tuning for pretrained backbones.

Example:
    >>> from pygeovision.ai.training.optimizers import build_optimizer, LayerWiseLRDecay
    >>> optimizer = build_optimizer(model, name="adamw", lr=1e-4)
    >>> optimizer = build_optimizer(
    ...     model, name="adamw", lr=1e-4,
    ...     layer_decay=LayerWiseLRDecay(decay=0.75, num_layers=12),
    ... )
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class OptimizerConfig:
    """Configuration for a PyGeoVision optimizer.

    Attributes:
        name: Optimizer name ('adamw', 'adam', 'sgd', 'lars', 'lion').
        lr: Base learning rate.
        weight_decay: L2 regularisation weight.
        momentum: SGD momentum (ignored for Adam variants).
        betas: Adam/AdamW beta coefficients.
        eps: Adam numerical stability epsilon.
        no_decay_keywords: Parameter name substrings that get zero weight decay.
        backbone_lr_multiplier: LR multiplier for backbone/encoder parameters.
    """

    name: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 1e-4
    momentum: float = 0.9
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    no_decay_keywords: List[str] = field(
        default_factory=lambda: ["bias", "norm", "LayerNorm", "layer_norm", "bn"]
    )
    backbone_lr_multiplier: float = 0.1


@dataclass
class LayerWiseLRDecay:
    """Layer-wise learning-rate decay for transformer models.

    Applies exponentially decreasing LR to earlier transformer layers,
    which is important for fine-tuning pretrained ViT/SegFormer encoders.

    Args:
        decay: Decay factor per layer (e.g. 0.75 means each earlier layer
            gets LR * 0.75 relative to the next deeper layer).
        num_layers: Total number of transformer layers in the encoder.

    Example:
        >>> # For SegFormer-B2 with 4 stages:
        >>> lrd = LayerWiseLRDecay(decay=0.8, num_layers=4)
        >>> optimizer = build_optimizer(model, "adamw", 1e-4, layer_decay=lrd)
    """

    decay: float = 0.75
    num_layers: int = 12


def build_optimizer(
    model: Any,
    name: str = "adamw",
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    momentum: float = 0.9,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    no_decay_keywords: Optional[List[str]] = None,
    backbone_lr_multiplier: float = 1.0,
    layer_decay: Optional[LayerWiseLRDecay] = None,
    **kwargs: Any,
) -> Any:
    """Build an optimizer with parameter group customisation.

    Supports:
    - Separate weight-decay groups (no decay for bias/norm parameters)
    - Discriminative fine-tuning (lower LR for backbone vs decoder)
    - Layer-wise LR decay for transformer encoders

    Args:
        model: PyTorch model whose parameters to optimise.
        name: Optimizer name: 'adamw', 'adam', 'sgd', 'lars', 'lion'.
        lr: Base learning rate for non-backbone parameters.
        weight_decay: Weight decay coefficient.
        momentum: SGD momentum.
        betas: Adam/AdamW beta1 and beta2.
        eps: Adam epsilon.
        no_decay_keywords: Parameter name substrings that get weight_decay=0.
        backbone_lr_multiplier: LR scale for encoder/backbone parameters.
            Values < 1.0 implement discriminative fine-tuning.
        layer_decay: Layer-wise LR decay config for transformer encoders.
        **kwargs: Extra kwargs forwarded to the optimizer constructor.

    Returns:
        Configured PyTorch optimizer.

    Example:
        >>> # Discriminative fine-tuning: backbone at 10× lower LR
        >>> opt = build_optimizer(
        ...     model, "adamw", lr=1e-4, backbone_lr_multiplier=0.1
        ... )
        >>> # ViT encoder with layer-wise decay
        >>> opt = build_optimizer(
        ...     vit_model, "adamw", lr=1e-4,
        ...     layer_decay=LayerWiseLRDecay(decay=0.8, num_layers=12),
        ... )
    """
    try:
        import torch.optim as optim
    except ImportError as exc:
        raise ImportError("build_optimizer requires torch. pip install torch") from exc

    no_decay = set(no_decay_keywords or ["bias", "norm", "LayerNorm", "layer_norm", "bn"])

    if layer_decay is not None:
        param_groups = _build_layer_wise_groups(
            model, lr=lr, weight_decay=weight_decay,
            layer_decay=layer_decay, no_decay_keywords=list(no_decay),
        )
    else:
        param_groups = _build_param_groups(
            model, lr=lr, weight_decay=weight_decay,
            backbone_lr_multiplier=backbone_lr_multiplier,
            no_decay_keywords=list(no_decay),
        )

    _name = name.lower()
    if _name == "adamw":
        optimizer = optim.AdamW(param_groups, lr=lr, betas=betas, eps=eps, **kwargs)
    elif _name == "adam":
        optimizer = optim.Adam(param_groups, lr=lr, betas=betas, eps=eps, **kwargs)
    elif _name == "sgd":
        optimizer = optim.SGD(param_groups, lr=lr, momentum=momentum, **kwargs)
    elif _name == "lars":
        optimizer = _build_lars(param_groups, lr=lr, momentum=momentum, **kwargs)
    elif _name == "lion":
        optimizer = _build_lion(param_groups, lr=lr, betas=betas, **kwargs)
    else:
        raise ValueError(
            f"Unknown optimizer '{name}'. Choose from: adamw, adam, sgd, lars, lion."
        )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Built %s optimizer | lr=%.2e | wd=%.2e | trainable params=%.2fM",
        name.upper(), lr, weight_decay, num_params / 1e6,
    )
    return optimizer


def build_lr_scheduler(
    optimizer: Any,
    name: str = "cosine",
    epochs: int = 100,
    warmup_epochs: int = 5,
    min_lr: float = 1e-6,
    **kwargs: Any,
) -> Any:
    """Build a learning-rate scheduler.

    Args:
        optimizer: Configured optimizer.
        name: Scheduler name: 'cosine', 'step', 'plateau', 'onecycle', 'linear'.
        epochs: Total training epochs (for cosine/onecycle).
        warmup_epochs: Warmup period (cosine with warmup).
        min_lr: Minimum LR for cosine annealing.
        **kwargs: Extra kwargs for the scheduler.

    Returns:
        PyTorch LR scheduler.

    Example:
        >>> scheduler = build_lr_scheduler(optimizer, "cosine", epochs=100, warmup_epochs=5)
    """
    try:
        import torch.optim.lr_scheduler as sched
    except ImportError as exc:
        raise ImportError("build_lr_scheduler requires torch.") from exc

    _name = name.lower()
    if _name == "cosine":
        if warmup_epochs > 0:
            return _CosineWithWarmup(optimizer, epochs, warmup_epochs, min_lr)
        return sched.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    elif _name == "step":
        step_size = kwargs.pop("step_size", epochs // 3)
        gamma = kwargs.pop("gamma", 0.1)
        return sched.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif _name == "plateau":
        return sched.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5,
            min_lr=min_lr, **kwargs
        )
    elif _name == "onecycle":
        steps_per_epoch = kwargs.pop("steps_per_epoch", 100)
        max_lr = kwargs.pop("max_lr", optimizer.param_groups[0]["lr"] * 10)
        return sched.OneCycleLR(
            optimizer, max_lr=max_lr,
            total_steps=epochs * steps_per_epoch, **kwargs
        )
    elif _name == "linear":
        return sched.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=epochs)
    else:
        raise ValueError(f"Unknown scheduler '{name}'. Choose: cosine, step, plateau, onecycle, linear.")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_param_groups(
    model: Any,
    lr: float,
    weight_decay: float,
    backbone_lr_multiplier: float,
    no_decay_keywords: List[str],
) -> List[Dict[str, Any]]:
    """Split parameters into decay/no-decay groups, with optional backbone LR."""
    _backbone_keys = ("encoder", "backbone", "patch_embed", "segformer.encoder")

    decay_head, no_decay_head = [], []
    decay_backbone, no_decay_backbone = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_backbone = any(k in name for k in _backbone_keys)
        is_no_decay = any(kw in name for kw in no_decay_keywords)

        if is_backbone:
            (no_decay_backbone if is_no_decay else decay_backbone).append(param)
        else:
            (no_decay_head if is_no_decay else decay_head).append(param)

    backbone_lr = lr * backbone_lr_multiplier
    groups = []
    if decay_head:
        groups.append({"params": decay_head, "lr": lr, "weight_decay": weight_decay})
    if no_decay_head:
        groups.append({"params": no_decay_head, "lr": lr, "weight_decay": 0.0})
    if decay_backbone:
        groups.append({"params": decay_backbone, "lr": backbone_lr, "weight_decay": weight_decay})
    if no_decay_backbone:
        groups.append({"params": no_decay_backbone, "lr": backbone_lr, "weight_decay": 0.0})

    logger.debug(
        "Param groups: head-decay=%d, head-nodecay=%d, bb-decay=%d, bb-nodecay=%d",
        len(decay_head), len(no_decay_head), len(decay_backbone), len(no_decay_backbone),
    )
    return groups


def _build_layer_wise_groups(
    model: Any,
    lr: float,
    weight_decay: float,
    layer_decay: LayerWiseLRDecay,
    no_decay_keywords: List[str],
) -> List[Dict[str, Any]]:
    """Build parameter groups with exponential LR decay per transformer layer."""
    num_layers = layer_decay.num_layers
    decay = layer_decay.decay

    # Assign each parameter a layer index
    layer_groups: Dict[int, List] = {i: [] for i in range(num_layers + 2)}
    no_decay_groups: Dict[int, List] = {i: [] for i in range(num_layers + 2)}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_no_decay = any(kw in name for kw in no_decay_keywords)
        layer_idx = _get_layer_index(name, num_layers)
        (no_decay_groups if is_no_decay else layer_groups)[layer_idx].append(param)

    groups = []
    for idx in range(num_layers + 2):
        layer_lr = lr * (decay ** (num_layers + 1 - idx))
        if layer_groups[idx]:
            groups.append({"params": layer_groups[idx], "lr": layer_lr, "weight_decay": weight_decay})
        if no_decay_groups[idx]:
            groups.append({"params": no_decay_groups[idx], "lr": layer_lr, "weight_decay": 0.0})

    logger.debug("Layer-wise LR: %d groups with decay=%.2f", len(groups), decay)
    return groups


def _get_layer_index(name: str, num_layers: int) -> int:
    """Map a parameter name to a transformer layer index."""
    # Patch embeddings / cls tokens -> layer 0
    if any(k in name for k in ("patch_embed", "cls_token", "pos_embed")):
        return 0
    # Decoder / head -> last layer
    if any(k in name for k in ("decode_head", "head", "classifier", "segmentation_head")):
        return num_layers + 1
    # Try to extract block/layer number from the name
    match = re.search(r"(?:blocks?|layer|encoder_layer)[.\[](\d+)", name)
    if match:
        return min(int(match.group(1)) + 1, num_layers)
    return num_layers + 1  # Default to head


def _build_lars(param_groups: List[Dict], lr: float, momentum: float, **kwargs: Any) -> Any:
    """Build LARS optimizer (requires apex or standalone implementation)."""
    try:
        from apex.optimizers import FusedLARS  # type: ignore[import]
        return FusedLARS(param_groups, lr=lr, momentum=momentum, **kwargs)
    except ImportError:
        pass

    # Minimal LARS fallback
    try:
        import torch
        import torch.optim as optim

        class LARS(optim.SGD):
            """Minimal LARS implementation."""
            def step(self, closure=None):
                for group in self.param_groups:
                    for p in group["params"]:
                        if p.grad is None:
                            continue
                        param_norm = p.data.norm()
                        grad_norm = p.grad.data.norm()
                        if param_norm > 0 and grad_norm > 0:
                            adaptive_lr = group["lr"] * param_norm / (grad_norm + 1e-8)
                            p.grad.data.mul_(adaptive_lr / group["lr"])
                return super().step(closure)

        return LARS(param_groups, lr=lr, momentum=momentum, **kwargs)
    except Exception as exc:
        raise ImportError("LARS requires apex or torch. pip install apex") from exc


def _build_lion(param_groups: List[Dict], lr: float, betas: Tuple[float, float], **kwargs: Any) -> Any:
    """Build Lion optimizer (Sign-based gradient descent)."""
    try:
        import torch
        import torch.optim as optim

        class Lion(optim.Optimizer):
            """Lion optimizer: EvoLved Sign Momentum (Chen et al., 2023)."""

            def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
                defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
                super().__init__(params, defaults)

            @torch.no_grad()
            def step(self, closure=None):
                loss = closure() if closure is not None else None
                for group in self.param_groups:
                    for p in group["params"]:
                        if p.grad is None:
                            continue
                        grad = p.grad
                        state = self.state[p]
                        if len(state) == 0:
                            state["exp_avg"] = torch.zeros_like(p)
                        exp_avg = state["exp_avg"]
                        beta1, beta2 = group["betas"]
                        # Update: w ← w - lr * (sign(β₁m + (1-β₁)g) + λw)
                        update = exp_avg.mul(beta1).add_(grad, alpha=1 - beta1).sign_()
                        if group["weight_decay"] > 0:
                            update.add_(p, alpha=group["weight_decay"])
                        p.add_(update, alpha=-group["lr"])
                        # Update moment
                        exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
                return loss

        return Lion(param_groups, lr=lr, betas=betas, **kwargs)
    except ImportError as exc:
        raise ImportError("Lion optimizer requires torch.") from exc


class _CosineWithWarmup:
    """Cosine annealing with linear warmup."""

    def __init__(self, optimizer: Any, epochs: int, warmup_epochs: int, min_lr: float):
        self.optimizer = optimizer
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self._base_lrs = [g["lr"] for g in optimizer.param_groups]
        self._step = 0

    def step(self, epoch: Optional[int] = None) -> None:
        import math
        e = epoch if epoch is not None else self._step
        self._step += 1

        for g, base_lr in zip(self.optimizer.param_groups, self._base_lrs):
            if e < self.warmup_epochs:
                g["lr"] = base_lr * (e + 1) / self.warmup_epochs
            else:
                progress = (e - self.warmup_epochs) / max(self.epochs - self.warmup_epochs, 1)
                g["lr"] = self.min_lr + 0.5 * (base_lr - self.min_lr) * (
                    1 + math.cos(math.pi * progress)
                )

    def get_last_lr(self) -> List[float]:
        return [g["lr"] for g in self.optimizer.param_groups]

    def state_dict(self) -> Dict:
        return {"step": self._step, "base_lrs": self._base_lrs}

    def load_state_dict(self, state: Dict) -> None:
        self._step = state["step"]
        self._base_lrs = state["base_lrs"]
