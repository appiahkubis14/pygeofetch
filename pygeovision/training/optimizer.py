"""Optimizer and LR scheduler builders (Phase 4.1)."""
from __future__ import annotations
from typing import Any, Optional


def build_optimizer(model: Any, cfg: Any) -> Any:
    """Build optimizer from TrainingConfig."""
    try:
        import torch.optim as optim
        params = [p for p in model.parameters() if p.requires_grad]
        name = cfg.optimizer.lower()
        if name == "adamw":
            return optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        elif name == "adam":
            return optim.Adam(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        elif name == "sgd":
            return optim.SGD(params, lr=cfg.learning_rate, momentum=cfg.momentum,
                             weight_decay=cfg.weight_decay, nesterov=cfg.nesterov)
        elif name == "lion":
            try:
                from lion_pytorch import Lion
                return Lion(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
            except ImportError:
                return optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        elif name == "lars":
            try:
                from torch.optim import SGD
                from torch.optim.lr_scheduler import LinearLR
                return SGD(params, lr=cfg.learning_rate, momentum=0.9, weight_decay=cfg.weight_decay)
            except Exception:
                return optim.AdamW(params, lr=cfg.learning_rate)
        else:
            return optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    except ImportError:
        raise ImportError("torch is required for training. pip install torch")


def build_scheduler(optimizer: Any, cfg: Any, steps_per_epoch: int = 100) -> Optional[Any]:
    """Build LR scheduler from TrainingConfig."""
    try:
        import torch.optim.lr_scheduler as sched
        name = cfg.scheduler.lower()
        total_steps = cfg.max_epochs * steps_per_epoch
        warmup_steps = cfg.warmup_epochs * steps_per_epoch

        if name == "cosine":
            scheduler = sched.CosineAnnealingLR(optimizer, T_max=cfg.max_epochs, eta_min=cfg.min_lr)
        elif name == "cosine_warmup":
            def _lr_lambda(step):
                if step < warmup_steps:
                    return cfg.warmup_lr_scale + (1.0 - cfg.warmup_lr_scale) * step / max(warmup_steps, 1)
                progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
                import math
                return cfg.min_lr / cfg.learning_rate + 0.5 * (1.0 - cfg.min_lr / cfg.learning_rate) * (1 + math.cos(math.pi * progress))
            scheduler = sched.LambdaLR(optimizer, _lr_lambda)
        elif name == "linear":
            scheduler = sched.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps)
        elif name == "step":
            scheduler = sched.StepLR(optimizer, step_size=cfg.lr_step_size, gamma=cfg.lr_gamma)
        elif name == "plateau":
            scheduler = sched.ReduceLROnPlateau(optimizer, mode="min", factor=cfg.lr_gamma,
                                                patience=5, min_lr=cfg.min_lr)
        elif name == "onecycle":
            scheduler = sched.OneCycleLR(optimizer, max_lr=cfg.learning_rate,
                                          total_steps=total_steps, pct_start=0.1)
        elif name == "warmup_cosine":
            from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
            warmup = LinearLR(optimizer, start_factor=cfg.warmup_lr_scale, end_factor=1.0, total_iters=warmup_steps)
            cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=cfg.min_lr)
            scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
        else:
            scheduler = sched.CosineAnnealingLR(optimizer, T_max=cfg.max_epochs, eta_min=cfg.min_lr)
        return scheduler
    except ImportError:
        return None
