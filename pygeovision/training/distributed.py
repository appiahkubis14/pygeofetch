"""Distributed training utilities — DDP, FSDP, gradient accumulation."""
from __future__ import annotations
import logging, os
from typing import Any, Dict, List, Optional, Tuple
logger = logging.getLogger(__name__)


def setup_distributed(backend: str = "nccl", rank: int = 0, world_size: int = 1,
                        init_method: str = "env://") -> None:
    """Initialise distributed process group."""
    try:
        import torch.distributed as dist
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, init_method=init_method,
                                     rank=rank, world_size=world_size)
            logger.info("Distributed: rank=%d/%d backend=%s", rank, world_size, backend)
    except ImportError:
        raise ImportError("torch required")


def cleanup_distributed() -> None:
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
    except ImportError:
        pass


def wrap_ddp(model: Any, device_ids: Optional[List[int]] = None,
              find_unused_parameters: bool = False) -> Any:
    """Wrap model with DistributedDataParallel."""
    try:
        from torch.nn.parallel import DistributedDataParallel as DDP
        return DDP(model, device_ids=device_ids,
                   find_unused_parameters=find_unused_parameters)
    except ImportError:
        raise ImportError("torch required")


def wrap_fsdp(model: Any, mixed_precision: bool = True) -> Any:
    """Wrap model with FullyShardedDataParallel for memory-efficient training."""
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        if mixed_precision:
            from torch.distributed.fsdp import MixedPrecision
            import torch
            policy = MixedPrecision(param_dtype=torch.float16)
            return FSDP(model, mixed_precision=policy)
        return FSDP(model)
    except ImportError:
        raise ImportError("torch>=1.12 required for FSDP")


class GradientAccumulator:
    """Gradient accumulation for effective large batch training on small GPUs.

    Example::

        acc = GradientAccumulator(steps=4)
        for batch in dataloader:
            loss = model(batch)
            if acc.step(loss, optimizer):
                scheduler.step()
    """

    def __init__(self, steps: int = 4) -> None:
        self.steps = steps
        self._count = 0

    def step(self, loss: Any, optimizer: Any, scaler: Any = None) -> bool:
        """Accumulate gradient. Returns True when optimizer.step() was called."""
        import torch
        scaled_loss = loss / self.steps
        if scaler:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        self._count += 1
        if self._count >= self.steps:
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for pg in optimizer.param_groups for p in pg["params"]], 1.0
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    [p for pg in optimizer.param_groups for p in pg["params"]], 1.0
                )
                optimizer.step()
            optimizer.zero_grad()
            self._count = 0
            return True
        return False


def auto_device_count() -> int:
    """Return the number of available CUDA devices."""
    try:
        import torch
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except ImportError:
        return 0


def launch_ddp(train_fn: Any, world_size: Optional[int] = None, **kwargs) -> None:
    """Launch DDP training using torch.multiprocessing.spawn."""
    try:
        import torch
        import torch.multiprocessing as mp
        n_gpus = world_size or torch.cuda.device_count()
        if n_gpus <= 1:
            train_fn(rank=0, world_size=1, **kwargs)
        else:
            mp.spawn(train_fn, args=(n_gpus,) + tuple(kwargs.values()), nprocs=n_gpus)
    except ImportError:
        raise ImportError("torch required")
