"""
Distributed training utilities for PyGeoVision.

Provides helper functions for setting up PyTorch DistributedDataParallel (DDP)
training across multiple GPUs or nodes.

Example:
    >>> # In a launch script (e.g. via torchrun):
    >>> from pygeovision.ai.training.distributed import setup_ddp, cleanup_ddp, wrap_ddp
    >>> setup_ddp()
    >>> model = wrap_ddp(model)
    >>> # ... training ...
    >>> cleanup_ddp()
"""

from __future__ import annotations

import logging
import os
from typing import Optional

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
if torch is not None:
    import torch.nn as nn
else:
    nn = None

logger = logging.getLogger(__name__)


def is_distributed() -> bool:
    """Return True if running inside a distributed process group."""
    if torch is None:
        return False
    return (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
    )


def get_rank() -> int:
    """Return the global rank of this process (0 if not distributed)."""
    if is_distributed():
        return torch.distributed.get_rank()
    return 0


def get_world_size() -> int:
    """Return the total number of distributed processes (1 if not distributed)."""
    if is_distributed():
        return torch.distributed.get_world_size()
    return 1


def is_main_process() -> bool:
    """Return True if this is the main (rank 0) process."""
    return get_rank() == 0


def setup_ddp(
    backend: str = "nccl",
    init_method: str = "env://",
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
) -> None:
    """Initialize the distributed process group.

    Call this at the start of each worker process when using DDP.
    Designed to work with ``torch.distributed.launch`` and ``torchrun``.

    Args:
        backend: Communication backend ('nccl' for GPU, 'gloo' for CPU).
        init_method: Process group initialization method.
        rank: Global rank of this process (auto-detected from env if None).
        world_size: Total number of processes (auto-detected if None).

    Example:
        >>> # torchrun --nproc_per_node=4 train_script.py
        >>> setup_ddp()  # reads RANK, WORLD_SIZE, LOCAL_RANK from env
    """
    if torch.distributed.is_initialized():
        logger.warning("Distributed process group already initialized; skipping.")
        return

    rank = rank if rank is not None else int(os.environ.get("RANK", 0))
    world_size = world_size if world_size is not None else int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size == 1:
        logger.info("Single process — DDP not initialized.")
        return

    torch.distributed.init_process_group(
        backend=backend,
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    torch.cuda.set_device(local_rank)
    logger.info(
        "DDP initialized: rank=%d / world_size=%d (local_rank=%d, backend=%s)",
        rank, world_size, local_rank, backend,
    )


def cleanup_ddp() -> None:
    """Destroy the distributed process group.

    Call this at the end of training when using DDP.
    """
    if is_distributed():
        torch.distributed.destroy_process_group()
        logger.info("DDP process group destroyed.")


def wrap_ddp(
    model: nn.Module,
    device_ids: Optional[list] = None,
    find_unused_parameters: bool = False,
    sync_batchnorm: bool = True,
) -> nn.Module:
    """Wrap a model with DistributedDataParallel if distributed is active.

    Also optionally converts BatchNorm layers to SyncBatchNorm for
    correct statistics across GPUs.

    Args:
        model: PyTorch model to wrap.
        device_ids: List of CUDA device IDs for this process. Defaults to
            [LOCAL_RANK].
        find_unused_parameters: Pass through to DDP (needed for models
            with optional graph branches).
        sync_batchnorm: Convert BatchNorm to SyncBatchNorm.

    Returns:
        DDP-wrapped model (or original model if not distributed).

    Example:
        >>> model = wrap_ddp(model, sync_batchnorm=True)
    """
    if not is_distributed():
        return model

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device_ids = device_ids or [local_rank]

    if sync_batchnorm and torch.cuda.is_available():
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logger.info("Converted BatchNorm to SyncBatchNorm.")

    model = nn.parallel.DistributedDataParallel(
        model.cuda(local_rank),
        device_ids=device_ids,
        output_device=local_rank,
        find_unused_parameters=find_unused_parameters,
    )
    logger.info("Model wrapped with DDP (device_ids=%s).", device_ids)
    return model


def barrier() -> None:
    """Synchronize all processes at a barrier (no-op if not distributed)."""
    if is_distributed():
        torch.distributed.barrier()


def reduce_tensor(tensor: torch.Tensor, average: bool = True) -> torch.Tensor:
    """All-reduce a tensor across all distributed processes.

    Args:
        tensor: Tensor to reduce.
        average: If True, divide by world_size after reducing.

    Returns:
        Reduced tensor.
    """
    if not is_distributed():
        return tensor
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    if average:
        rt /= get_world_size()
    return rt
