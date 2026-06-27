"""GPU optimisation utilities (Phase 8.1)."""
from __future__ import annotations
import logging
from typing import Any, Dict, Optional
logger = logging.getLogger(__name__)


def get_device(prefer: str = "cuda") -> Any:
    """Return the best available torch device."""
    try:
        import torch
        if prefer == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if prefer == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    except ImportError:
        return "cpu"


def gpu_info() -> Dict[str, Any]:
    """Return current GPU memory and utilisation stats."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"available": False}
        info: Dict[str, Any] = {
            "available": True,
            "n_gpus": torch.cuda.device_count(),
            "gpus": [],
        }
        for i in range(torch.cuda.device_count()):
            mem = torch.cuda.mem_get_info(i)
            info["gpus"].append({
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "free_gb":  round(mem[0] / 1e9, 2),
                "total_gb": round(mem[1] / 1e9, 2),
                "used_gb":  round((mem[1] - mem[0]) / 1e9, 2),
            })
        return info
    except ImportError:
        return {"available": False, "error": "torch not installed"}


def optimal_batch_size(
    model: Any,
    input_shape: tuple = (3, 512, 512),
    starting_batch: int = 1,
    max_batch: int = 128,
    device: Optional[Any] = None,
) -> int:
    """Binary-search for the largest batch size that fits in GPU memory."""
    try:
        import torch
        dev = device or get_device()
        model = model.to(dev).eval()
        lo, hi, best = starting_batch, max_batch, starting_batch
        while lo <= hi:
            mid = (lo + hi) // 2
            try:
                dummy = torch.randn(mid, *input_shape, device=dev)
                with torch.no_grad():
                    model(dummy)
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                best = mid
                lo = mid + 1
            except RuntimeError:  # OOM
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                hi = mid - 1
        logger.info("Optimal batch size: %d", best)
        return best
    except ImportError:
        return starting_batch


def enable_tf32() -> None:
    """Enable TF32 on Ampere+ GPUs for faster training."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            logger.info("TF32 + cuDNN benchmark enabled")
    except ImportError:
        pass


def set_memory_fraction(fraction: float = 0.9) -> None:
    """Reserve a fraction of GPU memory to avoid fragmentation."""
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                torch.cuda.set_per_process_memory_fraction(fraction, device=i)
    except (ImportError, Exception):
        pass
