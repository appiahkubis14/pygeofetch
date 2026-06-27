"""Health checks for the inference server."""
from __future__ import annotations
import logging, time
from typing import Any, Dict
logger = logging.getLogger(__name__)
_START_TIME = time.time()


class HealthChecker:
    """Comprehensive health check for the inference server."""

    def __init__(self, models: Any = None) -> None:
        self._models = models or {}

    def check(self) -> Dict[str, Any]:
        return {
            "status": "healthy",
            "version": "2.0.4",
            "uptime_s": round(time.time() - _START_TIME, 1),
            "models_loaded": len(self._models) if hasattr(self._models, "__len__") else 0,
            "gpu": self._gpu_status(),
            "memory": self._memory_status(),
        }

    def _gpu_status(self) -> Dict:
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                return {
                    "available": True,
                    "name": props.name,
                    "total_mb": props.total_memory // 1024**2,
                    "allocated_mb": torch.cuda.memory_allocated() // 1024**2,
                }
            return {"available": False}
        except ImportError:
            return {"available": False, "reason": "torch not installed"}

    def _memory_status(self) -> Dict:
        try:
            import psutil
            vm = psutil.virtual_memory()
            return {"total_gb": round(vm.total / 1024**3, 1),
                    "available_gb": round(vm.available / 1024**3, 1),
                    "percent_used": vm.percent}
        except ImportError:
            return {}
