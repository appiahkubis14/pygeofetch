"""Mixed precision training — FP16, BF16, and automatic loss scaling."""
from __future__ import annotations
import logging
from typing import Any, Optional
logger = logging.getLogger(__name__)


class MixedPrecisionManager:
    """Manages mixed precision training context and loss scaling.

    Example::

        mp = MixedPrecisionManager(precision="fp16")
        with mp.autocast():
            loss = model(batch)
        mp.scale_and_step(loss, optimizer)
    """

    def __init__(self, precision: str = "fp16", enabled: bool = True) -> None:
        if precision not in ("fp16", "bf16", "fp32"):
            raise ValueError(f"precision must be 'fp16', 'bf16', or 'fp32', got '{precision}'")
        self.precision = precision
        self.enabled = enabled and precision != "fp32"
        self._scaler = None
        self._dtype = None
        self._setup()

    def _setup(self) -> None:
        try:
            import torch
            if self.precision == "fp16":
                self._dtype = torch.float16
                if self.enabled:
                    self._scaler = torch.cuda.amp.GradScaler()
            elif self.precision == "bf16":
                self._dtype = torch.bfloat16
                # BF16 doesn't need gradient scaling
        except ImportError:
            logger.warning("torch not available — mixed precision disabled")
            self.enabled = False

    def autocast(self):
        """Return autocast context manager."""
        try:
            import torch
            return torch.cuda.amp.autocast(
                enabled=self.enabled,
                dtype=self._dtype,
            )
        except ImportError:
            import contextlib
            return contextlib.nullcontext()

    def scale_and_step(self, loss: Any, optimizer: Any,
                        clip_grad_norm: float = 1.0) -> None:
        """Scale loss, clip gradients, and update parameters."""
        try:
            import torch
            if self._scaler and self.enabled:
                self._scaler.scale(loss).backward()
                self._scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for pg in optimizer.param_groups for p in pg["params"]],
                    clip_grad_norm,
                )
                self._scaler.step(optimizer)
                self._scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for pg in optimizer.param_groups for p in pg["params"]],
                    clip_grad_norm,
                )
                optimizer.step()
        except ImportError:
            raise ImportError("torch required")

    @property
    def scale(self) -> float:
        if self._scaler:
            return self._scaler.get_scale()
        return 1.0

    def state_dict(self) -> dict:
        if self._scaler:
            return self._scaler.state_dict()
        return {}

    def load_state_dict(self, state: dict) -> None:
        if self._scaler and state:
            self._scaler.load_state_dict(state)
