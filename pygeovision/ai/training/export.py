"""
Model export utilities for PyGeoVision.

Export trained PyTorch models to ONNX and TorchScript for production
deployment, edge inference, or interoperability with other frameworks.

Example:
    >>> from pygeovision.ai.training.export import ModelExporter
    >>> exporter = ModelExporter(model, input_shape=(1, 3, 512, 512))
    >>> exporter.to_onnx("model.onnx")
    >>> exporter.to_torchscript("model.pt")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelExporter:
    """Export a PyGeoVision model to ONNX or TorchScript format.

    Args:
        model: Trained PyTorch model (should be in eval mode).
        input_shape: Expected input tensor shape as (B, C, H, W).
        device: Device for tracing/scripting.
        opset_version: ONNX opset version (default: 17).

    Example:
        >>> model.eval()
        >>> exporter = ModelExporter(model, input_shape=(1, 4, 512, 512))
        >>> exporter.to_onnx("unet.onnx", dynamic_axes={"input": [0, 2, 3]})
        >>> exporter.to_torchscript("unet.pt", method="trace")
    """

    def __init__(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...] = (1, 3, 512, 512),
        device: str = "cpu",
        opset_version: int = 17,
    ) -> None:
        self.model = model.to(device).eval()
        self.input_shape = input_shape
        self.device = device
        self.opset_version = opset_version

    def to_onnx(
        self,
        output_path: str | Path,
        input_names: Optional[Sequence[str]] = None,
        output_names: Optional[Sequence[str]] = None,
        dynamic_axes: Optional[dict] = None,
        simplify: bool = True,
    ) -> Path:
        """Export model to ONNX format.

        Args:
            output_path: Destination .onnx file path.
            input_names: Names for input nodes (default: ['input']).
            output_names: Names for output nodes (default: ['output']).
            dynamic_axes: Dict mapping node names to dynamic axis indices.
                Example: {'input': {0: 'batch', 2: 'height', 3: 'width'}}
            simplify: Run onnx-simplifier if available.

        Returns:
            Path to the saved ONNX file.

        Example:
            >>> exporter.to_onnx("model.onnx", dynamic_axes={"input": [0]})
        """
        try:
            import onnx
        except ImportError as exc:
            raise ImportError(
                "ONNX export requires onnx. Install: pip install onnx"
            ) from exc

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        dummy = torch.zeros(self.input_shape, device=self.device)
        input_names = list(input_names or ["input"])
        output_names = list(output_names or ["output"])

        # Build dynamic_axes if provided as list shorthand
        if isinstance(dynamic_axes, dict):
            dyn = {}
            for name, axes in dynamic_axes.items():
                if isinstance(axes, list):
                    dyn[name] = {ax: f"dim_{ax}" for ax in axes}
                else:
                    dyn[name] = axes
        else:
            dyn = None

        logger.info("Exporting to ONNX (opset=%d)…", self.opset_version)
        with torch.no_grad():
            torch.onnx.export(
                self.model,
                dummy,
                str(output_path),
                opset_version=self.opset_version,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dyn,
                do_constant_folding=True,
            )

        # Verify
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model verified OK.")

        if simplify:
            try:
                from onnxsim import simplify as onnx_simplify
                simplified, ok = onnx_simplify(onnx_model)
                if ok:
                    onnx.save(simplified, str(output_path))
                    logger.info("ONNX model simplified.")
            except ImportError:
                logger.debug("onnx-simplifier not installed; skipping simplification.")

        size_mb = output_path.stat().st_size / 1e6
        logger.info("ONNX export complete → %s (%.1f MB)", output_path, size_mb)
        return output_path

    def to_torchscript(
        self,
        output_path: str | Path,
        method: str = "trace",
        strict: bool = True,
    ) -> Path:
        """Export model to TorchScript format.

        Args:
            output_path: Destination .pt file path.
            method: 'trace' (fast, concrete input) or 'script' (full graph).
            strict: Enforce strict tracing mode.

        Returns:
            Path to the saved TorchScript file.

        Example:
            >>> exporter.to_torchscript("model.pt", method="script")
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Exporting to TorchScript (method=%s)…", method)
        with torch.no_grad():
            if method == "trace":
                dummy = torch.zeros(self.input_shape, device=self.device)
                scripted = torch.jit.trace(self.model, dummy, strict=strict)
            elif method == "script":
                scripted = torch.jit.script(self.model)
            else:
                raise ValueError(f"method must be 'trace' or 'script', got {method!r}")

        torch.jit.save(scripted, str(output_path))
        size_mb = output_path.stat().st_size / 1e6
        logger.info("TorchScript export complete → %s (%.1f MB)", output_path, size_mb)
        return output_path

    def benchmark(
        self,
        num_warmup: int = 10,
        num_runs: int = 100,
    ) -> dict:
        """Benchmark model inference latency and throughput.

        Args:
            num_warmup: Warmup iterations (not counted).
            num_runs: Measurement iterations.

        Returns:
            Dict with mean_ms, std_ms, fps, and input_shape.
        """
        import time
        import statistics

        dummy = torch.zeros(self.input_shape, device=self.device)

        with torch.no_grad():
            for _ in range(num_warmup):
                self.model(dummy)

            times = []
            for _ in range(num_runs):
                if self.device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                self.model(dummy)
                if self.device == "cuda":
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000)

        mean_ms = statistics.mean(times)
        std_ms = statistics.stdev(times) if len(times) > 1 else 0.0
        fps = 1000.0 / mean_ms * self.input_shape[0]

        result = {
            "mean_ms": round(mean_ms, 3),
            "std_ms": round(std_ms, 3),
            "fps": round(fps, 1),
            "input_shape": self.input_shape,
            "device": self.device,
        }
        logger.info(
            "Benchmark: %.2f ms ± %.2f ms | %.1f FPS | device=%s",
            mean_ms, std_ms, fps, self.device,
        )
        return result


def quantize_model(
    model: nn.Module,
    quantization: str = "dynamic",
    dtype: torch.dtype = torch.qint8,
) -> nn.Module:
    """Apply post-training quantization to reduce model size.

    Args:
        model: Trained model to quantize.
        quantization: 'dynamic' or 'static'.
        dtype: Quantized weight dtype.

    Returns:
        Quantized model.

    Example:
        >>> small_model = quantize_model(model, quantization="dynamic")
    """
    model = model.cpu().eval()

    if quantization == "dynamic":
        quantized = torch.quantization.quantize_dynamic(
            model,
            qconfig_spec={nn.Linear, nn.Conv2d},
            dtype=dtype,
        )
        logger.info("Applied dynamic quantization (%s).", dtype)
        return quantized

    if quantization == "static":
        model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
        torch.quantization.prepare(model, inplace=True)
        logger.info(
            "Model prepared for static quantization. "
            "Run calibration data through the model, then call "
            "torch.quantization.convert(model, inplace=True)."
        )
        return model

    raise ValueError(f"quantization must be 'dynamic' or 'static', got {quantization!r}")
