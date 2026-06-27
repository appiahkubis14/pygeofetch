"""ONNX Runtime inference — cross-platform, CPU/GPU/Jetson (F2, F3)."""
from __future__ import annotations
import logging, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
logger = logging.getLogger(__name__)


class ONNXRuntimeInference:
    """High-performance ONNX Runtime inference engine for geospatial models (F2).

    Supports CPU, CUDA, TensorRT (on Jetson), and CoreML (on Apple Silicon).
    Drop-in replacement for PyTorch inference — 2-4x faster on CPU,
    comparable speed on GPU with reduced memory.

    Example::

        engine = ONNXRuntimeInference("model.onnx", device="cuda")
        pred = engine.infer(image_array)   # numpy array in, numpy array out
        result = engine.infer_geotiff("./data/scene.tif", "./output/pred.tif")
        bench = engine.benchmark(n_runs=100)
    """

    def __init__(
        self,
        onnx_path: Union[str, Path],
        device: str = "cpu",       # cpu | cuda | tensorrt | coreml
        optimization_level: str = "all",
        intra_op_threads: int = 4,
        inter_op_threads: int = 1,
    ) -> None:
        self.onnx_path = str(onnx_path)
        self.device = device
        self.optimization_level = optimization_level
        self.intra_op_threads = intra_op_threads
        self.inter_op_threads = inter_op_threads
        self._session = None

    def _load(self) -> None:
        if self._session is not None:
            return
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime required:\n"
                "  CPU:  pip install onnxruntime\n"
                "  GPU:  pip install onnxruntime-gpu"
            )

        opt_levels = {
            "none":     ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
            "basic":    ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            "all":      ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        }

        opts = ort.SessionOptions()
        opts.graph_optimization_level = opt_levels.get(self.optimization_level,
                                                        ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
        opts.intra_op_num_threads = self.intra_op_threads
        opts.inter_op_num_threads = self.inter_op_threads

        # Select execution providers by device
        if self.device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif self.device == "tensorrt":
            providers = [
                ("TensorrtExecutionProvider", {"trt_fp16_enable": True}),
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
        elif self.device == "coreml":
            providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self._session = ort.InferenceSession(self.onnx_path, sess_options=opts, providers=providers)
        self._input_name  = self._session.get_inputs()[0].name
        self._input_shape = self._session.get_inputs()[0].shape
        self._output_name = self._session.get_outputs()[0].name
        logger.info("ONNXRuntime loaded: %s | providers=%s",
                    Path(self.onnx_path).name,
                    [p if isinstance(p, str) else p[0] for p in self._session.get_providers()])

    def infer(self, input_array: Any) -> Any:
        """Run inference on a numpy array.

        Args:
            input_array: numpy array (C, H, W) or (B, C, H, W)

        Returns:
            Output numpy array from the model
        """
        import numpy as np
        self._load()
        if hasattr(input_array, "numpy"):
            arr = input_array.numpy()
        else:
            arr = np.asarray(input_array)
        if arr.ndim == 3:
            arr = arr[np.newaxis]
        arr = arr.astype(np.float32)
        outputs = self._session.run([self._output_name], {self._input_name: arr})
        return outputs[0]

    def infer_geotiff(
        self,
        image_path: Union[str, Path],
        output_path: Union[str, Path] = "./output/onnx_pred.tif",
        chip_size: int = 512,
        overlap: int = 64,
        num_classes: int = 2,
        blend_mode: str = "gaussian",
        normalise: bool = True,
    ) -> Dict[str, Any]:
        """Run tiled ONNX inference over a full GeoTIFF (B1, B4).

        Wraps TiledInference with an ONNX-backed forward pass — handles
        arbitrarily large GeoTIFFs with Gaussian tile blending.
        """
        try:
            import numpy as np, rasterio, torch, torch.nn as nn
        except ImportError as exc:
            return {"success": False, "error": f"rasterio + torch required: {exc}"}

        # Wrap ONNX session as a PyTorch-compatible callable
        session = self
        class ONNXWrapper(nn.Module):
            def forward(self, x):
                import numpy as np
                arr = x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)
                result = session.infer(arr)
                return torch.tensor(result)

        from pygeovision.inference.tiled import TiledInference
        inf = TiledInference(
            model=ONNXWrapper(),
            chip_size=chip_size,
            overlap=overlap,
            blend_mode=blend_mode,
            num_classes=num_classes,
        )
        return inf.infer(image_path, output_path, normalise=normalise)

    def benchmark(self, input_shape: Optional[tuple] = None, n_runs: int = 100) -> Dict[str, Any]:
        """Benchmark inference speed.

        Args:
            input_shape: Override input shape (default: from model)
            n_runs: Number of inference runs

        Returns:
            Dict with mean_ms, p95_ms, fps, provider
        """
        import numpy as np
        self._load()

        if input_shape is None:
            # Build shape from model input — replace dynamic dims with 1 or 512
            raw_shape = self._input_shape
            shape = tuple(d if isinstance(d, int) and d > 0 else (1 if i == 0 else 512)
                          for i, d in enumerate(raw_shape))
        else:
            shape = input_shape

        dummy = np.random.randn(*shape).astype(np.float32)
        times = []

        # Warm-up
        for _ in range(5):
            self._session.run([self._output_name], {self._input_name: dummy})

        for _ in range(n_runs):
            t0 = time.perf_counter()
            self._session.run([self._output_name], {self._input_name: dummy})
            times.append((time.perf_counter() - t0) * 1000)

        import statistics
        return {
            "mean_ms":   round(statistics.mean(times), 2),
            "median_ms": round(statistics.median(times), 2),
            "p95_ms":    round(sorted(times)[int(n_runs * 0.95)], 2),
            "fps":       round(1000 / statistics.mean(times), 1),
            "n_runs":    n_runs,
            "input_shape": shape,
            "device":    self.device,
            "providers": self._session.get_providers(),
        }

    def model_info(self) -> Dict[str, Any]:
        """Return metadata about the loaded ONNX model."""
        self._load()
        inputs  = [{"name": i.name, "shape": i.shape, "dtype": i.type}
                   for i in self._session.get_inputs()]
        outputs = [{"name": o.name, "shape": o.shape, "dtype": o.type}
                   for o in self._session.get_outputs()]
        return {
            "onnx_path": self.onnx_path,
            "inputs":    inputs,
            "outputs":   outputs,
            "providers": self._session.get_providers(),
        }

    @staticmethod
    def from_pytorch(
        model: Any,
        output_path: Union[str, Path],
        input_shape: tuple = (1, 4, 512, 512),
        opset: int = 17,
        simplify: bool = True,
        dynamic_batch: bool = True,
        # alias accepted by tests
        opset_version: int = 0,
    ) -> "ONNXRuntimeInference":
        """Export a PyTorch model to ONNX and return an ONNXRuntimeInference wrapper.

        Args:
            model: PyTorch nn.Module in eval mode
            output_path: Output .onnx file path
            input_shape: Example input shape (B, C, H, W)
            opset: ONNX opset version (17 recommended)
            simplify: Run onnxsim to optimise the graph
            dynamic_batch: Allow dynamic batch dimension

        Returns:
            Ready-to-use ONNXRuntimeInference instance
        """
        try:
            import torch
        except ImportError:
            raise ImportError("torch required for ONNX export")

        # opset_version kwarg takes precedence (passed by some callers)
        effective_opset = opset_version if opset_version else opset

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model.eval()
        dummy = torch.randn(*input_shape)
        dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}} if dynamic_batch else None

        # Use the legacy torch.onnx.export API (compatible with torch<2.6 and >=2.6)
        # torch>=2.6 moved to a new exporter that requires onnxscript; the legacy
        # path is still available via torch.onnx.export with dynamo=False.
        _export_kwargs: Dict[str, Any] = dict(
            opset_version=effective_opset,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )
        try:
            # torch>=2.6: pass dynamo=False to force legacy path (no onnxscript dep)
            torch.onnx.export(
                model, dummy, str(output_path),
                dynamo=False,
                **_export_kwargs,
            )
        except TypeError:
            # torch<2.6 doesn't accept dynamo=
            torch.onnx.export(
                model, dummy, str(output_path),
                **_export_kwargs,
            )

        logger.info("ONNX exported: %s (opset=%d)", output_path, effective_opset)

        if simplify:
            try:
                import onnx
                from onnxsim import simplify as onnxsim
                model_onnx = onnx.load(str(output_path))
                simplified, ok = onnxsim(model_onnx)
                if ok:
                    onnx.save(simplified, str(output_path))
                    logger.info("ONNX graph simplified")
            except ImportError:
                logger.debug("pip install onnxsim for graph simplification")

        return ONNXRuntimeInference(output_path)
