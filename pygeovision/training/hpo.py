"""
Hyperparameter Optimisation — Optuna + Ray Tune integration (Phase 4.1).
"""
from __future__ import annotations
import logging
from typing import Any, Callable, Dict, List, Optional
logger = logging.getLogger(__name__)


class OptunaHPO:
    """Optuna-based hyperparameter optimisation for GeoTrainer.

    Example::

        hpo = OptunaHPO(n_trials=50, direction="maximize")
        best = hpo.run(
            objective_fn=lambda cfg: trainer.fit(train_ds, val_ds, config=cfg)["best_val_iou"],
            search_space={
                "learning_rate": ("float_log", 1e-5, 1e-2),
                "batch_size":    ("categorical", [8, 16, 32]),
                "optimizer":     ("categorical", ["adamw", "sgd"]),
                "weight_decay":  ("float_log", 1e-5, 1e-1),
            }
        )
        print("Best params:", best)
    """

    def __init__(
        self,
        n_trials: int = 50,
        direction: str = "maximize",
        study_name: str = "pgv_hpo",
        storage: Optional[str] = None,
        n_jobs: int = 1,
        timeout: Optional[float] = None,
        pruner: str = "median",          # median | hyperband | none
    ) -> None:
        self.n_trials = n_trials
        self.direction = direction
        self.study_name = study_name
        self.storage = storage
        self.n_jobs = n_jobs
        self.timeout = timeout
        self.pruner = pruner
        self._study: Any = None

    def _make_pruner(self) -> Any:
        try:
            import optuna
            if self.pruner == "median":
                return optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
            elif self.pruner == "hyperband":
                return optuna.pruners.HyperbandPruner()
            return optuna.pruners.NopPruner()
        except ImportError:
            return None

    def run(
        self,
        objective_fn: Callable[[Dict], float],
        search_space: Dict[str, tuple],
    ) -> Dict[str, Any]:
        """Run HPO and return the best hyperparameters found.

        search_space format:
            "param_name": ("type", *args)
            Types: float, float_log, int, int_log, categorical
            Example: "lr": ("float_log", 1e-5, 1e-2)
        """
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.error("Optuna not installed: pip install optuna")
            return {}

        pruner = self._make_pruner()
        self._study = optuna.create_study(
            study_name=self.study_name,
            direction=self.direction,
            storage=self.storage,
            pruner=pruner,
            load_if_exists=bool(self.storage),
        )

        def _wrapped_objective(trial: Any) -> float:
            params: Dict[str, Any] = {}
            for name, spec in search_space.items():
                kind = spec[0]
                if kind == "float":
                    params[name] = trial.suggest_float(name, spec[1], spec[2])
                elif kind == "float_log":
                    params[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
                elif kind == "int":
                    params[name] = trial.suggest_int(name, spec[1], spec[2])
                elif kind == "int_log":
                    params[name] = trial.suggest_int(name, spec[1], spec[2], log=True)
                elif kind == "categorical":
                    params[name] = trial.suggest_categorical(name, list(spec[1]))
                else:
                    params[name] = spec[1]
            return objective_fn(params)

        self._study.optimize(
            _wrapped_objective,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            timeout=self.timeout,
            show_progress_bar=True,
        )

        best = self._study.best_trial
        logger.info(
            "HPO complete: best value=%.4f | params=%s",
            best.value, best.params,
        )
        return {
            "best_params":  best.params,
            "best_value":   best.value,
            "n_trials":     len(self._study.trials),
            "study_name":   self.study_name,
        }

    @property
    def study(self) -> Any:
        return self._study

    def importance(self) -> Dict[str, float]:
        """Return hyperparameter importance scores."""
        try:
            import optuna
            if self._study is None:
                return {}
            return optuna.importance.get_param_importances(self._study)
        except Exception:
            return {}

    def plot_history(self, save_path: Optional[str] = None) -> None:
        """Plot optimisation history."""
        try:
            import optuna.visualization as vis
            if self._study is None:
                return
            fig = vis.plot_optimization_history(self._study)
            if save_path:
                fig.write_image(save_path)
                logger.info("HPO history saved → %s", save_path)
        except ImportError:
            logger.warning("plotly required for HPO plots")


class ModelOptimizer:
    """Model compression and optimisation utilities (Phase 4.2).

    Supports ONNX export, TensorRT, INT8 quantisation, knowledge distillation,
    and structured pruning.
    """

    def __init__(self, model: Any, num_classes: int = 2, in_channels: int = 3) -> None:
        self.model = model
        self.num_classes = num_classes
        self.in_channels = in_channels

    def export_onnx(
        self,
        output_path: str = "model.onnx",
        input_size: int = 512,
        opset: int = 17,
        dynamic_batch: bool = True,
        simplify: bool = True,
    ) -> str:
        """Export model to ONNX format."""
        try:
            import torch
            self.model.eval()
            dummy = torch.randn(1, self.in_channels, input_size, input_size)
            dynamic_axes: Dict[str, Any] = {}
            if dynamic_batch:
                dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}
            torch.onnx.export(
                self.model, dummy, output_path,
                opset_version=opset,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes=dynamic_axes,
                do_constant_folding=True,
            )
            # Optionally simplify
            if simplify:
                try:
                    import onnxsim, onnx
                    model_onnx = onnx.load(output_path)
                    simplified, check = onnxsim.simplify(model_onnx)
                    if check:
                        onnx.save(simplified, output_path)
                        logger.info("ONNX model simplified")
                except ImportError:
                    pass
            logger.info("ONNX exported → %s  (opset=%d)", output_path, opset)
            return output_path
        except ImportError as exc:
            raise ImportError(f"torch required: {exc}")

    def export_torchscript(
        self,
        output_path: str = "model.pt",
        method: str = "trace",
        input_size: int = 512,
    ) -> str:
        """Export to TorchScript (trace or script)."""
        try:
            import torch
            self.model.eval()
            if method == "trace":
                dummy = torch.randn(1, self.in_channels, input_size, input_size)
                scripted = torch.jit.trace(self.model, dummy)
            else:
                scripted = torch.jit.script(self.model)
            scripted.save(output_path)
            logger.info("TorchScript exported → %s  (method=%s)", output_path, method)
            return output_path
        except Exception as exc:
            raise RuntimeError(f"TorchScript export failed: {exc}")

    def quantize_int8(
        self,
        calibration_loader: Any,
        output_path: str = "model_int8.pth",
        backend: str = "fbgemm",   # fbgemm (x86) | qnnpack (ARM/mobile)
    ) -> Any:
        """Apply post-training static INT8 quantisation."""
        try:
            import torch
            import torch.quantization as tq
            torch.backends.quantized.engine = backend
            self.model.eval()
            # Fuse layers where possible
            try:
                self.model = tq.fuse_modules(
                    self.model,
                    [["conv", "bn", "relu"]],
                    inplace=True,
                )
            except Exception:
                pass
            qconfig = tq.get_default_qconfig(backend)
            self.model.qconfig = qconfig
            tq.prepare(self.model, inplace=True)
            # Calibrate
            with torch.no_grad():
                for batch in calibration_loader:
                    if isinstance(batch, (list, tuple)):
                        images = batch[0]
                    elif isinstance(batch, dict):
                        images = batch.get("image", batch.get("images"))
                    else:
                        break
                    self.model(images)
            tq.convert(self.model, inplace=True)
            torch.save(self.model.state_dict(), output_path)
            logger.info("INT8 quantised model → %s  (backend=%s)", output_path, backend)
            return self.model
        except ImportError as exc:
            raise ImportError(f"torch required for quantisation: {exc}")

    def prune(
        self,
        amount: float = 0.3,
        method: str = "l1",      # l1 | random | global
        structured: bool = False,
    ) -> Any:
        """Apply magnitude-based weight pruning."""
        try:
            import torch.nn.utils.prune as prune
            modules = [
                (m, "weight")
                for m in self.model.modules()
                if hasattr(m, "weight") and m.weight is not None
            ]
            if not modules:
                logger.warning("No prunable modules found")
                return self.model
            if method == "global":
                prune.global_unstructured(
                    modules,
                    pruning_method=prune.L1Unstructured,
                    amount=amount,
                )
            else:
                for module, name in modules:
                    if structured:
                        prune.ln_structured(module, name=name, amount=amount, n=2, dim=0)
                    else:
                        prune.l1_unstructured(module, name=name, amount=amount)
            # Count sparsity
            n_zeros = sum(
                (p == 0).sum().item()
                for p in self.model.parameters()
                if p is not None
            )
            n_total = sum(p.numel() for p in self.model.parameters())
            logger.info(
                "Pruning done: sparsity=%.1f%%  (%d/%d zeros)",
                100 * n_zeros / max(n_total, 1), n_zeros, n_total,
            )
            return self.model
        except ImportError as exc:
            raise ImportError(f"torch required for pruning: {exc}")

    def onnx_inference(
        self,
        input_array: Any,
        onnx_path: str = "model.onnx",
    ) -> Any:
        """Run inference with an ONNX Runtime session."""
        try:
            import onnxruntime as ort
            import numpy as np
            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            sess = ort.InferenceSession(onnx_path, sess_opts=sess_opts, providers=providers)
            inp_name = sess.get_inputs()[0].name
            if hasattr(input_array, "numpy"):
                input_array = input_array.numpy()
            if input_array.ndim == 3:
                input_array = input_array[None]
            return sess.run(None, {inp_name: input_array.astype(np.float32)})[0]
        except ImportError:
            raise ImportError("onnxruntime required: pip install onnxruntime-gpu")

    def benchmark_speed(
        self,
        input_size: int = 512,
        n_runs: int = 100,
        warmup: int = 10,
        device: str = "cuda",
    ) -> Dict[str, float]:
        """Benchmark model inference speed (ms/image, FPS)."""
        try:
            import torch, time
            dev = torch.device(device if torch.cuda.is_available() else "cpu")
            self.model.to(dev).eval()
            dummy = torch.randn(1, self.in_channels, input_size, input_size, device=dev)
            with torch.no_grad():
                for _ in range(warmup):
                    self.model(dummy)
            if str(dev) != "cpu":
                torch.cuda.synchronize()
            times = []
            with torch.no_grad():
                for _ in range(n_runs):
                    t0 = time.perf_counter()
                    self.model(dummy)
                    if str(dev) != "cpu":
                        torch.cuda.synchronize()
                    times.append((time.perf_counter() - t0) * 1000)
            import statistics
            mean_ms = statistics.mean(times)
            std_ms  = statistics.stdev(times) if len(times) > 1 else 0.0
            return {
                "mean_ms":   round(mean_ms, 2),
                "std_ms":    round(std_ms, 2),
                "fps":       round(1000 / mean_ms, 1),
                "p95_ms":    round(sorted(times)[int(0.95 * len(times))], 2),
                "device":    str(dev),
                "input_size": input_size,
                "n_runs":    n_runs,
            }
        except ImportError as exc:
            return {"error": str(exc)}
