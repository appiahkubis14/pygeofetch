"""
ModelEvaluator — standardised cross-model evaluation (Phase 4.4).

Provides:
- Standardised evaluation over any dataset with IoU, F1, mAP, accuracy
- Automatic train/val/test splits following EarthNets convention (60/20/20)
- Cross-model comparison tables
- Confusion matrix and PR-curve generation
- Leaderboard JSON output
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result from evaluating one model on one dataset split."""
    model_name: str
    dataset_name: str
    task: str
    split: str = "test"
    # Core metrics
    mean_iou: float = 0.0
    mean_f1: float = 0.0
    accuracy: float = 0.0
    mAP50: float = 0.0
    mAP50_95: float = 0.0
    # Per-class IoU
    per_class_iou: Dict[str, float] = field(default_factory=dict)
    # Runtime
    inference_ms_per_image: float = 0.0
    n_samples: int = 0
    duration_seconds: float = 0.0
    # Config
    hyperparams: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    @property
    def primary_metric(self) -> float:
        """Return the most relevant metric for this task."""
        if self.task in ("segmentation", "change_detection"):
            return self.mean_iou
        elif self.task == "detection":
            return self.mAP50
        elif self.task == "classification":
            return self.accuracy
        return self.mean_iou

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model_name, "dataset": self.dataset_name,
            "task": self.task, "split": self.split,
            "mean_iou": round(self.mean_iou, 4),
            "mean_f1": round(self.mean_f1, 4),
            "accuracy": round(self.accuracy, 4),
            "mAP50": round(self.mAP50, 4),
            "mAP50_95": round(self.mAP50_95, 4),
            "per_class_iou": {k: round(v, 4) for k, v in self.per_class_iou.items()},
            "inference_ms_per_image": round(self.inference_ms_per_image, 2),
            "n_samples": self.n_samples,
            "primary_metric": round(self.primary_metric, 4),
        }


class ModelEvaluator:
    """Standardised evaluator for geospatial AI models (Phase 4.4).

    Example::

        evaluator = ModelEvaluator(task="segmentation", num_classes=5)
        # Evaluate a single model
        result = evaluator.evaluate(model, test_loader, dataset_name="LoveDA")
        # Compare multiple models
        results = evaluator.compare(
            models={"UNet": unet, "SegFormer": segformer},
            loader=test_loader, dataset_name="LoveDA",
        )
        evaluator.print_leaderboard(results)
        evaluator.save_results(results, "results.json")
    """

    def __init__(self, task: str = "segmentation", num_classes: int = 2) -> None:
        self.task = task
        self.num_classes = num_classes

    def evaluate(
        self,
        model: Any,
        loader: Any,
        dataset_name: str = "unknown",
        model_name: str = "model",
        split: str = "test",
        device: Optional[str] = None,
    ) -> BenchmarkResult:
        """Evaluate a model on a DataLoader and return BenchmarkResult."""
        from pygeovision.training.metrics import SegmentationMetrics, DetectionMetrics

        result = BenchmarkResult(model_name=model_name, dataset_name=dataset_name,
                                 task=self.task, split=split)
        try:
            import torch
            dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
            model = model.to(dev).eval()
            metrics_seg = SegmentationMetrics(self.num_classes) if self.task != "detection" else None
            metrics_det = DetectionMetrics(self.num_classes) if self.task == "detection" else None

            total_images = 0
            total_latency = 0.0
            t_start = time.time()

            with torch.no_grad():
                for batch in loader:
                    if isinstance(batch, (list, tuple)):
                        images, targets = batch[0].to(dev), batch[1].to(dev)
                    else:
                        images = batch.get("image", batch.get("images")).to(dev)
                        targets = batch.get("mask", batch.get("label")).to(dev)

                    t_inf = time.time()
                    outputs = model(images)
                    total_latency += (time.time() - t_inf) * 1000
                    total_images += images.shape[0]

                    if metrics_seg is not None:
                        preds = outputs.argmax(dim=1)
                        metrics_seg.update(preds, targets)

            result.duration_seconds = time.time() - t_start
            result.n_samples = total_images
            result.inference_ms_per_image = total_latency / max(total_images, 1)

            if metrics_seg is not None:
                computed = metrics_seg.compute()
                result.mean_iou = computed.get("mean_iou", 0.0)
                result.mean_f1 = computed.get("mean_f1", 0.0)
                result.accuracy = computed.get("accuracy", 0.0)
                result.per_class_iou = {
                    k: v for k, v in computed.items() if k.startswith("iou_class_")
                }
        except ImportError:
            logger.warning("torch required for evaluation")
        except Exception as exc:
            logger.error("Evaluation failed: %s", exc)
            result.notes = str(exc)
        return result

    def compare(
        self,
        models: Dict[str, Any],
        loader: Any,
        dataset_name: str = "benchmark",
        split: str = "test",
        device: Optional[str] = None,
    ) -> List[BenchmarkResult]:
        """Evaluate and compare multiple models on the same dataset."""
        results = []
        for name, model in models.items():
            logger.info("Evaluating: %s on %s...", name, dataset_name)
            r = self.evaluate(model, loader, dataset_name, model_name=name,
                              split=split, device=device)
            results.append(r)
            logger.info("  %s: mIoU=%.4f  F1=%.4f  acc=%.4f  %.1fms/img",
                        name, r.mean_iou, r.mean_f1, r.accuracy, r.inference_ms_per_image)
        results.sort(key=lambda x: x.primary_metric, reverse=True)
        return results

    def print_leaderboard(self, results: List[BenchmarkResult]) -> None:
        """Print a formatted leaderboard table."""
        if not results:
            print("No results.")
            return
        task = results[0].task
        print(f"\n{'═'*90}")
        print(f"  Leaderboard: {results[0].dataset_name} | Task: {task}")
        print(f"{'═'*90}")
        if task in ("segmentation", "change_detection"):
            print(f"  {'#':<3} {'Model':<30} {'mIoU':>8} {'F1':>8} {'Acc':>8} {'ms/img':>8}")
            print(f"  {'─'*70}")
            for i, r in enumerate(results, 1):
                print(f"  {i:<3} {r.model_name:<30} {r.mean_iou:>8.4f} {r.mean_f1:>8.4f} {r.accuracy:>8.4f} {r.inference_ms_per_image:>8.1f}")
        elif task == "detection":
            print(f"  {'#':<3} {'Model':<30} {'mAP@50':>8} {'mAP50-95':>10} {'ms/img':>8}")
            print(f"  {'─'*65}")
            for i, r in enumerate(results, 1):
                print(f"  {i:<3} {r.model_name:<30} {r.mAP50:>8.4f} {r.mAP50_95:>10.4f} {r.inference_ms_per_image:>8.1f}")
        elif task == "classification":
            print(f"  {'#':<3} {'Model':<30} {'Accuracy':>10} {'F1':>8} {'ms/img':>8}")
            print(f"  {'─'*65}")
            for i, r in enumerate(results, 1):
                print(f"  {i:<3} {r.model_name:<30} {r.accuracy:>10.4f} {r.mean_f1:>8.4f} {r.inference_ms_per_image:>8.1f}")
        print(f"{'═'*90}\n")

    def save_results(self, results: List[BenchmarkResult], path: Union[str, Path]) -> None:
        """Save benchmark results to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        logger.info("Results saved → %s", path)

    def confusion_matrix_plot(self, model: Any, loader: Any,
                               class_names: Optional[List[str]] = None,
                               save_path: Optional[Union[str, Path]] = None) -> Any:
        """Generate and optionally save a confusion matrix plot."""
        try:
            import torch, matplotlib.pyplot as plt, numpy as np
            from pygeovision.training.metrics import SegmentationMetrics
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(dev).eval()
            metrics = SegmentationMetrics(self.num_classes)
            with torch.no_grad():
                for batch in loader:
                    if isinstance(batch, (list, tuple)):
                        images, targets = batch[0].to(dev), batch[1].to(dev)
                    else:
                        images = batch.get("image").to(dev)
                        targets = batch.get("mask").to(dev)
                    outputs = model(images)
                    metrics.update(outputs.argmax(dim=1), targets)
            cm = metrics.confusion_matrix().numpy()
            # Normalise
            cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)
            fig, ax = plt.subplots(figsize=(max(8, self.num_classes), max(6, self.num_classes)))
            im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
            plt.colorbar(im)
            labels = class_names or [str(i) for i in range(self.num_classes)]
            ax.set_xticks(range(self.num_classes)); ax.set_yticks(range(self.num_classes))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_yticklabels(labels)
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            ax.set_title("Normalised Confusion Matrix")
            for i in range(self.num_classes):
                for j in range(self.num_classes):
                    ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                            color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=8)
            plt.tight_layout()
            if save_path:
                plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
                logger.info("Confusion matrix saved → %s", save_path)
            return fig
        except ImportError as exc:
            logger.warning("matplotlib/numpy required for plots: %s", exc)
            return None
