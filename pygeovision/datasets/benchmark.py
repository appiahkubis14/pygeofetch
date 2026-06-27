"""
EarthNets Benchmark Builder (Phase 5.2).

Selects the canonical top-5 datasets per task, generates unified benchmark
configs with standard 60/20/20 splits, and runs cross-task / cross-domain
evaluation using the EarthNets ranking formula.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# EarthNets canonical tasks and their short names
BENCHMARK_TASKS = {
    "segmentation":     "Semantic segmentation of land cover / objects",
    "detection":        "Object detection with bounding boxes",
    "classification":   "Scene / patch-level classification",
    "change_detection": "Bi-temporal change detection",
    "multi_label":      "Multi-label scene classification",
    "regression":       "Pixel or scene-level regression (height, biomass…)",
    "prediction":       "Temporal sequence prediction",
    "self_supervised":  "Self-supervised pre-training datasets",
    "vqa":              "Visual question answering",
}

# Standard EarthNets split ratios
SPLIT_RATIOS = {"train": 0.60, "val": 0.20, "test": 0.20}


@dataclass
class BenchmarkConfig:
    """Configuration for a standardised EarthNets benchmark."""
    task: str
    dataset_names: List[str]
    split_ratios: Dict[str, float] = field(default_factory=lambda: dict(SPLIT_RATIOS))
    seed: int = 42
    metric: str = ""              # primary metric (auto-assigned by task)
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.metric:
            _metric_map = {
                "segmentation":     "mean_iou",
                "detection":        "mAP50",
                "classification":   "accuracy",
                "change_detection": "iou_change",
                "multi_label":      "mean_f1",
                "regression":       "rmse",
                "prediction":       "mae",
                "self_supervised":  "linear_probe_accuracy",
                "vqa":              "accuracy",
            }
            self.metric = _metric_map.get(self.task, "mean_iou")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "dataset_names": self.dataset_names,
            "split_ratios": self.split_ratios,
            "seed": self.seed,
            "primary_metric": self.metric,
            "notes": self.notes,
        }

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
        logger.info("Benchmark config saved → %s", path)


class BenchmarkBuilder:
    """Build EarthNets-style benchmark configurations (Phase 5.2).

    Selects the top-5 datasets per task using the EarthNets ranking formula
    and generates standardised benchmark configs with 60/20/20 splits.

    Example::

        builder = BenchmarkBuilder()
        # Get top-5 for segmentation
        cfg = builder.build("segmentation")
        print(cfg.dataset_names)

        # Print all task benchmarks
        builder.print_all()

        # Save configs for all tasks
        builder.save_all("./benchmarks/")

        # Cross-task evaluation matrix
        matrix = builder.cross_task_matrix()
    """

    def __init__(self, registry: Optional[Any] = None) -> None:
        if registry is None:
            from pygeovision.datasets.registry import dataset_registry
            registry = dataset_registry
        self.registry = registry

    def build(self, task: str, n: int = 5) -> BenchmarkConfig:
        """Build benchmark config for one task using EarthNets top-n selection."""
        top = self.registry.top_for_task(task, n=n)
        if not top:
            raise ValueError(f"No datasets found for task '{task}'. "
                             f"Available: {list(BENCHMARK_TASKS)}")
        cfg = BenchmarkConfig(task=task, dataset_names=[d.name for d in top])
        logger.info("Benchmark '%s': %s", task, cfg.dataset_names)
        return cfg

    def build_all(self, n: int = 5) -> Dict[str, BenchmarkConfig]:
        """Build benchmark configs for all supported tasks."""
        configs = {}
        for task in BENCHMARK_TASKS:
            try:
                configs[task] = self.build(task, n=n)
            except ValueError:
                pass  # No datasets for this task yet
        return configs

    def print_all(self, n: int = 5) -> None:
        """Print the EarthNets top-5 benchmark table for all tasks."""
        print(f"\n{'═'*75}")
        print(f"  EarthNets Benchmark Datasets (Top-{n} per Task)")
        print(f"{'═'*75}")
        for task, description in BENCHMARK_TASKS.items():
            top = self.registry.top_for_task(task, n=n)
            if not top:
                continue
            print(f"\n  {task.upper()} — {description}")
            print(f"  {'─'*65}")
            print(f"  {'#':<3} {'Dataset':<28} {'Domain':<14} {'Modality':<14} {'Year':>4} {'Samples':>10}")
            print(f"  {'─'*78}")
            for i, d in enumerate(top, 1):
                print(f"  {i:<3} {d.name:<28} {d.domain:<14} {d.modality:<14} {d.year:>4} {d.n_samples:>10,}")
        print(f"\n{'═'*75}")

    def save_all(self, output_dir: str = "./benchmarks", n: int = 5) -> List[str]:
        """Save benchmark configs for all tasks as JSON files."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        saved = []
        for task, cfg in self.build_all(n=n).items():
            path = str(out / f"benchmark_{task}.json")
            cfg.save(path)
            saved.append(path)
        logger.info("Saved %d benchmark configs → %s/", len(saved), output_dir)
        return saved

    def cross_task_matrix(self) -> Dict[str, Dict[str, List[str]]]:
        """Build a cross-task dataset matrix (which datasets appear in multiple tasks).

        Returns a dict mapping dataset_name → list of tasks it's benchmarked for.
        """
        matrix: Dict[str, List[str]] = {}
        for task in BENCHMARK_TASKS:
            top = self.registry.top_for_task(task, n=5)
            for d in top:
                matrix.setdefault(d.name, []).append(task)
        # Filter to multi-task datasets
        multi = {k: v for k, v in matrix.items() if len(v) > 1}
        logger.info("Multi-task datasets: %d", len(multi))
        return {"multi_task": multi, "all": matrix}

    def cross_domain_datasets(self) -> Dict[str, List[str]]:
        """Group benchmark datasets by domain for cross-domain evaluation."""
        by_domain: Dict[str, List[str]] = {}
        all_cfgs = self.build_all(n=5)
        for cfg in all_cfgs.values():
            for name in cfg.dataset_names:
                try:
                    d = self.registry[name]
                    by_domain.setdefault(d.domain, [])
                    if name not in by_domain[d.domain]:
                        by_domain[d.domain].append(name)
                except KeyError:
                    pass
        return by_domain

    def recommended_for_paper(self, task: str) -> Dict[str, Any]:
        """Return the full metadata for datasets recommended to benchmark in a paper."""
        cfg = self.build(task, n=5)
        datasets = []
        for name in cfg.dataset_names:
            try:
                d = self.registry[name]
                datasets.append({
                    "name": d.name,
                    "year": d.year,
                    "n_samples": d.n_samples,
                    "resolution_m": d.resolution_m,
                    "modality": d.modality,
                    "paper_url": d.paper_url,
                    "download_url": d.download_url,
                })
            except KeyError:
                pass
        return {
            "task": task,
            "primary_metric": cfg.metric,
            "split": "60 / 20 / 20 (EarthNets standard)",
            "datasets": datasets,
        }
