"""Persistent benchmark leaderboard (Phase 4.4)."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pygeovision.benchmark.evaluator import BenchmarkResult


class Leaderboard:
    """Persistent benchmark leaderboard with JSON storage.

    Example::

        lb = Leaderboard("leaderboard.json")
        lb.add(result)
        lb.print(task="segmentation", dataset="LoveDA")
        lb.export_csv("leaderboard.csv")
    """

    def __init__(self, path: Union[str, Path] = "pgv_leaderboard.json") -> None:
        self.path = Path(path)
        self._entries: List[Dict] = []
        if self.path.exists():
            with open(self.path) as f:
                self._entries = json.load(f)

    def add(self, result: BenchmarkResult) -> None:
        entry = result.to_dict()
        # Remove existing entry for same model+dataset+split
        self._entries = [e for e in self._entries
                         if not (e["model"] == result.model_name and
                                 e["dataset"] == result.dataset_name and
                                 e.get("split") == result.split)]
        self._entries.append(entry)
        self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self._entries, f, indent=2)

    def get(self, task: Optional[str] = None, dataset: Optional[str] = None) -> List[Dict]:
        entries = self._entries
        if task:
            entries = [e for e in entries if e.get("task") == task]
        if dataset:
            entries = [e for e in entries if e.get("dataset") == dataset]
        return sorted(entries, key=lambda x: x.get("primary_metric", 0.0), reverse=True)

    def print(self, task: Optional[str] = None, dataset: Optional[str] = None) -> None:
        entries = self.get(task=task, dataset=dataset)
        if not entries:
            print("No leaderboard entries.")
            return
        title = f"Leaderboard"
        if dataset: title += f" | {dataset}"
        if task: title += f" | {task}"
        print(f"\n{'═'*80}")
        print(f"  {title}")
        print(f"{'═'*80}")
        print(f"  {'#':<3} {'Model':<30} {'Dataset':<20} {'Primary':>10} {'mIoU':>8}")
        print(f"  {'─'*75}")
        for i, e in enumerate(entries[:20], 1):
            print(f"  {i:<3} {e['model']:<30} {e['dataset']:<20} {e.get('primary_metric',0):>10.4f} {e.get('mean_iou',0):>8.4f}")
        print(f"{'═'*80}")

    def export_csv(self, path: Union[str, Path]) -> None:
        import csv
        entries = self.get()
        if not entries:
            return
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(entries[0].keys()))
            writer.writeheader()
            writer.writerows(entries)
