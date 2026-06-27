"""Model performance tracking over time."""
from __future__ import annotations
import json, logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
logger = logging.getLogger(__name__)


class ModelPerformanceTracker:
    """Track model performance over time with automatic alerting."""

    def __init__(self, model_name: str, storage_path: str = "./monitoring/tracker.json") -> None:
        self.model_name = model_name
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._records: List[Dict] = []
        if self.storage_path.exists():
            with open(self.storage_path) as f:
                self._records = json.load(f)

    def record(self, metrics: Dict[str, float], split: str = "production",
                n_samples: int = 0, notes: str = "") -> None:
        entry = {
            "model": self.model_name,
            "timestamp": datetime.utcnow().isoformat(),
            "split": split,
            "n_samples": n_samples,
            "notes": notes,
            **{k: round(v, 6) for k, v in metrics.items()},
        }
        self._records.append(entry)
        with open(self.storage_path, "w") as f:
            json.dump(self._records, f, indent=2)

    def trend(self, metric: str = "mean_iou", window: int = 10) -> Dict[str, Any]:
        vals = [r[metric] for r in self._records if metric in r]
        if not vals:
            return {"metric": metric, "n_records": 0}
        recent = vals[-window:]
        if len(vals) > 1:
            direction = "improving" if vals[-1] > vals[-2] else "declining" if vals[-1] < vals[-2] else "stable"
        else:
            direction = "stable"
        return {
            "metric": metric,
            "n_records": len(vals),
            "current": round(vals[-1], 4) if vals else None,
            "mean_recent": round(sum(recent)/len(recent), 4),
            "direction": direction,
            "trend": direction,
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
        }

    def summary(self) -> Dict[str, Any]:
        if not self._records:
            return {"n_records": 0}
        metrics = [k for k in self._records[-1] if k not in ("model","timestamp","split","n_samples","notes")]
        return {
            "model": self.model_name,
            "n_records": len(self._records),
            "first_record": self._records[0]["timestamp"],
            "latest_record": self._records[-1]["timestamp"],
            "latest_metrics": {m: self._records[-1].get(m) for m in metrics},
            "trends": {m: self.trend(m)["trend"] for m in metrics},
        }
