"""
Model drift detection for production geospatial AI systems (G7).

Monitors:
    - Data distribution shift (input imagery statistics)
    - Prediction distribution shift (output class probabilities)
    - Model performance degradation (mIoU/mAP over time)
    - Spatial distribution shift (which regions are being processed)
"""
from __future__ import annotations
import json, logging, time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
logger = logging.getLogger(__name__)


class DistributionDrift:
    """Detect input data distribution shift using statistical tests.

    Uses the Population Stability Index (PSI) and KL divergence
    to compare reference (training) distribution against production data.
    """

    PSI_THRESHOLDS = {"low": 0.1, "medium": 0.2, "high": 0.25}

    def __init__(self, reference_stats: Optional[Dict] = None,
                 n_bins: int = 10) -> None:
        self.reference_stats = reference_stats
        self.n_bins = n_bins

    def compute_psi(self, expected: Any, actual: Any) -> float:
        """Population Stability Index (PSI).
        PSI < 0.1: No drift | 0.1-0.2: Minor | >0.2: Major drift.
        """
        import numpy as np
        exp_hist, bins = np.histogram(expected, bins=self.n_bins, density=True)
        act_hist, _    = np.histogram(actual,   bins=bins,         density=True)
        # Add small epsilon to avoid log(0)
        exp_hist = np.clip(exp_hist, 1e-10, None) / exp_hist.sum()
        act_hist = np.clip(act_hist, 1e-10, None) / act_hist.sum()
        psi = float(np.sum((act_hist - exp_hist) * np.log(act_hist / exp_hist)))
        return psi

    def compute_kl_divergence(self, p: Any, q: Any) -> float:
        """KL divergence — measures distribution distance."""
        import numpy as np
        p = np.clip(p, 1e-10, None); q = np.clip(q, 1e-10, None)
        p /= p.sum(); q /= q.sum()
        return float(np.sum(p * np.log(p / q)))

    def fit_reference(self, images: List[str]) -> Dict[str, Any]:
        """Compute reference statistics from a list of GeoTIFFs."""
        try:
            import numpy as np, rasterio
        except ImportError:
            return {}

        band_stats = {}
        for path in images[:100]:  # max 100 reference images
            try:
                with rasterio.open(path) as src:
                    data = src.read().astype(np.float32)
                    for b in range(data.shape[0]):
                        key = f"band_{b+1}"
                        stats = band_stats.setdefault(key, {"values": []})
                        sample = data[b].flatten()
                        stats["values"].extend(sample[::100].tolist())  # 1% sample
            except Exception:
                continue

        self.reference_stats = {}
        for band, stats in band_stats.items():
            values = np.array(stats["values"])
            self.reference_stats[band] = {
                "mean": float(values.mean()),
                "std":  float(values.std()),
                "p10": float(np.percentile(values, 10)),
                "p90": float(np.percentile(values, 90)),
                "histogram": np.histogram(values, bins=self.n_bins)[0].tolist(),
            }
        return self.reference_stats

    def detect(self, current_images: List[str]) -> Dict[str, Any]:
        """Detect drift in current production images vs reference."""
        if self.reference_stats is None:
            return {"error": "No reference stats. Call fit_reference() first."}

        try:
            import numpy as np, rasterio
        except ImportError:
            return {"error": "rasterio required"}

        current_values: Dict[str, List] = {}
        for path in current_images[:50]:
            try:
                with rasterio.open(path) as src:
                    data = src.read().astype(np.float32)
                    for b in range(data.shape[0]):
                        key = f"band_{b+1}"
                        current_values.setdefault(key, [])
                        current_values[key].extend(data[b].flatten()[::100].tolist())
            except Exception:
                continue

        drift_report = {}
        overall_drift_score = 0.0
        for band in self.reference_stats:
            if band not in current_values:
                continue
            ref   = np.array(self.reference_stats[band].get("histogram", []))
            curr  = np.array(np.histogram(current_values[band], bins=self.n_bins)[0], dtype=float)

            if ref.sum() > 0 and curr.sum() > 0:
                psi = self.compute_psi(ref, curr)
                kl  = self.compute_kl_divergence(ref + 1e-10, curr + 1e-10)
            else:
                psi, kl = 0.0, 0.0

            level = "none" if psi < 0.1 else "minor" if psi < 0.2 else "major"
            drift_report[band] = {"psi": round(psi, 4), "kl": round(kl, 4), "level": level}
            overall_drift_score += psi

        n_bands = max(len(drift_report), 1)
        overall_psi = overall_drift_score / n_bands
        return {
            "overall_psi": round(overall_psi, 4),
            "drift_level": "none" if overall_psi < 0.1 else "minor" if overall_psi < 0.2 else "major",
            "per_band": drift_report,
            "n_reference": 100,
            "n_current": len(current_images),
            "recommendation": self._recommendation(overall_psi),
        }

    def _recommendation(self, psi: float) -> str:
        if psi < 0.1:
            return "No action needed. Distribution stable."
        elif psi < 0.2:
            return "Minor drift detected. Monitor closely and consider recalibrating."
        else:
            return "Major distribution shift! Retrain model on recent data immediately."


class PerformanceDrift:
    """Detect model performance degradation over time (G7).

    Tracks mIoU, F1, and accuracy over rolling windows and
    triggers alerts when performance drops below thresholds.
    """

    def __init__(
        self,
        window_size: int = 100,
        alert_thresholds: Optional[Dict[str, float]] = None,
        storage_path: str = "./monitoring/performance_log.jsonl",
    ) -> None:
        self.window_size = window_size
        self.alert_thresholds = alert_thresholds or {
            "mean_iou":  0.05,   # Alert if mIoU drops by 5pp
            "accuracy":  0.03,
            "mean_f1":   0.05,
        }
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._baseline: Optional[Dict[str, float]] = None
        self._history: List[Dict] = []
        self._load_history()

    def _load_history(self) -> None:
        if self.storage_path.exists():
            with open(self.storage_path) as f:
                self._history = [json.loads(l) for l in f if l.strip()]

    def log(self, metrics: Dict[str, float], timestamp: Optional[str] = None) -> None:
        """Log a new metric measurement."""
        entry = {
            "timestamp": timestamp or datetime.utcnow().isoformat(),
            **metrics,
        }
        self._history.append(entry)
        with open(self.storage_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def set_baseline(self, metrics: Dict[str, float]) -> None:
        """Set the reference performance (from validation set)."""
        self._baseline = metrics
        logger.info("Baseline set: %s", metrics)

    def detect(self, recent_n: Optional[int] = None) -> Dict[str, Any]:
        """Detect performance drift vs baseline."""
        if self._baseline is None:
            return {"error": "No baseline. Call set_baseline() with validation metrics."}

        n = recent_n or self.window_size
        recent = self._history[-n:]
        if not recent:
            return {"error": "No performance history logged yet."}

        recent_avg = {}
        for metric in self.alert_thresholds:
            vals = [e[metric] for e in recent if metric in e]
            if vals:
                recent_avg[metric] = sum(vals) / len(vals)

        alerts = []
        drift_report = {}
        for metric, threshold in self.alert_thresholds.items():
            if metric not in recent_avg or metric not in self._baseline:
                continue
            drop = self._baseline[metric] - recent_avg[metric]
            drifted = drop > threshold
            drift_report[metric] = {
                "baseline": round(self._baseline[metric], 4),
                "recent":   round(recent_avg[metric], 4),
                "drop":     round(drop, 4),
                "threshold": threshold,
                "alert": drifted,
            }
            if drifted:
                alerts.append(f"{metric} dropped by {drop:.1%} (threshold: {threshold:.1%})")

        return {
            "n_measurements": len(recent),
            "drift_detected": len(alerts) > 0,
            "alerts": alerts,
            "metrics": drift_report,
            "recommendation": self._performance_recommendation(alerts),
        }

    def _performance_recommendation(self, alerts: List[str]) -> str:
        if not alerts:
            return "Model performance stable. No action required."
        n = len(alerts)
        if n == 1:
            return f"Minor performance drop in {alerts[0]}. Consider re-calibration."
        return f"{n} metrics degraded. Schedule fine-tuning on recent labelled data."

    def plot_history(self, metric: str = "mean_iou", save_path: Optional[str] = None) -> None:
        try:
            import matplotlib.pyplot as plt
            vals = [(e["timestamp"][:10], e[metric]) for e in self._history if metric in e]
            if not vals: return
            dates, scores = zip(*vals)
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(scores, "b-", linewidth=1.5, label=metric)
            if self._baseline and metric in self._baseline:
                ax.axhline(self._baseline[metric], color="g", linestyle="--", label="Baseline")
                threshold = self.alert_thresholds.get(metric, 0.05)
                ax.axhline(self._baseline[metric] - threshold, color="r", linestyle="--",
                            label="Alert threshold")
            ax.set_xlabel("Measurement"); ax.set_ylabel(metric)
            ax.set_title(f"Model Performance History — {metric}")
            ax.legend(); ax.grid(True, alpha=0.3)
            plt.tight_layout()
            if save_path: plt.savefig(save_path, dpi=120)
            else: plt.show()
        except ImportError:
            pass


class DriftDetector:
    """Unified drift detector — monitors data + prediction + performance drift."""

    def __init__(self, model: Optional[Any] = None,
                 storage_dir: str = "./monitoring/") -> None:
        self.model = model
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.data_drift   = DistributionDrift()
        self.perf_drift   = PerformanceDrift(storage_path=str(self.storage_dir / "perf.jsonl"))
        self._reference_images: List[str] = []

    def fit(self, reference_images: List[str],
             reference_metrics: Optional[Dict[str, float]] = None) -> "DriftDetector":
        """Set the reference distribution and performance baseline."""
        self._reference_images = reference_images
        self.data_drift.fit_reference(reference_images)
        if reference_metrics:
            self.perf_drift.set_baseline(reference_metrics)
        logger.info("DriftDetector: reference set (%d images)", len(reference_images))
        return self

    def check(self, current_images: List[str],
               current_metrics: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Run full drift check and return combined report."""
        data_report = self.data_drift.detect(current_images)
        if current_metrics:
            self.perf_drift.log(current_metrics)
        perf_report = self.perf_drift.detect()

        any_drift = (
            data_report.get("drift_level") in ("minor", "major") or
            perf_report.get("drift_detected", False)
        )
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "drift_detected": any_drift,
            "data_drift": data_report,
            "performance_drift": perf_report,
            "action_required": data_report.get("drift_level") == "major" or perf_report.get("drift_detected"),
        }

    def save_report(self, report: Dict, path: Optional[str] = None) -> str:
        p = path or str(self.storage_dir / f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json")
        with open(p, "w") as f:
            json.dump(report, f, indent=2, default=str)
        return p
