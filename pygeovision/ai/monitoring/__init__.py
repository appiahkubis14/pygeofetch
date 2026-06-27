"""
PyGeoVision model monitoring — drift detection and performance tracking.

Monitors deployed models for data drift and performance degradation over time.
Useful for satellite imagery workflows where seasonal and sensor changes
can significantly shift input distributions.

Example:
    >>> from pygeovision.ai.monitoring import DriftDetector, PerformanceTracker
    >>> detector = DriftDetector()
    >>> detector.fit_reference(reference_features)
    >>> drift_score = detector.score(new_features)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DriftReport:
    """Result of a drift detection check.

    Attributes:
        timestamp: When the check was performed.
        drift_score: Normalized drift score (0=no drift, 1=max drift).
        is_drifted: Whether drift exceeds the configured threshold.
        feature_drift: Per-feature drift scores.
        threshold: Detection threshold used.
    """
    timestamp: str
    drift_score: float
    is_drifted: bool
    feature_drift: Dict[str, float] = field(default_factory=dict)
    threshold: float = 0.1


class DriftDetector:
    """Detect distribution shift between reference and production data.

    Uses Maximum Mean Discrepancy (MMD) or simple statistical tests
    to detect when incoming satellite imagery features drift from the
    training distribution.

    Args:
        threshold: Drift score above which data is considered drifted.
        window_size: Number of recent samples to use for detection.
        method: Detection method: 'mmd' or 'ks' (Kolmogorov-Smirnov).

    Example:
        >>> detector = DriftDetector(threshold=0.1, method="ks")
        >>> detector.fit_reference(train_features)  # (N, D) numpy array
        >>> report = detector.check(production_features)
        >>> if report.is_drifted:
        ...     alert_team()
    """

    def __init__(
        self,
        threshold: float = 0.1,
        window_size: int = 1000,
        method: str = "ks",
    ) -> None:
        self.threshold = threshold
        self.window_size = window_size
        self.method = method
        self._reference: Optional[np.ndarray] = None
        self._history: List[DriftReport] = []

    def fit_reference(self, features: np.ndarray) -> None:
        """Set the reference distribution from training/validation data.

        Args:
            features: Feature array of shape (N, D).
        """
        self._reference = np.array(features, dtype=np.float32)
        logger.info(
            "DriftDetector: reference set (%d samples, %d features).",
            features.shape[0], features.shape[1] if features.ndim > 1 else 1,
        )

    def check(self, features: np.ndarray) -> DriftReport:
        """Check production features for drift against the reference.

        Args:
            features: Production feature array (N, D).

        Returns:
            DriftReport with drift score and per-feature breakdown.
        """
        if self._reference is None:
            raise RuntimeError("Call fit_reference() before check().")

        features = np.array(features, dtype=np.float32)

        if self.method == "ks":
            score, per_feature = self._ks_drift(features)
        else:
            score, per_feature = self._mmd_drift(features)

        report = DriftReport(
            timestamp=datetime.utcnow().isoformat(),
            drift_score=float(score),
            is_drifted=score > self.threshold,
            feature_drift=per_feature,
            threshold=self.threshold,
        )
        self._history.append(report)

        if report.is_drifted:
            logger.warning(
                "Data drift detected! score=%.4f > threshold=%.4f",
                score, self.threshold,
            )
        else:
            logger.debug("No drift detected (score=%.4f).", score)

        return report

    def _ks_drift(self, features: np.ndarray):
        """Kolmogorov-Smirnov test per feature dimension."""
        from scipy.stats import ks_2samp

        ref = self._reference
        scores = {}
        if features.ndim == 1:
            features = features[:, np.newaxis]
        if ref.ndim == 1:
            ref = ref[:, np.newaxis]

        for i in range(min(features.shape[1], ref.shape[1])):
            stat, _ = ks_2samp(ref[:, i], features[:, i])
            scores[f"feature_{i}"] = float(stat)

        mean_score = float(np.mean(list(scores.values()))) if scores else 0.0
        return mean_score, scores

    def _mmd_drift(self, features: np.ndarray):
        """Maximum Mean Discrepancy estimate."""
        ref = self._reference
        # Subsample for efficiency
        n = min(500, len(ref), len(features))
        idx_r = np.random.choice(len(ref), n, replace=False)
        idx_p = np.random.choice(len(features), n, replace=False)
        r = ref[idx_r]
        p = features[idx_p]

        # RBF kernel MMD
        gamma = 1.0 / r.shape[-1] if r.ndim > 1 else 1.0
        kxx = self._rbf_kernel(r, r, gamma)
        kyy = self._rbf_kernel(p, p, gamma)
        kxy = self._rbf_kernel(r, p, gamma)
        mmd = float(kxx.mean() + kyy.mean() - 2 * kxy.mean())
        mmd = max(0.0, mmd)
        return mmd, {"mmd": mmd}

    @staticmethod
    def _rbf_kernel(x: np.ndarray, y: np.ndarray, gamma: float) -> np.ndarray:
        diff = x[:, np.newaxis] - y[np.newaxis, :]
        return np.exp(-gamma * (diff ** 2).sum(-1))

    def get_history(self) -> List[DriftReport]:
        """Return the list of past drift reports."""
        return self._history


class PerformanceTracker:
    """Track model performance metrics over time in production.

    Stores per-batch and per-day metric snapshots for trend analysis
    and regression detection.

    Args:
        metrics: List of metric names to track.
        storage_path: Path for persisting metric history as JSON.
        alert_on_regression: Metric degradation threshold to trigger alerts.

    Example:
        >>> tracker = PerformanceTracker(
        ...     metrics=["miou", "f1", "precision"],
        ...     storage_path="~/.pygeovision/metrics.json",
        ... )
        >>> tracker.log({"miou": 0.82, "f1": 0.89}, timestamp="2024-06-15")
        >>> tracker.plot_trend("miou")
    """

    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        storage_path: Optional[Path] = None,
        alert_on_regression: float = 0.05,
    ) -> None:
        self.metrics = metrics or ["miou", "accuracy", "f1"]
        self.storage_path = Path(storage_path) if storage_path else (
            Path.home() / ".pygeovision" / "performance_history.json"
        )
        self.alert_on_regression = alert_on_regression
        self._history: List[Dict[str, Any]] = self._load_history()

    def log(
        self,
        metrics: Dict[str, float],
        timestamp: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        """Log a performance snapshot.

        Args:
            metrics: Dict of metric name → value.
            timestamp: ISO timestamp (defaults to now).
            model_name: Model identifier for multi-model tracking.
        """
        entry = {
            "timestamp": timestamp or datetime.utcnow().isoformat(),
            "model": model_name or "default",
            **metrics,
        }
        self._history.append(entry)
        self._save_history()

        # Check for regression
        self._check_regression(metrics)

    def get_trend(self, metric: str) -> List[float]:
        """Return historical values for a metric.

        Args:
            metric: Metric name.

        Returns:
            List of float values in chronological order.
        """
        return [e[metric] for e in self._history if metric in e]

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Return summary statistics for all tracked metrics.

        Returns:
            Dict mapping metric → {mean, min, max, last}.
        """
        result = {}
        for metric in self.metrics:
            values = self.get_trend(metric)
            if values:
                result[metric] = {
                    "mean": float(np.mean(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "last": float(values[-1]),
                }
        return result

    def _check_regression(self, metrics: Dict[str, float]) -> None:
        for metric, value in metrics.items():
            trend = self.get_trend(metric)[:-1]  # exclude current
            if len(trend) < 5:
                continue
            baseline = float(np.mean(trend[-10:]))
            if baseline > 0 and (baseline - value) / baseline > self.alert_on_regression:
                logger.warning(
                    "Performance regression detected: %s dropped from %.4f → %.4f (%.1f%%)",
                    metric, baseline, value, 100 * (baseline - value) / baseline,
                )

    def _load_history(self) -> List[Dict[str, Any]]:
        if self.storage_path.exists():
            try:
                return json.loads(self.storage_path.read_text())
            except Exception:
                pass
        return []

    def _save_history(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_path.write_text(json.dumps(self._history, indent=2))
