"""Tests for drift detection and monitoring (Phase 2+)."""
import pytest, json, tempfile
from pathlib import Path
from unittest.mock import patch


class TestDistributionDrift:
    def test_compute_psi_identical(self):
        from pygeovision.monitoring.drift import DistributionDrift
        import numpy as np
        d = DistributionDrift()
        data = np.random.rand(1000)
        psi = d.compute_psi(data, data)
        assert abs(psi) < 0.05  # identical distributions → PSI ≈ 0

    def test_compute_psi_different(self):
        from pygeovision.monitoring.drift import DistributionDrift
        import numpy as np
        d = DistributionDrift()
        data1 = np.random.randn(1000)
        data2 = np.random.randn(1000) + 5   # shifted distribution
        psi = d.compute_psi(data1, data2)
        assert psi > 0.2  # major drift

    def test_compute_kl_identical(self):
        from pygeovision.monitoring.drift import DistributionDrift
        import numpy as np
        d = DistributionDrift()
        p = np.ones(10) / 10
        kl = d.compute_kl_divergence(p, p)
        assert abs(kl) < 1e-6

    def test_psi_thresholds(self):
        from pygeovision.monitoring.drift import DistributionDrift
        assert DistributionDrift.PSI_THRESHOLDS["low"] == 0.1
        assert DistributionDrift.PSI_THRESHOLDS["medium"] == 0.2

    def test_recommendation_no_drift(self):
        from pygeovision.monitoring.drift import DistributionDrift
        d = DistributionDrift()
        rec = d._recommendation(0.05)
        assert "No action" in rec

    def test_recommendation_minor(self):
        from pygeovision.monitoring.drift import DistributionDrift
        d = DistributionDrift()
        rec = d._recommendation(0.15)
        assert "Minor" in rec

    def test_recommendation_major(self):
        from pygeovision.monitoring.drift import DistributionDrift
        d = DistributionDrift()
        rec = d._recommendation(0.30)
        assert "Major" in rec or "Retrain" in rec


class TestPerformanceDrift:
    def test_init_creates_storage(self, tmp_path):
        from pygeovision.monitoring.drift import PerformanceDrift
        storage = str(tmp_path / "perf.jsonl")
        pd = PerformanceDrift(storage_path=storage)
        assert Path(storage).parent.exists()

    def test_log_appends(self, tmp_path):
        from pygeovision.monitoring.drift import PerformanceDrift
        storage = str(tmp_path / "perf.jsonl")
        pd = PerformanceDrift(storage_path=storage)
        pd.log({"mean_iou": 0.85, "accuracy": 0.92})
        pd.log({"mean_iou": 0.83, "accuracy": 0.91})
        with open(storage) as f:
            lines = [l for l in f if l.strip()]
        assert len(lines) == 2

    def test_set_baseline_stores(self, tmp_path):
        from pygeovision.monitoring.drift import PerformanceDrift
        pd = PerformanceDrift(storage_path=str(tmp_path / "p.jsonl"))
        pd.set_baseline({"mean_iou": 0.85, "accuracy": 0.92})
        assert pd._baseline["mean_iou"] == 0.85

    def test_detect_no_baseline(self, tmp_path):
        from pygeovision.monitoring.drift import PerformanceDrift
        pd = PerformanceDrift(storage_path=str(tmp_path / "p.jsonl"))
        result = pd.detect()
        assert "error" in result

    def test_detect_no_drift(self, tmp_path):
        from pygeovision.monitoring.drift import PerformanceDrift
        pd = PerformanceDrift(storage_path=str(tmp_path / "p.jsonl"))
        pd.set_baseline({"mean_iou": 0.85})
        for _ in range(5):
            pd.log({"mean_iou": 0.85})  # stable
        result = pd.detect()
        assert result.get("drift_detected") is False

    def test_detect_drift_found(self, tmp_path):
        from pygeovision.monitoring.drift import PerformanceDrift
        pd = PerformanceDrift(storage_path=str(tmp_path / "p.jsonl"),
                               alert_thresholds={"mean_iou": 0.05})
        pd.set_baseline({"mean_iou": 0.85})
        for _ in range(5):
            pd.log({"mean_iou": 0.70})  # -15pp drop >> 5pp threshold
        result = pd.detect()
        assert result.get("drift_detected") is True
        assert len(result.get("alerts", [])) > 0

    def test_performance_recommendation_no_alerts(self, tmp_path):
        from pygeovision.monitoring.drift import PerformanceDrift
        pd = PerformanceDrift(storage_path=str(tmp_path / "p.jsonl"))
        rec = pd._performance_recommendation([])
        assert "stable" in rec.lower() or "No action" in rec


class TestDriftDetector:
    def test_init(self, tmp_path):
        from pygeovision.monitoring.drift import DriftDetector
        d = DriftDetector(storage_dir=str(tmp_path))
        assert d.data_drift is not None
        assert d.perf_drift is not None

    def test_fit_no_images(self, tmp_path):
        from pygeovision.monitoring.drift import DriftDetector
        d = DriftDetector(storage_dir=str(tmp_path))
        stats = d.fit([])   # empty list
        # Should not crash
        assert stats is d or stats is None or isinstance(stats, dict)

    def test_save_report(self, tmp_path):
        from pygeovision.monitoring.drift import DriftDetector
        d = DriftDetector(storage_dir=str(tmp_path))
        report = {"drift_detected": False, "data_drift": {}, "performance_drift": {}}
        path = d.save_report(report, str(tmp_path / "report.json"))
        assert Path(path).exists()
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["drift_detected"] is False


class TestModelPerformanceTracker:
    def test_record_and_summary(self, tmp_path):
        from pygeovision.monitoring.tracker import ModelPerformanceTracker
        tracker = ModelPerformanceTracker("test_model", str(tmp_path / "tracker.json"))
        tracker.record({"mean_iou": 0.82, "accuracy": 0.91}, split="val", n_samples=500)
        tracker.record({"mean_iou": 0.84, "accuracy": 0.93}, split="val", n_samples=600)
        summary = tracker.summary()
        assert summary["n_records"] == 2
        assert summary["model"] == "test_model"
        assert "mean_iou" in summary["latest_metrics"]

    def test_trend_improving(self, tmp_path):
        from pygeovision.monitoring.tracker import ModelPerformanceTracker
        tracker = ModelPerformanceTracker("m", str(tmp_path / "t.json"))
        for v in [0.70, 0.75, 0.80, 0.82]:
            tracker.record({"mean_iou": v})
        trend = tracker.trend("mean_iou")
        assert trend["direction"] == "improving"

    def test_trend_declining(self, tmp_path):
        from pygeovision.monitoring.tracker import ModelPerformanceTracker
        tracker = ModelPerformanceTracker("m", str(tmp_path / "t.json"))
        for v in [0.85, 0.82, 0.78, 0.75]:
            tracker.record({"mean_iou": v})
        trend = tracker.trend("mean_iou")
        assert trend["direction"] == "declining"


class TestAlertManager:
    def test_init_default(self):
        from pygeovision.monitoring.alerts import AlertManager
        am = AlertManager()
        assert "log" in am.channels

    def test_trigger_logs(self):
        from pygeovision.monitoring.alerts import AlertManager
        am = AlertManager(channels=["log"])
        am.trigger("warning", "Test alert")
        assert len(am.history) == 1
        assert am.history[0]["message"] == "Test alert"

    def test_register_custom_handler(self):
        from pygeovision.monitoring.alerts import AlertManager
        received = []
        am = AlertManager(channels=["custom"])
        am.register_handler("custom", lambda entry: received.append(entry))
        am.trigger("info", "Custom test")
        assert len(received) == 1
        assert received[0]["severity"] == "info"

    def test_check_drift_report_no_drift(self):
        from pygeovision.monitoring.alerts import AlertManager
        am = AlertManager()
        report = {"drift_detected": False, "data_drift": {"drift_level": "none"},
                   "performance_drift": {"drift_detected": False, "alerts": []}}
        am.check_drift_report(report)
        # No alerts should be triggered
        assert len(am.history) == 0

    def test_check_drift_report_major_drift(self):
        from pygeovision.monitoring.alerts import AlertManager
        received = []
        am = AlertManager(channels=["custom"])
        am.register_handler("custom", lambda e: received.append(e))
        report = {
            "drift_detected": True,
            "data_drift": {"drift_level": "major"},
            "performance_drift": {"drift_detected": True, "alerts": ["mean_iou dropped"]},
        }
        am.check_drift_report(report)
        severities = [r["severity"] for r in received]
        assert "critical" in severities or "warning" in severities
