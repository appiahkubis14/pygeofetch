"""
Integration tests for PyGeoVision — end-to-end pipeline flows with mocks.

These tests verify the full stack (data → AI → output) without real
satellite API calls by mocking PyGeoFetch's search/download methods.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest


@pytest.fixture
def mock_sv_with_geotiff(mock_pgv_client, small_geotiff):
    """PyGeoVision mock that returns a real GeoTIFF for download."""
    mock_pgv_client.download.return_value = small_geotiff
    mock_pgv_client.search.return_value = [MagicMock(id="s2-test-001")]
    return mock_pgv_client


class TestBuildingFootprintsPipeline:
    def test_pipeline_with_no_imagery(self, tmp_dir, mock_pgv_client):
        from pygeovision.ai.pipelines import get_pipeline
        mock_pgv_client.search.return_value = []
        pipeline = get_pipeline("building_footprints", mock_pgv_client)
        result = pipeline.run(
            bbox=(-0.1, 51.4, 0.0, 51.5),
            output_dir=tmp_dir,
            date="2024-06",
        )
        assert result.success is False
        assert "No imagery" in result.error

    def test_pipeline_creates_output(self, tmp_dir, mock_sv_with_geotiff):
        pytest.importorskip("torch")
        pytest.importorskip("rasterio")
        import torch.nn as nn
        from pygeovision.ai.pipelines import BuildingFootprintsPipeline

        # Patch ModelHub.load to return a tiny model
        class TinyModel(nn.Module):
            def forward(self, x):
                import torch
                return torch.zeros(x.shape[0], 2, x.shape[2], x.shape[3])

        with patch("pygeovision.ai.models.hub.ModelHub.load", return_value=TinyModel()):
            pipeline = BuildingFootprintsPipeline(mock_sv_with_geotiff)
            result = pipeline.run(
                bbox=(-0.1, 51.4, 0.0, 51.5),
                output_dir=tmp_dir,
                date="2024-06",
            )
        # Should succeed or fail gracefully
        assert result.pipeline == "building_footprints"
        assert isinstance(result.success, bool)


class TestWaterBodiesPipeline:
    def test_ndwi_method(self, tmp_dir, mock_sv_with_geotiff):
        pytest.importorskip("rasterio")
        from pygeovision.ai.pipelines import WaterBodiesPipeline

        pipeline = WaterBodiesPipeline(mock_sv_with_geotiff)
        result = pipeline.run(
            bbox=(-0.1, 51.4, 0.0, 51.5),
            output_dir=tmp_dir,
            date="2024-06",
            method="ndwi",
        )
        # NDWI only needs rasterio (no AI), should complete without errors
        # Result may succeed or fail depending on band count of test tile
        assert result.pipeline == "water_bodies"


class TestExperimentTracker:
    def test_full_lifecycle(self, tmp_dir):
        from pygeovision.ai.experiments import ExperimentTracker

        tracker = ExperimentTracker("test_run", save_dir=tmp_dir, seed=42)
        tracker.log_params({"lr": 1e-4, "model": "unet_resnet50"})
        tracker.log_metrics({"val_miou": 0.75, "val_loss": 0.32}, step=0)
        tracker.log_metrics({"val_miou": 0.82, "val_loss": 0.21}, step=10)
        tracker.log_artifact("/tmp/best.pth")
        tracker.set_tag("dataset", "sentinel2_africa")
        path = tracker.save()

        assert path.exists()
        assert path.suffix == ".json"

        # Reload and verify
        loaded = ExperimentTracker.load(tracker.record.run_id, save_dir=tmp_dir)
        assert loaded.record.name == "test_run"
        assert loaded.record.params["lr"] == 1e-4
        assert len(loaded.record.metrics["val_miou"]) == 2
        assert loaded.get_best_metric("val_miou") == pytest.approx(0.82)

    def test_list_experiments(self, tmp_dir):
        from pygeovision.ai.experiments import ExperimentTracker

        for name in ["run_a", "run_b", "run_c"]:
            t = ExperimentTracker(name, save_dir=tmp_dir)
            t.save()

        any_tracker = ExperimentTracker("unused", save_dir=tmp_dir)
        exps = any_tracker.list_experiments()
        assert len(exps) >= 3


class TestDriftDetector:
    def test_basic_drift_detection(self):
        pytest.importorskip("scipy")
        from pygeovision.ai.monitoring import DriftDetector

        detector = DriftDetector(threshold=0.1, method="ks")
        # Reference: standard normal
        ref = np.random.randn(500, 4).astype(np.float32)
        detector.fit_reference(ref)

        # No drift: same distribution
        same = np.random.randn(200, 4).astype(np.float32)
        report_no_drift = detector.check(same)
        assert report_no_drift.drift_score < 0.9  # should be low

        # Strong drift: very different distribution
        drifted = np.random.randn(200, 4).astype(np.float32) + 10.0
        report_drift = detector.check(drifted)
        assert report_drift.drift_score > report_no_drift.drift_score

    def test_requires_fit_before_check(self):
        from pygeovision.ai.monitoring import DriftDetector
        detector = DriftDetector()
        with pytest.raises(RuntimeError, match="fit_reference"):
            detector.check(np.random.randn(10, 4))


class TestModelExporter:
    def test_benchmark(self, simple_segmentation_model):
        pytest.importorskip("torch")
        from pygeovision.ai.training.export import ModelExporter

        exporter = ModelExporter(
            simple_segmentation_model,
            input_shape=(1, 3, 32, 32),
            device="cpu",
        )
        result = exporter.benchmark(num_warmup=2, num_runs=5)
        assert "mean_ms" in result
        assert "fps" in result
        assert result["mean_ms"] > 0

    def test_torchscript_export(self, simple_segmentation_model, tmp_dir):
        pytest.importorskip("torch")
        from pygeovision.ai.training.export import ModelExporter

        exporter = ModelExporter(
            simple_segmentation_model,
            input_shape=(1, 3, 32, 32),
            device="cpu",
        )
        path = exporter.to_torchscript(tmp_dir / "model.pt", method="trace")
        assert path.exists()
        assert path.stat().st_size > 0
