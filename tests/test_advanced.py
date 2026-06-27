"""Tests for advanced AI modules — few-shot, multi-task, AutoML, VLM, timeseries, 3D."""
import pytest
from unittest.mock import MagicMock, patch


# ── Few-Shot Learner ──────────────────────────────────────────────────────────

class TestFewShotLearner:
    def test_init(self):
        from pygeovision.advanced.few_shot import FewShotLearner
        learner = FewShotLearner(backbone="dinov2-base", method="prototypical")
        assert learner.backbone == "dinov2-base"
        assert learner.method == "prototypical"

    def test_hf_model_id_mapping(self):
        from pygeovision.advanced.few_shot import FewShotLearner
        learner = FewShotLearner(backbone="dinov2-large")
        assert "facebook" in learner.model_id
        assert "large" in learner.model_id

    def test_fit_support_requires_loading(self):
        from pygeovision.advanced.few_shot import FewShotLearner
        learner = FewShotLearner()
        # Without calling _load_backbone, _feature_extractor is None
        assert learner._feature_extractor is None

    def test_predict_without_fit_raises(self):
        from pygeovision.advanced.few_shot import FewShotLearner
        learner = FewShotLearner()
        with pytest.raises(RuntimeError, match="fit_support"):
            learner.predict(["img.tif"])

    def test_predict_with_mock_features(self):
        from pygeovision.advanced.few_shot import FewShotLearner
        import numpy as np
        learner = FewShotLearner()
        # Manually set prototypes
        learner._prototypes = {
            "forest":  np.array([1.0, 0.0, 0.0]),
            "water":   np.array([0.0, 1.0, 0.0]),
            "urban":   np.array([0.0, 0.0, 1.0]),
        }
        learner._class_names = ["forest", "water", "urban"]
        # Mock _extract_features
        with patch.object(learner, "_extract_features",
                           return_value=np.array([[1.0, 0.1, 0.0]])):
            results = learner.predict(["img.tif"])
        assert len(results) == 1
        assert results[0]["class"] == "forest"  # closest to [1, 0, 0]
        assert 0.0 <= results[0]["confidence"] <= 1.0

    def test_predict_probabilities_sum_to_one(self):
        from pygeovision.advanced.few_shot import FewShotLearner
        import numpy as np
        learner = FewShotLearner()
        learner._prototypes = {"a": np.ones(4), "b": np.zeros(4)}
        learner._class_names = ["a", "b"]
        with patch.object(learner, "_extract_features", return_value=np.array([[1., 1., 1., 1.]])):
            results = learner.predict(["x.tif"])
        probs = list(results[0]["probabilities"].values())
        assert abs(sum(probs) - 1.0) < 1e-5


# ── Multi-Task Learner ────────────────────────────────────────────────────────

class TestMultiTaskLearner:
    @pytest.fixture
    def learner(self):
        from pygeovision.advanced.multitask import MultiTaskLearner
        return MultiTaskLearner(backbone="resnet50",
                                 tasks=["segmentation", "classification"],
                                 n_classes={"segmentation": 2, "classification": 4})

    def test_init(self, learner):
        assert learner.backbone == "resnet50"
        assert "segmentation" in learner.tasks
        assert "classification" in learner.tasks
        assert learner.n_classes["segmentation"] == 2

    def test_task_weights_default(self, learner):
        # Default weights should sum to 1
        assert abs(sum(learner.task_weights.values()) - 1.0) < 1e-6

    def test_task_weights_correct_keys(self, learner):
        for task in learner.tasks:
            assert task in learner.task_weights

    @pytest.mark.skipif(not pytest.importorskip("torch", reason="torch not installed"), reason="needs torch")
    def test_build_returns_model(self, learner):
        import torch.nn as nn
        model = learner.build()
        assert isinstance(model, nn.Module)
        assert hasattr(model, "encoder")
        assert hasattr(model, "heads")

    @pytest.mark.skipif(not pytest.importorskip("torch", reason="torch"), reason="needs torch")
    def test_compute_loss_returns_scalar(self, learner):
        import torch
        learner.build()
        outputs = {
            "segmentation": torch.randn(1, 2, 4, 4),
            "classification": torch.randn(1, 4),
        }
        targets = {
            "segmentation": torch.zeros(1, 4, 4, dtype=torch.long),
            "classification": torch.zeros(1, dtype=torch.long),
        }
        loss, task_losses = learner.compute_loss(outputs, targets)
        assert loss.ndim == 0
        assert loss.item() >= 0
        assert "segmentation" in task_losses


# ── GeoAutoML ────────────────────────────────────────────────────────────────

class TestGeoAutoML:
    def test_init(self):
        from pygeovision.advanced.automl import GeoAutoML
        automl = GeoAutoML(metric="val_iou", n_trials=10, backend="optuna")
        assert automl.metric == "val_iou"
        assert automl.n_trials == 10
        assert automl.direction == "maximize"

    def test_importance_without_study(self):
        from pygeovision.advanced.automl import GeoAutoML
        automl = GeoAutoML()
        assert automl.importance() == {}

    @pytest.mark.skipif(not pytest.importorskip("optuna", reason="optuna"), reason="needs optuna")
    def test_search_optuna_basic(self):
        from pygeovision.advanced.automl import GeoAutoML
        results = []
        def train_fn(config):
            results.append(config)
            return config.get("lr", 0.001) * 100  # dummy metric

        automl = GeoAutoML(metric="val_iou", n_trials=3, backend="optuna")
        result = automl.search(train_fn, {"lr": ("float", 1e-4, 1e-2, "log")})
        assert result["success"] is True
        assert "best_params" in result
        assert "best_value" in result
        assert len(results) == 3


# ── CLIPGeo ─────────────────────────────────────────────────────────────────

class TestCLIPGeo:
    def test_init(self):
        from pygeovision.advanced.vlm.clip_geo import CLIPGeo
        clip = CLIPGeo(model="openclip-b32")
        assert clip.model_name == "openclip-b32"
        assert "clip" in clip.model_id.lower()

    def test_hf_models_registry(self):
        from pygeovision.advanced.vlm.clip_geo import CLIPGeo
        assert "remoteclip-b32" in CLIPGeo.HF_MODELS
        assert "openclip-b32" in CLIPGeo.HF_MODELS

    def test_search_empty_dir(self, tmp_path):
        from pygeovision.advanced.vlm.clip_geo import CLIPGeo
        clip = CLIPGeo()
        with patch.object(clip, "embed_text", return_value=None), \
             patch.object(clip, "embed_image", return_value=None):
            results = clip.search("deforestation", str(tmp_path))
        assert results == []


# ── GeoTimeSeries ────────────────────────────────────────────────────────────

class TestGeoTimeSeries:
    def test_init(self):
        from pygeovision.advanced.timeseries import GeoTimeSeries
        ts = GeoTimeSeries(sensor="sentinel2")
        assert ts.sensor == "sentinel2"
        assert "nir" in ts.band_map
        assert "red" in ts.band_map

    def test_supported_indices(self):
        from pygeovision.advanced.timeseries import GeoTimeSeries
        ts = GeoTimeSeries()
        assert "ndvi" in ts.INDICES
        assert "ndwi" in ts.INDICES
        assert "evi" in ts.INDICES
        assert "savi" in ts.INDICES

    def test_invalid_index(self):
        from pygeovision.advanced.timeseries import GeoTimeSeries
        ts = GeoTimeSeries()
        result = ts.compute_index_series([], index="invalid_xyz")
        assert "error" in result

    def test_empty_image_list(self):
        from pygeovision.advanced.timeseries import GeoTimeSeries
        ts = GeoTimeSeries()
        result = ts.compute_index_series([], "ndvi")
        assert result["n_images"] == 0
        assert result["mean"] == []

    def test_detect_anomalies_empty(self):
        from pygeovision.advanced.timeseries import GeoTimeSeries
        ts = GeoTimeSeries()
        series = {"mean": [], "dates": []}
        anomalies = ts.detect_anomalies(series)
        assert anomalies == []

    def test_detect_anomalies_zscore(self):
        from pygeovision.advanced.timeseries import GeoTimeSeries
        import numpy as np
        ts = GeoTimeSeries()
        # 9 normal values, 1 extreme outlier
        vals = [0.6, 0.62, 0.61, 0.63, 0.60, 0.62, 0.61, 0.60, 0.62, 0.10]
        series = {"mean": vals, "dates": [f"2024-{i+1:02d}" for i in range(10)]}
        anomalies = ts.detect_anomalies(series, threshold=2.0)
        assert len(anomalies) >= 1
        assert anomalies[-1]["type"] == "low"

    def test_compute_trend_increasing(self):
        from pygeovision.advanced.timeseries import GeoTimeSeries
        ts = GeoTimeSeries()
        series = {"mean": [0.5, 0.55, 0.60, 0.65, 0.70, 0.75], "dates": list(range(6))}
        trend = ts.compute_trend(series)
        assert trend["direction"] == "increasing"
        assert trend["slope"] > 0

    def test_compute_trend_insufficient_data(self):
        from pygeovision.advanced.timeseries import GeoTimeSeries
        ts = GeoTimeSeries()
        series = {"mean": [0.5, 0.6], "dates": [1, 2]}
        trend = ts.compute_trend(series)
        assert "insufficient" in trend.get("trend", "")


# ── PointCloudProcessor ──────────────────────────────────────────────────────

class TestPointCloudProcessor:
    def test_init(self):
        from pygeovision.advanced.pointcloud import PointCloudProcessor
        proc = PointCloudProcessor()
        assert proc is not None

    def test_read_missing_file(self):
        from pygeovision.advanced.pointcloud import PointCloudProcessor
        proc = PointCloudProcessor()
        result = proc.read("nonexistent.las")
        assert "error" in result or "n_points" in result  # laspy not installed → error

    def test_chm_missing_file(self, tmp_path):
        from pygeovision.advanced.pointcloud import PointCloudProcessor
        proc = PointCloudProcessor()
        result = proc.canopy_height_model("missing.las", str(tmp_path / "chm.tif"))
        assert "error" in result  # should fail gracefully
