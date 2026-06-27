"""Unit tests for PyGeoVision core components."""

from __future__ import annotations

import pytest
from pathlib import Path


class TestExceptions:
    def test_exception_hierarchy(self):
        from pygeovision.core.exceptions import (
            PyGeoVisionError, AIEngineError, ModelNotFoundError,
            TrainingError, InferenceError, PipelineError, LabelingError,
        )
        assert issubclass(AIEngineError, PyGeoVisionError)
        assert issubclass(ModelNotFoundError, AIEngineError)
        assert issubclass(TrainingError, AIEngineError)
        assert issubclass(InferenceError, AIEngineError)
        assert issubclass(PipelineError, PyGeoVisionError)
        assert issubclass(LabelingError, PyGeoVisionError)

    def test_raise_with_message(self):
        from pygeovision.core.exceptions import PyGeoVisionError
        with pytest.raises(PyGeoVisionError, match="test error"):
            raise PyGeoVisionError("test error")


class TestConfig:
    def test_default_config(self):
        from pygeovision.core.config import PyGeoVisionConfig
        cfg = PyGeoVisionConfig()
        assert cfg.gpu.mixed_precision in (True, False, "auto")
        assert cfg.training.batch_size > 0
        assert cfg.model_hub.cache_dir is not None

    def test_config_env_override(self, monkeypatch):
        monkeypatch.setenv("PYGEOVISION_GPU_DEVICE", "cpu")
        from pygeovision.core import config as cfg_module
        import importlib
        importlib.reload(cfg_module)
        # Config should respect env var

    def test_gpu_config_defaults(self):
        from pygeovision.core.config import GPUConfig
        gpu = GPUConfig()
        assert gpu.device in ("auto", "cuda", "mps", "cpu")


class TestModelRegistry:
    def test_registry_has_builtin_models(self):
        from pygeovision.ai.models.registry import registry
        assert len(registry) > 0

    def test_list_segmentation_models(self):
        from pygeovision.ai.models.registry import registry
        seg_models = registry.list_models(task="segmentation")
        assert len(seg_models) > 0
        assert all(m.task == "segmentation" for m in seg_models)

    def test_get_known_model(self):
        from pygeovision.ai.models.registry import registry
        info = registry.get("unet_resnet50")
        assert info.name == "unet_resnet50"
        assert info.task == "segmentation"
        assert info.pretrained_available is True

    def test_get_unknown_model_raises(self):
        from pygeovision.ai.models.registry import registry
        with pytest.raises(KeyError):
            registry.get("not_a_real_model_xyz")

    def test_registry_contains(self):
        from pygeovision.ai.models.registry import registry
        assert "unet_resnet50" in registry
        assert "not_real" not in registry

    def test_registry_summary(self):
        from pygeovision.ai.models.registry import registry
        summary = registry.summary()
        assert "unet_resnet50" in summary
        assert "segmentation" in summary


class TestPipelineRouter:
    def test_list_pipelines(self):
        from pygeovision.ai.pipelines import list_pipelines
        pipelines = list_pipelines()
        assert "building_footprints" in pipelines
        assert "land_cover" in pipelines
        assert "change_detection" in pipelines
        assert len(pipelines) == 10

    def test_get_unknown_pipeline(self):
        from pygeovision.ai.pipelines import get_pipeline
        from unittest.mock import MagicMock
        with pytest.raises(ValueError, match="Unknown pipeline"):
            get_pipeline("not_a_pipeline", MagicMock())

    def test_get_valid_pipeline(self):
        from pygeovision.ai.pipelines import get_pipeline, BuildingFootprintsPipeline
        from unittest.mock import MagicMock
        pipeline = get_pipeline("building_footprints", MagicMock())
        assert isinstance(pipeline, BuildingFootprintsPipeline)
