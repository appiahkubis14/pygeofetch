"""Tests for PyGeoVision advanced inference engine (Phase 2+)."""
import pytest, tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

_TORCH_AVAILABLE = False
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    pass

_RASTERIO_AVAILABLE = False
try:
    import rasterio
    _RASTERIO_AVAILABLE = True
except ImportError:
    pass


@pytest.fixture
def dummy_model():
    """A trivial PyTorch model that outputs 2-class logits."""
    if not _TORCH_AVAILABLE:
        pytest.skip("torch not installed")
    import torch, torch.nn as nn
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(4, 2, 1)
        def forward(self, x):
            return self.conv(x)
    return DummyModel()


@pytest.fixture
def synthetic_geotiff(tmp_path):
    """Create a small synthetic GeoTIFF for testing."""
    if not _RASTERIO_AVAILABLE:
        pytest.skip("rasterio not installed")
    import numpy as np, rasterio
    from rasterio.transform import from_bounds
    H, W = 64, 64
    data = (np.random.rand(4, H, W) * 10000).astype(np.float32)
    transform = from_bounds(-74.0, 40.7, -73.9, 40.8, W, H)
    p = tmp_path / "test_scene.tif"
    with rasterio.open(str(p), "w", driver="GTiff", height=H, width=W,
                        count=4, dtype="float32", crs="EPSG:4326", transform=transform) as dst:
        dst.write(data)
    return str(p)


class TestGaussianBlend:
    def test_window_shape(self):
        from pygeovision.inference.tiled import GaussianBlend
        w = GaussianBlend.window(64)
        assert w.shape == (64, 64)

    def test_window_max_is_one(self):
        from pygeovision.inference.tiled import GaussianBlend
        import numpy as np
        w = GaussianBlend.window(128, sigma_ratio=0.25)
        assert abs(w.max() - 1.0) < 1e-6

    def test_window_centre_highest(self):
        from pygeovision.inference.tiled import GaussianBlend
        w = GaussianBlend.window(64)
        centre = w[32, 32]
        corner = w[0, 0]
        assert centre > corner

    def test_window_symmetric(self):
        from pygeovision.inference.tiled import GaussianBlend
        import numpy as np
        w = GaussianBlend.window(64)
        assert np.allclose(w, w[::-1])   # vertical symmetry
        assert np.allclose(w, w[:, ::-1]) # horizontal symmetry

    def test_apply_to_prediction(self):
        from pygeovision.inference.tiled import GaussianBlend
        import numpy as np
        pred = np.ones((2, 64, 64), dtype=np.float32)
        weighted = GaussianBlend.apply_to_prediction(pred, 64)
        assert weighted.shape == (2, 64, 64)
        # Centre pixels should be near 1, corners near 0
        assert weighted[0, 32, 32] > 0.5
        assert weighted[0, 0, 0] < 0.5

    def test_blend_mode_linear(self):
        from pygeovision.inference.tiled import TiledInference
        inf = TiledInference(model=MagicMock(), blend_mode="linear")
        w = inf._get_blend_window(64, 64)
        assert w.shape == (64, 64)

    def test_blend_mode_constant(self):
        from pygeovision.inference.tiled import TiledInference
        import numpy as np
        inf = TiledInference(model=MagicMock(), blend_mode="constant")
        w = inf._get_blend_window(32, 32)
        assert np.allclose(w, 1.0)


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")
@pytest.mark.skipif(not _RASTERIO_AVAILABLE, reason="rasterio not installed")
class TestTiledInference:
    def test_init_defaults(self, dummy_model):
        from pygeovision.inference.tiled import TiledInference
        inf = TiledInference(model=dummy_model)
        assert inf.chip_size == 512
        assert inf.overlap == 128
        assert inf.blend_mode == "gaussian"
        assert inf.num_classes == 2

    def test_init_custom(self, dummy_model):
        from pygeovision.inference.tiled import TiledInference
        inf = TiledInference(model=dummy_model, chip_size=256, overlap=32,
                              blend_mode="linear", num_classes=5)
        assert inf.chip_size == 256
        assert inf.num_classes == 5

    def test_estimate_memory(self, dummy_model):
        from pygeovision.inference.tiled import TiledInference
        inf = TiledInference(model=dummy_model, chip_size=512, overlap=64)
        mem = inf.estimate_memory(H=2048, W=2048)
        assert "n_chips" in mem
        assert mem["n_chips"] > 0
        assert "batch_gpu_mb" in mem
        assert "accumulation_ram_mb" in mem

    def test_predict_chip_shape(self, dummy_model):
        import numpy as np
        from pygeovision.inference.tiled import TiledInference
        inf = TiledInference(model=dummy_model, num_classes=2)
        chip = np.random.randn(4, 32, 32).astype(np.float32)
        pred = inf._predict_chip(chip)
        assert pred.shape[0] == 2  # num_classes

    def test_infer_small_geotiff(self, dummy_model, synthetic_geotiff, tmp_path):
        from pygeovision.inference.tiled import TiledInference
        inf = TiledInference(model=dummy_model, chip_size=32, overlap=4,
                              num_classes=2, batch_tiles=2, half_precision=False)
        out = str(tmp_path / "prediction.tif")
        result = inf.infer(synthetic_geotiff, out)
        assert result["success"] is True
        assert Path(out).exists()
        assert result["n_chips"] > 0

    def test_infer_creates_output(self, dummy_model, synthetic_geotiff, tmp_path):
        from pygeovision.inference.tiled import TiledInference
        import rasterio
        inf = TiledInference(model=dummy_model, chip_size=32, overlap=8,
                              num_classes=2, half_precision=False)
        out = str(tmp_path / "out.tif")
        inf.infer(synthetic_geotiff, out)
        with rasterio.open(out) as src:
            assert src.count == 1
            assert src.read(1).dtype.itemsize == 1  # uint8


class TestEnsembleInference:
    def test_init(self):
        from pygeovision.inference.stream import EnsembleInference
        models = [MagicMock(), MagicMock()]
        ens = EnsembleInference(models, weights=[0.6, 0.4], fusion="mean")
        assert ens.fusion == "mean"
        assert len(ens.models) == 2

    def test_default_equal_weights(self):
        from pygeovision.inference.stream import EnsembleInference
        models = [MagicMock()] * 3
        ens = EnsembleInference(models)
        assert all(abs(w - 1/3) < 1e-6 for w in ens.weights)


class TestBatchInferenceEngine:
    def test_init(self, dummy_model):
        from pygeovision.inference.batch import BatchInferenceEngine
        engine = BatchInferenceEngine(dummy_model, n_workers=2, chip_size=256)
        assert engine.n_workers == 2
        assert engine.chip_size == 256

    def test_empty_directory(self, tmp_path, dummy_model):
        from pygeovision.inference.batch import BatchInferenceEngine
        engine = BatchInferenceEngine(dummy_model)
        result = engine.run_directory(str(tmp_path), str(tmp_path / "out"))
        assert result.get("success") is False or result.get("n_success", 0) == 0
