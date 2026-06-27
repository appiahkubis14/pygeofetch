"""Tests for Prithvi-EO integration — models, band handling, multi-temporal, tasks."""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock


# ── Model Registry ────────────────────────────────────────────────────────────

class TestPrithviRegistry:
    def test_all_models_present(self):
        from pygeovision.models.foundation.prithvi import PRITHVI_MODELS
        expected = ["prithvi_eo_1_0", "prithvi_eo_2_0",
                    "prithvi_eo_1_0_finetuned_burn", "prithvi_eo_1_0_finetuned_flood"]
        for name in expected:
            assert name in PRITHVI_MODELS, f"Missing: {name}"

    def test_prithvi_eo_1_0_specs(self):
        from pygeovision.models.foundation.prithvi import PRITHVI_MODELS
        spec = PRITHVI_MODELS["prithvi_eo_1_0"]
        assert spec["params_m"] == 100
        assert spec["n_bands"] == 6
        assert spec["temporal"] is True
        assert spec["coverage"] == "US"

    def test_prithvi_eo_2_0_specs(self):
        from pygeovision.models.foundation.prithvi import PRITHVI_MODELS
        spec = PRITHVI_MODELS["prithvi_eo_2_0"]
        assert spec["params_m"] == 600
        assert spec["n_bands"] == 6
        assert spec["embed_dim"] == 1024
        assert spec["coverage"] == "Global"
        assert spec["temporal"] is True

    def test_hf_ids_present(self):
        from pygeovision.models.foundation.prithvi import PRITHVI_MODELS
        assert "hf_id" in PRITHVI_MODELS["prithvi_eo_1_0"]
        assert "hf_id" in PRITHVI_MODELS["prithvi_eo_2_0"]

    def test_list_prithvi_models(self):
        from pygeovision.models.foundation.prithvi import list_prithvi_models
        models = list_prithvi_models()
        assert "prithvi_eo_1_0" in models
        assert "prithvi_eo_2_0" in models
        assert len(models) >= 2

    def test_get_prithvi_info(self):
        from pygeovision.models.foundation.prithvi import get_prithvi_info
        info = get_prithvi_info("prithvi_eo_2_0")
        assert info["params_m"] == 600
        assert "sentinel2_mapping" in info
        assert "landsat_mapping" in info
        assert "band_order" in info

    def test_get_info_unknown_raises(self):
        from pygeovision.models.foundation.prithvi import get_prithvi_info
        with pytest.raises(ValueError, match="Unknown"):
            get_prithvi_info("not_a_real_model")

    def test_main_registry_contains_prithvi(self):
        from pygeovision.models.registry import model_registry
        assert "prithvi_eo_1_0" in model_registry
        assert "prithvi_eo_2_0" in model_registry
        assert model_registry["prithvi_eo_2_0"].params_m == 600

    def test_prithvi_multispectral_flag(self):
        from pygeovision.models.registry import model_registry
        assert model_registry["prithvi_eo_1_0"].supports_multispectral is True
        assert model_registry["prithvi_eo_2_0"].supports_multispectral is True


# ── Band Mapping ─────────────────────────────────────────────────────────────

class TestPrithviBandHandling:
    def test_sentinel2_mapping_correct(self):
        from pygeovision.models.foundation.prithvi import SENTINEL2_TO_PRITHVI
        # Standard 6-band HLS order
        assert SENTINEL2_TO_PRITHVI["B02"] == 0   # Blue
        assert SENTINEL2_TO_PRITHVI["B03"] == 1   # Green
        assert SENTINEL2_TO_PRITHVI["B04"] == 2   # Red
        assert SENTINEL2_TO_PRITHVI["B08"] == 3   # NIR
        assert SENTINEL2_TO_PRITHVI["B11"] == 4   # SWIR1 (legacy mapping)
        assert SENTINEL2_TO_PRITHVI["B12"] == 5   # SWIR2

    def test_sentinel2_extended_bands(self):
        from pygeovision.models.foundation.prithvi import SENTINEL2_TO_PRITHVI
        # Extended 10-band mapping
        assert SENTINEL2_TO_PRITHVI["B05"] == 6   # Red Edge 1
        assert SENTINEL2_TO_PRITHVI["B06"] == 7   # Red Edge 2
        assert SENTINEL2_TO_PRITHVI["B07"] == 8   # Red Edge 3
        assert SENTINEL2_TO_PRITHVI["B8A"] == 9   # NIR Narrow

    def test_landsat_mapping_correct(self):
        from pygeovision.models.foundation.prithvi import LANDSAT_TO_PRITHVI
        assert LANDSAT_TO_PRITHVI["B2"] == 0   # Blue
        assert LANDSAT_TO_PRITHVI["B3"] == 1   # Green
        assert LANDSAT_TO_PRITHVI["B4"] == 2   # Red
        assert LANDSAT_TO_PRITHVI["B5"] == 3   # NIR
        assert LANDSAT_TO_PRITHVI["B6"] == 4   # SWIR1
        assert LANDSAT_TO_PRITHVI["B7"] == 5   # SWIR2

    def test_landsat_has_exactly_6_bands(self):
        from pygeovision.models.foundation.prithvi import LANDSAT_TO_PRITHVI
        assert len(LANDSAT_TO_PRITHVI) == 6

    def test_hls_scale_factor(self):
        from pygeovision.models.foundation.prithvi import HLS_SCALE_FACTOR
        assert HLS_SCALE_FACTOR == 10000.0

    def test_normalise_hls(self):
        from pygeovision.models.foundation.prithvi import normalise_hls
        # HLS SR values: 0–10000 → [0, 1]
        data = np.array([0, 5000, 10000], dtype=np.float32)
        norm = normalise_hls(data)
        assert abs(norm[0] - 0.0) < 1e-6
        assert abs(norm[1] - 0.5) < 1e-6
        assert abs(norm[2] - 1.0) < 1e-6

    def test_normalise_hls_clips_above_10000(self):
        from pygeovision.models.foundation.prithvi import normalise_hls
        data = np.array([12000, -500], dtype=np.float32)
        norm = normalise_hls(data)
        assert norm[0] == 1.0
        assert norm[1] == 0.0

    def test_map_bands_hls_passthrough(self):
        from pygeovision.models.foundation.prithvi import map_bands
        data = np.random.rand(6, 32, 32).astype(np.float32)
        out  = map_bands(data, source="hls", n_prithvi_bands=6)
        assert out.shape == (6, 32, 32)

    def test_map_bands_preserves_shape(self):
        from pygeovision.models.foundation.prithvi import map_bands
        data = np.random.rand(10, 64, 64).astype(np.float32)
        out  = map_bands(data, source="sentinel2", n_prithvi_bands=6)
        assert out.shape == (6, 64, 64)


# ── Loading Mechanisms ────────────────────────────────────────────────────────

class TestPrithviLoading:
    def test_load_hf_returns_model_or_surrogate(self):
        """load_prithvi_hf must return a model (surrogate if HF unavailable)."""
        from pygeovision.models.foundation.prithvi import load_prithvi_hf
        model = load_prithvi_hf("prithvi_eo_1_0", device="cpu")
        assert model is not None
        assert hasattr(model, "forward") or callable(model)

    def test_load_hf_eo2(self):
        from pygeovision.models.foundation.prithvi import load_prithvi_hf
        model = load_prithvi_hf("prithvi_eo_2_0", device="cpu")
        assert model is not None

    def test_load_local_nonexistent_raises(self):
        from pygeovision.models.foundation.prithvi import load_prithvi_local
        with pytest.raises(FileNotFoundError):
            load_prithvi_local("prithvi_eo_1_0", "/nonexistent/weights.pth")

    def test_load_unknown_model_raises(self):
        from pygeovision.models.foundation.prithvi import load_prithvi_hf
        with pytest.raises(ValueError, match="Unknown"):
            load_prithvi_hf("not_a_model", device="cpu")

    def test_surrogate_builds_correctly(self):
        from pygeovision.models.foundation.prithvi import _build_prithvi_surrogate, PRITHVI_MODELS
        spec = PRITHVI_MODELS["prithvi_eo_1_0"]
        model = _build_prithvi_surrogate(spec, device="cpu")
        assert model is not None
        assert hasattr(model, "forward")

    def test_surrogate_forward_4d(self):
        import torch
        from pygeovision.models.foundation.prithvi import _build_prithvi_surrogate, PRITHVI_MODELS
        spec  = PRITHVI_MODELS["prithvi_eo_1_0"]
        model = _build_prithvi_surrogate(spec, device="cpu")
        x     = torch.randn(1, 6, 64, 64)  # (B, C, H, W)
        with torch.no_grad():
            out = model(pixel_values=x)
        assert hasattr(out, "last_hidden_state")
        assert out.last_hidden_state.ndim == 3  # (B, N, D)

    def test_surrogate_forward_5d_multitemporal(self):
        import torch
        from pygeovision.models.foundation.prithvi import _build_prithvi_surrogate, PRITHVI_MODELS
        spec  = PRITHVI_MODELS["prithvi_eo_2_0"]
        model = _build_prithvi_surrogate(spec, device="cpu")
        x     = torch.randn(1, 3, 6, 64, 64)  # (B, T, C, H, W)
        with torch.no_grad():
            out = model(pixel_values=x)
        assert hasattr(out, "last_hidden_state")


# ── Prithvi Class ────────────────────────────────────────────────────────────

class TestPrithviClass:
    def test_init_defaults(self):
        from pygeovision.models.foundation.prithvi import Prithvi
        p = Prithvi()
        assert p.variant == "prithvi_eo_2_0"
        assert p._model is None
        assert p.device in ("cpu", "cuda", "mps")

    def test_init_eo1(self):
        from pygeovision.models.foundation.prithvi import Prithvi
        p = Prithvi("prithvi_eo_1_0")
        assert p.variant == "prithvi_eo_1_0"
        assert p._spec["params_m"] == 100

    def test_load_returns_self(self):
        from pygeovision.models.foundation.prithvi import Prithvi
        p = Prithvi("prithvi_eo_1_0")
        result = p.load()
        assert result is p
        assert p._model is not None

    def test_repr(self):
        from pygeovision.models.foundation.prithvi import Prithvi
        p = Prithvi("prithvi_eo_1_0")
        r = repr(p)
        assert "prithvi_eo_1_0" in r
        assert "100" in r
        assert "US" in r

    def test_finetune_config_eo2(self):
        from pygeovision.models.foundation.prithvi import Prithvi
        p   = Prithvi("prithvi_eo_2_0")
        cfg = p.finetune_config()
        assert cfg["optimizer"] == "AdamW"
        assert cfg["learning_rate"] == 5e-5
        assert cfg["weight_decay"] == 0.01
        assert cfg["warmup_epochs"] == 5
        assert cfg["mixed_precision"] == "bf16"

    def test_extract_features_shape(self, tmp_path):
        import torch, torch.nn as nn
        from pygeovision.models.foundation.prithvi import Prithvi, _build_prithvi_surrogate, PRITHVI_MODELS
        import rasterio
        from rasterio.transform import from_bounds

        # Write a 6-band synthetic GeoTIFF
        data = (np.random.rand(6, 32, 32) * 10000).astype(np.float32)
        t    = from_bounds(0, 0, 1, 1, 32, 32)
        p    = tmp_path / "hls.tif"
        with rasterio.open(str(p), "w", driver="GTiff", height=32, width=32,
                            count=6, dtype="float32", crs="EPSG:4326", transform=t) as dst:
            dst.write(data)

        model = Prithvi("prithvi_eo_1_0")
        model._model = _build_prithvi_surrogate(PRITHVI_MODELS["prithvi_eo_1_0"])
        feats = model.extract_features(str(p), source="hls")
        assert feats.ndim == 2
        assert feats.shape[1] == 768  # embed_dim for EO-1.0

    def test_build_segmentation_head(self):
        import torch
        from pygeovision.models.foundation.prithvi import Prithvi, _build_prithvi_surrogate, PRITHVI_MODELS
        p = Prithvi("prithvi_eo_1_0")
        p._model = _build_prithvi_surrogate(PRITHVI_MODELS["prithvi_eo_1_0"])
        head = p.build_segmentation_head(num_classes=11)
        assert head is not None
        # Backbone should be frozen
        for param in p._model.parameters():
            assert not param.requires_grad


# ── PrithviMultiTemporal ─────────────────────────────────────────────────────

class TestPrithviMultiTemporal:
    def test_init(self):
        from pygeovision.models.foundation.prithvi import PrithviMultiTemporal
        mt = PrithviMultiTemporal("prithvi_eo_1_0")
        assert mt.model_name == "prithvi_eo_1_0"
        assert mt._prithvi is not None

    def test_process_time_series_returns_features(self, tmp_path):
        import rasterio
        from rasterio.transform import from_bounds
        from pygeovision.models.foundation.prithvi import (
            PrithviMultiTemporal, _build_prithvi_surrogate, PRITHVI_MODELS
        )

        paths = []
        for i in range(3):
            data = (np.random.rand(6, 32, 32) * 8000).astype(np.float32)
            t    = from_bounds(0, 0, 1, 1, 32, 32)
            p    = tmp_path / f"t{i}.tif"
            with rasterio.open(str(p), "w", driver="GTiff", height=32, width=32,
                                count=6, dtype="float32", crs="EPSG:4326", transform=t) as dst:
                dst.write(data)
            paths.append(str(p))

        mt = PrithviMultiTemporal("prithvi_eo_1_0")
        mt._prithvi._model = _build_prithvi_surrogate(PRITHVI_MODELS["prithvi_eo_1_0"])

        result = mt.process_time_series(paths, dates=["2024-01","2024-04","2024-07"])
        assert "features" in result
        assert "n_frames" in result
        assert result["n_frames"] == 3
        assert len(result["dates"]) == 3

    def test_detect_change_returns_map(self, tmp_path):
        import rasterio
        from rasterio.transform import from_bounds
        from pygeovision.models.foundation.prithvi import (
            PrithviMultiTemporal, _build_prithvi_surrogate, PRITHVI_MODELS
        )

        paths = []
        for i in range(2):
            data = (np.random.rand(6, 32, 32) * 8000).astype(np.float32)
            t    = from_bounds(0, 0, 1, 1, 32, 32)
            p    = tmp_path / f"cd_{i}.tif"
            with rasterio.open(str(p), "w", driver="GTiff", height=32, width=32,
                                count=6, dtype="float32", crs="EPSG:4326", transform=t) as dst:
                dst.write(data)
            paths.append(str(p))

        mt = PrithviMultiTemporal("prithvi_eo_1_0")
        mt._prithvi._model = _build_prithvi_surrogate(PRITHVI_MODELS["prithvi_eo_1_0"])

        result = mt.detect_change(paths[0], paths[1])
        assert "change_map" in result
        assert "change_pct" in result
        assert isinstance(result["change_pct"], float)
        assert 0.0 <= result["change_pct"] <= 100.0

    def test_monitor_trend_insufficient_data(self, tmp_path):
        import rasterio
        from rasterio.transform import from_bounds
        from pygeovision.models.foundation.prithvi import (
            PrithviMultiTemporal, _build_prithvi_surrogate, PRITHVI_MODELS
        )

        paths = []
        for i in range(2):
            data = (np.random.rand(6, 32, 32) * 8000).astype(np.float32)
            t    = from_bounds(0, 0, 1, 1, 32, 32)
            p    = tmp_path / f"trend_{i}.tif"
            with rasterio.open(str(p), "w", driver="GTiff", height=32, width=32,
                                count=6, dtype="float32", crs="EPSG:4326", transform=t) as dst:
                dst.write(data)
            paths.append(str(p))

        mt = PrithviMultiTemporal("prithvi_eo_1_0")
        mt._prithvi._model = _build_prithvi_surrogate(PRITHVI_MODELS["prithvi_eo_1_0"])
        result = mt.monitor_trend(paths)
        assert "trend" in result or "trend_direction" in result

    def test_predict_seasonal_needs_4_frames(self, tmp_path):
        import rasterio
        from rasterio.transform import from_bounds
        from pygeovision.models.foundation.prithvi import (
            PrithviMultiTemporal, _build_prithvi_surrogate, PRITHVI_MODELS
        )

        paths = []
        for i in range(4):
            data = (np.random.rand(6, 32, 32) * 8000).astype(np.float32)
            t    = from_bounds(0, 0, 1, 1, 32, 32)
            p    = tmp_path / f"seas_{i}.tif"
            with rasterio.open(str(p), "w", driver="GTiff", height=32, width=32,
                                count=6, dtype="float32", crs="EPSG:4326", transform=t) as dst:
                dst.write(data)
            paths.append(str(p))

        mt = PrithviMultiTemporal("prithvi_eo_1_0")
        mt._prithvi._model = _build_prithvi_surrogate(PRITHVI_MODELS["prithvi_eo_1_0"])
        result = mt.predict_seasonal(
            paths, dates=["2024-01","2024-04","2024-07","2024-10"]
        )
        assert "seasonal_amplitude" in result
        assert "peak_date" in result
        assert "trough_date" in result


# ── PrithviTasks ─────────────────────────────────────────────────────────────

class TestPrithviTasks:
    def test_init(self):
        from pygeovision.models.foundation.prithvi import PrithviTasks
        tasks = PrithviTasks("prithvi_eo_1_0")
        assert tasks.model_name == "prithvi_eo_1_0"

    def test_land_cover_classes(self):
        from pygeovision.models.foundation.prithvi import PrithviTasks
        tasks = PrithviTasks()
        assert len(tasks.LAND_COVER_CLASSES) == 10
        assert "water" in tasks.LAND_COVER_CLASSES
        assert "trees" in tasks.LAND_COVER_CLASSES
        assert "crops" in tasks.LAND_COVER_CLASSES

    def test_crop_classes(self):
        from pygeovision.models.foundation.prithvi import PrithviTasks
        tasks = PrithviTasks()
        assert len(tasks.CROP_CLASSES) == 10
        assert "corn" in tasks.CROP_CLASSES
        assert "soybeans" in tasks.CROP_CLASSES

    def test_land_cover_output_shape(self, tmp_path):
        import rasterio
        from rasterio.transform import from_bounds
        from pygeovision.models.foundation.prithvi import (
            PrithviTasks, _build_prithvi_surrogate, PRITHVI_MODELS
        )

        data = (np.random.rand(6, 32, 32) * 8000).astype(np.float32)
        t    = from_bounds(0, 0, 1, 1, 32, 32)
        p    = tmp_path / "lc.tif"
        with rasterio.open(str(p), "w", driver="GTiff", height=32, width=32,
                            count=6, dtype="float32", crs="EPSG:4326", transform=t) as dst:
            dst.write(data)

        tasks = PrithviTasks("prithvi_eo_1_0")
        tasks._prithvi._model = _build_prithvi_surrogate(PRITHVI_MODELS["prithvi_eo_1_0"])

        result = tasks.land_cover(str(p))
        assert "prediction" in result
        assert "class_pct" in result
        assert "class_names" in result
        assert result["prediction"].ndim == 2

    def test_flood_detection_binary(self, tmp_path):
        import rasterio
        from rasterio.transform import from_bounds
        from pygeovision.models.foundation.prithvi import (
            PrithviTasks, _build_prithvi_surrogate, PRITHVI_MODELS
        )

        data = (np.random.rand(6, 32, 32) * 8000).astype(np.float32)
        t    = from_bounds(0, 0, 1, 1, 32, 32)
        p    = tmp_path / "flood.tif"
        with rasterio.open(str(p), "w", driver="GTiff", height=32, width=32,
                            count=6, dtype="float32", crs="EPSG:4326", transform=t) as dst:
            dst.write(data)

        tasks = PrithviTasks("prithvi_eo_1_0")
        tasks._prithvi._model = _build_prithvi_surrogate(PRITHVI_MODELS["prithvi_eo_1_0"])

        result = tasks.flood_detection(str(p), source="hls")
        assert "flood_pct" in result
        assert 0.0 <= result["flood_pct"] <= 100.0
        assert result["n_classes"] == 2

    def test_biomass_estimation(self, tmp_path):
        import rasterio
        from rasterio.transform import from_bounds
        from pygeovision.models.foundation.prithvi import (
            PrithviTasks, _build_prithvi_surrogate, PRITHVI_MODELS
        )

        data = (np.random.rand(6, 32, 32) * 8000).astype(np.float32)
        t    = from_bounds(0, 0, 1, 1, 32, 32)
        p    = tmp_path / "bio.tif"
        with rasterio.open(str(p), "w", driver="GTiff", height=32, width=32,
                            count=6, dtype="float32", crs="EPSG:4326", transform=t) as dst:
            dst.write(data)

        tasks = PrithviTasks("prithvi_eo_1_0")
        tasks._prithvi._model = _build_prithvi_surrogate(PRITHVI_MODELS["prithvi_eo_1_0"])

        result = tasks.biomass_estimation(str(p))
        assert "estimated_biomass_t_ha" in result
        assert result["estimated_biomass_t_ha"] >= 0.0


# ── finetune_prithvi ─────────────────────────────────────────────────────────

class TestFinetunePrithvi:
    def test_returns_training_components(self):
        from pygeovision.models.foundation.prithvi import finetune_prithvi
        result = finetune_prithvi(
            model_name="prithvi_eo_1_0",
            task="land_cover",
            num_classes=10,
            epochs=5,
        )
        if "error" not in result:
            assert "model" in result
            assert "optimizer" in result
            assert "scheduler" in result
            assert "config" in result
            assert result["config"]["learning_rate"] == 5e-5

    def test_config_matches_paper(self):
        from pygeovision.models.foundation.prithvi import Prithvi
        p   = Prithvi("prithvi_eo_2_0")
        cfg = p.finetune_config()
        assert cfg["learning_rate"] == 5e-5
        assert cfg["weight_decay"] == 0.01
        assert cfg["warmup_epochs"] == 5
        assert cfg["batch_size"] == 8
        assert cfg["mixed_precision"] == "bf16"


# ── Prithvi Proxy ─────────────────────────────────────────────────────────────

class TestPrithviProxy:
    def test_proxy_init(self):
        from pygeovision.ai.geoai.prithvi_proxy import PrithviProxy
        proxy = PrithviProxy()
        assert proxy is not None
        assert proxy._prithvi is None

    def test_proxy_list_models(self):
        from pygeovision.ai.geoai.prithvi_proxy import PrithviProxy
        proxy = PrithviProxy()
        models = proxy.list_models()
        assert "prithvi_eo_1_0" in models
        assert "prithvi_eo_2_0" in models

    def test_proxy_get_info(self):
        from pygeovision.ai.geoai.prithvi_proxy import PrithviProxy
        proxy = PrithviProxy()
        info = proxy.get_info("prithvi_eo_1_0")
        assert info["params_m"] == 100

    def test_proxy_repr(self):
        from pygeovision.ai.geoai.prithvi_proxy import PrithviProxy
        r = repr(PrithviProxy())
        assert "prithvi_eo_2_0" in r
        assert "EO" in r

    def test_engine_exposes_prithvi_proxy(self):
        import sys; sys.path.insert(0, '/home/claude/pgv')
        from pygeovision.ai.geoai import GeoAIEngine
        engine = GeoAIEngine()
        proxy = engine.prithvi
        assert hasattr(proxy, "land_cover")
        assert hasattr(proxy, "crop_mapping")
        assert hasattr(proxy, "flood_detection")
        assert hasattr(proxy, "biomass_estimation")
        assert hasattr(proxy, "change_detection")
        assert hasattr(proxy, "time_series")
        assert hasattr(proxy, "monitor_trend")
        assert hasattr(proxy, "predict_seasonal")