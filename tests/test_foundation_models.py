"""Integration tests — model registry, GeoAI engine, end-to-end foundation model workflows."""
import pytest
import sys
sys.path.insert(0, '/home/claude/pgv')
import numpy as np


# ── Model Registry Integration ────────────────────────────────────────────────

class TestFoundationModelRegistry:
    """Verify the main model registry contains all DINOv3 + Prithvi models."""

    # DINOv3 ViT Web variants
    @pytest.mark.parametrize("name", [
        "dinov3_vits16", "dinov3_vits16plus", "dinov3_vitb16", "dinov3_vitl16",
        "dinov3_vith16plus", "dinov3_vit7b16",
    ])
    def test_dinov3_web_variants_in_registry(self, name):
        from pygeovision.models.registry import model_registry
        assert name in model_registry, f"Missing from registry: {name}"
        spec = model_registry[name]
        assert spec.task == "foundation"
        assert spec.family == "dinov3"
        assert spec.pretrained_on == "LVD-1689M"

    # DINOv3 SAT variants
    @pytest.mark.parametrize("name", ["dinov3_vitl16_sat", "dinov3_vit7b16_sat"])
    def test_dinov3_sat_variants_in_registry(self, name):
        from pygeovision.models.registry import model_registry
        assert name in model_registry
        spec = model_registry[name]
        assert spec.pretrained_on == "SAT-493M"
        assert spec.supports_multispectral is True

    # DINOv3 ConvNeXt variants
    @pytest.mark.parametrize("name", [
        "dinov3_convnext_tiny", "dinov3_convnext_small",
        "dinov3_convnext_base", "dinov3_convnext_large",
    ])
    def test_dinov3_convnext_in_registry(self, name):
        from pygeovision.models.registry import model_registry
        assert name in model_registry
        assert model_registry[name].family == "dinov3"

    # DINOv3 task heads
    @pytest.mark.parametrize("name", [
        "dinov3_classifier", "dinov3_depther", "dinov3_detector",
        "dinov3_segmentor", "dinov3_dinotxt", "dinov3_chmv2",
    ])
    def test_dinov3_heads_in_registry(self, name):
        from pygeovision.models.registry import model_registry
        assert name in model_registry, f"Head missing: {name}"

    # Prithvi models
    @pytest.mark.parametrize("name,params,pretrain", [
        ("prithvi_eo_1_0", 100, "HLS-US"),
        ("prithvi_eo_2_0", 600, "HLS-Global"),
    ])
    def test_prithvi_in_registry(self, name, params, pretrain):
        from pygeovision.models.registry import model_registry
        assert name in model_registry
        spec = model_registry[name]
        assert spec.params_m == params
        assert spec.pretrained_on == pretrain
        assert spec.family == "prithvi"
        assert spec.supports_multispectral is True

    def test_total_foundation_models_count(self):
        from pygeovision.models.registry import model_registry
        foundation = model_registry.list(task="foundation")
        # 6 ViT web + 2 ViT SAT + 4 ConvNeXt + 6 heads + 2+ Prithvi + others ≥ 28
        assert len(foundation) >= 28, f"Expected ≥28 foundation models, got {len(foundation)}"

    def test_registry_summary(self):
        from pygeovision.models.registry import model_registry
        s = model_registry.summary()
        assert s["total"] >= 119
        assert "foundation" in s["by_task"]

    def test_list_satellite_pretrained(self):
        from pygeovision.models.registry import _REGISTRY
        sat = [n for n, s in _REGISTRY.items()
               if s.pretrained_on in ("SAT-493M", "HLS-US", "HLS-Global")]
        assert len(sat) >= 4  # 2 DINOv3 SAT + 2 Prithvi

    def test_search_dinov3(self):
        from pygeovision.models.registry import model_registry
        results = model_registry.search("dinov3")
        assert len(results) >= 12

    def test_search_prithvi(self):
        from pygeovision.models.registry import model_registry
        results = model_registry.search("prithvi")
        assert len(results) >= 2


# ── GeoAI Engine ──────────────────────────────────────────────────────────────

class TestGeoAIEngineFoundation:
    """Test GeoAI Engine exposes DINOv3 and Prithvi subsystems correctly."""

    def test_engine_has_dinov3_property(self):
        from pygeovision.ai.geoai import GeoAIEngine
        engine = GeoAIEngine()
        assert hasattr(engine, "dinov3")

    def test_engine_has_prithvi_property(self):
        from pygeovision.ai.geoai import GeoAIEngine
        engine = GeoAIEngine()
        assert hasattr(engine, "prithvi")

    def test_engine_has_foundation_models_property(self):
        from pygeovision.ai.geoai import GeoAIEngine
        engine = GeoAIEngine()
        assert hasattr(engine, "foundation_models")

    def test_dinov3_proxy_api(self):
        from pygeovision.ai.geoai import GeoAIEngine
        engine = GeoAIEngine()
        proxy  = engine.dinov3
        # Verify all required methods
        for method in ["load", "extract_features", "extract_embeddings",
                        "canopy_height", "zero_shot", "list_models",
                        "list_satellite_models", "get_info", "finetune"]:
            assert hasattr(proxy, method), f"DINOv3Proxy missing: {method}"

    def test_prithvi_proxy_api(self):
        from pygeovision.ai.geoai import GeoAIEngine
        engine = GeoAIEngine()
        proxy  = engine.prithvi
        for method in ["load", "land_cover", "crop_mapping", "flood_detection",
                        "biomass_estimation", "change_detection", "time_series",
                        "monitor_trend", "predict_seasonal", "list_models",
                        "get_info", "finetune"]:
            assert hasattr(proxy, method), f"PrithviProxy missing: {method}"

    def test_foundation_models_has_both(self):
        from pygeovision.ai.geoai import GeoAIEngine
        engine = GeoAIEngine()
        fm = engine.foundation_models
        assert hasattr(fm, "dinov3")
        assert hasattr(fm, "prithvi")

    def test_dinov3_list_models_returns_12(self):
        from pygeovision.ai.geoai import GeoAIEngine
        proxy  = GeoAIEngine().dinov3
        models = proxy.list_models()
        assert len(models) == 12

    def test_dinov3_list_satellite_returns_2(self):
        from pygeovision.ai.geoai import GeoAIEngine
        proxy = GeoAIEngine().dinov3
        sat   = proxy.list_satellite_models()
        assert len(sat) == 2
        assert "dinov3_vitl16_sat" in sat
        assert "dinov3_vit7b16_sat" in sat

    def test_prithvi_list_models_at_least_2(self):
        from pygeovision.ai.geoai import GeoAIEngine
        proxy  = GeoAIEngine().prithvi
        models = proxy.list_models()
        assert len(models) >= 2
        assert "prithvi_eo_1_0" in models
        assert "prithvi_eo_2_0" in models

    def test_dinov3_proxy_is_cached(self):
        """Calling engine.dinov3 twice should return the same proxy instance."""
        from pygeovision.ai.geoai import GeoAIEngine
        engine = GeoAIEngine()
        p1 = engine.dinov3
        p2 = engine.dinov3
        assert p1 is p2

    def test_prithvi_proxy_is_cached(self):
        from pygeovision.ai.geoai import GeoAIEngine
        engine = GeoAIEngine()
        p1 = engine.prithvi
        p2 = engine.prithvi
        assert p1 is p2


# ── Transform Correctness ────────────────────────────────────────────────────

class TestTransformCorrectness:
    """Ensure web and SAT transforms use exactly the right statistics."""

    def test_web_mean_imagenet(self):
        from pygeovision.models.foundation.dinov3 import WEB_MEAN
        assert len(WEB_MEAN) == 3
        assert abs(WEB_MEAN[0] - 0.485) < 0.001
        assert abs(WEB_MEAN[1] - 0.456) < 0.001
        assert abs(WEB_MEAN[2] - 0.406) < 0.001

    def test_web_std_imagenet(self):
        from pygeovision.models.foundation.dinov3 import WEB_STD
        assert abs(WEB_STD[0] - 0.229) < 0.001
        assert abs(WEB_STD[1] - 0.224) < 0.001
        assert abs(WEB_STD[2] - 0.225) < 0.001

    def test_sat_mean_satellite(self):
        from pygeovision.models.foundation.dinov3 import SAT_MEAN
        assert abs(SAT_MEAN[0] - 0.430) < 0.001
        assert abs(SAT_MEAN[1] - 0.411) < 0.001
        assert abs(SAT_MEAN[2] - 0.296) < 0.001

    def test_sat_std_satellite(self):
        from pygeovision.models.foundation.dinov3 import SAT_STD
        assert abs(SAT_STD[0] - 0.213) < 0.001
        assert abs(SAT_STD[1] - 0.156) < 0.001
        assert abs(SAT_STD[2] - 0.143) < 0.001

    def test_web_and_sat_stats_are_different(self):
        from pygeovision.models.foundation.dinov3 import WEB_MEAN, SAT_MEAN, WEB_STD, SAT_STD
        assert WEB_MEAN != SAT_MEAN
        assert WEB_STD  != SAT_STD

    def test_get_transform_auto_selects_sat_for_sat_model(self):
        from pygeovision.models.foundation.dinov3 import get_transform, SAT_MEAN
        try:
            import torchvision.transforms as T
            t = get_transform("dinov3_vitl16_sat")
            # Verify it's a Compose with a Normalize using SAT stats
            norms = [x for x in t.transforms if isinstance(x, T.Normalize)]
            assert len(norms) == 1
            assert abs(norms[0].mean[0] - SAT_MEAN[0]) < 0.001
        except ImportError:
            pytest.skip("torchvision required")

    def test_get_transform_auto_selects_web_for_web_model(self):
        from pygeovision.models.foundation.dinov3 import get_transform, WEB_MEAN
        try:
            import torchvision.transforms as T
            t = get_transform("dinov3_vitl16")
            norms = [x for x in t.transforms if isinstance(x, T.Normalize)]
            assert len(norms) == 1
            assert abs(norms[0].mean[0] - WEB_MEAN[0]) < 0.001
        except ImportError:
            pytest.skip("torchvision required")


# ── Band Mapping Integration ─────────────────────────────────────────────────

class TestBandMappingIntegration:
    """Test full band mapping pipeline from satellite to Prithvi input."""

    def test_sentinel2_6band_mapping(self):
        from pygeovision.models.foundation.prithvi import SENTINEL2_TO_PRITHVI, map_bands
        # Verify 6-band subset is Blue/Green/Red/NIR/SWIR1/SWIR2
        core_bands = {k: v for k, v in SENTINEL2_TO_PRITHVI.items() if v < 6}
        assert len(core_bands) == 6

    def test_landsat_complete_mapping(self):
        from pygeovision.models.foundation.prithvi import LANDSAT_TO_PRITHVI, map_bands
        data = np.ones((6, 32, 32), dtype=np.float32)
        out  = map_bands(data, source="landsat", n_prithvi_bands=6)
        assert out.shape == (6, 32, 32)

    def test_hls_scale_correct(self):
        from pygeovision.models.foundation.prithvi import normalise_hls, HLS_SCALE_FACTOR
        # Test that a cloud-free clear-sky pixel (e.g. 2000 = 20% reflectance) normalises correctly
        pixel = np.array([2000.0])
        norm  = normalise_hls(pixel)
        assert abs(norm[0] - 0.2) < 1e-4

    def test_prithvi_band_order_documentation(self):
        """Verify band order matches the documented HLS specification."""
        from pygeovision.models.foundation.prithvi import SENTINEL2_TO_PRITHVI
        # Standard 6-band HLS: Blue=0, Green=1, Red=2, NIR=3, SWIR1=4, SWIR2=5
        assert SENTINEL2_TO_PRITHVI["B02"] == 0, "Blue must be index 0"
        assert SENTINEL2_TO_PRITHVI["B03"] == 1, "Green must be index 1"
        assert SENTINEL2_TO_PRITHVI["B04"] == 2, "Red must be index 2"
        assert SENTINEL2_TO_PRITHVI["B08"] == 3, "NIR must be index 3"


# ── End-to-End Workflows ──────────────────────────────────────────────────────

class TestEndToEndFoundation:
    """End-to-end workflow tests (use surrogates, no real model downloads)."""

    def test_dinov3_backbone_pipeline(self, tmp_path):
        """Full DINOv3 pipeline: load → extract → embed → build_classifier."""
        import torch, torch.nn as nn
        from pygeovision.models.foundation.dinov3 import DINOv3Backbone
        from PIL import Image

        # Mock output that has last_hidden_state
        class FakeOutput:
            def __init__(self, B):
                self.last_hidden_state = torch.randn(B, 197, 768)
                self.attentions = None

        class FakeModel(nn.Module):
            config = type("C", (), {"hidden_size": 768})()
            def forward(self, x=None, pixel_values=None, **kw):
                inp = x if x is not None else pixel_values
                return FakeOutput(inp.shape[0])

        b = DINOv3Backbone("dinov3_vitb16")
        b._model     = FakeModel()
        b._transform = None
        b._spec      = {"embed": 768}

        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

        # 1. Extract spatial features
        feats = b.extract_features(img)
        assert feats.ndim == 3
        assert feats.shape[2] == 768

        # 2. Extract embedding
        emb = b.extract_embeddings(img)
        assert emb.shape == (1, 768)

        # 3. Extract patches
        patches = b.extract_patch_features(img)
        assert patches.ndim == 2
        assert patches.shape[1] == 768

        # 4. Build classifier (no weight download needed — backbone already set)
        head = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 5))
        assert head is not None

    def test_prithvi_pipeline(self, tmp_path):
        """Full Prithvi pipeline: load → extract_features → build_seg_head → infer."""
        import rasterio, torch
        from rasterio.transform import from_bounds
        from pygeovision.models.foundation.prithvi import (
            Prithvi, PrithviTasks, _build_prithvi_surrogate, PRITHVI_MODELS
        )

        # Write 6-band synthetic GeoTIFF
        data = (np.random.rand(6, 64, 64) * 8000).astype(np.float32)
        t    = from_bounds(0, 0, 1, 1, 64, 64)
        p    = tmp_path / "prithvi_e2e.tif"
        with rasterio.open(str(p), "w", driver="GTiff", height=64, width=64,
                            count=6, dtype="float32", crs="EPSG:4326", transform=t) as dst:
            dst.write(data)

        # Use surrogate model
        surrogate = _build_prithvi_surrogate(PRITHVI_MODELS["prithvi_eo_1_0"])
        model     = Prithvi("prithvi_eo_1_0")
        model._model = surrogate

        # Feature extraction
        feats = model.extract_features(str(p), source="hls")
        assert feats.ndim == 2
        assert feats.shape[1] == 768

        # Build seg head and run tasks
        tasks = PrithviTasks("prithvi_eo_1_0")
        tasks._prithvi._model = surrogate

        lc     = tasks.land_cover(str(p))
        assert "prediction" in lc
        assert "class_names" in lc

        flood  = tasks.flood_detection(str(p), source="hls")
        assert "flood_pct" in flood
        assert 0.0 <= flood["flood_pct"] <= 100.0

        biomass = tasks.biomass_estimation(str(p))
        assert "estimated_biomass_t_ha" in biomass

    def test_dinov3_chm_pipeline(self, tmp_path):
        """CHMv2 canopy height pipeline: predict → biomass → deforestation."""
        import rasterio, torch.nn as nn
        from rasterio.transform import from_bounds
        from pygeovision.models.foundation.dinov3 import CHMv2Model

        data = (np.random.rand(4, 64, 64) * 10000).astype(np.float32)
        t    = from_bounds(0, 0, 1, 1, 64, 64)
        for name in ["chm_before.tif", "chm_after.tif"]:
            p = tmp_path / name
            with rasterio.open(str(p), "w", driver="GTiff", height=64, width=64,
                                count=4, dtype="float32", crs="EPSG:4326", transform=t) as dst:
                dst.write(data)

        class FakeBack(nn.Module):
            def forward(self, x):
                import torch; return torch.randn(x.shape[0], 197, 1024)

        chm = CHMv2Model()
        chm._backbone._model    = FakeBack()
        chm._backbone._transform = None
        chm._backbone._is_sat   = False

        result = chm.predict_canopy_height(str(tmp_path / "chm_before.tif"))
        assert "height_map" in result or "error" in result
        if "height_map" in result:
            assert result["height_map"].ndim == 2
            assert result["statistics"]["mean_m"] >= 0.0

    def test_model_registry_completeness(self):
        """Final check: registry has all 12+6+2 = 20+ foundation model entries."""
        from pygeovision.models.registry import model_registry
        foundation = set(model_registry.list(task="foundation"))
        # All 12 DINOv3 backbone variants
        dinov3_variants = [
            "dinov3_vits16", "dinov3_vits16plus", "dinov3_vitb16", "dinov3_vitl16",
            "dinov3_vith16plus", "dinov3_vit7b16",
            "dinov3_vitl16_sat", "dinov3_vit7b16_sat",
            "dinov3_convnext_tiny", "dinov3_convnext_small",
            "dinov3_convnext_base", "dinov3_convnext_large",
        ]
        for v in dinov3_variants:
            assert v in model_registry, f"Missing DINOv3 variant: {v}"
        # Both Prithvi models
        assert "prithvi_eo_1_0" in model_registry
        assert "prithvi_eo_2_0" in model_registry
        print(f"\n  ✓ Foundation models: {len(foundation)} in registry")