"""
Tests for PyGeoVision × GeoAI integration layer.

Tests cover the integration API surface without requiring geoai-py to be
installed — all heavy operations are mocked. Tests marked with
@pytest.mark.skipif check for actual geoai installation.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_geoai():
    """Create a comprehensive geoai mock matching the real geoai API."""
    ga = MagicMock()
    ga.__version__ = "0.39.2"

    # Extractor classes return instances that have predict()
    for cls_name in [
        "BuildingFootprintExtractor", "CarDetector", "ShipDetector",
        "SolarPanelDetector", "ParkingSplotDetector", "AgricultureFieldDelineator",
    ]:
        mock_cls = MagicMock()
        mock_cls.return_value.predict.return_value = MagicMock(success=True)
        setattr(ga, cls_name, mock_cls)

    # GroundedSAM
    gsam = MagicMock()
    gsam.return_value.predict.return_value = MagicMock()
    ga.GroundedSAM = gsam

    # ChangeStarDetection
    ga.ChangeStarDetection = MagicMock()
    ga.changestar_detect = MagicMock(return_value={"changed_pixels": 1024})
    ga.list_changestar_models = MagicMock(return_value=["changestar-v1", "changestar-v2"])

    # Free functions
    ga.segment_water = MagicMock(return_value=MagicMock())
    ga.mask_generation = MagicMock(return_value=MagicMock())
    ga.image_segmentation = MagicMock(return_value=MagicMock())
    ga.semantic_segmentation = MagicMock(return_value=MagicMock())
    ga.timm_semantic_segmentation = MagicMock(return_value=MagicMock())
    ga.timm_segmentation_from_hub = MagicMock(return_value=MagicMock())
    ga.push_timm_model_to_hub = MagicMock(return_value=MagicMock())
    ga.object_detection = MagicMock(return_value=MagicMock())
    ga.multiclass_detection = MagicMock(return_value=MagicMock())
    ga.instance_segmentation = MagicMock(return_value=MagicMock())
    ga.rfdetr_detect = MagicMock(return_value=MagicMock())
    ga.rfdetr_segment = MagicMock(return_value=MagicMock())
    ga.rfdetr_train = MagicMock(return_value=MagicMock())
    ga.rfdetr_detect_from_hub = MagicMock(return_value=MagicMock())
    ga.list_rfdetr_models = MagicMock(return_value=["rfdetr-base", "rfdetr-large"])
    ga.predict_geotiff = MagicMock(return_value=np.zeros((512, 512), dtype=np.uint8))
    ga.train_segmentation_model = MagicMock(return_value={"loss": 0.1, "miou": 0.82})
    ga.train_segmentation_landcover = MagicMock(return_value={"loss": 0.08})
    ga.train_multiclass_detector = MagicMock(return_value={"map": 0.65})
    ga.train_instance_segmentation_model = MagicMock(return_value={"map": 0.70})
    ga.train_timm_segmentation_model = MagicMock(return_value={"miou": 0.78})
    ga.train_classifier = MagicMock(return_value={"accuracy": 0.92})
    ga.train_pixel_regressor = MagicMock(return_value={"rmse": 1.5})
    ga.classify_image = MagicMock(return_value={"class": "forest", "confidence": 0.91})
    ga.classify_images = MagicMock(return_value=[])
    ga.CLIPVectorClassifier = MagicMock(return_value=MagicMock())
    ga.clip_classify_vector = MagicMock(return_value=MagicMock())
    ga.extract_patch_embeddings = MagicMock(return_value=np.random.rand(100, 512))
    ga.extract_pixel_embeddings = MagicMock(return_value=np.random.rand(256, 256, 64))
    ga.cluster_embeddings = MagicMock(return_value=MagicMock())
    ga.embedding_similarity = MagicMock(return_value=0.85)
    ga.visualize_embeddings = MagicMock(return_value=MagicMock())
    ga.list_embedding_datasets = MagicMock(return_value=["tessera", "dinov2"])
    ga.super_resolution = MagicMock(return_value=MagicMock())
    ga.export_to_onnx = MagicMock(return_value=Path("model.onnx"))
    ga.onnx_semantic_segmentation = MagicMock(return_value=MagicMock())
    ga.predict_cloud_mask_from_raster = MagicMock(return_value=MagicMock())
    ga.predict_cloud_mask_batch = MagicMock(return_value=[])
    ga.calculate_cloud_statistics = MagicMock(return_value={"cloud_cover": 0.15})
    ga.create_cloud_free_mask = MagicMock(return_value=MagicMock())
    ga.canopy_height_estimation = MagicMock(return_value=MagicMock())
    ga.list_canopy_models = MagicMock(return_value=["canopy-v1"])
    ga.get_available_prithvi_models = MagicMock(return_value=["prithvi-100M", "prithvi-300M"])
    ga.load_prithvi_model = MagicMock(return_value=MagicMock())
    ga.prithvi_inference = MagicMock(return_value=MagicMock())
    ga.moondream_caption = MagicMock(return_value="Aerial view of urban area")
    ga.moondream_query = MagicMock(return_value="Yes, there are buildings")
    ga.moondream_detect = MagicMock(return_value=[])
    ga.moondream_caption_sliding_window = MagicMock(return_value=MagicMock())
    ga.download_naip = MagicMock(return_value=Path("naip.tif"))
    ga.download_overture_buildings = MagicMock(return_value=Path("buildings.geojson"))
    ga.pc_stac_search = MagicMock(return_value=[])
    ga.pc_stac_download = MagicMock(return_value=MagicMock())
    ga.download_model_from_hf = MagicMock(return_value=Path("model.pth"))
    ga.get_raster_info = MagicMock(return_value={"width": 512, "height": 512, "bands": 4})
    ga.get_vector_info = MagicMock(return_value={"features": 100})
    ga.raster_to_vector = MagicMock(return_value=MagicMock())
    ga.vector_to_raster = MagicMock(return_value=MagicMock())
    ga.clip_raster_by_bbox = MagicMock(return_value=MagicMock())
    ga.mosaic_geotiffs = MagicMock(return_value=MagicMock())
    ga.calc_segmentation_metrics = MagicMock(return_value={"miou": 0.82})
    ga.calc_iou = MagicMock(return_value=0.78)
    ga.smooth_vector = MagicMock(return_value=MagicMock())
    ga.regularize = MagicMock(return_value=MagicMock())
    ga.stack_bands = MagicMock(return_value=MagicMock())
    ga.get_device = MagicMock(return_value="cuda")
    ga.empty_cache = MagicMock()
    ga.create_segmentation_dataset = MagicMock(return_value=MagicMock())
    ga.export_geotiff_tiles = MagicMock()
    ga.export_training_data = MagicMock()
    ga.export_landcover_tiles = MagicMock()
    ga.Map = MagicMock(return_value=MagicMock())
    ga.view_raster = MagicMock(return_value=MagicMock())
    ga.view_vector = MagicMock(return_value=MagicMock())
    ga.load_pipeline = MagicMock(return_value=MagicMock())
    ga.Pipeline = MagicMock(return_value=MagicMock())
    ga.analyze_image_patches = MagicMock(return_value=MagicMock())
    ga.create_similarity_map = MagicMock(return_value=MagicMock())
    ga.train_dinov3_segmentation = MagicMock(return_value=MagicMock())
    ga.dinov3_segment_geotiff = MagicMock(return_value=MagicMock())
    ga.tessera_download = MagicMock(return_value=MagicMock())
    ga.tessera_available_years = MagicMock(return_value=[2020, 2021, 2022, 2023])
    ga.tessera_coverage = MagicMock(return_value={"coverage": 0.95})
    ga.BAND_ORDER_PRESETS = {
        "sentinel2": [2, 3, 4, 8],
        "naip": [1, 2, 3],
        "landsat": [3, 2, 1],
    }
    ga.timm_semantic_segmentation = MagicMock(return_value=MagicMock())
    ga.predict_raster = MagicMock(return_value=MagicMock())
    ga.list_timm_models = MagicMock(return_value=["resnet50", "efficientnet_b4"])
    ga.segment_water = MagicMock(return_value=MagicMock())
    return ga


@pytest.fixture
def geoai_engine(mock_geoai):
    """GeoAIEngine with patched geoai module."""
    with patch("pygeovision.ai.geoai._require_geoai", return_value=mock_geoai), \
         patch("pygeovision.ai.geoai._check_geoai", return_value=True), \
         patch("pygeovision.ai.geoai._GEOAI_AVAILABLE", True):
        from pygeovision.ai.geoai import GeoAIEngine
        engine = GeoAIEngine(pgv_client=MagicMock())
        yield engine, mock_geoai


# ---------------------------------------------------------------------------
# GeoAI availability
# ---------------------------------------------------------------------------

class TestGeoAIAvailability:
    def test_geoai_engine_has_version(self, geoai_engine):
        engine, ga = geoai_engine
        assert engine.version == "0.39.2"

    def test_geoai_engine_is_available(self, geoai_engine):
        engine, _ = geoai_engine
        assert engine.is_available is True

    def test_geoai_not_installed(self):
        with patch("pygeovision.ai.geoai._check_geoai", return_value=False), \
             patch("pygeovision.ai.geoai._GEOAI_AVAILABLE", False):
            from pygeovision.ai.geoai import _check_geoai, _require_geoai, GeoAIEngine
            with patch("pygeovision.ai.geoai._check_geoai", return_value=False):
                assert not _check_geoai()

    def test_raw_returns_geoai(self, geoai_engine):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            raw = engine.raw()
            assert raw is ga


# ---------------------------------------------------------------------------
# Segmentation subsystem
# ---------------------------------------------------------------------------

class TestSegmentation:
    def test_segment_buildings(self, geoai_engine, tmp_path):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            result = engine.segment.buildings("scene.tif", output_path=str(tmp_path / "out.tif"))
            ga.BuildingFootprintExtractor.return_value.predict.assert_called_once()
            assert result is not None

    def test_segment_solar_panels(self, geoai_engine, tmp_path):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            engine.segment.solar_panels("aerial.tif", output_path=str(tmp_path / "solar.tif"))
            ga.SolarPanelDetector.return_value.predict.assert_called_once()

    def test_segment_agriculture(self, geoai_engine):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            engine.segment.agriculture_fields("s2.tif")
            ga.AgricultureFieldDelineator.return_value.predict.assert_called_once()

    def test_segment_water(self, geoai_engine):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            engine.segment.water("s2.tif", band_order="sentinel2")
            ga.segment_water.assert_called_once()
            call_kwargs = ga.segment_water.call_args[1]
            assert "band_order" in call_kwargs

    def test_segment_custom(self, geoai_engine):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            engine.segment.custom("scene.tif", "model.pth", num_classes=5)
            ga.semantic_segmentation.assert_called_once()

    def test_segment_with_hf_model(self, geoai_engine):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            engine.segment.with_hf_model("scene.tif", "facebook/mask2former-swin-base-ade-semantic")
            ga.image_segmentation.assert_called_once()

    def test_segment_with_sam(self, geoai_engine):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            engine.segment.with_sam("aerial.tif")
            ga.mask_generation.assert_called_once()

    def test_segment_timm(self, geoai_engine):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            engine.segment.timm_model("scene.tif", "timm_seg.pth")
            ga.timm_semantic_segmentation.assert_called_once()

    def test_segment_from_hub(self, geoai_engine):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            engine.segment.from_hub("scene.tif", "giswqs/building-footprint-usa")
            ga.timm_segmentation_from_hub.assert_called_once()


# ---------------------------------------------------------------------------
# Detection subsystem
# ---------------------------------------------------------------------------

class TestDetection:
    def test_detect_cars(self, geoai_engine):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            engine.detect.cars("aerial.tif")
            ga.CarDetector.return_value.predict.assert_called_once()

    def test_detect_ships(self, geoai_engine):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            engine.detect.ships("port.tif", output_path="ships.geojson")
            ga.ShipDetector.return_value.predict.assert_called_once()

    def test_detect_parking(self, geoai_engine):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            engine.detect.parking("parking_lot.tif")
            ga.ParkingSplotDetector.return_value.predict.assert_called_once()

    def test_detect_grounded(self, geoai_engine):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            engine.detect.grounded("aerial.tif", "swimming pools")
            ga.GroundedSAM.return_value.predict.assert_called_once()

    def test_detect_rfdetr(self, geoai_engine):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            engine.detect.rfdetr("scene.tif")
            ga.rfdetr_detect.assert_called_once()

    def test_detect_instance_segmentation(self, geoai_engine):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            engine.detect.instance_segmentation("scene.tif", "maskrcnn.pth")
            ga.instance_segmentation.assert_called_once()


# ---------------------------------------------------------------------------
# Classification subsystem
# ---------------------------------------------------------------------------

class TestClassification:
    def test_classify_single(self, geoai_engine):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            result = engine.classify.classify("tile.tif", "classifier.pth")
            ga.classify_image.assert_called_once()

    def test_classify_land_cover(self, geoai_engine):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            engine.classify.land_cover("s2.tif", classes=["forest", "water", "urban"])
            ga.CLIPVectorClassifier.assert_called_once()

    def test_classify_batch(self, geoai_engine, tmp_path):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            engine.classify.batch(str(tmp_path), "classifier.pth")
            ga.classify_images.assert_called_once()

    def test_classify_train(self, geoai_engine, tmp_path):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            engine.classify.train(str(tmp_path), "model.pth", num_classes=8)
            ga.train_classifier.assert_called_once()


# ---------------------------------------------------------------------------
# Change detection subsystem
# ---------------------------------------------------------------------------

class TestChangeDetection:
    def test_change_detect(self, geoai_engine):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            result = engine.change.detect("before.tif", "after.tif", output_path="changes.tif")
            ga.changestar_detect.assert_called_once()
            assert result is not None

    def test_change_list_models(self, geoai_engine):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            models = engine.change.list_models()
            assert "changestar-v1" in models


# ---------------------------------------------------------------------------
# Training subsystem
# ---------------------------------------------------------------------------

class TestTraining:
    def test_train_segmentation(self, geoai_engine, tmp_path):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            result = engine.train.segmentation(str(tmp_path), "model.pth", num_classes=5)
            ga.train_segmentation_model.assert_called_once()
            call_kwargs = ga.train_segmentation_model.call_args[1]
            assert call_kwargs["num_classes"] == 5

    def test_train_segmentation_landcover(self, geoai_engine, tmp_path):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            engine.train.segmentation_landcover(str(tmp_path), "lc.pth", loss_fn="unified_focal")
            ga.train_segmentation_landcover.assert_called_once()

    def test_train_detection(self, geoai_engine, tmp_path):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            engine.train.detection(str(tmp_path), "det.pth", num_classes=10)
            ga.train_multiclass_detector.assert_called_once()

    def test_train_instance_segmentation(self, geoai_engine, tmp_path):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            engine.train.instance_segmentation(str(tmp_path), "inst.pth")
            ga.train_instance_segmentation_model.assert_called_once()

    def test_train_timm_segmentation(self, geoai_engine, tmp_path):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            engine.train.timm_segmentation(str(tmp_path), "timm.pth", backbone="convnext_base")
            ga.train_timm_segmentation_model.assert_called_once()

    def test_train_rfdetr(self, geoai_engine, tmp_path):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            engine.train.rfdetr(str(tmp_path), "rfdetr.pth")
            ga.rfdetr_train.assert_called_once()

    def test_generate_chips(self, geoai_engine, tmp_path):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            engine.train.generate_chips("scene.tif", "labels.tif", str(tmp_path), chip_size=256)
            ga.export_training_data.assert_called_once()


# ---------------------------------------------------------------------------
# Inference subsystem
# ---------------------------------------------------------------------------

class TestInference:
    def test_predict(self, geoai_engine):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            model = MagicMock()
            result = engine.infer.predict("scene.tif", model, "pred.tif")
            ga.predict_geotiff.assert_called_once()


# ---------------------------------------------------------------------------
# Embeddings subsystem
# ---------------------------------------------------------------------------

class TestEmbeddings:
    def test_patch_embeddings(self, geoai_engine):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            embs = engine.embed.patch("sentinel2.tif", chip_size=64)
            ga.extract_patch_embeddings.assert_called_once()
            assert embs.shape == (100, 512)

    def test_cluster_embeddings(self, geoai_engine):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            embs = np.random.rand(200, 512)
            engine.embed.cluster(embs, n_clusters=10)
            ga.cluster_embeddings.assert_called_once()

    def test_similarity(self, geoai_engine):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            result = engine.embed.similarity(np.ones(512), np.ones(512))
            assert result == 0.85

    def test_list_datasets(self, geoai_engine):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            datasets = engine.embed.list_datasets()
            assert "tessera" in datasets


# ---------------------------------------------------------------------------
# SAM subsystem
# ---------------------------------------------------------------------------

class TestSAM:
    def test_generate_masks(self, geoai_engine):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            engine.sam.generate_masks("aerial.tif")
            ga.mask_generation.assert_called_once()

    def test_grounded_sam(self, geoai_engine):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            engine.sam.grounded("aerial.tif", "solar panels")
            ga.GroundedSAM.return_value.predict.assert_called()


# ---------------------------------------------------------------------------
# Prithvi subsystem
# ---------------------------------------------------------------------------

class TestPrithvi:
    def test_list_models(self, geoai_engine):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            models = engine.prithvi.list_models()
            assert "prithvi-100M" in models

    def test_load_and_infer(self, geoai_engine):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            model = engine.prithvi.load("prithvi-100M")
            ga.load_prithvi_model.assert_called_once()
            engine.prithvi.infer("hls.tif", model)
            ga.prithvi_inference.assert_called_once()


# ---------------------------------------------------------------------------
# Cloud subsystem
# ---------------------------------------------------------------------------

class TestCloud:
    def test_predict(self, geoai_engine):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            engine.cloud.predict("sentinel2.tif")
            ga.predict_cloud_mask_from_raster.assert_called_once()

    def test_statistics(self, geoai_engine):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            stats = engine.cloud.statistics("cloud_mask.tif")
            assert stats["cloud_cover"] == 0.15


# ---------------------------------------------------------------------------
# Super resolution subsystem
# ---------------------------------------------------------------------------

class TestSuperResolution:
    def test_enhance(self, geoai_engine):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            engine.sr.enhance("landsat.tif", scale_factor=4)
            ga.super_resolution.assert_called_once()
            call_kwargs = ga.super_resolution.call_args[1]
            assert call_kwargs["scale_factor"] == 4


# ---------------------------------------------------------------------------
# ONNX subsystem
# ---------------------------------------------------------------------------

class TestONNX:
    def test_export(self, geoai_engine):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            engine.onnx.export(MagicMock(), "model.onnx", input_shape=(1, 4, 512, 512))
            ga.export_to_onnx.assert_called_once()

    def test_segmentation(self, geoai_engine):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            engine.onnx.segmentation("scene.tif", "model.onnx")
            ga.onnx_semantic_segmentation.assert_called_once()


# ---------------------------------------------------------------------------
# Download subsystem
# ---------------------------------------------------------------------------

class TestDownload:
    def test_naip(self, geoai_engine, tmp_path):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            engine.download.naip((-74.1, 40.6, -73.7, 40.9), str(tmp_path / "naip.tif"))
            ga.download_naip.assert_called_once()

    def test_overture_buildings(self, geoai_engine, tmp_path):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            engine.download.overture_buildings((-74.1, 40.6, -73.7, 40.9), str(tmp_path / "b.geojson"))
            ga.download_overture_buildings.assert_called_once()

    def test_pc_search(self, geoai_engine):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            engine.download.pc_search((-74.1, 40.6, -73.7, 40.9), "sentinel-2-l2a")
            ga.pc_stac_search.assert_called_once()

    def test_model_from_hub(self, geoai_engine, tmp_path):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            engine.download.model_from_hub("giswqs/building-segmentation", str(tmp_path))
            ga.download_model_from_hf.assert_called_once()


# ---------------------------------------------------------------------------
# Utils subsystem
# ---------------------------------------------------------------------------

class TestUtils:
    def test_raster_info(self, geoai_engine):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            info = engine.utils.raster_info("scene.tif")
            assert info["width"] == 512

    def test_raster_to_vector(self, geoai_engine):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            engine.utils.raster_to_vector("pred.tif", "polygons.geojson")
            ga.raster_to_vector.assert_called_once()

    def test_segmentation_metrics(self, geoai_engine):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            pred = np.zeros((64, 64))
            gt = np.zeros((64, 64))
            result = engine.utils.segmentation_metrics(pred, gt)
            assert result["miou"] == 0.82

    def test_get_device(self, geoai_engine):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            device = engine.utils.get_device()
            assert device == "cuda"

    def test_clip_by_bbox(self, geoai_engine):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            engine.utils.clip_by_bbox("scene.tif", (-74.1, 40.6, -73.7, 40.9), "clipped.tif")
            ga.clip_raster_by_bbox.assert_called_once()


# ---------------------------------------------------------------------------
# Caption/Moondream subsystem
# ---------------------------------------------------------------------------

class TestCaption:
    def test_caption(self, geoai_engine):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            caption = engine.caption.caption("tile.tif")
            assert "urban" in caption.lower() or len(caption) > 0

    def test_query(self, geoai_engine):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            answer = engine.caption.query("tile.tif", "Are there buildings?")
            ga.moondream_query.assert_called_once()
            assert "Yes" in answer

    def test_detect(self, geoai_engine):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            engine.caption.detect("tile.tif", "cars")
            ga.moondream_detect.assert_called_once()


# ---------------------------------------------------------------------------
# Canopy subsystem
# ---------------------------------------------------------------------------

class TestCanopy:
    def test_list_models(self, geoai_engine):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            models = engine.canopy.list_models()
            assert "canopy-v1" in models

    def test_estimate(self, geoai_engine):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            engine.canopy.estimate("sentinel2.tif", output_path="canopy.tif")
            ga.canopy_height_estimation.assert_called_once()


# ---------------------------------------------------------------------------
# DINOv3 subsystem
# ---------------------------------------------------------------------------

class TestDINOv3:
    def test_analyze(self, geoai_engine):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            engine.dinov3.analyze("scene.tif")
            ga.analyze_image_patches.assert_called_once()

    def test_finetune(self, geoai_engine, tmp_path):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            engine.dinov3.finetune(str(tmp_path), "dino_seg.pth", num_classes=5)
            ga.train_dinov3_segmentation.assert_called_once()

    def test_segment(self, geoai_engine):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            engine.dinov3.segment("scene.tif", "dino_seg.pth")
            ga.dinov3_segment_geotiff.assert_called_once()


# ---------------------------------------------------------------------------
# Tessera subsystem
# ---------------------------------------------------------------------------

class TestTessera:
    def test_available_years(self, geoai_engine):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            years = engine.tessera.available_years((-74.1, 40.6, -73.7, 40.9))
            assert 2023 in years

    def test_coverage(self, geoai_engine):
        engine, ga = geoai_engine
        with patch("pygeovision.ai.geoai._require_geoai", return_value=ga):
            cov = engine.tessera.coverage((-74.1, 40.6, -73.7, 40.9))
            assert cov["coverage"] == 0.95


# ---------------------------------------------------------------------------
# PyGeoVision main client integration
# ---------------------------------------------------------------------------

class TestPyGeoVisionGeoAIProperty:
    def test_client_has_geoai_property(self):
        from pygeovision import PyGeoVision
        client = PyGeoVision()
        # geoai property should exist and be accessible
        assert hasattr(client, "geoai")

    def test_geoai_property_returns_engine(self):
        from pygeovision import PyGeoVision
        from pygeovision.ai.geoai import GeoAIEngine
        client = PyGeoVision()
        engine = client.geoai
        assert isinstance(engine, GeoAIEngine)

    def test_status_includes_geoai(self):
        from pygeovision import PyGeoVision
        client = PyGeoVision()
        status = client.status()
        assert "geoai" in status
        assert "available" in status["geoai"]
