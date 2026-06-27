"""Tests for PyGeoVision auto-labeling layer (Phase 2+)."""
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile, os


# ── OSM Labeler ───────────────────────────────────────────────────────────────

class TestOSMLabeler:
    def test_list_categories(self):
        from pygeovision.labeling.osm import OSMLabeler, OSM_CATEGORIES
        labeler = OSMLabeler()
        cats = labeler.list_categories()
        assert isinstance(cats, dict)
        assert len(cats) == len(OSM_CATEGORIES)
        assert "buildings" in cats
        assert "water" in cats
        assert "roads" in cats

    def test_preview_query(self):
        from pygeovision.labeling.osm import OSMLabeler
        labeler = OSMLabeler()
        q = labeler.preview_query((-74.05, 40.70, -73.95, 40.80), ["buildings"])
        assert "building" in q
        assert "out body geom" in q
        assert "-74.05" in q and "40.7" in q  # lat-min,lon-min Overpass format

    def test_build_overpass_query(self):
        from pygeovision.labeling.osm import OSMLabeler
        labeler = OSMLabeler()
        q = labeler._build_overpass_query((-74.05, 40.70, -73.95, 40.80), ["buildings", "water"])
        assert "[out:json]" in q
        assert "building" in q
        assert "water" in q

    def test_osm_to_geojson_empty(self):
        from pygeovision.labeling.osm import OSMLabeler
        labeler = OSMLabeler()
        geojson = labeler._osm_to_geojson({"elements": []}, ["buildings"])
        assert geojson["type"] == "FeatureCollection"
        assert geojson["features"] == []

    def test_osm_to_geojson_with_buildings(self):
        from pygeovision.labeling.osm import OSMLabeler
        labeler = OSMLabeler()
        data = {"elements": [{
            "type": "way", "id": 1,
            "tags": {"building": "yes"},
            "geometry": [
                {"lon": -74.0, "lat": 40.7}, {"lon": -74.01, "lat": 40.7},
                {"lon": -74.01, "lat": 40.71}, {"lon": -74.0, "lat": 40.71},
                {"lon": -74.0, "lat": 40.7},
            ]
        }]}
        geojson = labeler._osm_to_geojson(data, ["buildings"])
        assert len(geojson["features"]) == 1
        f = geojson["features"][0]
        assert f["properties"]["category"] == "buildings"
        assert f["properties"]["label_value"] == 1

    def test_label_network_failure_returns_error(self):
        from pygeovision.labeling.osm import OSMLabeler
        labeler = OSMLabeler(retry_attempts=1)
        with patch.object(labeler, "_fetch_overpass", return_value=None):
            result = labeler.label((-74.05, 40.70, -73.95, 40.80), ["buildings"])
        assert result["success"] is False
        assert "error" in result

    @pytest.mark.skipif(not pytest.importorskip("rasterio", reason="rasterio not installed"), reason="needs rasterio")
    def test_rasterise_empty_geojson(self, tmp_path):
        from pygeovision.labeling.osm import OSMLabeler
        labeler = OSMLabeler()
        geojson = {"type": "FeatureCollection", "features": []}
        out = tmp_path / "test.tif"
        result = labeler._rasterise(geojson, (-74.05, 40.70, -73.95, 40.80), out, 1000.0)
        assert result == out
        assert out.exists()


# ── Label Quality Assessor ────────────────────────────────────────────────────

class TestLabelQualityAssessor:
    def test_recommendations_empty(self):
        from pygeovision.labeling.quality import LabelQualityAssessor
        qa = LabelQualityAssessor()
        results = {"quality_score": 0.9, "quality_grade": "A", "checks": {}}
        recs = qa._recommendations(results)
        assert isinstance(recs, list)

    def test_report_html_structure(self):
        from pygeovision.labeling.quality import LabelQualityAssessor
        qa = LabelQualityAssessor()
        results = {
            "label_path": "test.tif",
            "quality_score": 0.85,
            "quality_grade": "B",
            "checks": {"class_balance": {"status": "ok", "score": 0.9}},
            "recommendations": ["Test recommendation"],
        }
        html = qa.report_html(results)
        assert "<html>" in html
        assert "quality_grade" in html.lower() or "B" in html
        assert "Test recommendation" in html

    @pytest.mark.skipif(not pytest.importorskip("rasterio", reason="rasterio"), reason="needs rasterio")
    def test_assess_synthetic_label(self, tmp_path):
        import numpy as np, rasterio
        from rasterio.transform import from_bounds
        from pygeovision.labeling.quality import LabelQualityAssessor

        # Create a synthetic label raster
        label = np.zeros((100, 100), dtype=np.uint8)
        label[20:50, 20:50] = 1   # class 1 patch
        label[60:80, 60:80] = 1   # another patch
        transform = from_bounds(0, 0, 1, 1, 100, 100)
        p = tmp_path / "label.tif"
        with rasterio.open(str(p), "w", driver="GTiff", height=100, width=100,
                            count=1, dtype="uint8", crs="EPSG:4326", transform=transform) as dst:
            dst.write(label[np.newaxis])

        qa = LabelQualityAssessor(num_classes=2)
        result = qa.assess(str(p), checks=["class_balance", "coverage"])
        assert "quality_score" in result
        assert 0.0 <= result["quality_score"] <= 1.0
        assert "quality_grade" in result
        assert "class_balance" in result.get("checks", {})


# ── Active Learner ────────────────────────────────────────────────────────────

class TestActiveLearner:
    def test_init_valid_strategies(self):
        from pygeovision.labeling.active import ActiveLearner
        for strategy in ["entropy", "least_confidence", "margin", "coreset", "committee", "random"]:
            learner = ActiveLearner(strategy=strategy)
            assert learner.strategy == strategy

    def test_init_invalid_strategy(self):
        from pygeovision.labeling.active import ActiveLearner
        with pytest.raises(ValueError, match="strategy"):
            ActiveLearner(strategy="invalid_xyz")

    def test_update_adds_samples(self):
        from pygeovision.labeling.active import ActiveLearner
        learner = ActiveLearner()
        samples = [{"path": f"img{i}.tif", "label": i % 2} for i in range(5)]
        learner.update(samples)
        assert len(learner._labeled) == 5
        assert learner._iteration == 1

    def test_random_selection(self):
        from pygeovision.labeling.active import ActiveLearner
        learner = ActiveLearner(strategy="random", seed=42)
        pool = [{"path": f"img{i}.tif"} for i in range(20)]
        selected = learner.select(None, pool, n_select=5)
        assert len(selected) == 5

    def test_history_empty_initially(self):
        from pygeovision.labeling.active import ActiveLearner
        learner = ActiveLearner()
        assert learner.history == []

    def test_train_iteration_returns_dict(self):
        from pygeovision.labeling.active import ActiveLearner
        learner = ActiveLearner(strategy="random")
        pool = [{"path": f"img{i}.tif"} for i in range(10)]
        result = learner.train_iteration(None, None, pool)
        assert "n_selected" in result
        assert "selected_samples" in result
        assert result["strategy"] == "random"


# ── AutoLabelPipeline ────────────────────────────────────────────────────────

class TestAutoLabelPipeline:
    def test_init_defaults(self):
        from pygeovision.labeling.pipeline import AutoLabelPipeline
        pipeline = AutoLabelPipeline()
        assert "osm" in pipeline.sources
        assert pipeline.fusion in ("majority_vote", "union", "intersection", "priority")

    def test_init_custom_sources(self):
        from pygeovision.labeling.pipeline import AutoLabelPipeline
        pipeline = AutoLabelPipeline(sources=["osm", "esa_worldcover"])
        assert pipeline.sources == ["osm", "esa_worldcover"]

    def test_run_returns_dict_on_network_failure(self, tmp_path):
        from pygeovision.labeling.pipeline import AutoLabelPipeline
        pipeline = AutoLabelPipeline(sources=["osm"])
        with patch("pygeovision.labeling.osm.OSMLabeler.label",
                   return_value={"success": False, "error": "Network"}):
            result = pipeline.run((-74.05, 40.70, -73.95, 40.80),
                                   output_dir=str(tmp_path))
        assert "sources_succeeded" in result
        assert "sources_failed" in result

    def test_supported_sources_list(self):
        from pygeovision.labeling.pipeline import AutoLabelPipeline
        assert "osm" in AutoLabelPipeline.SOURCES
        assert "microsoft_buildings" in AutoLabelPipeline.SOURCES
        assert "esa_worldcover" in AutoLabelPipeline.SOURCES
        assert "sam_auto" in AutoLabelPipeline.SOURCES


# ── Foundation Model Labeler ──────────────────────────────────────────────────

class TestFoundationModelLabeler:
    def test_init(self):
        from pygeovision.labeling.foundation import FoundationModelLabeler
        lab = FoundationModelLabeler(model="dinov2-base")
        assert lab.model_name == "dinov2-base"

    def test_hf_model_mapping(self):
        # Verify all model names map to HF IDs (via FewShotLearner which shares mapping)
        from pygeovision.advanced.few_shot import FewShotLearner
        for name in ["dinov2-small", "dinov2-base", "dinov2-large"]:
            learner = FewShotLearner(backbone=name)
            assert learner is not None  # FewShotLearner init succeeds
