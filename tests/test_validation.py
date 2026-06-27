"""Tests for inference validation and vectorization."""
from __future__ import annotations
import numpy as np
import pytest
from pathlib import Path


class TestValidationReport:
    def test_report_creation(self):
        from pygeovision.ai.inference.validation import ValidationReport
        r = ValidationReport(
            metrics={"mean_iou": 0.82, "accuracy": 0.94},
            per_class_iou=[0.90, 0.74],
            class_names=["background", "building"],
        )
        assert r.metrics["mean_iou"] == 0.82
        assert len(r.per_class_iou) == 2

    def test_report_summary(self):
        from pygeovision.ai.inference.validation import ValidationReport
        r = ValidationReport(
            metrics={"mean_iou": 0.75, "accuracy": 0.90},
            class_names=["bg", "tree"],
            per_class_iou=[0.85, 0.65],
        )
        summary = r.summary()
        assert "mean_iou" in summary
        assert "0.7500" in summary

    def test_report_to_dict(self):
        from pygeovision.ai.inference.validation import ValidationReport
        r = ValidationReport(metrics={"accuracy": 0.95}, per_class_iou=[0.95])
        d = r.to_dict()
        assert d["metrics"]["accuracy"] == 0.95
        assert d["per_class_iou"] == [0.95]

    def test_report_save(self, tmp_path):
        from pygeovision.ai.inference.validation import ValidationReport
        r = ValidationReport(metrics={"mean_iou": 0.80})
        path = r.save(tmp_path / "report.json")
        assert path.exists()
        import json
        data = json.loads(path.read_text())
        assert data["metrics"]["mean_iou"] == 0.80


class TestPredictionValidator:
    def test_init_defaults(self):
        from pygeovision.ai.inference.validation import PredictionValidator
        v = PredictionValidator(num_classes=5)
        assert v.num_classes == 5
        assert v.ignore_index == 255

    def test_validate_arrays_perfect(self):
        from pygeovision.ai.inference.validation import PredictionValidator
        v = PredictionValidator(num_classes=3)
        pred = np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int64)
        gt = np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int64)
        report = v.validate_arrays(pred, gt)
        assert report.metrics["accuracy"] == pytest.approx(1.0, abs=1e-6)
        assert report.metrics["mean_iou"] == pytest.approx(1.0, abs=1e-6)

    def test_validate_arrays_all_wrong(self):
        from pygeovision.ai.inference.validation import PredictionValidator
        v = PredictionValidator(num_classes=2)
        pred = np.ones((4, 4), dtype=np.int64)
        gt = np.zeros((4, 4), dtype=np.int64)
        report = v.validate_arrays(pred, gt)
        assert report.metrics["accuracy"] == pytest.approx(0.0, abs=1e-6)
        assert report.error_rate == pytest.approx(1.0, abs=1e-6)

    def test_validate_arrays_ignore_index(self):
        from pygeovision.ai.inference.validation import PredictionValidator
        v = PredictionValidator(num_classes=2, ignore_index=255)
        pred = np.array([[0, 1, 255]], dtype=np.int64)
        gt   = np.array([[0, 1, 255]], dtype=np.int64)
        report = v.validate_arrays(pred, gt)
        # 255 pixels should be ignored — only 2 valid pixels
        assert report.num_pixels == 2
        assert report.metrics["accuracy"] == pytest.approx(1.0, abs=1e-6)

    def test_validate_arrays_with_class_names(self):
        from pygeovision.ai.inference.validation import PredictionValidator
        v = PredictionValidator(num_classes=3, class_names=["bg", "tree", "water"])
        pred = np.zeros((8, 8), dtype=np.int64)
        gt   = np.zeros((8, 8), dtype=np.int64)
        report = v.validate_arrays(pred, gt)
        assert report.class_names == ["bg", "tree", "water"]

    def test_validate_arrays_no_valid_pixels(self):
        from pygeovision.ai.inference.validation import PredictionValidator
        v = PredictionValidator(num_classes=2, ignore_index=255)
        pred = np.full((4, 4), 255, dtype=np.int64)
        gt   = np.full((4, 4), 255, dtype=np.int64)
        report = v.validate_arrays(pred, gt)
        assert report.num_pixels == 0
        assert report.metrics.get("mean_iou", 0.0) == 0.0

    def test_validate_geotiff_pair(self, tmp_path):
        pytest.importorskip("rasterio")
        import rasterio
        from rasterio.transform import from_bounds
        from pygeovision.ai.inference.validation import PredictionValidator

        transform = from_bounds(0, 0, 1, 1, 32, 32)
        profile = {"driver": "GTiff", "dtype": "uint8", "count": 1,
                   "width": 32, "height": 32, "crs": "EPSG:4326", "transform": transform}

        pred_path = tmp_path / "pred.tif"
        gt_path   = tmp_path / "gt.tif"
        data = np.random.randint(0, 3, (1, 32, 32), dtype=np.uint8)

        for p in (pred_path, gt_path):
            with rasterio.open(p, "w", **profile) as dst:
                dst.write(data)   # identical → perfect score

        v = PredictionValidator(num_classes=3)
        report = v.validate(pred_path, gt_path)
        assert report.metrics["accuracy"] == pytest.approx(1.0, abs=1e-6)
        assert report.num_pixels == 32 * 32

    def test_validate_saves_error_map(self, tmp_path):
        pytest.importorskip("rasterio")
        import rasterio
        from rasterio.transform import from_bounds
        from pygeovision.ai.inference.validation import PredictionValidator

        transform = from_bounds(0, 0, 1, 1, 16, 16)
        profile = {"driver": "GTiff", "dtype": "uint8", "count": 1,
                   "width": 16, "height": 16, "crs": "EPSG:4326", "transform": transform}

        pred = np.zeros((1, 16, 16), dtype=np.uint8)
        gt   = np.ones((1, 16, 16), dtype=np.uint8)
        pred_p = tmp_path / "pred.tif"
        gt_p   = tmp_path / "gt.tif"
        with rasterio.open(pred_p, "w", **profile) as dst: dst.write(pred)
        with rasterio.open(gt_p,   "w", **profile) as dst: dst.write(gt)

        err_p = tmp_path / "error.tif"
        v = PredictionValidator(num_classes=2)
        v.validate(pred_p, gt_p, output_error_map=err_p)
        assert err_p.exists()

    def test_cross_validate(self, tmp_path):
        pytest.importorskip("rasterio")
        import rasterio
        from rasterio.transform import from_bounds
        from pygeovision.ai.inference.validation import PredictionValidator

        transform = from_bounds(0, 0, 1, 1, 16, 16)
        profile = {"driver": "GTiff", "dtype": "uint8", "count": 1,
                   "width": 16, "height": 16, "crs": "EPSG:4326", "transform": transform}

        preds, gts = [], []
        for i in range(3):
            data = np.random.randint(0, 2, (1, 16, 16), dtype=np.uint8)
            pp = tmp_path / f"pred_{i}.tif"
            gp = tmp_path / f"gt_{i}.tif"
            for p in (pp, gp):
                with rasterio.open(p, "w", **profile) as dst:
                    dst.write(data)
            preds.append(pp); gts.append(gp)

        v = PredictionValidator(num_classes=2)
        report = v.cross_validate(preds, gts)
        assert "mean_iou" in report.metrics
        assert report.num_pixels == 3 * 16 * 16


class TestVectorizer:
    def test_init_defaults(self):
        from pygeovision.ai.inference.vectorization import Vectorizer
        v = Vectorizer()
        assert v.config.output_crs == "EPSG:4326"
        assert v.config.simplify_tolerance == 1.0

    def test_array_to_geojson(self):
        pytest.importorskip("rasterio")
        pytest.importorskip("shapely")
        from pygeovision.ai.inference.vectorization import Vectorizer
        from rasterio.transform import from_bounds

        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[16:48, 16:48] = 1   # solid 32x32 square of class 1
        transform = from_bounds(0, 0, 1, 1, 64, 64)

        v = Vectorizer(simplify_tolerance=0.0, output_crs=None)
        result = v.array_to_geojson(mask, transform, crs="EPSG:4326",
                                    class_names=["bg", "building"])
        assert result["type"] == "FeatureCollection"
        assert len(result["features"]) >= 1
        feat = result["features"][0]
        assert feat["properties"]["class_id"] == 1
        assert feat["properties"]["class_name"] == "building"

    def test_array_to_geojson_with_class_filter(self):
        pytest.importorskip("rasterio")
        pytest.importorskip("shapely")
        from pygeovision.ai.inference.vectorization import Vectorizer
        from rasterio.transform import from_bounds

        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[0:16, 0:16] = 1
        mask[16:32, 16:32] = 2
        transform = from_bounds(0, 0, 1, 1, 32, 32)

        v = Vectorizer(simplify_tolerance=0.0, output_crs=None, class_filter=[1])
        result = v.array_to_geojson(mask, transform, crs="EPSG:4326")
        class_ids = {f["properties"]["class_id"] for f in result["features"]}
        assert 2 not in class_ids  # class 2 filtered out

    def test_raster_to_geojson(self, tmp_path):
        pytest.importorskip("rasterio")
        pytest.importorskip("shapely")
        import rasterio
        from rasterio.transform import from_bounds
        from pygeovision.ai.inference.vectorization import Vectorizer

        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[8:24, 8:24] = 1
        transform = from_bounds(0, 0, 1, 1, 32, 32)
        profile = {"driver": "GTiff", "dtype": "uint8", "count": 1,
                   "width": 32, "height": 32, "crs": "EPSG:4326", "transform": transform}
        mask_path = tmp_path / "mask.tif"
        with rasterio.open(mask_path, "w", **profile) as dst:
            dst.write(mask[np.newaxis, ...])

        v = Vectorizer(simplify_tolerance=0.0, output_crs=None)
        result = v.raster_to_geojson(mask_path, class_names=["bg", "building"])
        assert len(result["features"]) >= 1

    def test_geojson_to_mask(self, tmp_path):
        pytest.importorskip("rasterio")
        pytest.importorskip("shapely")
        import rasterio
        from rasterio.transform import from_bounds
        from pygeovision.ai.inference.vectorization import Vectorizer

        transform = from_bounds(0, 0, 1, 1, 32, 32)
        profile = {"driver": "GTiff", "dtype": "uint8", "count": 1,
                   "width": 32, "height": 32, "crs": "EPSG:4326", "transform": transform}
        ref_path = tmp_path / "ref.tif"
        with rasterio.open(ref_path, "w", **profile) as dst:
            dst.write(np.zeros((1, 32, 32), dtype=np.uint8))

        from shapely.geometry import box, mapping
        geojson = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": mapping(box(0.25, 0.25, 0.75, 0.75)),
                "properties": {"class_id": 1},
            }]
        }
        v = Vectorizer()
        out_path = tmp_path / "out_mask.tif"
        mask = v.geojson_to_mask(geojson, ref_path, out_path)
        assert out_path.exists()
        assert mask.shape == (32, 32)
        assert mask[16, 16] == 1  # center inside the box

    def test_geojson_saved_to_file(self, tmp_path):
        pytest.importorskip("rasterio")
        pytest.importorskip("shapely")
        from pygeovision.ai.inference.vectorization import Vectorizer
        from rasterio.transform import from_bounds

        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[4:12, 4:12] = 1
        transform = from_bounds(0, 0, 1, 1, 16, 16)
        v = Vectorizer(simplify_tolerance=0.0, output_crs=None)
        out = tmp_path / "out.geojson"
        result = v.array_to_geojson(mask, transform, crs="EPSG:4326", output_path=out)
        assert out.exists()
        import json
        loaded = json.loads(out.read_text())
        assert loaded["type"] == "FeatureCollection"


class TestOptimizerConfig:
    def test_optimizer_config_defaults(self):
        from pygeovision.ai.training.optimizers import OptimizerConfig
        cfg = OptimizerConfig()
        assert cfg.name == "adamw"
        assert cfg.lr > 0
        assert cfg.weight_decay >= 0

    def test_build_optimizer_no_torch(self):
        """build_optimizer should raise ImportError if torch not available."""
        pytest.importorskip("torch", reason="torch not installed")
        # If torch is installed, test the actual build
        import torch.nn as nn
        from pygeovision.ai.training.optimizers import build_optimizer
        model = nn.Linear(10, 2)
        opt = build_optimizer(model, name="adamw", lr=1e-4)
        assert opt is not None
