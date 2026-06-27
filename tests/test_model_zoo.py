"""Tests for Phase 2: ModelZoo — 98+ architectures."""
from __future__ import annotations
import pytest
import sys; sys.path.insert(0, '/home/claude/pgv')


class TestModelZoo:
    @pytest.fixture(autouse=True)
    def _zoo(self):
        from pygeovision.ai.models.zoo import ModelZoo
        self.zoo = ModelZoo()

    def test_zoo_size(self):
        assert len(self.zoo) >= 50

    def test_contains(self):
        assert "segformer_b2" in self.zoo
        assert "yolov8_s" in self.zoo
        assert "sam_vit_h" in self.zoo

    def test_getitem(self):
        m = self.zoo["segformer_b2"]
        assert m.name == "segformer_b2"
        assert m.task == "segmentation"
        assert m.params_m > 0

    def test_getitem_missing_raises(self):
        with pytest.raises(KeyError):
            _ = self.zoo["nonexistent_model_xyz"]

    def test_filter_task(self):
        seg = self.zoo.filter(task="segmentation")
        assert all(m.task == "segmentation" for m in seg)
        assert len(seg) >= 10

    def test_filter_detection(self):
        det = self.zoo.filter(task="detection")
        assert all(m.task == "detection" for m in det)
        assert len(det) >= 5

    def test_filter_tag(self):
        transformers = self.zoo.filter(tag="transformer")
        assert all("transformer" in m.tags for m in transformers)
        assert len(transformers) >= 5

    def test_filter_max_params(self):
        light = self.zoo.filter(max_params_m=30.0)
        assert all(m.params_m <= 30.0 for m in light)

    def test_filter_pretrained_only(self):
        pretrained = self.zoo.filter(pretrained_only=True)
        assert all(m.pretrained_available for m in pretrained)

    def test_search(self):
        results = self.zoo.search("transformer")
        assert len(results) >= 3

    def test_search_returns_empty_for_unknown(self):
        results = self.zoo.search("xyzzy9999nonexistent")
        assert results == []

    def test_tasks(self):
        tasks = self.zoo.tasks()
        assert "segmentation" in tasks
        assert "detection" in tasks
        assert "classification" in tasks
        assert "change_detection" in tasks
        assert "foundation" in tasks
        assert "vlm" in tasks

    def test_top_for_task(self):
        top = self.zoo.top_for_task("segmentation", n=5)
        assert len(top) == 5
        assert all(m.task == "segmentation" for m in top)
        assert all(m.pretrained_available for m in top)

    def test_hf_models_have_ids(self):
        hf_models = [m for m in self.zoo.all() if m.hf_model_id]
        assert len(hf_models) >= 15
        for m in hf_models:
            assert "/" in m.hf_model_id, f"Invalid HF model ID: {m.hf_model_id}"

    def test_summary(self):
        s = self.zoo.summary()
        assert s["total_models"] >= 50
        assert s["with_hf_weights"] >= 15
        assert s["pretrained"] >= 50
        assert "segmentation" in s["tasks"]

    def test_all_models_have_required_fields(self):
        for m in self.zoo.all():
            assert m.name, f"Model missing name"
            assert m.task, f"Model '{m.name}' missing task"
            assert m.architecture, f"Model '{m.name}' missing architecture"
            assert m.params_m >= 0, f"Model '{m.name}' invalid params_m"

    def test_tasks_are_known(self):
        KNOWN = {"segmentation","detection","classification","change_detection",
                 "foundation","vlm","3d","timeseries","super_resolution"}
        for m in self.zoo.all():
            assert m.task in KNOWN, f"Model '{m.name}' has unknown task '{m.task}'"

    def test_print_table_no_error(self, capsys):
        items = self.zoo.filter(task="segmentation")[:5]
        self.zoo.print_table(items)
        captured = capsys.readouterr()
        assert "segmentation" in captured.out.lower()

    def test_foundation_models_exist(self):
        foundation = self.zoo.filter(task="foundation")
        assert len(foundation) >= 5
        names = [m.name for m in foundation]
        assert any("prithvi" in n for n in names)
        assert any("dinov" in n for n in names)

    def test_vlm_models_exist(self):
        vlms = self.zoo.filter(task="vlm")
        assert len(vlms) >= 3
        names = [m.name for m in vlms]
        assert any("clip" in n for n in names)
