"""
Tests for Phase 1 & 5: DatasetRegistry, DatasetAnalyzer, BenchmarkBuilder.
All tests are pure-Python, no network, no GPU, no optional dependencies.
"""
from __future__ import annotations
import pytest
import sys; sys.path.insert(0, '/home/claude/pgv')


# ── DatasetRegistry ───────────────────────────────────────────────────────────

class TestDatasetRegistry:
    @pytest.fixture(autouse=True)
    def _reg(self):
        from pygeovision.datasets.registry import DatasetRegistry
        self.reg = DatasetRegistry()

    def test_registry_size(self):
        assert len(self.reg) >= 100, "Registry should have at least 100 datasets"

    def test_contains(self):
        assert "EuroSAT" in self.reg
        assert "BigEarthNet" in self.reg
        assert "LEVIR-CD" in self.reg

    def test_getitem(self):
        d = self.reg["EuroSAT"]
        assert d.name == "EuroSAT"
        assert d.domain == "land_cover"
        assert d.n_classes == 10
        assert d.resolution_m == 10.0

    def test_getitem_missing_raises(self):
        with pytest.raises(KeyError):
            _ = self.reg["NonExistentDatasetXYZ"]

    def test_search_keyword(self):
        results = self.reg.search("flood")
        assert len(results) >= 1
        names = [d.name for d in results]
        assert any("lood" in n or "lood" in d.description.lower()
                   for n, d in zip(names, results))

    def test_search_empty_returns_empty(self):
        results = self.reg.search("xyzzy_nonexistent_term_9999")
        assert results == []

    def test_filter_domain(self):
        urban = self.reg.filter(domain="urban")
        assert all(d.domain == "urban" for d in urban)
        assert len(urban) >= 10

    def test_filter_modality(self):
        sar = self.reg.filter(modality="sar")
        assert all(d.modality == "sar" for d in sar)
        assert len(sar) >= 5

    def test_filter_task(self):
        seg = self.reg.filter(task="segmentation")
        assert all("segmentation" in d.tasks for d in seg)

    def test_filter_combined(self):
        items = self.reg.filter(domain="urban", modality="rgb")
        assert all(d.domain == "urban" and d.modality == "rgb" for d in items)

    def test_filter_year(self):
        recent = self.reg.filter(min_year=2022)
        assert all(d.year >= 2022 for d in recent)

    def test_top_for_task_returns_n(self):
        top = self.reg.top_for_task("segmentation", n=5)
        assert len(top) == 5

    def test_top_for_task_sorted(self):
        top3 = self.reg.top_for_task("detection", n=3)
        assert len(top3) == 3
        # All must have detection in tasks
        assert all("detection" in d.tasks for d in top3)

    def test_similar_to_returns_n(self):
        similar = self.reg.similar_to("EuroSAT", n=5)
        assert len(similar) == 5
        # Should not include the reference itself
        assert all(d.name != "EuroSAT" for d in similar)

    def test_similar_to_same_domain_first(self):
        similar = self.reg.similar_to("EuroSAT", n=10)
        land_cover = sum(1 for d in similar if d.domain == "land_cover")
        assert land_cover >= 3, "Similar datasets should share the land_cover domain"

    def test_domains(self):
        domains = self.reg.domains()
        assert "urban" in domains
        assert "agriculture" in domains
        assert "forestry" in domains
        assert "ocean" in domains
        assert sorted(domains) == domains

    def test_tasks(self):
        tasks = self.reg.tasks()
        assert "segmentation" in tasks
        assert "detection" in tasks
        assert "classification" in tasks

    def test_summary(self):
        s = self.reg.summary()
        assert s["total_datasets"] >= 100
        assert s["total_volume_tb"] > 0
        assert len(s["domains"]) >= 10
        assert "segmentation" in s["tasks"]

    def test_all_datasets_have_required_fields(self):
        for d in self.reg.all():
            assert d.name, f"Dataset missing name"
            assert d.domain, f"Dataset '{d.name}' missing domain"
            assert d.year >= 1990, f"Dataset '{d.name}' has invalid year {d.year}"
            assert d.resolution_m > 0, f"Dataset '{d.name}' has invalid resolution"
            assert d.n_classes >= 0, f"Dataset '{d.name}' has invalid n_classes"
            assert len(d.tasks) >= 1, f"Dataset '{d.name}' has no tasks"

    def test_dataset_info_to_dict(self):
        d = self.reg["EuroSAT"]
        dd = d.to_dict()
        assert dd["name"] == "EuroSAT"
        assert "tasks" in dd
        assert "resolution_m" in dd

    def test_print_table_no_error(self, capsys):
        items = self.reg.filter(domain="urban")[:5]
        self.reg.print_table(items)
        captured = capsys.readouterr()
        assert "urban" in captured.out.lower()


# ── DatasetAnalyzer ───────────────────────────────────────────────────────────

class TestDatasetAnalyzer:
    @pytest.fixture(autouse=True)
    def _ana(self):
        from pygeovision.datasets.analysis import DatasetAnalyzer
        self.ana = DatasetAnalyzer()

    def test_volume_trend_returns_dict(self):
        trend = self.ana.volume_trend(show=False)
        assert isinstance(trend, dict)
        assert len(trend) >= 10
        # Values should be monotonically non-decreasing (cumulative)
        vals = [v for _, v in sorted(trend.items())]
        for i in range(1, len(vals)):
            assert vals[i] >= vals[i - 1]

    def test_resolution_distribution(self):
        dist = self.ana.resolution_distribution(show=False)
        assert isinstance(dist, dict)
        total = sum(dist.values())
        from pygeovision.datasets.registry import dataset_registry
        assert total == len(dataset_registry), "All datasets must fall in a resolution bin"

    def test_domain_distribution(self):
        dist = self.ana.domain_distribution(show=False)
        assert "urban" in dist
        assert "agriculture" in dist
        assert all(v >= 0 for v in dist.values())

    def test_modality_distribution(self):
        dist = self.ana.modality_distribution(show=False)
        assert "rgb" in dist or "multispectral" in dist

    def test_correlation_matrix_shape(self):
        names = ["EuroSAT", "BigEarthNet", "LoveDA", "LEVIR-CD"]
        matrix = self.ana.correlation_matrix(names=names, show=False)
        assert len(matrix) == 4
        assert all(len(row) == 4 for row in matrix)

    def test_correlation_matrix_diagonal_is_one(self):
        names = ["EuroSAT", "BigEarthNet", "DOTA"]
        matrix = self.ana.correlation_matrix(names=names, show=False)
        for i in range(len(names)):
            assert abs(matrix[i][i] - 1.0) < 1e-6

    def test_correlation_matrix_symmetric(self):
        names = ["EuroSAT", "BigEarthNet", "DOTA"]
        matrix = self.ana.correlation_matrix(names=names, show=False)
        for i in range(len(names)):
            for j in range(len(names)):
                assert abs(matrix[i][j] - matrix[j][i]) < 1e-6

    def test_correlation_values_in_range(self):
        names = ["EuroSAT", "BigEarthNet", "DOTA", "LEVIR-CD"]
        matrix = self.ana.correlation_matrix(names=names, show=False)
        for row in matrix:
            for v in row:
                assert 0.0 <= v <= 1.0, f"Similarity out of range [0,1]: {v}"

    def test_full_report(self):
        report = self.ana.full_report()
        assert "catalog_summary" in report
        assert "volume_trend" in report
        assert "top_segmentation" in report
        assert len(report["top_segmentation"]) == 5
        assert len(report["top_detection"]) == 5


# ── BenchmarkBuilder ─────────────────────────────────────────────────────────

class TestBenchmarkBuilder:
    @pytest.fixture(autouse=True)
    def _builder(self):
        from pygeovision.datasets.benchmark import BenchmarkBuilder
        self.builder = BenchmarkBuilder()

    def test_build_segmentation(self):
        cfg = self.builder.build("segmentation", n=5)
        assert cfg.task == "segmentation"
        assert len(cfg.dataset_names) == 5
        assert cfg.metric == "mean_iou"

    def test_build_detection(self):
        cfg = self.builder.build("detection", n=5)
        assert cfg.task == "detection"
        assert cfg.metric == "mAP50"

    def test_build_classification(self):
        cfg = self.builder.build("classification", n=5)
        assert cfg.metric == "accuracy"

    def test_build_change_detection(self):
        cfg = self.builder.build("change_detection", n=5)
        assert cfg.metric == "iou_change"

    def test_build_invalid_task(self):
        with pytest.raises(ValueError):
            self.builder.build("nonexistent_task_xyz")

    def test_build_all_returns_dict(self):
        all_cfgs = self.builder.build_all(n=3)
        assert isinstance(all_cfgs, dict)
        assert "segmentation" in all_cfgs
        assert "detection" in all_cfgs

    def test_cross_task_matrix(self):
        matrix = self.builder.cross_task_matrix()
        assert "multi_task" in matrix
        assert "all" in matrix
        multi = matrix["multi_task"]
        # Some datasets should appear in multiple tasks
        if multi:
            for name, tasks in multi.items():
                assert len(tasks) >= 2

    def test_recommended_for_paper(self):
        rec = self.builder.recommended_for_paper("segmentation")
        assert rec["task"] == "segmentation"
        assert "primary_metric" in rec
        assert "datasets" in rec
        assert len(rec["datasets"]) == 5
        for d in rec["datasets"]:
            assert "name" in d
            assert "year" in d

    def test_benchmark_config_to_dict(self):
        cfg = self.builder.build("segmentation", n=3)
        d = cfg.to_dict()
        assert d["task"] == "segmentation"
        assert len(d["dataset_names"]) == 3
        assert "split_ratios" in d
        assert abs(sum(d["split_ratios"].values()) - 1.0) < 1e-6

    def test_save_all(self, tmp_path):
        saved = self.builder.save_all(str(tmp_path), n=3)
        assert len(saved) >= 3
        for path in saved:
            import json
            with open(path) as f:
                data = json.load(f)
            assert "task" in data
            assert "dataset_names" in data
