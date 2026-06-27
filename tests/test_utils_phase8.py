"""Tests for Phase 8: GPU utils, data pipeline, streaming dataset."""
from __future__ import annotations
import pytest
import sys; sys.path.insert(0, '/home/claude/pgv')


class TestGPUUtils:
    def test_get_device_returns_something(self):
        from pygeovision.utils.gpu import get_device
        dev = get_device()
        assert dev is not None

    def test_gpu_info_structure(self):
        from pygeovision.utils.gpu import gpu_info
        info = gpu_info()
        assert isinstance(info, dict)
        assert "available" in info

    def test_enable_tf32_no_error(self):
        from pygeovision.utils.gpu import enable_tf32
        enable_tf32()  # Should not raise

    def test_set_memory_fraction_no_error(self):
        from pygeovision.utils.gpu import set_memory_fraction
        set_memory_fraction(0.8)  # Should not raise


class TestDataPipelineUtils:
    def test_optimal_num_workers(self):
        from pygeovision.utils.data_pipeline import optimal_num_workers
        nw = optimal_num_workers()
        assert isinstance(nw, int)
        assert nw >= 1

    def test_optimal_num_workers_safety_factor(self):
        from pygeovision.utils.data_pipeline import optimal_num_workers
        import os
        expected = max(1, int((os.cpu_count() or 1) * 0.5))
        result = optimal_num_workers(safety_factor=0.5)
        assert result == expected

    def test_parallel_raster_read_no_files(self):
        from pygeovision.utils.data_pipeline import parallel_raster_read
        # Should return empty list for empty input
        results = parallel_raster_read([], n_workers=2, fn=lambda p: p)
        assert results == []

    def test_parallel_raster_read_custom_fn(self):
        from pygeovision.utils.data_pipeline import parallel_raster_read
        paths = ["/tmp/a.tif", "/tmp/b.tif", "/tmp/c.tif"]
        results = parallel_raster_read(paths, n_workers=2, fn=lambda p: p.upper())
        assert len(results) == 3
        assert all(r is None or r.endswith(".TIF") for r in results)


class TestStreamingRasterDataset:
    def test_init_no_rasterio(self, tmp_path):
        """StreamingRasterDataset.__init__ should not raise even without rasterio."""
        from pygeovision.utils.data_pipeline import StreamingRasterDataset
        fake_path = str(tmp_path / "fake.tif")
        # Creates but _compute_chips will warn and set to []
        ds = StreamingRasterDataset(fake_path, chip_size=256)
        assert isinstance(ds._chips, list)

    def test_len_with_no_chips(self, tmp_path):
        from pygeovision.utils.data_pipeline import StreamingRasterDataset
        ds = StreamingRasterDataset(str(tmp_path / "fake.tif"))
        # Without rasterio, chips = [] so len is 0
        assert len(ds) == 0
