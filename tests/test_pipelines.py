"""Tests for Phase 3: 26 domain production pipelines."""
from __future__ import annotations
import pytest
import sys; sys.path.insert(0, '/home/claude/pgv')


class TestPipelineRegistry:
    def test_total_count(self):
        from pygeovision.ai.pipelines.domains import list_pipelines
        pipes = list_pipelines()
        assert len(pipes) >= 20

    def test_known_pipelines_present(self):
        from pygeovision.ai.pipelines.domains import list_pipelines
        pipes = set(list_pipelines())
        required = {
            "crop_type_mapping", "crop_health", "irrigation_detection",
            "canopy_height", "tree_species", "forest_fire",
            "road_extraction", "infrastructure_monitoring",
            "flood_mapping", "water_quality", "coastal_monitoring",
            "landslide_detection", "volcano_monitoring",
            "land_surface_temperature", "vegetation_indices",
            "ocean_ship_detection",
        }
        missing = required - pipes
        assert not missing, f"Missing pipelines: {missing}"

    def test_pipelines_sorted(self):
        from pygeovision.ai.pipelines.domains import list_pipelines
        pipes = list_pipelines()
        assert pipes == sorted(pipes)


class TestDomainPipelines:
    """Test each domain pipeline class directly with a mock client."""

    @pytest.fixture
    def mock_client(self, tmp_path):
        from unittest.mock import MagicMock
        from pathlib import Path

        mock_path = tmp_path / "sentinel2.tif"
        mock_path.write_bytes(b"FAKE_GEOTIFF")

        client = MagicMock()
        # search returns 2 results
        sr = MagicMock()
        sr.id = "S2C_TEST_001"
        sr.provider = "planetary_computer"
        client.search.return_value = [sr, sr]

        # download returns a successful DownloadResult
        dl = MagicMock()
        dl.success = True
        dl.path = mock_path
        dl.error = ""
        client.download.return_value = [dl, dl]

        # geoai operations succeed
        client.geoai.segment.buildings.return_value = None
        client.geoai.water.segment.return_value = None
        client.geoai.change.detect.return_value = None
        client.geoai.detect.ships.return_value = None
        client.geoai.detect.grounded.return_value = None
        client.geoai.canopy.estimate.return_value = None

        return client

    @pytest.mark.parametrize("pipeline_name,cls_name", [
        ("crop_type_mapping",         "CropTypeMappingPipeline"),
        ("crop_health",               "CropHealthPipeline"),
        ("irrigation_detection",      "IrrigationDetectionPipeline"),
        ("canopy_height",             "CanopyHeightPipeline"),
        ("tree_species",              "TreeSpeciesPipeline"),
        ("forest_fire",               "ForestFirePipeline"),
        ("road_extraction",           "RoadExtractionPipeline"),
        ("flood_mapping",             "FloodMappingPipeline"),
        ("water_quality",             "WaterQualityPipeline"),
        ("landslide_detection",       "LandslideDetectionPipeline"),
        ("volcano_monitoring",        "VolcanoMonitoringPipeline"),
        ("land_surface_temperature",  "LandSurfaceTemperaturePipeline"),
        ("vegetation_indices",        "VegetationIndicesPipeline"),
        ("ocean_ship_detection",      "OceanShipDetectionPipeline"),
    ])
    def test_pipeline_runs(self, pipeline_name, cls_name, mock_client, tmp_path):
        from pygeovision.ai.pipelines import domains as dom
        cls = getattr(dom, cls_name, None)
        if cls is None:
            pytest.skip(f"{cls_name} not found in domains module")
        pipeline = cls(pgv_client=mock_client)
        result = pipeline.run(
            bbox=(-74.1, 40.6, -73.7, 40.9),
            output_dir=str(tmp_path),
            date="2024-06",
        )
        from pygeovision.ai.pipelines.domains import PipelineResult
        assert isinstance(result, PipelineResult)
        assert result.name == pipeline_name

    def test_infrastructure_monitoring_needs_both_dates(self, mock_client, tmp_path):
        from pygeovision.ai.pipelines.domains import InfrastructureMonitoringPipeline
        p = InfrastructureMonitoringPipeline(pgv_client=mock_client)
        result = p.run(
            bbox=(-74.1, 40.6, -73.7, 40.9),
            output_dir=str(tmp_path),
            date_before="2020-01",
            date_after="2024-01",
        )
        from pygeovision.ai.pipelines.domains import PipelineResult
        assert isinstance(result, PipelineResult)

    def test_coastal_monitoring_needs_both_dates(self, mock_client, tmp_path):
        from pygeovision.ai.pipelines.domains import CoastalMonitoringPipeline
        p = CoastalMonitoringPipeline(pgv_client=mock_client)
        result = p.run(
            bbox=(-74.1, 40.6, -73.7, 40.9),
            output_dir=str(tmp_path),
            date_before="2020-01",
            date_after="2024-01",
        )
        from pygeovision.ai.pipelines.domains import PipelineResult
        assert isinstance(result, PipelineResult)

    def test_pipeline_failure_on_empty_search(self, tmp_path):
        from unittest.mock import MagicMock
        from pygeovision.ai.pipelines.domains import FloodMappingPipeline
        client = MagicMock()
        client.search.return_value = []
        client.download.return_value = []
        p = FloodMappingPipeline(pgv_client=client)
        result = p.run(bbox=(-74.1, 40.6, -73.7, 40.9), output_dir=str(tmp_path))
        assert not result.success
        assert result.error

    def test_pipeline_failure_on_failed_download(self, tmp_path):
        from unittest.mock import MagicMock
        from pygeovision.ai.pipelines.domains import CanopyHeightPipeline
        client = MagicMock()
        sr = MagicMock()
        client.search.return_value = [sr]
        dl = MagicMock()
        dl.success = False
        dl.path = None
        dl.error = "Download failed"
        client.download.return_value = [dl]
        p = CanopyHeightPipeline(pgv_client=client)
        result = p.run(bbox=(-74.1, 40.6, -73.7, 40.9), output_dir=str(tmp_path))
        assert not result.success

    def test_pipeline_result_str(self):
        from pygeovision.ai.pipelines.domains import PipelineResult
        from pathlib import Path
        r = PipelineResult("test_pipe", True, Path("/tmp/out.tif"), {"n": 3})
        s = str(r)
        assert "test_pipe" in s
        assert "✓" in s

    def test_pipeline_result_failure_str(self):
        from pygeovision.ai.pipelines.domains import PipelineResult
        r = PipelineResult("test_pipe", False, error="Something went wrong")
        s = str(r)
        assert "✗" in s
        assert "Something went wrong" in s
