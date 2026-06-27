"""
PyGeoVision test configuration and shared fixtures.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dir():
    """Temporary directory that is cleaned up after each test."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def mock_pgv_client():
    """Mock PyGeoVision client that simulates PyGeoFetch data operations."""
    client = MagicMock()
    client.search.return_value = [MagicMock(id="test-item-001")]
    client.download.return_value = Path("/tmp/test_scene.tif")
    return client


@pytest.fixture
def small_geotiff(tmp_dir):
    """Create a tiny 64×64 GeoTIFF for testing (3 bands, float32)."""
    pytest.importorskip("rasterio")
    import rasterio
    from rasterio.transform import from_bounds

    path = tmp_dir / "test_tile.tif"
    transform = from_bounds(west=0.0, south=0.0, east=0.1, north=0.1, width=64, height=64)
    data = (np.random.rand(3, 64, 64) * 3000).astype(np.float32)

    with rasterio.open(
        path, "w", driver="GTiff", dtype="float32",
        width=64, height=64, count=3, crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)

    return path


@pytest.fixture
def label_geotiff(tmp_dir):
    """Create a tiny 64×64 uint8 label GeoTIFF."""
    pytest.importorskip("rasterio")
    import rasterio
    from rasterio.transform import from_bounds

    path = tmp_dir / "test_label.tif"
    transform = from_bounds(west=0.0, south=0.0, east=0.1, north=0.1, width=64, height=64)
    data = np.random.randint(0, 5, (1, 64, 64), dtype=np.uint8)

    with rasterio.open(
        path, "w", driver="GTiff", dtype="uint8",
        width=64, height=64, count=1, crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)

    return path


@pytest.fixture
def tile_metadata(small_geotiff):
    """TileMetadata for the small test GeoTIFF."""
    pytest.importorskip("rasterio")
    import rasterio

    with rasterio.open(small_geotiff) as src:
        bounds = src.bounds
        crs = src.crs

    # Import here to avoid circular imports
    from pygeovision.ai.data.dataset import TileMetadata
    crs_str = crs.to_string() if hasattr(crs, 'to_string') else str(crs)
    return TileMetadata(
        tile_id="test_tile_001",
        source_file=small_geotiff,
        bounds=(bounds.left, bounds.bottom, bounds.right, bounds.top),
        crs=crs_str,
        transform=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        row_off=0, col_off=0,
        height=64, width=64,
        bands=[1, 2, 3],
    )


@pytest.fixture
def simple_segmentation_model():
    """A tiny segmentation model for testing (no GPU required)."""
    pytest.importorskip("torch")
    import torch
    import torch.nn as nn

    class TinySegModel(nn.Module):
        def __init__(self, in_channels=3, num_classes=5):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, num_classes, 1)

        def forward(self, x):
            return self.conv(x)

    return TinySegModel()
