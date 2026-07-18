"""
Tests for pygeofetch.insar.extraction.SLCExtractor.

Verified against a synthetic multi-sub-swath .SAFE.zip with known GCP
footprints, so correctness is checked against ground truth (which
sub-swath SHOULD match a given AOI), not just "doesn't crash".
"""

from __future__ import annotations

import zipfile

import pytest


def _build_synthetic_slc_zip(tmp_path, footprints, polarisation="vv"):
    """
    Build a synthetic Sentinel-1-style SLC .SAFE.zip with one measurement
    TIFF per sub-swath, each carrying GCPs matching the given footprint.

    footprints: dict of {swath_name: (min_lon, min_lat, max_lon, max_lat)}
    """
    rasterio = pytest.importorskip("rasterio")
    import numpy as np

    tiff_dir = tmp_path / "measurement"
    tiff_dir.mkdir(exist_ok=True)

    zip_path = tmp_path / "S1A_IW_SLC__1SDV_test.SAFE.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for swath, (minlon, minlat, maxlon, maxlat) in footprints.items():
            tiff_path = tiff_dir / f"s1a-{swath}-slc-{polarisation}-test.tiff"
            h, w = 8, 8
            data = np.ones((h, w), dtype=np.complex64)
            gcps = [
                rasterio.control.GroundControlPoint(row=0, col=0, x=minlon, y=maxlat),
                rasterio.control.GroundControlPoint(row=0, col=w, x=maxlon, y=maxlat),
                rasterio.control.GroundControlPoint(row=h, col=0, x=minlon, y=minlat),
                rasterio.control.GroundControlPoint(row=h, col=w, x=maxlon, y=minlat),
            ]
            with rasterio.open(
                tiff_path,
                "w",
                driver="GTiff",
                dtype="complex_int16",
                count=1,
                width=w,
                height=h,
                gcps=gcps,
                crs="EPSG:4326",
            ) as ds:
                ds.write(data, 1)
            zf.write(
                tiff_path,
                f"S1A_test.SAFE/measurement/s1a-{swath}-slc-{polarisation}-test.tiff",
            )
    return zip_path


class TestSLCExtractor:
    def test_picks_the_overlapping_subswath(self, tmp_path):
        """With 3 candidate sub-swaths, only the truly-overlapping one is extracted."""
        from pygeofetch.insar import SLCExtractor
        from pygeofetch.models.search_query import BoundingBox

        footprints = {
            "iw1": (-100.0, 18.0, -99.5, 19.0),
            "iw2": (-99.3, 19.2, -98.9, 19.6),  # matches AOI below
            "iw3": (-98.5, 20.0, -98.0, 20.5),
        }
        zip_path = _build_synthetic_slc_zip(tmp_path, footprints)
        aoi = BoundingBox(min_lon=-99.2, min_lat=19.3, max_lon=-99.0, max_lat=19.5)

        extractor = SLCExtractor(polarisation="VV")
        out = extractor.extract_scene(zip_path, aoi, tmp_path / "out", label="test")

        assert out is not None
        assert out.exists()

    def test_returns_none_when_no_subswath_overlaps(self, tmp_path):
        from pygeofetch.insar import SLCExtractor
        from pygeofetch.models.search_query import BoundingBox

        footprints = {"iw1": (-100.0, 18.0, -99.5, 19.0)}
        zip_path = _build_synthetic_slc_zip(tmp_path, footprints)
        far_aoi = BoundingBox(min_lon=50.0, min_lat=50.0, max_lon=51.0, max_lat=51.0)

        extractor = SLCExtractor(polarisation="VV")
        out = extractor.extract_scene(zip_path, far_aoi, tmp_path / "out")

        assert out is None

    def test_returns_none_for_missing_archive(self, tmp_path):
        from pygeofetch.insar import SLCExtractor
        from pygeofetch.models.search_query import BoundingBox

        extractor = SLCExtractor(polarisation="VV")
        aoi = BoundingBox(min_lon=-99.2, min_lat=19.3, max_lon=-99.0, max_lat=19.5)
        out = extractor.extract_scene(
            tmp_path / "nonexistent.zip", aoi, tmp_path / "out"
        )

        assert out is None

    def test_resolves_download_result_output_path(self, tmp_path):
        """extract_pair should use DownloadResult.output_path directly, no guessing."""
        from pygeofetch.insar import SLCExtractor
        from pygeofetch.models.download_task import DownloadResult, DownloadStatus
        from pygeofetch.models.search_query import BoundingBox

        footprints = {"iw2": (-99.3, 19.2, -98.9, 19.6)}
        zip_path = _build_synthetic_slc_zip(tmp_path, footprints)
        aoi = BoundingBox(min_lon=-99.2, min_lat=19.3, max_lon=-99.0, max_lat=19.5)

        dl_result = DownloadResult(
            status=DownloadStatus.COMPLETED,
            data_id="test-id",
            provider="copernicus",
            output_path=zip_path,
            output_paths=[zip_path],
        )

        extractor = SLCExtractor(polarisation="VV")
        ref_tif, sec_tif = extractor.extract_pair(
            dl_result, dl_result, aoi, tmp_path / "out"
        )

        assert ref_tif is not None
        assert sec_tif is not None
        assert ref_tif.exists()
        assert sec_tif.exists()

    def test_extract_pair_returns_none_none_for_failed_download(self, tmp_path):
        """A failed DownloadResult (no output_path) should resolve to (None, None)."""
        from pygeofetch.insar import SLCExtractor
        from pygeofetch.models.download_task import DownloadResult, DownloadStatus
        from pygeofetch.models.search_query import BoundingBox

        failed_result = DownloadResult(
            status=DownloadStatus.FAILED,
            data_id="test-id",
            provider="copernicus",
            error="network error",
        )
        aoi = BoundingBox(min_lon=-99.2, min_lat=19.3, max_lon=-99.0, max_lat=19.5)

        extractor = SLCExtractor(polarisation="VV")
        ref_tif, sec_tif = extractor.extract_pair(
            failed_result, failed_result, aoi, tmp_path / "out"
        )

        assert ref_tif is None
        assert sec_tif is None

    def test_list_subswaths(self, tmp_path):
        from pygeofetch.insar import SLCExtractor

        footprints = {
            "iw1": (-100.0, 18.0, -99.5, 19.0),
            "iw2": (-99.3, 19.2, -98.9, 19.6),
            "iw3": (-98.5, 20.0, -98.0, 20.5),
        }
        zip_path = _build_synthetic_slc_zip(tmp_path, footprints)

        extractor = SLCExtractor(polarisation="VV")
        subswaths = extractor.list_subswaths(zip_path)

        assert len(subswaths) == 3

    def test_vh_polarisation_selection(self, tmp_path):
        """Requesting VH should find VH tiffs, not VV."""
        from pygeofetch.insar import SLCExtractor

        footprints = {"iw1": (-100.0, 18.0, -99.5, 19.0)}
        zip_path = _build_synthetic_slc_zip(tmp_path, footprints, polarisation="vh")

        vv_extractor = SLCExtractor(polarisation="VV")
        vh_extractor = SLCExtractor(polarisation="VH")

        assert vv_extractor.list_subswaths(zip_path) == []
        assert len(vh_extractor.list_subswaths(zip_path)) == 1

    def test_bad_zip_returns_empty_list(self, tmp_path):
        """A corrupt/non-zip file should not crash, just return no sub-swaths."""
        from pygeofetch.insar import SLCExtractor

        bad_zip = tmp_path / "not_a_zip.zip"
        bad_zip.write_bytes(b"not a real zip file")

        extractor = SLCExtractor(polarisation="VV")
        result = extractor.list_subswaths(bad_zip)

        assert result == []

    def test_direct_string_path_resolution(self, tmp_path):
        """extract_pair should also accept plain string/Path arguments, not just DownloadResult."""
        from pygeofetch.insar import SLCExtractor
        from pygeofetch.models.search_query import BoundingBox

        footprints = {"iw2": (-99.3, 19.2, -98.9, 19.6)}
        zip_path = _build_synthetic_slc_zip(tmp_path, footprints)
        aoi = BoundingBox(min_lon=-99.2, min_lat=19.3, max_lon=-99.0, max_lat=19.5)

        extractor = SLCExtractor(polarisation="VV")
        ref_tif, sec_tif = extractor.extract_pair(
            str(zip_path), zip_path, aoi, tmp_path / "out"
        )

        assert ref_tif is not None
        assert sec_tif is not None
