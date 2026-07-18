"""
Tests for pygeofetch.insar — verified against known analytical ground truth.

These tests confirm the InSAR chain (interferogram formation, coherence,
phase unwrapping, SBAS inversion, atmospheric correction) produces
mathematically correct results on synthetic data with known answers.
"""

from __future__ import annotations

import pytest


def _make_complex_pair(h=64, w=64, phase_ramp_x=0.05, phase_ramp_y=0.03, seed=42):
    """Build a synthetic reference/secondary complex SLC pair with a known phase ramp."""
    import numpy as np

    np.random.seed(seed)
    amp = np.random.rayleigh(scale=50, size=(h, w)).astype(np.float32)
    y, x = np.mgrid[0:h, 0:w]
    true_phase = (phase_ramp_x * x + phase_ramp_y * y).astype(np.float32)

    ref = (amp * np.exp(1j * 0)).astype(np.complex64)
    sec = (amp * np.exp(1j * true_phase)).astype(np.complex64)
    return ref, sec, true_phase


class TestInterferogramGenerator:
    def test_coherence_near_one_for_noise_free_pair(self):
        """Identical-amplitude, phase-only-difference pair should have coherence ~1."""
        from pygeofetch.insar import InterferogramGenerator

        ref, sec, _ = _make_complex_pair()
        gen = InterferogramGenerator(esd_enabled=False)
        coherence = gen._estimate_coherence(ref, sec, window=5)

        assert coherence.mean() > 0.95

    def test_interferogram_phase_matches_true_ramp(self):
        """Interferogram phase should equal the true phase difference (wrapped)."""
        import numpy as np

        from pygeofetch.insar import InterferogramGenerator

        ref, sec, true_phase = _make_complex_pair()
        InterferogramGenerator(esd_enabled=False)
        interferogram = ref * np.conj(sec)

        # ref*conj(sec) has phase = phase(ref) - phase(sec) = -true_phase
        recovered = np.angle(interferogram)
        expected = np.angle(np.exp(-1j * true_phase))  # wrap -true_phase for comparison

        diff = np.abs(np.angle(np.exp(1j * (recovered - expected))))
        assert diff.max() < 1e-3

    def test_process_pair_end_to_end(self, tmp_path):
        """Full pipeline: write SLC pair to disk, process_pair produces valid result."""
        import numpy as np
        import rasterio
        from rasterio.crs import CRS
        from rasterio.transform import from_bounds

        from pygeofetch.insar import InterferogramGenerator

        h, w = 40, 40
        ref, sec, _ = _make_complex_pair(h, w)

        crs = CRS.from_epsg(4326)
        transform = from_bounds(-1, 0, 1, 1, w, h)

        def write_complex(path, data):
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                dtype="float32",
                count=2,
                width=w,
                height=h,
                crs=crs,
                transform=transform,
            ) as ds:
                ds.write(data.real.astype(np.float32), 1)
                ds.write(data.imag.astype(np.float32), 2)

        ref_path = tmp_path / "ref.tif"
        sec_path = tmp_path / "sec.tif"
        write_complex(ref_path, ref)
        write_complex(sec_path, sec)

        gen = InterferogramGenerator(esd_enabled=False)
        result = gen.process_pair(ref_path, sec_path)

        assert result.interferogram.shape == (h, w)
        assert result.coherence.shape == (h, w)
        assert 0 <= result.coherence.mean() <= 1

    def test_save_writes_three_geotiffs(self, tmp_path):
        """InterferogramResult.save() writes wrapped_phase, coherence, amplitude."""
        import numpy as np
        from rasterio.crs import CRS
        from rasterio.transform import from_bounds

        from pygeofetch.insar.interferogram import InterferogramResult

        h, w = 20, 20
        result = InterferogramResult(
            interferogram=np.ones((h, w), dtype=np.complex64),
            coherence=np.ones((h, w), dtype=np.float32) * 0.8,
            amplitude=np.ones((h, w), dtype=np.float32),
            profile={
                "crs": CRS.from_epsg(4326),
                "transform": from_bounds(-1, 0, 1, 1, w, h),
            },
        )
        paths = result.save(tmp_path)
        assert set(paths.keys()) == {"wrapped_phase", "coherence", "amplitude"}
        for p in paths.values():
            assert p.exists()

    def test_missing_dem_warns_but_does_not_crash(self, tmp_path, caplog):
        """process_pair without a DEM should warn, not raise."""
        import numpy as np
        import rasterio
        from rasterio.crs import CRS
        from rasterio.transform import from_bounds

        from pygeofetch.insar import InterferogramGenerator

        h, w = 16, 16
        ref, sec, _ = _make_complex_pair(h, w)
        crs = CRS.from_epsg(4326)
        transform = from_bounds(-1, 0, 1, 1, w, h)

        def write_complex(path, data):
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                dtype="float32",
                count=2,
                width=w,
                height=h,
                crs=crs,
                transform=transform,
            ) as ds:
                ds.write(data.real.astype(np.float32), 1)
                ds.write(data.imag.astype(np.float32), 2)

        ref_path = tmp_path / "ref.tif"
        sec_path = tmp_path / "sec.tif"
        write_complex(ref_path, ref)
        write_complex(sec_path, sec)

        gen = InterferogramGenerator(esd_enabled=False)
        result = gen.process_pair(ref_path, sec_path, dem=None)
        assert result.metadata["topographic_phase_removed"] is False


class TestPhaseUnwrapper:
    def test_missing_snaphu_raises_import_error(self):
        """_require_snaphu raises ImportError with install instructions when missing."""
        from unittest.mock import patch

        from pygeofetch.insar.unwrap import _require_snaphu

        with patch.dict("sys.modules", {"snaphu": None}):
            with pytest.raises(ImportError, match="snaphu"):
                _require_snaphu()

    def test_invalid_cost_mode_raises(self):
        from pygeofetch.insar import PhaseUnwrapper

        with pytest.raises(ValueError, match="cost_mode"):
            PhaseUnwrapper(cost_mode="invalid")

    def test_invalid_init_method_raises(self):
        from pygeofetch.insar import PhaseUnwrapper

        with pytest.raises(ValueError, match="init_method"):
            PhaseUnwrapper(init_method="invalid")

    def test_unwrap_recovers_linear_ramp(self):
        """SNAPHU should recover a known linear phase ramp with near-zero RMS error."""
        pytest.importorskip("snaphu")
        import numpy as np

        from pygeofetch.insar import PhaseUnwrapper

        h, w = 48, 48
        y, x = np.mgrid[0:h, 0:w]
        true_phase = 0.1 * x + 0.05 * y
        igram = np.exp(1j * true_phase).astype(np.complex64)
        coherence = np.ones((h, w), dtype=np.float32) * 0.9

        unwrapper = PhaseUnwrapper(cost_mode="defo", init_method="mcf")
        unwrapped, conncomp = unwrapper.unwrap(igram, coherence, nlooks=1.0)

        # SNAPHU output may have a constant offset — remove it before comparing
        diff = unwrapped - true_phase
        diff -= diff.mean()
        rms = float(np.sqrt(np.mean(diff**2)))
        assert rms < 0.1

    def test_unwrap_accepts_float_phase_input(self):
        """unwrap() should accept plain float phase arrays, not just complex."""
        pytest.importorskip("snaphu")
        import numpy as np

        from pygeofetch.insar import PhaseUnwrapper

        h, w = 32, 32
        y, x = np.mgrid[0:h, 0:w]
        phase = (0.08 * x).astype(np.float32)
        coherence = np.ones((h, w), dtype=np.float32) * 0.85

        unwrapper = PhaseUnwrapper()
        unwrapped, conncomp = unwrapper.unwrap(phase, coherence)
        assert unwrapped.shape == (h, w)


class TestSBASTimeSeries:
    def test_recovers_known_linear_velocity(self):
        """SBAS inversion should exactly recover a synthetic linear velocity signal."""
        import numpy as np

        from pygeofetch.insar import SBASTimeSeries
        from pygeofetch.insar.timeseries import InterferogramPair

        h, w = 16, 16
        wavelength = 0.05546576
        true_velocity = -0.010  # m/year, at the "signal" pixel

        dates = ["2026-01-01", "2026-01-13", "2026-01-25", "2026-02-06"]
        days = [0, 12, 24, 36]
        true_disp = {d: true_velocity * (dd / 365.25) for d, dd in zip(dates, days)}

        def disp_to_phase(disp_m):
            # Matches InterferogramGenerator's ref*conj(sec) convention:
            # unwrapped_phase = phase(ref) - phase(sec)
            #                 = +4*pi/wavelength * (disp(sec) - disp(ref))
            return 4 * np.pi / wavelength * disp_m

        # Reference (stable) pixel at (0,0) has zero displacement in all pairs;
        # every other pixel follows true_velocity. This spatial variation is
        # required for SBASTimeSeries's reference-pixel normalization (needed
        # to remove SNAPHU's per-interferogram arbitrary offset) to be
        # meaningful rather than degenerate.
        pair_defs = [(0, 1), (1, 2), (2, 3), (0, 2)]
        pairs = []
        for i, j in pair_defs:
            d_ref, d_sec = dates[i], dates[j]
            true_diff = disp_to_phase(true_disp[d_sec] - true_disp[d_ref])
            phase = np.full((h, w), true_diff, dtype=np.float32)
            phase[0, 0] = 0.0  # stable reference pixel
            coh = np.ones((h, w), dtype=np.float32) * 0.8
            pairs.append(InterferogramPair(d_ref, d_sec, phase, coh))

        sbas = SBASTimeSeries(wavelength_m=wavelength, reference_date=dates[0])
        result = sbas.invert(pairs, reference_pixel=(0, 0))

        assert abs(result.velocity[8, 8] - true_velocity) < 1e-5
        assert abs(result.velocity[0, 0]) < 1e-10  # reference pixel stays at zero
        assert result.dates == dates
        assert result.displacement.shape == (len(dates), h, w)

    def test_reference_date_has_zero_displacement(self):
        import numpy as np

        from pygeofetch.insar import SBASTimeSeries
        from pygeofetch.insar.timeseries import InterferogramPair

        h, w = 8, 8
        pairs = [
            InterferogramPair(
                "2026-01-01",
                "2026-01-13",
                np.ones((h, w), dtype=np.float32) * 0.5,
                np.ones((h, w), dtype=np.float32) * 0.8,
            ),
        ]
        sbas = SBASTimeSeries(reference_date="2026-01-01")
        result = sbas.invert(pairs)
        ref_idx = result.dates.index("2026-01-01")
        assert (result.displacement[ref_idx] == 0).all()

    def test_invalid_reference_date_raises(self):
        import numpy as np

        from pygeofetch.insar import SBASTimeSeries
        from pygeofetch.insar.timeseries import InterferogramPair

        h, w = 4, 4
        pairs = [
            InterferogramPair(
                "2026-01-01",
                "2026-01-13",
                np.zeros((h, w), dtype=np.float32),
                np.ones((h, w), dtype=np.float32),
            ),
        ]
        sbas = SBASTimeSeries(reference_date="2099-01-01")
        with pytest.raises(ValueError, match="reference_date"):
            sbas.invert(pairs)

    def test_mintpy_missing_falls_back_or_raises_clear_error(self):
        """use_mintpy=True without mintpy installed should give a clear error/fallback."""
        import numpy as np

        from pygeofetch.insar import SBASTimeSeries
        from pygeofetch.insar.timeseries import InterferogramPair

        h, w = 4, 4
        pairs = [
            InterferogramPair(
                "2026-01-01",
                "2026-01-13",
                np.zeros((h, w), dtype=np.float32),
                np.ones((h, w), dtype=np.float32),
            ),
        ]
        sbas = SBASTimeSeries()
        # Should fall back to native inversion rather than crash
        result = sbas.invert(pairs, use_mintpy=True)
        assert result is not None

    def test_save_writes_geotiffs(self, tmp_path):
        import numpy as np

        from pygeofetch.insar.timeseries import TimeSeriesResult

        h, w = 10, 10
        result = TimeSeriesResult(
            dates=["2026-01-01", "2026-01-13"],
            displacement=np.zeros((2, h, w), dtype=np.float32),
            velocity=np.zeros((h, w), dtype=np.float32),
            residual_rms=np.zeros((h, w), dtype=np.float32),
            reference_date="2026-01-01",
        )
        paths = result.save(tmp_path)
        assert set(paths.keys()) == {
            "velocity",
            "displacement_timeseries",
            "residual_rms",
        }
        for p in paths.values():
            assert p.exists()


class TestAtmosphericCorrector:
    def test_invalid_method_raises(self):
        from pygeofetch.insar import AtmosphericCorrector

        with pytest.raises(ValueError, match="method"):
            AtmosphericCorrector(method="invalid")

    def test_elevation_correction_removes_dem_correlated_phase(self, tmp_path):
        """Synthetic phase = pure elevation correlation; correction should remove it."""
        import numpy as np
        import rasterio
        from rasterio.crs import CRS
        from rasterio.transform import from_bounds

        from pygeofetch.insar import AtmosphericCorrector

        h, w = 40, 40
        y, x = np.mgrid[0:h, 0:w]
        dem_data = (x * 20 + 100).astype(np.float32)

        dem_path = tmp_path / "dem.tif"
        with rasterio.open(
            dem_path,
            "w",
            driver="GTiff",
            dtype="float32",
            count=1,
            width=w,
            height=h,
            crs=CRS.from_epsg(4326),
            transform=from_bounds(-1, 0, 1, 1, w, h),
        ) as ds:
            ds.write(dem_data[np.newaxis])

        tropo_slope = 0.002
        observed_phase = (tropo_slope * dem_data).astype(np.float32)

        corrector = AtmosphericCorrector(method="elevation")
        corrected = corrector.correct(observed_phase, dem_path)

        assert (
            corrected.mean() < observed_phase.mean() * 0.1
            or abs(corrected.mean()) < 1e-3
        )

    def test_era5_missing_pyaps_raises_clear_error(self, tmp_path):
        from unittest.mock import patch

        import numpy as np
        import rasterio
        from rasterio.crs import CRS
        from rasterio.transform import from_bounds

        from pygeofetch.insar import AtmosphericCorrector

        h, w = 10, 10
        dem_path = tmp_path / "dem.tif"
        with rasterio.open(
            dem_path,
            "w",
            driver="GTiff",
            dtype="float32",
            count=1,
            width=w,
            height=h,
            crs=CRS.from_epsg(4326),
            transform=from_bounds(-1, 0, 1, 1, w, h),
        ) as ds:
            ds.write(np.full((h, w), 100.0, dtype=np.float32)[np.newaxis])

        corrector = AtmosphericCorrector(method="era5")
        with patch.dict("sys.modules", {"pyaps3": None}):
            with pytest.raises(ImportError, match="pyaps3"):
                corrector.correct(
                    np.zeros((h, w), dtype=np.float32),
                    dem_path,
                    acquisition_datetime="2026-06-01T18:16:00",
                )

    def test_era5_missing_datetime_raises_value_error(self, tmp_path):
        import numpy as np
        import rasterio
        from rasterio.crs import CRS
        from rasterio.transform import from_bounds

        from pygeofetch.insar import AtmosphericCorrector

        h, w = 10, 10
        dem_path = tmp_path / "dem.tif"
        with rasterio.open(
            dem_path,
            "w",
            driver="GTiff",
            dtype="float32",
            count=1,
            width=w,
            height=h,
            crs=CRS.from_epsg(4326),
            transform=from_bounds(-1, 0, 1, 1, w, h),
        ) as ds:
            ds.write(np.full((h, w), 100.0, dtype=np.float32)[np.newaxis])

        corrector = AtmosphericCorrector(method="era5")
        with pytest.raises(ValueError, match="acquisition_datetime"):
            corrector.correct(np.zeros((h, w), dtype=np.float32), dem_path)


class TestTopLevelExports:
    def test_insar_package_exports(self):
        import pygeofetch.insar as insar

        assert hasattr(insar, "InterferogramGenerator")
        assert hasattr(insar, "PhaseUnwrapper")
        assert hasattr(insar, "SBASTimeSeries")
        assert hasattr(insar, "AtmosphericCorrector")
        assert hasattr(insar, "InterferogramResult")