"""
PyGeoFetch InSAR — a state-of-the-art Interferometric SAR processing chain.

Install with: pip install "pygeofetch[insar]"
Adds time-series inversion:  pip install "pygeofetch[insar-full]"

This module implements the full InSAR processing chain as practiced by
ASF HyP3, ISCE2/3, GAMMA, and SNAP, using pure-Python, pip-installable
components wherever a proven one exists:

  1. Coregistration    — geometric (orbit + DEM) + Enhanced Spectral
                          Diversity (ESD) refinement to <0.001 px accuracy,
                          required for TOPS burst-overlap phase continuity
                          (Prats-Iraola et al. 2012; Yagüe-Martínez et al. 2016)
  2. Interferogram      — complex conjugate multiplication + topographic
                          phase removal using a reference DEM
  3. Coherence          — already implemented in pygeofetch.processing.sar
  4. Phase unwrapping   — SNAPHU (Chen & Zebker 2001) via the official
                          snaphu-py bindings — the same algorithm used by
                          ASF, ISCE2/3, and GAMMA
  5. Atmospheric        — ERA5-based tropospheric delay correction
     correction           (Jolivet et al. 2011, 2014 — the PyAPS method)
  6. Time series         — Small BAseline Subset (SBAS) inversion
                          (Berardino et al. 2002; Yunjun et al. 2019 — MintPy)

References:
  Chen, C.W. & Zebker, H.A. (2001). Two-dimensional phase unwrapping with
    use of statistical models for cost functions in a network programming
    framework. J. Opt. Soc. Am. A, 18(2), 338-351.
  Yunjun, Z., Fattahi, H., Amelung, F. (2019). Small baseline InSAR time
    series analysis: unwrapping error correction and noise reduction.
    Computers & Geosciences, 133, 104331.
  Prats-Iraola, P. et al. (2012). TOPS interferometry with TerraSAR-X.
    IEEE TGRS, 50(8), 3179-3188.
  Jolivet, R. et al. (2014). Improving InSAR geodesy using Global
    Atmospheric Models. JGR Solid Earth, 119(3), 2019-2034.

Usage::

    from pygeofetch.insar import InterferogramGenerator, PhaseUnwrapper

    gen = InterferogramGenerator()
    result = gen.process_pair(
        reference="slc_20260601.tif",
        secondary="slc_20260613.tif",
        dem="dem.tif",
    )

    unwrapper = PhaseUnwrapper()
    unwrapped = unwrapper.unwrap(result.interferogram, result.coherence)
"""

from pygeofetch.insar.atmosphere import AtmosphericCorrector
from pygeofetch.insar.extraction import SLCExtractor
from pygeofetch.insar.interferogram import InterferogramGenerator, InterferogramResult
from pygeofetch.insar.timeseries import SBASTimeSeries
from pygeofetch.insar.unwrap import PhaseUnwrapper

__all__ = [
    "InterferogramGenerator",
    "InterferogramResult",
    "PhaseUnwrapper",
    "SBASTimeSeries",
    "AtmosphericCorrector",
    "SLCExtractor",
]