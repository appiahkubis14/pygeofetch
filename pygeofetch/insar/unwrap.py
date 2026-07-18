"""
PhaseUnwrapper — production-grade phase unwrapping via SNAPHU.

Uses snaphu-py (https://github.com/isce-framework/snaphu-py), the official
Python bindings for SNAPHU (Statistical-cost, Network-flow Algorithm for
Phase Unwrapping), maintained by the same JPL/Caltech team behind ISCE2/3.

SNAPHU is the algorithm used in production by:
  - ASF HyP3's On Demand InSAR products (via GAMMA's MCF variant)
  - ESA SNAP (bundled as an external unwrapping step)
  - ISCE2/ISCE3 (native binding, same as snaphu-py)
  - GMTSAR

Reference:
  Chen, C.W. & Zebker, H.A. (2001). Two-dimensional phase unwrapping with
  use of statistical models for cost functions in a network programming
  framework. Journal of the Optical Society of America A, 18(2), 338-351.

Install: pip install "pygeofetch[insar]"   (installs snaphu, scipy)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Tuple, Union

logger = logging.getLogger("pygeofetch.insar.unwrap")


def _require_snaphu():
    try:
        import snaphu

        return snaphu
    except ImportError:
        raise ImportError(
            "snaphu-py is not installed.\n"
            'Install with: pip install "pygeofetch[insar]"\n'
            "Or directly:  pip install snaphu\n\n"
            "snaphu-py provides Python bindings for SNAPHU, the same "
            "phase-unwrapping algorithm (Chen & Zebker 2001) used by "
            "ASF, ISCE2/3, GAMMA, and SNAP."
        )


class PhaseUnwrapper:
    """
    Phase unwrapping via SNAPHU (Statistical-cost, Network-flow Algorithm).

    Args:
        cost_mode: SNAPHU cost function — ``"topo"`` (default, for terrain),
                   ``"defo"`` (for deformation — less smoothing bias),
                   ``"smooth"`` (generic smoothness prior),
                   ``"nostatcosts"`` (uniform cost, fastest, least accurate).
        init_method: ``"mcf"`` (default — Minimum Cost Flow, matches ASF/GAMMA)
                     or ``"mst"`` (Minimum Spanning Tree — faster, less optimal).

    Example::

        from pygeofetch.insar import InterferogramGenerator, PhaseUnwrapper

        gen    = InterferogramGenerator()
        result = gen.process_pair("ref.tif", "sec.tif", dem="dem.tif")

        unwrapper = PhaseUnwrapper(cost_mode="defo")
        unwrapped, conncomp = unwrapper.unwrap(
            result.interferogram, result.coherence
        )

    Choosing cost_mode:
        - Use ``"topo"`` for DEM generation / topographic mapping tasks
          where phase gradients follow terrain.
        - Use ``"defo"`` for deformation monitoring (subsidence, volcanic,
          earthquake) where phase gradients follow displacement, not
          terrain — this is the standard choice for MintPy/SBAS workflows.
    """

    def __init__(self, cost_mode: str = "defo", init_method: str = "mcf") -> None:
        valid_costs = ("topo", "defo", "smooth", "nostatcosts")
        if cost_mode not in valid_costs:
            raise ValueError(
                f"cost_mode must be one of {valid_costs}, got {cost_mode!r}"
            )
        valid_inits = ("mcf", "mst")
        if init_method not in valid_inits:
            raise ValueError(
                f"init_method must be one of {valid_inits}, got {init_method!r}"
            )
        self._cost_mode = cost_mode
        self._init_method = init_method

    def unwrap(
        self,
        interferogram: Any,
        coherence: Any,
        nlooks: float = 1.0,
        mask: Optional[Any] = None,
    ) -> Tuple[Any, Any]:
        """
        Unwrap a wrapped interferogram phase.

        Args:
            interferogram: Complex64 array (wrapped interferogram) OR
                           float32 array of wrapped phase in radians.
            coherence:     Float32 coherence array, 0-1, same shape.
            nlooks:        Effective number of looks (affects statistical
                           cost weighting). Use the multilook factor applied
                           during interferogram formation.
            mask:          Optional boolean array — True = valid, False = masked out
                           (e.g. water bodies, layover/shadow regions).

        Returns:
            (unwrapped_phase, conncomp) — both same shape as input.
            unwrapped_phase: float32 radians, continuous (not wrapped to [-pi, pi))
            conncomp:        int32 connected-component labels (0 = unreliable/masked)

        Example::

            unwrapped, conncomp = unwrapper.unwrap(igram, coherence, nlooks=4.0)
            # conncomp == 0 marks pixels SNAPHU could not confidently unwrap
            reliable = conncomp > 0
        """
        np = self._np()
        sx = _require_snaphu()

        if np.iscomplexobj(interferogram):
            igram = interferogram.astype(np.complex64)
        else:
            # Treat as wrapped phase in radians — convert to unit-magnitude complex
            igram = np.exp(1j * interferogram).astype(np.complex64)

        corr = np.clip(coherence, 0.0, 1.0).astype(np.float32)

        if mask is not None:
            corr = np.where(mask, corr, 0.0).astype(np.float32)

        logger.info(
            "Unwrapping %s pixels (cost=%s, init=%s, nlooks=%.1f)",
            f"{igram.shape[0]}x{igram.shape[1]}",
            self._cost_mode,
            self._init_method,
            nlooks,
        )

        try:
            unwrapped, conncomp = sx.unwrap(
                igram,
                corr,
                nlooks=nlooks,
                cost=self._cost_mode,
                init=self._init_method,
            )
        except Exception as exc:
            raise RuntimeError(
                f"SNAPHU unwrapping failed: {exc}\n"
                "Common causes: incompatible array shapes, all-zero coherence, "
                "or insufficient memory for large scenes. Consider multilooking "
                "the interferogram first to reduce pixel count."
            ) from exc

        n_unreliable = int(np.sum(conncomp == 0))
        pct = 100 * n_unreliable / conncomp.size
        if pct > 30:
            logger.warning(
                "%.1f%% of pixels are in the unreliable connected component "
                "(conncomp==0). Consider improving coherence via multilooking "
                "or filtering, or check for large decorrelated areas.",
                pct,
            )
        else:
            logger.info("Unwrapping complete — %.1f%% unreliable pixels", pct)

        return unwrapped.astype(np.float32), conncomp

    def unwrap_files(
        self,
        interferogram_path: Union[str, Path],
        coherence_path: Union[str, Path],
        output_path: Union[str, Path],
        nlooks: float = 1.0,
        mask_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """
        Unwrap directly from/to GeoTIFF files, preserving georeferencing.

        Args:
            interferogram_path: Wrapped phase or complex interferogram GeoTIFF.
            coherence_path:     Coherence GeoTIFF (0-1).
            output_path:        Output path for the unwrapped phase GeoTIFF.
            nlooks:              Effective number of looks.
            mask_path:           Optional binary mask GeoTIFF (1=valid, 0=masked).

        Returns:
            Path to the unwrapped phase GeoTIFF.

        Example::

            unwrapper.unwrap_files(
                "wrapped_phase.tif", "coherence.tif",
                output_path="unwrapped.tif", nlooks=4.0,
            )
        """
        np = self._np()
        try:
            import rasterio
        except ImportError:
            raise ImportError('rasterio required: pip install "pygeofetch[geo]"')

        with rasterio.open(interferogram_path) as src:
            profile = src.profile.copy()
            phase = src.read(1).astype(np.float32)

        with rasterio.open(coherence_path) as src:
            coherence = src.read(1).astype(np.float32)

        mask = None
        if mask_path:
            with rasterio.open(mask_path) as src:
                mask = src.read(1).astype(bool)

        unwrapped, conncomp = self.unwrap(phase, coherence, nlooks=nlooks, mask=mask)

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        out_profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "count": 1,
            "height": unwrapped.shape[0],
            "width": unwrapped.shape[1],
            "crs": profile.get("crs"),
            "transform": profile.get("transform"),
            "nodata": -9999.0,
            "compress": "deflate",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        }
        with rasterio.open(out_path, "w", **out_profile) as dst:
            dst.write(unwrapped[np.newaxis])
            dst.update_tags(1, description="unwrapped_phase_radians")

        conncomp_path = out_path.parent / f"{out_path.stem}_conncomp.tif"
        cc_profile = dict(out_profile, dtype="int32", nodata=0)
        with rasterio.open(conncomp_path, "w", **cc_profile) as dst:
            dst.write(conncomp.astype(np.int32)[np.newaxis])

        logger.info(
            "Unwrapped phase → %s (conncomp → %s)", out_path.name, conncomp_path.name
        )
        return out_path

    def _np(self):
        import numpy as np

        return np
