"""
SARProcessor — factory selecting the appropriate SAR processing backend.

Backends:
  "native"    — PyGeoFetch built-in (always available, no extra deps)
  "sarxarray" — sarxarray integration (pip install pygeofetch[sar])
  "ost"       — OpenSARToolkit/SNAP (pip install pygeofetch[ost], requires SNAP)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Union

logger = logging.getLogger("pygeofetch.sar.processor")


class SARProcessor:
    """
    SAR processing with pluggable backends.

    Args:
        backend: ``"native"`` (default), ``"sarxarray"``, or ``"ost"``.

    Example::

        from pygeofetch.sar import SARProcessor

        # Native (always works, no extra deps)
        proc = SARProcessor()
        r    = proc.despeckle("s1_vv.tif", filter="lee")

        # sarxarray backend (richer xarray-native workflow)
        proc = SARProcessor(backend="sarxarray")
        r    = proc.calibrate("s1_dn.tif", output_type="sigma0")

        # OST/SNAP backend (production-grade but requires SNAP)
        proc = SARProcessor(backend="ost")
        r    = proc.terrain_correct("s1_cal.tif", dem="srtm")
    """

    def __init__(self, backend: str = "native") -> None:
        if backend not in ("native", "sarxarray", "ost"):
            raise ValueError(
                f"Unknown SAR backend: {backend!r}. "
                "Choose from: 'native', 'sarxarray', 'ost'"
            )
        self._backend_name = backend
        self._backend = None

    def _load_backend(self):
        if self._backend is not None:
            return self._backend

        if self._backend_name == "native":
            from pygeofetch.sar._native import NativeSARBackend

            self._backend = NativeSARBackend()

        elif self._backend_name == "sarxarray":
            from pygeofetch.sar._sarxarray import SARXarrayBackend

            self._backend = SARXarrayBackend()

        elif self._backend_name == "ost":
            from pygeofetch.sar._ost import OSTBackend

            self._backend = OSTBackend()

        return self._backend

    # ── Public API — delegates to backend ─────────────────────────────────────

    def despeckle(
        self,
        input: Union[str, Path],
        filter: str = "lee",
        window: int = 5,
        output: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """Apply speckle filter. See backend docs for full parameter list."""
        return self._load_backend().despeckle(
            Path(input), filter=filter, window=window, output=output, **kwargs
        )

    def calibrate(
        self,
        input: Union[str, Path],
        output_type: str = "sigma0",
        in_db: bool = True,
        output: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """Radiometric calibration (DN → sigma0/gamma0/beta0)."""
        return self._load_backend().calibrate(
            Path(input), output_type=output_type, in_db=in_db, output=output, **kwargs
        )

    def terrain_correct(
        self,
        input: Union[str, Path],
        dem: str = "srtm",
        output: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """Terrain correction / Range-Doppler geocoding."""
        return self._load_backend().terrain_correct(
            Path(input), dem=dem, output=output, **kwargs
        )

    def flood_map(
        self,
        input: Union[str, Path],
        threshold: float = -15.0,
        reference: Optional[Union[str, Path]] = None,
        output: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """Flood mapping via backscatter threshold or change detection."""
        return self._load_backend().flood_map(
            Path(input),
            threshold=threshold,
            reference=Path(reference) if reference else None,
            output=output,
            **kwargs,
        )

    def coherence(
        self,
        image1: Union[str, Path],
        image2: Union[str, Path],
        window: int = 7,
        output: Optional[str] = None,
        **kwargs,
    ) -> Any:
        """InSAR coherence estimation between two co-registered SLC images."""
        return self._load_backend().coherence(
            Path(image1), Path(image2), window=window, output=output, **kwargs
        )

    @property
    def backend(self) -> str:
        return self._backend_name
