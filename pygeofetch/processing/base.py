"""Base classes and result types for the processing engine."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Outcome of a single processing step."""
    success: bool
    output_path: Optional[Path] = None
    input_path: Optional[Path] = None
    operation: str = ""
    duration_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def path(self) -> Optional[Path]:
        return self.output_path

    def __str__(self) -> str:
        status = "✓" if self.success else "✗"
        out = str(self.output_path.name) if self.output_path else "none"
        return f"{status} {self.operation} → {out} ({self.duration_seconds:.2f}s)"


def _require_rasterio():
    """Raise ImportError with install hint if rasterio is missing."""
    try:
        import rasterio
        return rasterio
    except ImportError:
        raise ImportError(
            "rasterio is required for raster processing.\n"
            "Install with: pip install rasterio\n"
            "Or: pip install \"PyGeoFetch[geo]\""
        )


def _require_numpy():
    try:
        import numpy as np
        return np
    except ImportError:
        raise ImportError("numpy is required: pip install numpy")


def _require_geopandas():
    try:
        import geopandas as gpd
        return gpd
    except ImportError:
        raise ImportError(
            "geopandas is required for vector operations.\n"
            "Install with: pip install geopandas"
        )


def _require_shapely():
    try:
        import shapely
        return shapely
    except ImportError:
        raise ImportError("shapely is required: pip install shapely")


def _timed(func):
    """Decorator to measure execution time and wrap result."""
    from functools import wraps
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        try:
            result = func(*args, **kwargs)
            if isinstance(result, ProcessingResult):
                result.duration_seconds = time.time() - t0
            return result
        except Exception as exc:
            elapsed = time.time() - t0
            logger.error(f"{func.__name__} failed after {elapsed:.2f}s: {exc}")
            raise
    return wrapper


def _resolve_output(input_path: Path, output: Optional[str], suffix: str) -> Path:
    """Build an output path from input + suffix if output not specified."""
    if output:
        out = Path(output)
        if out.is_dir() or str(output).endswith("/"):
            out.mkdir(parents=True, exist_ok=True)
            stem = input_path.stem if input_path else "output"
            return out / f"{stem}_{suffix}.tif"
        out.parent.mkdir(parents=True, exist_ok=True)
        return out
    p = input_path or Path("output.tif")
    return p.parent / f"{p.stem}_{suffix}.tif"
