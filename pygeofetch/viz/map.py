"""MapViewer — interactive web maps via leafmap/folium."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional, Union

logger = logging.getLogger("pygeofetch.viz.map")


def _require_leafmap():
    try:
        import leafmap

        return leafmap
    except ImportError:
        raise ImportError(
            "leafmap is not installed.\n"
            'Install with: pip install "pygeofetch[viz]"\n'
            "Or directly:  pip install leafmap"
        )


class MapViewer:
    """
    Interactive map viewer backed by leafmap (which supports folium, ipyleaflet, etc.).

    Args:
        backend: ``"leafmap"`` (default) or ``"folium"``.
        center:  (lat, lon) map center. Auto-detected from first layer if None.
        zoom:    Initial zoom level.

    Example::

        from pygeofetch.viz import MapViewer

        mv = MapViewer()
        mv.add_raster("ndvi.tif",  colormap="RdYlGn", layer_name="NDVI")
        mv.add_vector("flood.geojson", style={"color": "blue"})
        mv.show()                    # Jupyter
        mv.save("map.html")          # Static HTML
    """

    def __init__(
        self,
        backend: str = "leafmap",
        center: Optional[tuple] = None,
        zoom: int = 8,
    ) -> None:
        self._backend = backend
        self._center = center or [0, 0]
        self._zoom = zoom
        self._map = None
        self._layers: List[dict] = []

    def _get_map(self):
        if self._map is None:
            lm = _require_leafmap()
            self._map = lm.Map(center=self._center, zoom=self._zoom)
        return self._map

    def add_raster(
        self,
        path: Union[str, Path],
        colormap: str = "viridis",
        layer_name: Optional[str] = None,
        opacity: float = 0.8,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> "MapViewer":
        """
        Add a raster layer to the map.

        Args:
            path:       Path to GeoTIFF.
            colormap:   Matplotlib colormap name.
            layer_name: Display name (defaults to filename stem).
            opacity:    Layer opacity 0–1.
            vmin/vmax:  Value range for colormap scaling.
        """
        p = Path(path)
        name = layer_name or p.stem
        m = self._get_map()
        try:
            m.add_raster(
                str(p),
                colormap=colormap,
                layer_name=name,
                opacity=opacity,
                vmin=vmin,
                vmax=vmax,
            )
            logger.info("Raster layer added: %s", name)
        except Exception as exc:
            logger.warning("Could not add raster %s: %s", p.name, exc)
        self._layers.append({"type": "raster", "path": str(p), "name": name})
        return self

    def add_vector(
        self,
        path: Union[str, Path],
        layer_name: Optional[str] = None,
        style: Optional[dict] = None,
    ) -> "MapViewer":
        """Add a vector layer (GeoJSON, GeoPackage, Shapefile)."""
        p = Path(path)
        name = layer_name or p.stem
        m = self._get_map()
        style = style or {"color": "red", "weight": 1, "fillOpacity": 0.3}
        try:
            m.add_vector(str(p), layer_name=name, style=style)
            logger.info("Vector layer added: %s", name)
        except Exception as exc:
            logger.warning("Could not add vector %s: %s", p.name, exc)
        self._layers.append({"type": "vector", "path": str(p), "name": name})
        return self

    def add_basemap(self, name: str = "OpenStreetMap") -> "MapViewer":
        """Add a basemap tile layer."""
        m = self._get_map()
        try:
            m.add_basemap(name)
        except Exception as exc:
            logger.warning("Could not add basemap %r: %s", name, exc)
        return self

    def show(self) -> Any:
        """Display the map (Jupyter only)."""
        return self._get_map()

    def save(self, output: Union[str, Path]) -> Path:
        """Save the map as a standalone HTML file."""
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        self._get_map().to_html(str(out))
        logger.info("Map saved → %s", out)
        return out

    @property
    def layers(self) -> List[dict]:
        return self._layers
