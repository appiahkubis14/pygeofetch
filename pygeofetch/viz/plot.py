"""Plotter — static and time-series plots via matplotlib."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Tuple, Union

logger = logging.getLogger("pygeofetch.viz.plot")


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is not installed.\n"
            'Install with: pip install "pygeofetch[viz]"\n'
            "Or directly:  pip install matplotlib"
        )


class Plotter:
    """
    Static and time-series plots for satellite data.

    Example::

        from pygeofetch.viz import Plotter

        pl = Plotter()

        # Single raster
        pl.plot_raster("ndvi.tif", title="NDVI 2024-06-01",
                       colormap="RdYlGn", output="ndvi.png")

        # RGB composite
        pl.plot_rgb("B04.tif", "B03.tif", "B02.tif",
                    title="True Colour", output="rgb.png")

        # Time series
        pl.plot_timeseries(
            {"Jan": 0.35, "Feb": 0.38, "Mar": 0.61, "Jun": 0.72},
            title="Mean NDVI 2024", ylabel="NDVI", output="ts.png"
        )
    """

    def __init__(self, figsize: Tuple[int, int] = (10, 8), dpi: int = 150) -> None:
        self._figsize = figsize
        self._dpi = dpi

    def plot_raster(
        self,
        path: Union[str, Path],
        title: str = "",
        colormap: str = "viridis",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        output: Optional[Union[str, Path]] = None,
        band: int = 1,
    ) -> Any:
        """
        Plot a single-band raster.

        Args:
            path:     Path to GeoTIFF.
            title:    Plot title.
            colormap: Matplotlib colormap.
            vmin/vmax: Color scale range (auto if None).
            output:   Save path (.png/.pdf/.svg). Show in notebook if None.
            band:     1-based band index.
        """
        import numpy as np

        plt = _require_matplotlib()

        try:
            import rasterio

            with rasterio.open(path) as src:
                data = src.read(band).astype(np.float32)
                nodata = src.nodata
                if nodata is not None:
                    data = np.where(data == nodata, np.nan, data)
                extent = [
                    src.bounds.left,
                    src.bounds.right,
                    src.bounds.bottom,
                    src.bounds.top,
                ]
        except ImportError:
            raise ImportError('rasterio required: pip install "pygeofetch[geo]"')

        fig, ax = plt.subplots(1, 1, figsize=self._figsize)
        im = ax.imshow(
            data,
            cmap=colormap,
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            origin="upper",
            aspect="auto",
        )
        plt.colorbar(im, ax=ax, fraction=0.03)
        ax.set_title(title or Path(path).stem, fontsize=13)
        ax.set_xlabel("Longitude" if extent[0] < 360 else "Easting")
        ax.set_ylabel("Latitude" if extent[2] < 90 else "Northing")
        plt.tight_layout()

        return self._save_or_show(fig, output)

    def plot_rgb(
        self,
        red: Union[str, Path],
        green: Union[str, Path],
        blue: Union[str, Path],
        title: str = "RGB Composite",
        percentile: Tuple[float, float] = (2, 98),
        output: Optional[Union[str, Path]] = None,
    ) -> Any:
        """
        Plot an RGB true/false colour composite.

        Args:
            red, green, blue: Paths to single-band rasters.
            title:            Plot title.
            percentile:       Stretch percentiles for display.
            output:           Save path.
        """
        import numpy as np

        plt = _require_matplotlib()

        try:
            import rasterio
        except ImportError:
            raise ImportError('rasterio required: pip install "pygeofetch[geo]"')

        bands = []
        extent = None
        for p in [red, green, blue]:
            with rasterio.open(p) as src:
                data = src.read(1).astype(np.float32)
                if extent is None:
                    extent = [
                        src.bounds.left,
                        src.bounds.right,
                        src.bounds.bottom,
                        src.bounds.top,
                    ]
                lo, hi = np.nanpercentile(data, percentile)
                data = np.clip((data - lo) / (hi - lo + 1e-10), 0, 1)
                bands.append(data)

        rgb = np.stack(bands, axis=-1)
        fig, ax = plt.subplots(1, 1, figsize=self._figsize)
        ax.imshow(rgb, extent=extent, origin="upper", aspect="auto")
        ax.set_title(title, fontsize=13)
        plt.tight_layout()
        return self._save_or_show(fig, output)

    def plot_timeseries(
        self,
        data: dict,
        title: str = "Time Series",
        ylabel: str = "Value",
        output: Optional[Union[str, Path]] = None,
        color: str = "steelblue",
    ) -> Any:
        """
        Plot a time series from a dict of {date_label: value}.

        Args:
            data:   Ordered dict of {label: scalar_value}.
            title:  Plot title.
            ylabel: Y-axis label.
            output: Save path.
        """
        plt = _require_matplotlib()

        labels = list(data.keys())
        values = list(data.values())

        fig, ax = plt.subplots(1, 1, figsize=self._figsize)
        ax.plot(labels, values, marker="o", color=color, linewidth=2, markersize=6)
        ax.fill_between(labels, values, alpha=0.15, color=color)
        ax.set_title(title, fontsize=13)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        return self._save_or_show(fig, output)

    def plot_histogram(
        self,
        path: Union[str, Path],
        title: str = "",
        bins: int = 50,
        output: Optional[Union[str, Path]] = None,
        band: int = 1,
    ) -> Any:
        """Plot a histogram of raster values."""
        import numpy as np

        plt = _require_matplotlib()

        try:
            import rasterio

            with rasterio.open(path) as src:
                data = src.read(band).astype(np.float32).ravel()
                nodata = src.nodata
                if nodata is not None:
                    data = data[data != nodata]
        except ImportError:
            raise ImportError('rasterio required: pip install "pygeofetch[geo]"')

        data = data[np.isfinite(data)]
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.hist(data, bins=bins, color="steelblue", edgecolor="white", linewidth=0.3)
        ax.set_title(title or f"{Path(path).stem} histogram", fontsize=13)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        plt.tight_layout()
        return self._save_or_show(fig, output)

    def _save_or_show(self, fig: Any, output: Optional[Union[str, Path]]) -> Any:
        import matplotlib.pyplot as plt

        if output:
            out = Path(output)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(out), dpi=self._dpi, bbox_inches="tight")
            plt.close(fig)
            logger.info("Plot saved → %s", out)
            return out
        return fig
