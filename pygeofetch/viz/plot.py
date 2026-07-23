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

    def plot_3d_terrain(
        self,
        data: Union[str, Path, Any],
        title: str = "",
        colormap: str = "terrain",
        drape: Optional[Any] = None,
        drape_colormap: str = "Blues",
        drape_alpha: float = 0.6,
        azimuth: float = 315.0,
        altitude: float = 45.0,
        vert_exaggeration: float = 1.5,
        max_render_dim: int = 250,
        elev_angle: float = 35.0,
        azim_angle: float = -60.0,
        output: Optional[Union[str, Path]] = None,
        colorbar_label: Optional[str] = None,
    ) -> Any:
        """
        3D-render a DEM as an illuminated terrain surface — from a file
        path OR directly from an in-memory numpy array, same convention
        as plot_raster().

        Args:
            data:      Path to a DEM GeoTIFF, OR a 2D numpy array of
                       elevation values.
            title:     Plot title.
            colormap:  Matplotlib colormap for the terrain surface itself.
            drape:     Optional second array or path to overlay ON the 3D
                       surface — e.g. a flood susceptibility classification,
                       TWI, or observed flood extent — so risk/analysis
                       results can be shown directly on the real terrain
                       shape rather than as a flat 2D map. Must share the
                       DEM's grid (same shape).
            drape_colormap: Colormap for the drape layer.
            drape_alpha: Opacity of the drape layer over the base terrain shading.
            azimuth:   Sun azimuth for surface illumination, degrees
                       (cartographic convention: 0=N, 90=E, 180=S, 270=W).
            altitude:  Sun altitude above the horizon, degrees.
            vert_exaggeration: Vertical exaggeration factor — real terrain
                       relief is often subtle relative to its horizontal
                       extent; 1.5-3x is typical for a readable 3D plot
                       without visually lying about the real proportions.
            max_render_dim: Downsample the grid so its largest dimension
                       doesn't exceed this — full-resolution DEMs (often
                       thousands of pixels per side) render extremely
                       slowly and are not visually distinguishable from a
                       downsampled version in a 3D view.
            elev_angle: Camera elevation angle for the 3D view.
            azim_angle: Camera azimuth angle for the 3D view.
            output:    Save path (.png/.pdf/.svg). Show in notebook if None.
            colorbar_label: Label for the terrain colorbar.

        Example::

            pl.plot_3d_terrain("dem.tif", drape="flood_susceptibility.tif",
                               title="Flood Susceptibility Draped on Terrain")
        """
        import numpy as np

        plt = _require_matplotlib()
        from matplotlib.colors import LightSource
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)

        def _load(source, band=1):
            if isinstance(source, (str, Path)):
                import rasterio

                with rasterio.open(source) as src:
                    arr = src.read(band).astype(np.float32)
                    nodata = src.nodata
                    if nodata is not None:
                        arr = np.where(arr == nodata, np.nan, arr)
                    ext = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
                return arr, ext
            return np.asarray(source, dtype=np.float32), None

        dem, plot_extent = _load(data)
        drape_arr = None
        if drape is not None:
            drape_arr, drape_extent = _load(drape)
            if drape_arr.shape != dem.shape:
                raise ValueError(
                    f"drape shape {drape_arr.shape} doesn't match the DEM's "
                    f"shape {dem.shape} — they must share the same grid. "
                    f"Use Preprocessor.resample(reference=...) to align them first."
                )
            plot_extent = plot_extent or drape_extent

        step = max(1, max(dem.shape) // max_render_dim)
        dem_ds = dem[::step, ::step]
        h, w = dem_ds.shape

        if plot_extent is not None:
            lon = np.linspace(plot_extent[0], plot_extent[1], w)
            lat = np.linspace(plot_extent[3], plot_extent[2], h)
        else:
            lon = np.arange(w)
            lat = np.arange(h)
        lon_mesh, lat_mesh = np.meshgrid(lon, lat)

        ls = LightSource(azdeg=azimuth, altdeg=altitude)
        rgb = ls.shade(dem_ds, cmap=plt.get_cmap(colormap), vert_exag=vert_exaggeration, blend_mode="soft")

        if drape_arr is not None:
            drape_ds = drape_arr[::step, ::step]
            drape_norm = plt.Normalize(vmin=np.nanmin(drape_ds), vmax=np.nanmax(drape_ds))
            drape_rgba = plt.get_cmap(drape_colormap)(drape_norm(drape_ds))
            valid = ~np.isnan(drape_ds)
            blended_rgb = np.where(
                valid[..., None],
                rgb[..., :3] * (1 - drape_alpha) + drape_rgba[..., :3] * drape_alpha,
                rgb[..., :3],
            )
            rgb = np.concatenate([blended_rgb, rgb[..., 3:4]], axis=-1)

        fig = plt.figure(figsize=self._figsize)
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(
            lon_mesh, lat_mesh, dem_ds, facecolors=rgb,
            rstride=1, cstride=1, antialiased=True, shade=False,
        )
        ax.set_title(title or "3D Terrain", fontsize=13)
        if plot_extent is not None:
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
        ax.set_zlabel(colorbar_label or "Elevation (m)")
        ax.view_init(elev=elev_angle, azim=azim_angle)
        plt.tight_layout()

        return self._save_or_show(fig, output)

    def plot_raster(
        self,
        data: Union[str, Path, Any],
        title: str = "",
        colormap: str = "viridis",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        output: Optional[Union[str, Path]] = None,
        band: int = 1,
        extent: Optional[Tuple[float, float, float, float]] = None,
        colorbar_label: Optional[str] = None,
    ) -> Any:
        """
        Plot a single-band raster — from a file path OR directly from an
        in-memory numpy array, so results from LandsatExtractor,
        SpectralIndex, SBASTimeSeries, etc. can be plotted straight away
        without a round-trip through disk first.

        Args:
            data:      Path to a GeoTIFF, OR a 2D numpy array.
            title:     Plot title.
            colormap:  Matplotlib colormap.
            vmin/vmax: Color scale range (auto if None).
            output:    Save path (.png/.pdf/.svg). Show in notebook if None.
            band:      1-based band index (only used when `data` is a path).
            extent:    (min_lon, max_lon, min_lat, max_lat) — required for
                       correct axis labelling when passing an array
                       directly; ignored (derived from the file) when
                       `data` is a path.
            colorbar_label: Label for the colorbar (e.g. "NDVI", "m/year").
        """
        import numpy as np

        plt = _require_matplotlib()

        if isinstance(data, (str, Path)):
            try:
                import rasterio

                with rasterio.open(data) as src:
                    arr = src.read(band).astype(np.float32)
                    nodata = src.nodata
                    if nodata is not None:
                        arr = np.where(arr == nodata, np.nan, arr)
                    plot_extent = [
                        src.bounds.left,
                        src.bounds.right,
                        src.bounds.bottom,
                        src.bounds.top,
                    ]
            except ImportError:
                raise ImportError('rasterio required: pip install "pygeofetch[geo]"')
            default_title = Path(data).stem
        else:
            arr = np.asarray(data, dtype=np.float32)
            plot_extent = list(extent) if extent is not None else None
            default_title = title or "Raster"

        fig, ax = plt.subplots(1, 1, figsize=self._figsize)
        im = ax.imshow(
            arr,
            cmap=colormap,
            vmin=vmin,
            vmax=vmax,
            extent=plot_extent,
            origin="upper",
            aspect="auto",
        )
        cbar = plt.colorbar(im, ax=ax, fraction=0.03)
        if colorbar_label:
            cbar.set_label(colorbar_label)
        ax.set_title(title or default_title, fontsize=13)
        if plot_extent is not None:
            ax.set_xlabel("Longitude" if plot_extent[0] < 360 else "Easting")
            ax.set_ylabel("Latitude" if plot_extent[2] < 90 else "Northing")
        plt.tight_layout()

        return self._save_or_show(fig, output)

    def plot_comparison(
        self,
        panels: dict,
        suptitle: str = "",
        colormap: str = "RdYlGn",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        output: Optional[Union[str, Path]] = None,
        extent: Optional[Tuple[float, float, float, float]] = None,
        colorbar_label: Optional[str] = None,
        per_panel_cmap: Optional[dict] = None,
        per_panel_range: Optional[dict] = None,
    ) -> Any:
        """
        Side-by-side comparison plot — e.g. before / after / change — the
        standard layout for any change-detection result (NDVI change,
        flood extent change, deformation time steps).

        Args:
            panels:    Ordered dict of {panel_title: array_or_path}.
            suptitle:  Overall figure title.
            colormap:  Default colormap applied to every panel.
            vmin/vmax: Default shared color scale (auto per-panel if None).
            output:    Save path.
            extent:    Shared (min_lon, max_lon, min_lat, max_lat) for
                       array inputs (ignored for path inputs, which carry
                       their own georeferencing).
            colorbar_label: Shared colorbar label.
            per_panel_cmap:  Optional {panel_title: colormap} override —
                       e.g. a diverging colormap just for a "Change" panel
                       while other panels use a sequential one.
            per_panel_range: Optional {panel_title: (vmin, vmax)} override.

        Example::

            pl.plot_comparison({
                "Baseline (2016)": ndvi_before,
                "Recent (2024)": ndvi_after,
                "Change": ndvi_change,
            }, per_panel_cmap={"Change": "RdBu"}, per_panel_range={"Change": (-0.5, 0.5)})
        """
        import numpy as np

        plt = _require_matplotlib()
        per_panel_cmap = per_panel_cmap or {}
        per_panel_range = per_panel_range or {}

        n = len(panels)
        fig, axes = plt.subplots(
            1, n, figsize=(self._figsize[0] * n / 2, self._figsize[1] * 0.75)
        )
        if n == 1:
            axes = [axes]

        for ax, (panel_title, item) in zip(axes, panels.items()):
            if isinstance(item, (str, Path)):
                try:
                    import rasterio

                    with rasterio.open(item) as src:
                        arr = src.read(1).astype(np.float32)
                        nodata = src.nodata
                        if nodata is not None:
                            arr = np.where(arr == nodata, np.nan, arr)
                        panel_extent = [
                            src.bounds.left,
                            src.bounds.right,
                            src.bounds.bottom,
                            src.bounds.top,
                        ]
                except ImportError:
                    raise ImportError(
                        'rasterio required: pip install "pygeofetch[geo]"'
                    )
            else:
                arr = np.asarray(item, dtype=np.float32)
                panel_extent = list(extent) if extent is not None else None

            cmap = per_panel_cmap.get(panel_title, colormap)
            pv = per_panel_range.get(panel_title, (vmin, vmax))
            im = ax.imshow(
                arr,
                cmap=cmap,
                vmin=pv[0],
                vmax=pv[1],
                extent=panel_extent,
                origin="upper",
                aspect="auto",
            )
            ax.set_title(panel_title, fontsize=12)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046)
            if colorbar_label:
                cbar.set_label(colorbar_label, fontsize=9)

        if suptitle:
            fig.suptitle(suptitle, fontsize=14, y=1.02)
        plt.tight_layout()
        return self._save_or_show(fig, output)

    def plot_classification(
        self,
        data: Any,
        class_labels: dict,
        class_colors: dict,
        title: str = "",
        output: Optional[Union[str, Path]] = None,
        extent: Optional[Tuple[float, float, float, float]] = None,
        show_percentages: bool = True,
    ) -> Any:
        """
        Plot a categorical/classified map with a discrete legend — for
        severity classification (e.g. NDVI change classes) or extent
        classification (e.g. flood / no-flood).

        Args:
            data:         2D array of integer class codes, OR a string
                          array already containing class labels directly.
            class_labels: {class_code: display_label}.
            class_colors: {class_code: matplotlib color}.
            title:        Plot title.
            output:       Save path.
            extent:       (min_lon, max_lon, min_lat, max_lat) for axis labels.
            show_percentages: Annotate the legend with the % of the scene
                          in each class.

        Example::

            pl.plot_classification(
                classified_array,
                class_labels={0: "Stable", 1: "Moderate decline", 2: "Severe decline"},
                class_colors={0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"},
                title="Land Degradation Severity",
            )
        """
        import numpy as np
        from matplotlib.colors import BoundaryNorm, ListedColormap
        from matplotlib.patches import Patch

        plt = _require_matplotlib()

        arr = np.asarray(data)
        codes = sorted(class_labels.keys())
        colors = [class_colors[c] for c in codes]
        cmap = ListedColormap(colors)
        bounds = codes + [codes[-1] + 1]
        norm = BoundaryNorm(bounds, cmap.N)

        fig, ax = plt.subplots(1, 1, figsize=self._figsize)
        ax.imshow(
            arr, cmap=cmap, norm=norm, extent=extent, origin="upper", aspect="auto"
        )
        ax.set_title(title, fontsize=13)
        if extent is not None:
            ax.set_xlabel("Longitude" if extent[0] < 360 else "Easting")
            ax.set_ylabel("Latitude" if extent[2] < 90 else "Northing")

        total = arr.size
        legend_handles = []
        for c in codes:
            label = class_labels[c]
            if show_percentages:
                pct = 100 * np.sum(arr == c) / total
                label = f"{label} ({pct:.1f}%)"
            legend_handles.append(Patch(facecolor=class_colors[c], label=label))
        ax.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=10,
            frameon=True,
        )

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

    def quicklook(
        self,
        source: Union[str, Path, Any],
        title: str = "",
        output: Optional[Union[str, Path]] = None,
        mode: Optional[str] = None,
        colormap: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        band: int = 1,
    ) -> Any:
        """
        One call that visualizes almost anything PyGeoFetch can produce —
        a Landsat band, a Sentinel-1 backscatter raster, an NDVI array, a
        classification map, an RGB composite — without you needing to
        know which specific plot_*() method or colormap fits.

        Inspects the data and picks a sensible visualization:

        - **Multi-band raster (3+ bands)** → RGB composite (first 3 bands,
          percentile-stretched). This is a best-effort default, not
          guaranteed band-order correctness — for a specific sensor's
          true RGB bands, use plot_rgb() directly with named bands.
        - **Few distinct values relative to pixel count** (e.g. a
          classification/severity/flood mask) → categorical display with
          an auto-generated legend.
        - **Values mostly negative, roughly -40 to +10** → treated as SAR
          backscatter in dB → grayscale.
        - **Values roughly within [-1, 1]** → treated as a spectral index
          (NDVI, NDWI, etc.) → diverging colormap centered at zero.
        - **Everything else** → continuous data, percentile-stretched,
          default colormap.

        These are heuristics based on typical value ranges, not format
        detection — they can guess wrong for unusual data. Override with
        `mode=` ("rgb", "categorical", "sar", "index", "continuous"),
        or fall back to plot_raster()/plot_rgb()/plot_classification()
        directly when you know exactly what you have.

        Args:
            source:   Path to a raster, a numpy array, or a DownloadResult
                     (uses .output_path).
            title:    Plot title.
            output:   Save path. Shown inline if None.
            mode:     Force a specific interpretation instead of
                     auto-detecting (see above).
            colormap: Override the auto-selected colormap.
            vmin/vmax: Override the auto-selected value range.
            band:     Which band to read as the "single band" case, if
                     `source` is a multi-band file being forced into a
                     non-RGB mode.

        Example::

            pl = Plotter()
            pl.quicklook(ndvi_array)                    # -> index mode, RdYlGn
            pl.quicklook("sentinel1_sigma0_db.tif")      # -> SAR mode, grayscale
            pl.quicklook(flood_mask)                     # -> categorical mode
            pl.quicklook("landsat_multiband.tif")        # -> RGB composite
            pl.quicklook(download_result)                # DownloadResult resolved automatically
        """
        import numpy as np

        _require_matplotlib()

        # Resolve DownloadResult-like objects to a path first
        if hasattr(source, "output_path") and not isinstance(source, (str, Path)):
            resolved = getattr(source, "output_path", None)
            if resolved is None:
                raise ValueError(
                    "source has no output_path to visualize (download may have failed)"
                )
            source = resolved

        band_count = 1
        if isinstance(source, (str, Path)):
            try:
                import rasterio

                with rasterio.open(source) as src:
                    band_count = src.count
            except ImportError:
                raise ImportError('rasterio required: pip install "pygeofetch[geo]"')

        effective_mode = mode
        if effective_mode is None and band_count >= 3:
            effective_mode = "rgb"

        if effective_mode == "rgb":
            import rasterio

            with rasterio.open(source) as src:
                band_paths_or_arrays = [src.read(i) for i in (1, 2, 3)]
            # plot_rgb expects paths OR we replicate its array-stretch logic here
            r, g, b = band_paths_or_arrays
            stacked = np.stack(
                [self._percentile_stretch(a) for a in (r, g, b)], axis=-1
            )
            plt = _require_matplotlib()
            fig, ax = plt.subplots(1, 1, figsize=self._figsize)
            ax.imshow(stacked, origin="upper", aspect="auto")
            ax.set_title(
                title or "RGB composite (first 3 bands, auto-stretched)", fontsize=13
            )
            plt.tight_layout()
            return self._save_or_show(fig, output)

        # Single-band path: load the array to inspect its value distribution
        if isinstance(source, (str, Path)):
            import rasterio

            with rasterio.open(source) as src:
                arr = src.read(band).astype(np.float32)
                nodata = src.nodata
                if nodata is not None:
                    arr = np.where(arr == nodata, np.nan, arr)
                extent = [
                    src.bounds.left,
                    src.bounds.right,
                    src.bounds.bottom,
                    src.bounds.top,
                ]
        else:
            arr = np.asarray(source, dtype=np.float32)
            extent = None

        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            raise ValueError("Data has no finite values to plot.")

        if effective_mode is None:
            n_unique = len(np.unique(finite))
            frac_unique = n_unique / finite.size
            if n_unique <= 12 and frac_unique < 0.01:
                effective_mode = "categorical"
            elif (
                np.nanmean(finite) < -3
                and np.nanmin(finite) > -60
                and np.nanmax(finite) < 15
            ):
                effective_mode = "sar"
            elif np.nanmin(finite) >= -1.05 and np.nanmax(finite) <= 1.05:
                effective_mode = "index"
            else:
                effective_mode = "continuous"

        logger.debug("quicklook: auto-detected mode=%r", effective_mode)

        if effective_mode == "categorical":
            codes = sorted(int(c) for c in np.unique(finite))
            import matplotlib.cm as cm

            palette = cm.get_cmap("tab10" if len(codes) <= 10 else "tab20")
            class_labels = {c: f"Class {c}" for c in codes}
            class_colors = {
                c: palette(i / max(len(codes) - 1, 1)) for i, c in enumerate(codes)
            }
            return self.plot_classification(
                arr,
                class_labels=class_labels,
                class_colors=class_colors,
                title=title or "Classification (auto-detected)",
                output=output,
                extent=extent,
            )

        if effective_mode == "sar":
            cmap = colormap or "gray"
            lo, hi = (
                (vmin, vmax)
                if vmin is not None and vmax is not None
                else np.nanpercentile(finite, [2, 98])
            )
            return self.plot_raster(
                arr,
                title=title or "SAR backscatter (dB, auto-detected)",
                colormap=cmap,
                vmin=lo,
                vmax=hi,
                output=output,
                extent=extent,
                colorbar_label="dB",
            )

        if effective_mode == "index":
            cmap = colormap or "RdYlGn"
            lo = vmin if vmin is not None else -1.0
            hi = vmax if vmax is not None else 1.0
            return self.plot_raster(
                arr,
                title=title or "Spectral index (auto-detected)",
                colormap=cmap,
                vmin=lo,
                vmax=hi,
                output=output,
                extent=extent,
            )

        # continuous
        cmap = colormap or "viridis"
        lo, hi = (
            (vmin, vmax)
            if vmin is not None and vmax is not None
            else np.nanpercentile(finite, [2, 98])
        )
        return self.plot_raster(
            arr,
            title=title or "Raster (auto-detected)",
            colormap=cmap,
            vmin=lo,
            vmax=hi,
            output=output,
            extent=extent,
        )

    def _percentile_stretch(
        self, arr: Any, percentile: Tuple[float, float] = (2, 98)
    ) -> Any:
        import numpy as np

        arr = arr.astype(np.float32)
        lo, hi = np.nanpercentile(arr, percentile)
        return np.clip((arr - lo) / (hi - lo + 1e-10), 0, 1)

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