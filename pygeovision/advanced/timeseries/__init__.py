"""Time series analysis for geospatial data (G4)."""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Union
logger = logging.getLogger(__name__)


class GeoTimeSeries:
    """Temporal analysis of satellite image stacks (G4).

    Analyse time series of satellite images for:
        - Vegetation phenology (NDVI time series)
        - Crop type mapping from temporal signatures
        - Urban change over time
        - Anomaly detection (drought, floods, fires)
        - Trend analysis (climate indicators)

    Example::

        ts = GeoTimeSeries()
        ndvi_ts = ts.compute_index_series(
            image_paths=sorted(glob("./data/s2_*.tif")),
            index="ndvi",
            date_strings=["2024-01", ..., "2024-12"],
        )
        ts.plot_pixel_timeseries(ndvi_ts, lat=40.7, lon=-74.0)
    """

    INDICES = {
        "ndvi":  ("nir", "red",   lambda n, r: (n - r) / (n + r + 1e-8)),
        "ndwi":  ("green", "nir", lambda g, n: (g - n) / (g + n + 1e-8)),
        "ndbi":  ("swir", "nir",  lambda s, n: (s - n) / (s + n + 1e-8)),
        "evi":   ("nir", "red",   lambda n, r: 2.5 * (n - r) / (n + 6*r - 7.5*0.01 + 1)),
        "savi":  ("nir", "red",   lambda n, r: 1.5 * (n - r) / (n + r + 0.5)),
    }

    BAND_PRESETS = {
        "sentinel2":  {"blue": 1, "green": 2, "red": 3, "nir": 4, "swir": 10},
        "landsat8":   {"blue": 1, "green": 2, "red": 3, "nir": 4, "swir": 5},
        "planet":     {"blue": 1, "green": 2, "red": 3, "nir": 4},
        "naip":       {"red": 1, "green": 2, "blue": 3, "nir": 4},
    }

    def __init__(self, sensor: str = "sentinel2") -> None:
        self.sensor = sensor
        self.band_map = self.BAND_PRESETS.get(sensor, self.BAND_PRESETS["sentinel2"])

    def compute_index_series(
        self,
        image_paths: List[str],
        index: str = "ndvi",
        date_strings: Optional[List[str]] = None,
        cloud_mask: bool = True,
    ) -> Dict[str, Any]:
        """Compute a spectral index time series from multiple images."""
        try:
            import numpy as np, rasterio
        except ImportError:
            return {"error": "rasterio required"}

        if index not in self.INDICES:
            return {"error": f"Unknown index. Choose from {list(self.INDICES)}"}

        b1_name, b2_name, fn = self.INDICES[index]
        b1_idx = self.band_map.get(b1_name, 4)
        b2_idx = self.band_map.get(b2_name, 3)

        series = {"dates": date_strings or list(range(len(image_paths))),
                  "mean": [], "std": [], "p10": [], "p90": [], "valid_fraction": []}

        for path in image_paths:
            try:
                with rasterio.open(path) as src:
                    b1 = src.read(b1_idx).astype(np.float32)
                    b2 = src.read(b2_idx).astype(np.float32)
                    # Scale surface reflectance (0-10000 for SR products)
                    if b1.max() > 1: b1 /= 10000.0; b2 /= 10000.0

                idx_vals = fn(b1, b2)
                valid = (idx_vals >= -1) & (idx_vals <= 1)
                idx_valid = idx_vals[valid]

                series["mean"].append(round(float(idx_valid.mean()), 4) if valid.sum() > 0 else None)
                series["std"].append(round(float(idx_valid.std()), 4) if valid.sum() > 0 else None)
                series["p10"].append(round(float(np.percentile(idx_valid, 10)), 4) if valid.sum() > 0 else None)
                series["p90"].append(round(float(np.percentile(idx_valid, 90)), 4) if valid.sum() > 0 else None)
                series["valid_fraction"].append(round(float(valid.mean()), 3))
            except Exception as exc:
                logger.warning("Time series step failed for %s: %s", path, exc)
                for k in ("mean", "std", "p10", "p90", "valid_fraction"):
                    series[k].append(None)

        series["index"] = index
        series["n_images"] = len(image_paths)
        return series

    def detect_anomalies(
        self, series: Dict, method: str = "zscore", threshold: float = 2.5
    ) -> List[Dict]:
        """Detect anomalous time steps (drought, flood, fire)."""
        import numpy as np
        vals = [v for v in series.get("mean", []) if v is not None]
        dates = series.get("dates", [])
        if len(vals) < 5:
            return []

        arr = np.array(vals)
        if method == "zscore":
            zscores = (arr - arr.mean()) / (arr.std() + 1e-8)
            anomaly_mask = np.abs(zscores) > threshold
            return [
                {"date": dates[i], "value": vals[i], "zscore": round(float(zscores[i]), 2),
                 "type": "low" if zscores[i] < 0 else "high"}
                for i in range(len(vals)) if anomaly_mask[i]
            ]
        return []

    def compute_trend(self, series: Dict) -> Dict[str, Any]:
        """Compute linear trend in the time series."""
        import numpy as np
        vals = [v for v in series.get("mean", []) if v is not None]
        if len(vals) < 3:
            return {"trend": "insufficient_data"}
        x = np.arange(len(vals))
        y = np.array(vals)
        slope, intercept = np.polyfit(x, y, 1)
        r2 = float(1 - np.sum((y - (slope * x + intercept))**2) / np.sum((y - y.mean())**2 + 1e-8))
        return {
            "slope": round(float(slope), 6),
            "intercept": round(float(intercept), 4),
            "r_squared": round(r2, 4),
            "direction": "increasing" if slope > 0.001 else "decreasing" if slope < -0.001 else "stable",
        }

    def plot(self, series: Dict, title: str = "NDVI Time Series",
              save_path: Optional[str] = None) -> None:
        try:
            import matplotlib.pyplot as plt, numpy as np
            dates = series.get("dates", [])
            means = series.get("mean", [])
            p10 = series.get("p10", [None]*len(means))
            p90 = series.get("p90", [None]*len(means))

            valid = [(i, d, m) for i, (d, m) in enumerate(zip(dates, means)) if m is not None]
            if not valid: return
            idxs, ds, ms = zip(*valid)

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(ds, ms, "b-o", linewidth=2, markersize=4, label=series.get("index", "index").upper())

            # Confidence band
            p10v = [p10[i] for i in idxs if p10[i] is not None]
            p90v = [p90[i] for i in idxs if p90[i] is not None]
            if len(p10v) == len(ds):
                ax.fill_between(ds, p10v, p90v, alpha=0.2, color="blue", label="P10-P90 range")

            ax.set_title(title); ax.set_xlabel("Date"); ax.set_ylabel(series.get("index", "Index"))
            ax.legend(); ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45); plt.tight_layout()
            if save_path: plt.savefig(save_path, dpi=120)
            else: plt.show()
        except ImportError:
            logger.warning("matplotlib required for plot")


__all__ = ["GeoTimeSeries"]
