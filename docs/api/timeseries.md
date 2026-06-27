# Time Series Analysis

Compute spectral index time series from multi-date satellite imagery, detect anomalies, and analyse vegetation and land-cover dynamics.

---

## `GeoTimeSeries`

```python
from pygeovision.advanced.timeseries import GeoTimeSeries

ts = GeoTimeSeries(
    sensor="sentinel2",    # "sentinel2" | "landsat" | "modis" | "planet"
    device="cpu",
)
```

---

## Spectral Indices

### Compute Index Series

```python
images = ["jan.tif", "feb.tif", "mar.tif", "apr.tif",
           "may.tif", "jun.tif", "jul.tif", "aug.tif",
           "sep.tif", "oct.tif", "nov.tif", "dec.tif"]

series = ts.compute_index_series(
    image_paths=images,
    index="ndvi",   # See table below
)

# series = {
#   "index": "ndvi",
#   "values": [...],     per-date values (spatially averaged)
#   "mean":  [...],      mean per date
#   "std":   [...],      standard deviation per date
#   "min":   [...],
#   "max":   [...],
# }
```

### Supported Indices

| Index | Formula | Application |
|-------|---------|------------|
| `ndvi` | (NIR-Red)/(NIR+Red) | Vegetation health |
| `ndwi` | (Green-NIR)/(Green+NIR) | Surface water |
| `ndbi` | (SWIR-NIR)/(SWIR+NIR) | Built-up area |
| `evi` | 2.5×(NIR-Red)/(NIR+6×Red-7.5×Blue+1) | Enhanced vegetation |
| `savi` | 1.5×(NIR-Red)/(NIR+Red+0.5) | Soil-adjusted vegetation |
| `mndwi` | (Green-SWIR)/(Green+SWIR) | Modified water index |
| `bai` | 1/((0.1+Red)²+(0.06+NIR)²) | Burn area |
| `lai` | Leaf Area Index (model-based) | Canopy density |
| `fapar` | Fraction of absorbed PAR | Photosynthetic activity |

---

## Trend Analysis

```python
# Compute linear trend from a time series
trend = ts.compute_trend(series)

print(f"Direction:  {trend['direction']}")    # "increasing" | "decreasing" | "stable"
print(f"Slope:      {trend['slope']:.5f}")    # Change per time step
print(f"R²:         {trend['r_squared']:.3f}")
print(f"P-value:    {trend['p_value']:.4f}")
print(f"Significant: {trend['significant']}")  # p < 0.05
```

---

## Anomaly Detection

Detect dates with unusual spectral values (drought, flood, fire, crop failure).

```python
anomalies = ts.detect_anomalies(
    series=series,
    threshold=2.5,         # Z-score threshold (±2.5σ from rolling mean)
    window=12,             # Rolling window size (months)
    method="zscore",       # "zscore" | "iqr" | "isolation_forest"
)

for a in anomalies:
    print(f"Date:    {a['date']}")
    print(f"Value:   {a['value']:.4f}  (z={a['zscore']:.2f})")
    print(f"Type:    {a['type']}")    # "low" (drought/burn) or "high" (flood/cloud)
```

---

## Seasonal Decomposition

Separate trend, seasonal, and residual components.

```python
decomp = ts.decompose(
    series=series,
    period=12,         # Annual cycle (12 months)
    model="additive",  # "additive" | "multiplicative"
)

print(f"Peak month:    {decomp['peak_month']}")
print(f"Trough month:  {decomp['trough_month']}")
print(f"Amplitude:     {decomp['seasonal_amplitude']:.4f}")
trend_comp    = decomp['trend']      # Trend component
seasonal_comp = decomp['seasonal']   # Seasonal component
residual_comp = decomp['residual']   # Residual noise
```

---

## Change Point Detection

Identify abrupt changes in the time series (deforestation, flood, crop change).

```python
change_points = ts.detect_change_points(
    series=series,
    method="pelt",       # "pelt" | "binseg" | "window"
    penalty=3,
)

for cp in change_points:
    print(f"Change at index {cp['index']}  ({cp['date']})")
    print(f"  Before: mean={cp['before_mean']:.4f}")
    print(f"  After:  mean={cp['after_mean']:.4f}")
    print(f"  Magnitude: {cp['magnitude']:.4f}")
```

---

## Spatial Time Series (Per-Pixel)

For spatial trend maps, apply the analysis per pixel across the full GeoTIFF stack.

```python
# Compute NDVI trend map — one slope value per pixel
trend_map = ts.compute_spatial_trend(
    image_paths=images,
    index="ndvi",
    output_path="./output/ndvi_trend.tif",
)
print(f"Trend map shape: {trend_map['shape']}")
print(f"Greening area:   {trend_map['positive_trend_pct']:.1f}%")
print(f"Browning area:   {trend_map['negative_trend_pct']:.1f}%")
```

---

## Prithvi Foundation Model (Multi-Temporal)

For deep multi-temporal analysis, use the Prithvi integration:

```python
from pygeovision.models.foundation.prithvi import PrithviMultiTemporal

mt = PrithviMultiTemporal("prithvi_eo_2_0")
ts_result = mt.process_time_series(images, dates=["2024-01", ..., "2024-12"])
trend = mt.monitor_trend(images)
```
