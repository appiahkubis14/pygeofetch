# Agriculture

Crop type mapping, yield prediction, disease detection, and precision farming analytics from satellite imagery.

---

## Crop Type Mapping

Map 10 major crop types from Sentinel-2 time series using Prithvi-EO-2.0.

```python
import pygeovision as pgv
from pygeovision.models.foundation.prithvi import PrithviTasks, PrithviMultiTemporal

client = pgv.PyGeoVision()
bbox   = [-92.5, 41.5, -92.0, 42.0]   # Iowa, USA — Corn Belt

# Monthly time series (April through October)
months = ["04","05","06","07","08","09","10"]
scenes = []
for m in months:
    r = client.search(bbox=bbox, date_range=[f"2024-{m}-01", f"2024-{m}-28"],
                      providers=["planetary_computer"], cloud_cover_max=15)
    if r:
        d = client.download(r[:1], f"./data/iowa/{m}/",
                            bands=["B02","B03","B04","B08","B11","B12"])[0]
        scenes.append(d.path)

# Multi-temporal feature extraction
mt    = PrithviMultiTemporal("prithvi_eo_2_0", device="cuda")
trend = mt.monitor_trend(scenes)
print(f"Growing season trend: {trend['trend_direction']}")

# Crop type classification
tasks = PrithviTasks("prithvi_eo_2_0", device="cuda")
crops = tasks.crop_mapping(scenes[3], source="hls",  # Mid-season (July)
                            output_path="./output/crop_map.tif")

print("Crop distribution — Iowa County:")
for cls, pct in sorted(crops['class_pct'].items(), key=lambda x: -x[1])[:5]:
    name = crops['class_names'][cls]
    print(f"  {name:<15} {pct:.1f}%")
```

---

## NDVI Time Series & Anomaly Detection

```python
from pygeovision.advanced.timeseries import GeoTimeSeries

ts = GeoTimeSeries(sensor="sentinel2")

# Compute NDVI for each monthly scene
series = ts.compute_index_series(scenes, index="ndvi")

# Detect stress events (drought, disease, flood)
anomalies = ts.detect_anomalies(series, threshold=2.0)
for a in anomalies:
    print(f"Anomaly: {a['date']}  NDVI={a['value']:.3f}  type={a['type']}")

# Seasonal decomposition
decomp = ts.decompose(series, period=len(months))
print(f"Peak NDVI month: {decomp['peak_month']}")
```

---

## Field Boundary Detection

```python
from pygeovision.models import get_model
from pygeovision.inference.tiled import TiledInference

# SegFormer fine-tuned for field boundaries
model = get_model("segformer-b5", num_classes=3,   # background, crop, boundary
                   in_channels=4, pretrained=True)

inf    = TiledInference(model=model, chip_size=512, overlap=64)
result = inf.infer(scenes[3], "./output/field_boundaries.tif")
print(f"Processed {result['n_chips']} tiles")
```

---

## Auto-Label Cropland

```python
# ESA WorldCover cropland mask (class 40)
labels = client.labeling.esa_worldcover(bbox, output_path="./labels/cropland.tif")

# OpenStreetMap farmland polygons
osm_labels = client.labeling.osm(bbox,
    categories=["farmland","orchard","vineyard"],
    output_path="./labels/osm_farm.tif")
```

---

## Key Datasets

| Dataset | Samples | Task |
|---------|---------|------|
| CropHarvest | 70,000 | Global crop classification |
| BreizhCrops | 614,000 | Time series crop types |
| PASTIS-R | 2,433 | Panoptic crop mapping |
| AgriSen | 45,000 | Sentinel-2 time series |
| TimeSen2Crop | 589,700 | 16-class crop classification |
