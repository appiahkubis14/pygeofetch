# Water & Floods

Flood mapping, water body detection, coastal monitoring, and water quality estimation.

---

## Rapid Flood Mapping

```python
import pygeovision as pgv
from pygeovision.models.foundation.prithvi import PrithviTasks

client = pgv.PyGeoVision()
bbox   = [14.4, 50.0, 14.7, 50.2]   # Prague, Czech Republic (2024 flood event)

# Acquire post-flood SAR (works through clouds)
results = client.search(bbox=bbox,
    date_range=["2024-09-14","2024-09-18"],
    providers=["copernicus"], collections=["sentinel-1-rtc"],
    cloud_cover_max=100)   # SAR — cloud cover irrelevant

post_scene = client.download(results[:1], "./data/prague/post/")[0].path

# Prithvi flood detection
tasks = PrithviTasks("prithvi_eo_2_0", device="cuda")
flood = tasks.flood_detection(post_scene, source="sentinel2",
                               output_path="./output/flood_mask.tif")

print(f"Flooded area: {flood['flood_pct']:.1f}%")
print(f"Flood extent: {flood['class_pct'].get(1, 0):.1f}% of study area")
```

---

## Water Body Mapping

```python
from pygeovision.models import get_model
from pygeovision.inference.tiled import TiledInference
from pygeovision.labeling.dynamic_world import DynamicWorldLabeler

bbox = [-122.5, 37.2, -122.1, 37.6]   # San Francisco Bay

# Dynamic World water labels (9 classes, real-time)
dw = DynamicWorldLabeler()
labels = dw.label(bbox, date_range=["2024-01-01","2024-06-30"],
                   output_path="./labels/bay_water.tif")

# U-Net water segmentation
model = get_model("unet-r101", num_classes=3,  # water, wetland, other
                   in_channels=4)
inf   = TiledInference(model=model, chip_size=512)
result = inf.infer("./data/bay_s2.tif", "./output/water_seg.tif")
```

---

## NDWI Water Index Time Series

```python
from pygeovision.advanced.timeseries import GeoTimeSeries

ts = GeoTimeSeries(sensor="sentinel2")
scenes = ["jan.tif","apr.tif","jul.tif","oct.tif"]

ndwi_series = ts.compute_index_series(scenes, index="ndwi")

print("Seasonal water index:")
for i, (date, val) in enumerate(zip(["Jan","Apr","Jul","Oct"],
                                      ndwi_series["mean"])):
    trend_char = "▲" if i > 0 and val > ndwi_series["mean"][i-1] else "▼"
    print(f"  {date}: {val:.4f} {trend_char}")
```

---

## Coastal Erosion Monitoring

```python
from pygeovision.models.change_detection.changeformer import ChangeFormer

cd = ChangeFormer(num_classes=2, in_channels=4)
result = cd.detect("coast_2015.tif", "coast_2024.tif",
                    output_path="./output/coastal_change.tif")
print(f"Shoreline change detected: {result['change_pct']:.2f}%")
```

---

## Key Datasets

| Dataset | Task | Notes |
|---------|------|-------|
| SEN1Floods11 | Flood segmentation | 11 flood events globally |
| WorldFloods | Flood mapping | 119 Sentinel-2 flood events |
| CoastalSeg | Coastal zones | 6-class segmentation |
| WaterNet | Water bodies | Global Landsat |
| KelpNet | Kelp forest | Multispectral |
