# Land Cover Mapping

Map land cover at 10m resolution using ESA WorldCover labels and SegFormer-B5.

---

## Setup

```python
import pygeovision as pgv
from pygeovision.models import get_model
from pygeovision.inference.tiled import TiledInference

client = pgv.PyGeoVision()
bbox   = [8.45, 47.30, 8.60, 47.45]   # Zurich, Switzerland
```

---

## 1. Auto-Label from ESA WorldCover

```python
labels = client.labeling.esa_worldcover(
    bbox=bbox,
    output_path="./labels/worldcover_zurich.tif",
    year=2021,
)
print(f"Land cover classes: {labels['n_classes']}")
```

**11 ESA WorldCover classes:**

| Class | ID | Area in Zurich |
|-------|----|---------------|
| Tree cover | 10 | 38% |
| Grassland | 30 | 15% |
| Cropland | 40 | 12% |
| Built-up | 50 | 22% |
| Permanent water | 80 | 8% |
| ... | ... | ... |

---

## 2. Download Sentinel-2

```python
results = client.search(
    bbox=bbox,
    date_range=["2024-06-01", "2024-08-31"],
    providers=["planetary_computer"],
    cloud_cover_max=5,
)

downloads = client.download(
    results[:1],
    output_dir="./data/zurich/",
    bands=["B02","B03","B04","B08","B11","B12"],   # 6 bands
    post_process=["reproject:EPSG:32632", "cog"],
)
scene = downloads[0].path
```

---

## 3. Run SegFormer-B5

```python
model = get_model("segformer-b5", num_classes=11, in_channels=6)

inf = TiledInference(model=model, chip_size=512, overlap=64,
                      blend_mode="gaussian", num_classes=11)

result = inf.infer(scene, "./output/lc_zurich.tif")
print(f"Done in {result['duration_seconds']:.0f}s")
```

---

## 4. Compute Class Statistics

```python
import rasterio, numpy as np

class_names = ["tree_cover","shrubland","grassland","cropland","built_up",
               "bare_sparse","snow_ice","water","herb_wetland","mangroves","moss"]

with rasterio.open("./output/lc_zurich.tif") as src:
    pred = src.read(1)

classes, counts = np.unique(pred, return_counts=True)
total = pred.size
print("\nLand Cover Map — Zurich")
print("-" * 35)
for cls, n in zip(classes, counts):
    if cls < len(class_names):
        print(f"  {class_names[cls]:<18} {100*n/total:5.1f}%")
```
