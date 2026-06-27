# Change Detection

Detect land-cover and structural changes between two satellite acquisitions.

---

## Setup

```python
import pygeovision as pgv
from pygeovision.models.change_detection.changeformer import ChangeFormer

client = pgv.PyGeoVision()
bbox   = [10.95, 48.35, 11.05, 48.45]   # Munich, Germany
```

---

## 1. Acquire Two-Date Imagery

```python
def get_best_scene(bbox, year):
    results = client.search(
        bbox=bbox,
        date_range=[f"{year}-06-01", f"{year}-08-31"],
        providers=["planetary_computer"],
        cloud_cover_max=5,
    )
    return client.download(results[:1], f"./data/{year}/",
                           post_process=["reproject:EPSG:32632"])[0].path

before_path = get_best_scene(bbox, 2020)   # Pre-development
after_path  = get_best_scene(bbox, 2024)   # Post-development
print(f"Before: {before_path}")
print(f"After:  {after_path}")
```

---

## 2. Run ChangeFormer

```python
cd = ChangeFormer(num_classes=2, in_channels=4, device="cuda")
result = cd.detect(before_path, after_path, "./output/change_map.tif")

print(f"Changed area: {result['change_pct']:.2f}%")
```

---

## 3. Prithvi Multi-Temporal Change Detection

For higher accuracy with temporal context:

```python
from pygeovision.models.foundation.prithvi import PrithviMultiTemporal

mt = PrithviMultiTemporal("prithvi_eo_2_0", device="cuda")
result = mt.detect_change(before_path, after_path,
                           output_path="./output/prithvi_change.tif")

print(f"Change detected: {result['change_pct']:.2f}%")
```

---

## 4. DINOv3 CHMv2 Deforestation Check

```python
from pygeovision.models.foundation.dinov3 import CHMv2Model

chm = CHMv2Model(device="cuda")
result = chm.detect_deforestation(before_path, after_path,
                                    min_height_loss_m=3.0,
                                    output_path="./output/deforestation.tif")

print(f"Deforested: {result['deforested_pct']:.2f}%  ({result['area_ha']:.0f} ha)")
```

---

## 5. Visualise Results

```python
import rasterio, matplotlib.pyplot as plt, numpy as np

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, path, title in zip(axes,
        [before_path, after_path, "./output/change_map.tif"],
        ["Before (2020)", "After (2024)", "Change Map"]):
    with rasterio.open(path) as src:
        data = src.read([1,2,3] if src.count >= 3 else [1])
    data = data.transpose(1,2,0) if data.ndim==3 else data[0]
    ax.imshow(data, cmap="RdYlGn" if "change" in title.lower() else None)
    ax.set_title(title); ax.axis("off")

plt.tight_layout()
plt.savefig("./output/change_comparison.png", dpi=150, bbox_inches="tight")
print("Comparison plot saved")
```
