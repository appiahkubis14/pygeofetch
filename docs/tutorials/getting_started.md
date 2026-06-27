# Getting Started

This tutorial gets you from zero to running your first satellite AI prediction in under 5 minutes.

---

## Prerequisites

- Python 3.10+
- 4 GB RAM minimum (8 GB recommended)
- GPU optional (CPU works for small images)

---

## Install

```bash
# Minimal install (data layer only)
pip install pygeovision

# With AI models and geospatial tools
pip install "pygeovision[geo,train]"

# Full platform (recommended for this tutorial)
pip install "pygeovision[geo,train,foundation,labeling]"
```

Verify:

```bash
python -c "import pygeovision; print(pygeovision.__version__)"
# 2.0.4
```

---

## Step 1: Initialize the Client

```python
import pygeovision as pgv

client = pgv.PyGeoVision()
print(client)
# PyGeoVision(v2.0 | datasets=503 | models=119 | geoai=independent)
```

---

## Step 2: Load a Pre-trained Model

```python
from pygeovision.models import get_model

# SegFormer-B2 for 7-class land cover (Sentinel-2 input)
model = get_model("segformer-b2", num_classes=7, in_channels=4)
print(model)
```

---

## Step 3: Create a Synthetic Test Image

```python
import numpy as np
import rasterio
from rasterio.transform import from_bounds

# Create a small synthetic 4-band image
data = np.random.rand(4, 256, 256).astype(np.float32)
transform = from_bounds(0, 0, 1, 1, 256, 256)

with rasterio.open("test_scene.tif", "w", driver="GTiff",
                    height=256, width=256, count=4,
                    dtype="float32", crs="EPSG:4326",
                    transform=transform) as dst:
    dst.write(data)

print("Test image created: test_scene.tif")
```

---

## Step 4: Run Inference

```python
from pygeovision.inference.tiled import TiledInference

inf = TiledInference(
    model=model,
    chip_size=128,
    overlap=16,
    num_classes=7,
    device="cpu",
)

result = inf.infer("test_scene.tif", "./output/prediction.tif")
print(f"Tiles processed: {result['n_chips']}")
print(f"Time:            {result['duration_seconds']:.1f}s")
print(f"Output saved:    {result['output_path']}")
```

---

## Step 5: Inspect the Output

```python
import rasterio
import numpy as np

with rasterio.open("./output/prediction.tif") as dst:
    pred = dst.read(1)

classes, counts = np.unique(pred, return_counts=True)
print("Class distribution:")
class_names = ["water","trees","grass","crops","built","bare","snow"]
for cls, n in zip(classes, counts):
    pct = 100 * n / pred.size
    print(f"  {class_names[cls]:<10} {pct:.1f}%")
```

---

## What's Next

- [Authentication](authentication.md) — connect to real satellite providers
- [Data Search](data_search.md) — search Sentinel-2 scenes with STAC
- [Building Extraction](building_extraction.md) — a real end-to-end example
