# Quick Start

Get from zero to your first geospatial AI prediction in 5 minutes.

---

## 1. Install

```bash
pip install "pygeovision[geo,train,foundation]"
```

---

## 2. Initialize

```python
import pygeovision as pgv

client = pgv.PyGeoVision()
print(client)
# PyGeoVision(v2.0 | datasets=503 | models=119 | geoai=independent)
```

---

## 3. Search for Satellite Imagery

```python
results = client.search(
    bbox=[-74.1, 40.6, -73.7, 40.9],          # New York City (min_lon, min_lat, max_lon, max_lat)
    date_range=["2024-06-01", "2024-08-31"],
    providers=["planetary_computer"],
    cloud_cover_max=10,
    collections=["sentinel-2-l2a"],
)

print(f"Found {len(results)} scenes")
for r in results[:3]:
    print(f"  {r.date}  cloud={r.cloud_cover:.0f}%  id={r.id}")
```

---

## 4. Download

```python
downloads = client.download(
    results[:1],
    output_dir="./data/",
    post_process=["reproject:EPSG:32618", "cog"],
)

scene_path = downloads[0].path
print(f"Downloaded: {scene_path}")
```

---

## 5. Auto-Label

```python
# Generate labels automatically from OpenStreetMap
labels = client.labeling.osm(
    bbox=[-74.1, 40.6, -73.7, 40.9],
    categories=["buildings", "roads", "water"],
    output_path="./labels/osm_labels.tif",
)

print(f"Generated {labels['n_features']} labelled features")
```

---

## 6. Load a Model and Run Inference

```python
from pygeovision.models import get_model
from pygeovision.inference.tiled import TiledInference

# Load SegFormer-B2 for 7-class land cover
model = get_model("segformer-b2", num_classes=7, in_channels=4)

# Tiled inference with Gaussian blending (handles any image size)
inf    = TiledInference(model=model, chip_size=512, overlap=64, blend_mode="gaussian")
result = inf.infer(scene_path, "./output/prediction.tif")

print(f"Processed {result['n_chips']} tiles in {result['duration_seconds']:.1f}s")
```

---

## 7. Explain the Prediction

```python
from pygeovision.explainability.gradcam import GradCAM

cam    = GradCAM(model)
result = cam.batch_explain(scene_path, "./output/gradcam.tif", class_idx=1)
print("GradCAM saliency map saved")
```

---

## Next Steps

- [Authentication](tutorials/authentication.md) — connect to all 22 satellite providers
- [Data Search](tutorials/data_search.md) — advanced STAC filtering with CQL2
- [Building Extraction](tutorials/building_extraction.md) — full end-to-end example
- [Foundation Models](tutorials/foundation_models.md) — DINOv3 and Prithvi-EO-2.0
- [Deployment](tutorials/deployment.md) — ONNX, Jetson, AWS, Azure, GCP
