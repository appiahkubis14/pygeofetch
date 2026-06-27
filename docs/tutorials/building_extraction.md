# Building Extraction

End-to-end tutorial: search Sentinel-2 imagery, auto-label with Microsoft Building Footprints, run SegFormer, and export to GeoJSON.

---

## Overview

**Area:** New York City (Manhattan + Brooklyn)  
**Model:** SegFormer-B2 (27.5M params)  
**Labels:** Microsoft Building Footprints  
**Output:** Building footprint GeoJSON

---

## 1. Setup

```python
import pygeovision as pgv
from pygeovision.models import get_model
from pygeovision.inference.tiled import TiledInference

client = pgv.PyGeoVision()
bbox   = [-74.05, 40.69, -73.97, 40.76]   # Lower Manhattan + DUMBO
```

---

## 2. Search and Download

```python
results = client.search(
    bbox=bbox,
    date_range=["2024-06-01", "2024-08-31"],
    providers=["planetary_computer"],
    cloud_cover_max=5,
    sort_by="cloud_cover",
)

print(f"Found {len(results)} scenes, downloading best...")

downloads = client.download(
    results[:1],
    output_dir="./data/nyc/",
    post_process=["reproject:EPSG:32618", "cog"],
    bands=["B02", "B03", "B04", "B08"],
)

scene_path = downloads[0].path
print(f"Scene: {scene_path}")
```

---

## 3. Auto-Label with Microsoft Building Footprints

```python
labels = client.labeling.microsoft_buildings(
    bbox=bbox,
    output_path="./labels/ms_buildings.tif",
    resolution_m=10.0,
    crs="EPSG:32618",
)

print(f"Buildings labelled: {labels['n_features']}")
print(f"Label file: {labels['output_path']}")
```

---

## 4. Load Model

```python
# SegFormer-B2 pre-trained on ImageNet, head for 2-class segmentation
model = get_model(
    "segformer-b2",
    num_classes=2,       # background, building
    in_channels=4,       # B, G, R, NIR
    pretrained=True,
    device="cuda",
)
```

---

## 5. Run Inference

```python
inf = TiledInference(
    model=model,
    chip_size=512,
    overlap=64,
    blend_mode="gaussian",
    num_classes=2,
    device="cuda",
    batch_size=4,
)

result = inf.infer(scene_path, "./output/buildings_pred.tif")

print(f"Tiles processed: {result['n_chips']}")
print(f"Time:            {result['duration_seconds']:.1f}s")
print(f"Speed:           {result['chips_per_second']:.1f} chips/s")
```

---

## 6. Post-Process: Vectorise to GeoJSON

```python
import rasterio, numpy as np
from rasterio.features import shapes
from shapely.geometry import shape
import geopandas as gpd

# Load raster prediction
with rasterio.open("./output/buildings_pred.tif") as src:
    pred      = src.read(1).astype(np.uint8)
    transform = src.transform
    crs       = src.crs

# Extract building polygons (class=1)
polys = []
for geom, val in shapes(pred, mask=(pred == 1), transform=transform):
    if val == 1:
        s = shape(geom)
        if s.area > 25:   # Minimum 25 m² (~5×5 pixels)
            polys.append({"geometry": s, "area_m2": round(s.area, 1)})

gdf = gpd.GeoDataFrame(polys, crs=crs)
gdf = gdf.to_crs("EPSG:4326")
gdf.to_file("./output/buildings.geojson", driver="GeoJSON")

print(f"Buildings extracted: {len(gdf)}")
print(f"GeoJSON saved: ./output/buildings.geojson")
```

---

## 7. Explain the Prediction

```python
from pygeovision.explainability.gradcam import GradCAM

cam    = GradCAM(model)
result = cam.batch_explain(
    scene_path, "./output/gradcam.tif",
    class_idx=1, colormap="jet"
)
print("GradCAM saliency map saved")
```

---

## 8. Assess Label Quality

```python
report = client.labeling.quality("./labels/ms_buildings.tif")
print(f"Quality grade: {report['quality_grade']}")
print(f"Score:         {report['quality_score']:.0%}")
```

---

## Results

| Metric | Value |
|--------|-------|
| Scene area | ~60 km² |
| Buildings detected | ~45,000 |
| Processing time | ~8 min (GPU) |
| Output size | ~2 MB GeoJSON |
