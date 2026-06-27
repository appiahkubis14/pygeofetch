# Urban Mapping

Building extraction, road detection, land-use classification, and urban growth monitoring.

---

## Building Footprint Extraction

```python
import pygeovision as pgv
from pygeovision.models import get_model
from pygeovision.inference.tiled import TiledInference

client = pgv.PyGeoVision()
bbox   = [2.28, 48.82, 2.42, 48.92]   # Paris, France

# Auto-label from Microsoft Building Footprints
labels = client.labeling.microsoft_buildings(bbox, output_path="./labels/buildings.tif")
print(f"Buildings labelled: {labels['n_features']}")

# SegFormer inference
model  = get_model("segformer-b2", num_classes=2, in_channels=4)
scene  = "./data/paris_s2.tif"
inf    = TiledInference(model=model, chip_size=512, blend_mode="gaussian")
result = inf.infer(scene, "./output/buildings_pred.tif")
```

---

## Multi-Class Urban Segmentation

8 urban land-use classes: building, road, vegetation, water, parking, impervious, soil, background.

```python
model = get_model("segformer-b5", num_classes=8, in_channels=4)
inf   = TiledInference(model=model, chip_size=512, overlap=64)
result = inf.infer(scene, "./output/urban_lc.tif")
```

---

## Road Network Extraction

```python
from pygeovision.models import get_model
from pygeovision.labeling.osm import OSMLabeler

# Auto-label roads from OSM
road_labels = OSMLabeler().label(bbox, categories=["roads"],
                                  output_path="./labels/roads.tif")

# Train or infer with U-Net
road_model = get_model("unet-r50", num_classes=2, in_channels=4)
inf        = TiledInference(road_model, chip_size=512, blend_mode="gaussian")
result     = inf.infer(scene, "./output/roads.tif")
```

---

## Urban Growth Monitoring

```python
from pygeovision.models.change_detection.changeformer import ChangeFormer

cd = ChangeFormer(num_classes=4, in_channels=4)   # no-change, new building, demolition, other
result = cd.detect("paris_2018.tif", "paris_2024.tif", "./output/urban_growth.tif")
print(f"New construction: {result['change_pct']:.1f}% of area")
```

---

## Object Detection (Vehicles, Aircraft, Ships)

```python
from pygeovision.models.detection.yolo import GeoYOLO

# YOLOv8 for vehicle detection
detector = GeoYOLO("yolov8-l", num_classes=3,
                    class_names=["car","truck","bus"])
result   = detector.detect("urban_vhr.tif", conf=0.35)
print(f"Vehicles detected: {result['n_detections']}")
for d in result['detections'][:5]:
    print(f"  {d['class']:<8} conf={d['confidence']:.2f}  bbox={d['bbox_px']}")
```

---

## Key Datasets

| Dataset | Task | Notes |
|---------|------|-------|
| iSAID | Instance segmentation | 15 classes |
| DOTA-v2 | Detection | 18 classes, 195K instances |
| SpaceNet 1-8 | Buildings, roads | High-res VHR |
| Inria Aerial | Building segmentation | 5 cities |
| OpenCities | Building footprints | 790K buildings |
