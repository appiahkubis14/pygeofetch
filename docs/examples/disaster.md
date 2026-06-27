# Disaster Response

Rapid damage assessment, building damage classification, and recovery monitoring after natural disasters.

---

## Building Damage Assessment (xBD)

Post-disaster building damage classification using 4 classes: no damage, minor, major, destroyed.

```python
import pygeovision as pgv
from pygeovision.models import get_model
from pygeovision.inference.tiled import TiledInference

client = pgv.PyGeoVision()

# Before/after imagery
before_path = "./data/turkey_earthquake_before.tif"
after_path  = "./data/turkey_earthquake_after.tif"

# 4-class damage model
model  = get_model("segformer-b5", num_classes=4, in_channels=4)
inf    = TiledInference(model=model, chip_size=512, overlap=64)
result = inf.infer(after_path, "./output/damage_map.tif")

# Change-based damage assessment
from pygeovision.models.change_detection.changeformer import ChangeFormer
cd = ChangeFormer(num_classes=4, in_channels=4, device="cuda")
damage = cd.detect(before_path, after_path, "./output/damage_change.tif")
print(f"Damaged area: {damage['change_pct']:.1f}%")
```

---

## Rapid Post-Event Mapping

```python
from pygeovision.models.foundation.prithvi import PrithviMultiTemporal

mt = PrithviMultiTemporal("prithvi_eo_2_0", device="cuda")

# Compare pre and post event
result = mt.detect_change(before_path, after_path,
                           output_path="./output/rapid_change.tif")
print(f"Changed:  {result['change_pct']:.1f}%")
```

---

## Landslide Detection

```python
model = get_model("unet-r101", num_classes=2, in_channels=4)  # landslide vs stable
inf   = TiledInference(model=model, chip_size=512)
result = inf.infer("post_rain_dem_deriv.tif", "./output/landslide_risk.tif")
```

---

## Pipeline for Disaster Response

```python
from pygeovision.pipelines import Pipeline

# Load the built-in disaster assessment template
p = Pipeline.from_yaml("pygeovision/pipelines/templates/disaster.yaml")

result = p.run(context={
    "bbox":        [36.0, 36.5, 37.0, 37.5],   # Affected area
    "event_date":  "2024-09-01",
    "output_dir":  "./output/disaster_2024/",
})

print(f"Pipeline complete: {result.success}")
print(f"Steps: {result.steps_completed}")
```

---

## Key Datasets

| Dataset | Task | Events |
|---------|------|--------|
| xBD | Building damage (4 classes) | 19 disaster events |
| SpaceNet 8 | Flood + building | Hurricane Harvey, etc. |
| FloodNet | Flood + VQA (10 classes) | Hurricane Harvey |
| RescueNet | Post-disaster segmentation | 7 classes |
| BARD | Building damage from SAR | Multiple events |
