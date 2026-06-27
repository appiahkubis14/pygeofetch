# Foundation Models

Practical guide to DINOv3 (12 variants) and Prithvi-EO-2.0 (600M) for geospatial analysis.

---

## When to Use Which Model

| Task | Recommended Model |
|------|------------------|
| Feature extraction / retrieval | DINOv3 SAT (`dinov3_vitl16_sat`) |
| Canopy height estimation | DINOv3 CHMv2 |
| Zero-shot segmentation | DINOv3Text |
| Multi-temporal analysis | Prithvi-EO-2.0 |
| Land cover / crop mapping | Prithvi-EO-2.0 |
| Flood detection | Prithvi-EO-2.0 |
| Biomass estimation | Prithvi-EO-2.0 or CHMv2 |

---

## DINOv3 Feature Extraction

```python
from pygeovision.models.foundation.dinov3 import DINOv3Backbone

backbone = DINOv3Backbone("dinov3_vitl16_sat", device="cuda")

# Dense spatial features — useful for segmentation decoders
features = backbone.extract_features("sentinel2.tif")
print(f"Feature map: {features.shape}")   # (14, 14, 1024)

# Global embedding — useful for image retrieval
embedding = backbone.extract_embeddings("sentinel2.tif")
print(f"Embedding: {embedding.shape}")    # (1, 1024)

# Visualise attention
attn = backbone.get_attention_maps("sentinel2.tif")
print(f"Attention map: {attn.shape}")     # (14, 14)
```

---

## DINOv3 Canopy Height (CHMv2)

```python
from pygeovision.models.foundation.dinov3 import CHMv2Model

chm = CHMv2Model(device="cuda")

# Predict height
result = chm.predict_canopy_height(
    "sentinel2_amazon.tif",
    output_path="./output/canopy_height.tif",
)
print(f"Mean canopy: {result['statistics']['mean_m']:.1f}m")
print(f"Coverage:    {result['statistics']['coverage_pct']:.1f}%")

# Biomass
bio = chm.estimate_biomass("sentinel2_amazon.tif")
print(f"Biomass: {bio['statistics']['mean_t_ha']:.0f} t/ha")

# Deforestation (two dates)
defor = chm.detect_deforestation(
    "amazon_2020.tif", "amazon_2024.tif",
    min_height_loss_m=3.0,
    output_path="./output/deforestation.tif",
)
print(f"Deforested: {defor['area_ha']:.0f} ha")
```

---

## DINOv3 Zero-Shot Segmentation

```python
from pygeovision.models.foundation.dinov3 import DINOv3Text

txt  = DINOv3Text("dinov3_vitl16_sat")

# Segment by text
mask = txt.segment_by_text("urban_scene.tif", "solar panels on rooftops")
# Binary mask: 1 = solar panel, 0 = background

# Detect objects by text
ships = txt.detect_by_text("harbour.tif", "cargo ship")
print(f"Detected: {len(ships)} ships")

# Classify the scene
probs = txt.classify_by_text("scene.tif",
    ["dense urban", "forest", "cropland", "water"])
best = max(probs, key=probs.get)
print(f"Scene type: {best} ({probs[best]:.0%})")
```

---

## Prithvi-EO-2.0 Multi-Temporal Analysis

```python
from pygeovision.models.foundation.prithvi import Prithvi, PrithviMultiTemporal, PrithviTasks

# Load model
model = Prithvi("prithvi_eo_2_0", device="cuda").load()
```

### Land Cover

```python
tasks = PrithviTasks("prithvi_eo_2_0", device="cuda")

lc = tasks.land_cover("hls_scene.tif", output_path="./output/land_cover.tif")
print("Class distribution:")
for cls, pct in sorted(lc['class_pct'].items(), key=lambda x: -x[1])[:5]:
    print(f"  {lc['class_names'][cls]:<15} {pct:.1f}%")
```

### Crop Mapping

```python
crops = tasks.crop_mapping(
    "hls_agriculture.tif",
    source="hls",
    output_path="./output/crop_map.tif",
)
print("Crop distribution:")
for cls, pct in sorted(crops['class_pct'].items(), key=lambda x: -x[1])[:3]:
    print(f"  {crops['class_names'][cls]:<15} {pct:.1f}%")
```

### Flood Detection

```python
flood = tasks.flood_detection(
    "sentinel2_after_rain.tif",
    source="sentinel2",
    output_path="./output/flood_mask.tif",
)
print(f"Flooded area: {flood['flood_pct']:.1f}%")
```

### Time Series Analysis

```python
mt = PrithviMultiTemporal("prithvi_eo_2_0", device="cuda")

monthly_images = [f"hls_{m:02d}.tif" for m in range(1, 13)]
dates = [f"2024-{m:02d}" for m in range(1, 13)]

result = mt.process_time_series(monthly_images, dates=dates)
trend  = mt.monitor_trend(monthly_images)
seasonal = mt.predict_seasonal(monthly_images, dates=dates)

print(f"Annual trend: {trend['trend_direction']}")
print(f"Peak month:   {seasonal['peak_date']}")
```

---

## Fine-Tuning DINOv3

```python
from pygeovision.models.foundation.dinov3 import finetune_dinov3

result = finetune_dinov3(
    model_name="dinov3_vitl16_sat",
    task="segmentation",
    num_classes=2,          # Building vs background
    epochs=50,
    learning_rate=1e-5,     # Low LR for SAT models
)

print("Training config:", result["config"])
print("Status:", result["status"])
```

---

## Fine-Tuning Prithvi

```python
from pygeovision.models.foundation.prithvi import finetune_prithvi

result = finetune_prithvi(
    model_name="prithvi_eo_2_0",
    task="crop_mapping",
    num_classes=10,
    epochs=30,
    learning_rate=5e-5,     # Paper-recommended LR
    batch_size=8,           # Memory-limited for 600M params
)
```

---

## Transform Correctness

Always verify you're using the right transform:

```python
from pygeovision.models.foundation.dinov3 import (
    WEB_MEAN, WEB_STD, SAT_MEAN, SAT_STD, get_transform
)

# These are DIFFERENT — wrong transform = silently bad results
print("Web stats:", WEB_MEAN, WEB_STD)   # ImageNet
print("SAT stats:", SAT_MEAN, SAT_STD)   # Satellite

# Auto-select
t = get_transform("dinov3_vitl16_sat")   # → satellite stats
t = get_transform("dinov3_vitl16")       # → web/ImageNet stats
```
