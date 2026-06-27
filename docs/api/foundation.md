# Foundation Models

PyGeoVision provides first-class integration of two leading geospatial foundation models: **DINOv3** (12 variants) and **Prithvi-EO-2.0** (600M). Both are completely independent of GeoAI — pure PyTorch + HuggingFace Transformers.

---

## DINOv3

### Model Variants

| Name | Params | Pretrained On | Transform | Embed Dim |
|------|--------|--------------|-----------|-----------|
| `dinov3_vits16` | 21M | LVD-1689M (web) | ImageNet | 384 |
| `dinov3_vits16plus` | 29M | LVD-1689M (web) | ImageNet | 384 |
| `dinov3_vitb16` | 86M | LVD-1689M (web) | ImageNet | 768 |
| `dinov3_vitl16` | 300M | LVD-1689M (web) | ImageNet | 1024 |
| `dinov3_vith16plus` | 840M | LVD-1689M (web) | ImageNet | 1280 |
| `dinov3_vit7b16` | 6.7B | LVD-1689M (web) | ImageNet | 4096 |
| **`dinov3_vitl16_sat`** | **300M** | **SAT-493M** | **Satellite** | **1024** |
| **`dinov3_vit7b16_sat`** | **6.7B** | **SAT-493M** | **Satellite** | **4096** |
| `dinov3_convnext_tiny` | 29M | LVD-1689M | ImageNet | 768 |
| `dinov3_convnext_small` | 50M | LVD-1689M | ImageNet | 768 |
| `dinov3_convnext_base` | 89M | LVD-1689M | ImageNet | 1024 |
| `dinov3_convnext_large` | 198M | LVD-1689M | ImageNet | 1024 |

> **SAT models recommended for satellite imagery.** They use different normalisation statistics — PyGeoVision applies the correct transform automatically.

---

### Normalisation Transforms

```python
from pygeovision.models.foundation.dinov3 import (
    dinov3_web_transform, dinov3_sat_transform, get_transform,
    WEB_MEAN, WEB_STD, SAT_MEAN, SAT_STD,
)

# Web (LVD-1689M) — ImageNet statistics
# mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)
web_t = dinov3_web_transform(resize_size=256, crop_size=224)

# SAT (SAT-493M) — satellite statistics
# mean = (0.430, 0.411, 0.296), std = (0.213, 0.156, 0.143)
sat_t = dinov3_sat_transform(resize_size=256, crop_size=224)

# Auto-select based on model name
transform = get_transform("dinov3_vitl16_sat")   # → satellite stats
transform = get_transform("dinov3_vitl16")        # → ImageNet stats
```

---

### `DINOv3Backbone` — Feature Extraction

```python
from pygeovision.models.foundation.dinov3 import DINOv3Backbone

backbone = DINOv3Backbone(
    model_name="dinov3_vitl16_sat",
    method="hf",             # "hf" | "hub" | "local"
    device="cuda",
)

# Dense spatial feature map
features = backbone.extract_features("sentinel2.tif")
# Shape: (H_p, W_p, 1024) — 14×14 patches for 224px input

# Global CLS embedding for retrieval / similarity
embedding = backbone.extract_embeddings("sentinel2.tif")
# Shape: (1, 1024)

# All patch tokens for dense prediction
patches = backbone.extract_patch_features("sentinel2.tif")
# Shape: (196, 1024)

# Multi-head attention maps (explainability)
attn_maps = backbone.get_attention_maps("sentinel2.tif")
# Shape: (H_p, W_p) — mean over all attention heads
```

### Loading Methods

```python
from pygeovision.models.foundation.dinov3 import (
    load_dinov3_hub, load_dinov3_hf, load_dinov3_local
)

# Method 1: HuggingFace (recommended)
model = load_dinov3_hf("dinov3_vitl16_sat", device="cuda")

# Method 2: PyTorch Hub (official FacebookResearch repo)
model = load_dinov3_hub("dinov3_vitl16", device="cuda")

# Method 3: Local checkpoint (air-gapped deployments)
model = load_dinov3_local("dinov3_vitl16_sat",
                            weights_path="./weights/dinov3_sat.pth",
                            device="cuda")
```

### Build a Classifier (Linear Probing)

```python
clf = backbone.build_classifier(num_classes=10, freeze_backbone=True)
# → GeoClassifier: DINOv3Backbone + LayerNorm + Linear(1024, 10)

# Recommended fine-tuning config
cfg = backbone.finetune_config()
# {'optimizer': 'AdamW', 'learning_rate': 1e-4, 'weight_decay': 0.05, ...}
```

---

### `CHMv2Model` — Canopy Height Maps

```python
from pygeovision.models.foundation.dinov3 import CHMv2Model

chm = CHMv2Model(device="cuda")

# Predict canopy height from Sentinel-2 imagery
result = chm.predict_canopy_height(
    image_path="sentinel2_forest.tif",
    output_path="./output/canopy_height.tif",
    max_height_m=70.0,
)
# result['height_map']   — (H, W) float32 array, metres
# result['statistics']   — {'mean_m': 18.3, 'max_m': 54.2, 'coverage_pct': 72.1}

# Biomass estimation (allometric equations: AGB = 0.112 × H^2.40)
biomass = chm.estimate_biomass("sentinel2_forest.tif")
# {'biomass_map': ..., 'statistics': {'mean_t_ha': 145.2, 'max_t_ha': 380.0}}

# Deforestation detection (compare two dates)
result = chm.detect_deforestation(
    before_path="2021_forest.tif",
    after_path="2024_forest.tif",
    min_height_loss_m=2.0,
    output_path="./output/deforestation.tif",
)
# {'deforested_pct': 3.7, 'area_ha': 142.8}
```

---

### `DINOv3Text` — Zero-Shot Open-Vocabulary

```python
from pygeovision.models.foundation.dinov3 import DINOv3Text

txt = DINOv3Text(backbone="dinov3_vitl16_sat")

# Zero-shot segmentation via text prompt
mask = txt.segment_by_text("scene.tif", "solar panels on rooftops")
# Returns binary mask (H, W) uint8

# Zero-shot object detection
boxes = txt.detect_by_text("scene.tif", "cargo ships")
# Returns list of {'bbox': [x1,y1,x2,y2], 'area_px': N, 'class': 'cargo ships'}

# Scene classification (softmax over text prompts)
probs = txt.classify_by_text("scene.tif", ["dense urban", "forest", "cropland", "water"])
# Returns {'dense urban': 0.72, 'forest': 0.18, ...}
```

---

### Fine-Tuning

```python
from pygeovision.models.foundation.dinov3 import finetune_dinov3

result = finetune_dinov3(
    model_name="dinov3_vitl16_sat",
    dataset=my_dataset,
    task="segmentation",           # or "classification", "detection"
    num_classes=7,
    epochs=100,
    learning_rate=1e-4,            # Paper recommendation
    weight_decay=0.05,
    batch_size=16,
    mixed_precision=True,          # BF16 recommended
    distributed=False,
    output_dir="./checkpoints/dinov3/",
)
# Returns: {'model', 'optimizer', 'scheduler', 'mp_manager', 'config'}
```

---

## Prithvi-EO

### Model Variants

| Name | Params | Coverage | Bands | Frames |
|------|--------|----------|-------|--------|
| `prithvi_eo_1_0` | 100M | USA | 6 (HLS) | 3 |
| `prithvi_eo_2_0` | 600M | Global | 6 (HLS) | 4 |

---

### Band Ordering

Prithvi expects HLS (Harmonized Landsat Sentinel-2) band order. PyGeoVision handles remapping automatically:

| Position | HLS Band | Sentinel-2 | Landsat |
|----------|----------|-----------|---------|
| 0 | Blue | B02 | B2 |
| 1 | Green | B03 | B3 |
| 2 | Red | B04 | B4 |
| 3 | NIR | B08 | B5 |
| 4 | SWIR1 | B11 | B6 |
| 5 | SWIR2 | B12 | B7 |

```python
from pygeovision.models.foundation.prithvi import map_bands, normalise_hls

# Rearrange Sentinel-2 bands to HLS order
data_hls = map_bands(sentinel2_array, source="sentinel2", n_prithvi_bands=6)

# Normalise HLS surface reflectance (integer → float [0,1])
data_norm = normalise_hls(data_hls)   # Divides by HLS_SCALE_FACTOR = 10000.0
```

---

### `Prithvi` — Feature Extraction

```python
from pygeovision.models.foundation.prithvi import Prithvi

model = Prithvi("prithvi_eo_2_0", device="cuda").load()

# CLS token embedding
features = model.extract_features("hls_scene.tif", source="hls")
# Shape: (1, 1024)  — Prithvi-EO-2.0 embed dim

# All patch tokens
patches = model.extract_patch_features("hls_scene.tif")
# Shape: (N_patches, 1024)

# Segmentation head
seg_model = model.build_segmentation_head(num_classes=11, freeze_backbone=True)
```

---

### `PrithviMultiTemporal` — Time Series

```python
from pygeovision.models.foundation.prithvi import PrithviMultiTemporal

mt = PrithviMultiTemporal("prithvi_eo_2_0", device="cuda")

# Multi-temporal feature extraction
result = mt.process_time_series(
    image_paths=["jan.tif", "apr.tif", "jul.tif", "oct.tif"],
    dates=["2024-01", "2024-04", "2024-07", "2024-10"],
    source="hls",
)
# result['features'] — temporal features, result['n_frames'] = 4

# Change detection (temporal attention)
change = mt.detect_change("before.tif", "after.tif", output_path="change.tif")
# {'change_map': ..., 'change_pct': 4.2}

# Trend analysis (linear fit on features)
trend = mt.monitor_trend(["2020.tif","2021.tif","2022.tif","2023.tif","2024.tif"])
# {'trend_direction': 'decreasing', 'trend_magnitude': 0.034}

# Seasonal cycle detection
seasonal = mt.predict_seasonal(images, dates=["jan","apr","jul","oct"])
# {'seasonal_amplitude': 0.18, 'peak_date': 'jul', 'trough_date': 'jan'}
```

---

### `PrithviTasks` — Task-Specific Inference

```python
from pygeovision.models.foundation.prithvi import PrithviTasks

tasks = PrithviTasks("prithvi_eo_2_0", device="cuda")

# Land cover (10 classes)
lc = tasks.land_cover("hls.tif", source="hls", output_path="lc.tif")
# {'prediction': array(...), 'class_pct': {0: 34.2, 1: 21.8, ...}}
# classes: water, trees, grass, flooded_veg, crops, shrub, built, bare, snow_ice, clouds

# Crop mapping (10 major crop types)
crops = tasks.crop_mapping("hls.tif", source="hls")

# Flood detection (binary)
flood = tasks.flood_detection("scene.tif", source="sentinel2")
# {'flood_pct': 12.4, 'n_classes': 2}

# Biomass estimation
bio = tasks.biomass_estimation("hls.tif")
# {'estimated_biomass_t_ha': 143.6}

# Deforestation
defor = tasks.deforestation_detection("2021.tif", "2024.tif", output_path="defor.tif")
```

---

### Fine-Tuning

```python
from pygeovision.models.foundation.prithvi import finetune_prithvi

result = finetune_prithvi(
    model_name="prithvi_eo_2_0",
    task="land_cover",
    num_classes=10,
    epochs=50,
    learning_rate=5e-5,     # Paper recommendation
    weight_decay=0.01,
    batch_size=8,
    mixed_precision=True,
)
```

---

## GeoAI Engine

Access foundation models through the high-level engine:

```python
import pygeovision as pgv

client = pgv.PyGeoVision()

# DINOv3
client.geoai.dinov3.load("dinov3_vitl16_sat")
features = client.geoai.dinov3.extract_features("sentinel2.tif")
height   = client.geoai.dinov3.canopy_height("forest.tif")
mask     = client.geoai.dinov3.zero_shot("scene.tif", "solar panels")

# Prithvi
client.geoai.prithvi.load("prithvi_eo_2_0")
lc     = client.geoai.prithvi.land_cover("hls.tif")
change = client.geoai.prithvi.change_detection("2021.tif", "2024.tif")

# All at once
print(client.geoai.foundation_models.list())
# {'dinov3': [...12 models...], 'prithvi': [...4 models...]}
```
