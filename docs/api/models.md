# Model Layer

119 production-ready geospatial model architectures — fully independent of GeoAI.

---

## Model Registry

### `model_registry`

The central registry of all available model architectures.

```python
from pygeovision.models.registry import model_registry

# Summary
summary = model_registry.summary()
print(f"Total architectures: {summary['total']}")
# {'total': 119, 'by_task': {'segmentation': 19, 'detection': 18, ...}}

# List all models
all_models = model_registry.list()

# Filter by task
seg_models = model_registry.list(task="segmentation")
# ['unet-r50', 'unet-r101', 'segformer-b0', ...]

# Filter by family
vit_models = model_registry.list(family="vit")

# Filter by parameter budget
small_models = model_registry.list(max_params_m=30.0)

# Search by keyword
results = model_registry.search("unet")
# Returns list of ModelSpec objects

# Get spec for one model
spec = model_registry["segformer-b2"]
print(spec.name, spec.task, spec.params_m, spec.hf_id)

# Top-N models for a task (sorted by param count)
top5 = model_registry.top_by_task("segmentation", n=5)
```

---

## `get_model()`

Load any registered model with geospatial configuration.

```python
from pygeovision.models import get_model

model = get_model(
    name="segformer-b2",
    num_classes=7,       # Segmentation classes
    in_channels=4,       # Input bands (4 = Sentinel-2 BGRN)
    pretrained=True,     # Load ImageNet/HF pretrained weights
    device="cuda",       # Optional: move to device immediately
)
```

**Loading priority:**
1. `timm` registry (fastest, most models)
2. HuggingFace Hub (transformers-based models)
3. Built-in PyTorch fallback (torchvision + custom U-Net)

---

## Models by Task

### Segmentation (19 models)

| Name | Family | Params | Backbone | Notes |
|------|--------|--------|----------|-------|
| `unet-r50` | unet | 31M | ResNet-50 | Classic encoder-decoder |
| `unet-r101` | unet | 50M | ResNet-101 | Deeper encoder |
| `unet-efficientb4` | unet | 24M | EfficientNet-B4 | Efficient backbone |
| `segformer-b0` | segformer | 3.8M | MiT-B0 | Lightweight transformer |
| `segformer-b2` | segformer | 27.5M | MiT-B2 | Balanced — recommended |
| `segformer-b5` | segformer | 84.7M | MiT-B5 | Highest accuracy |
| `deeplab-r50` | deeplab | 43M | ResNet-50 | Atrous convolutions |
| `deeplab-r101` | deeplab | 63M | ResNet-101 | Multi-scale context |
| `pspnet-r50` | pspnet | 47M | ResNet-50 | Pyramid pooling |
| `fcn-r50` | fcn | 35M | ResNet-50 | Fully convolutional |
| `mask2former-swin-t` | mask2former | 47M | Swin-T | Panoptic capable |
| `mask2former-swin-b` | mask2former | 102M | Swin-B | Best panoptic |
| `sam-vit-h` | sam | 636M | ViT-H | Zero-shot prompting |
| `sam-vit-l` | sam | 308M | ViT-L | Balanced zero-shot |
| `sam-vit-b` | sam | 94M | ViT-B | Lightweight zero-shot |
| `sam2-hiera-l` | sam2 | 224M | Hiera-L | Video + image SAM |
| `upernet-swin-b` | upernet | 121M | Swin-B | Dense prediction |

```python
# Example: load SegFormer for 4-band multispectral input
model = get_model("segformer-b2", num_classes=11, in_channels=4)
```

---

### Detection (18 models)

| Name | Family | Params | Notes |
|------|--------|--------|-------|
| `yolov8-n/s/m/l/x` | yolo | 3–68M | Real-time detection |
| `yolov9-c` | yolo | 25M | Programmable gradient |
| `rf-detr-b` | detr | 29M | Roboflow DETR base |
| `rf-detr-l` | detr | 128M | Roboflow DETR large |
| `rt-detr-l` | detr | 32M | Real-time DETR |
| `detr-r50` | detr | 41M | Original DETR |
| `detr-r101` | detr | 60M | DETR larger backbone |
| `dino-detr-r50` | detr | 47M | DINO-DETR |
| `grounding-dino` | vlm | 172M | Text-driven detection |
| `faster-rcnn-r50` | rcnn | 42M | Two-stage classic |
| `mask-rcnn-r50` | rcnn | 44M | Instance segmentation |
| `fcos-r50` | anchor_free | 32M | Anchor-free |
| `centernet-r50` | anchor_free | 33M | Heatmap-based |

```python
from pygeovision.models.detection.yolo import GeoYOLO

detector = GeoYOLO("yolov8-m", num_classes=3,
                    class_names=["building","vehicle","ship"])
result = detector.detect("scene.tif", conf=0.35)
print(f"Detections: {result['n_detections']}")
```

---

### Classification (21 models)

| Name | Family | Params | Pretrained on |
|------|--------|--------|---------------|
| `vit-b16` | vit | 86M | ImageNet-21k |
| `vit-l16` | vit | 307M | ImageNet-21k |
| `swin-t/s/b/l` | swin | 28–197M | ImageNet |
| `convnext-t/b/l` | convnext | 29–198M | ImageNet |
| `resnet50/101/152` | resnet | 26–60M | ImageNet |
| `efficientnet-b4/b7` | efficientnet | 19–66M | ImageNet |
| `densenet121/201` | densenet | 8–20M | ImageNet |
| `dinov2-s/b/l/g` | dinov2 | 21–1100M | LVD-142M |
| `clip-vit-b32` | clip | 151M | LAION-400M |

```python
model = get_model("swin-b", num_classes=45, in_channels=4)
```

---

### Change Detection (10 models)

| Name | Family | Notes |
|------|--------|-------|
| `changeformer-mit-b0` | changeformer | Siamese transformer |
| `changeformer-mit-b4` | changeformer | Higher accuracy |
| `changestar-r18/r50` | changestar | Change + semantic |
| `bit-r50` | bit | Binary change |
| `dsamnet` | dsamnet | Depth-separable |
| `snunet-32/128` | snunet | Dense skip connections |
| `tinycd` | lightweight | 0.3M — mobile-ready |

```python
from pygeovision.models.change_detection.changeformer import ChangeFormer

cd = ChangeFormer(num_classes=2, in_channels=4)
result = cd.detect("before.tif", "after.tif", "change.tif")
print(f"Changed area: {result['change_pct']:.1f}%")
```

---

### Foundation Models (28 entries)

See the dedicated [Foundation Models](foundation.md) page for full DINOv3 and Prithvi documentation.

| Name | Params | Pretrained on |
|------|--------|---------------|
| `dinov3_vitl16_sat` | 300M | SAT-493M |
| `dinov3_vit7b16_sat` | 6.7B | SAT-493M |
| `prithvi_eo_2_0` | 600M | HLS Global (10yr) |
| `remoteclip-l14` | 428M | RS5M (5M RS pairs) |
| `dofa-base` | 86M | Sentinel-1/2, Landsat |
| ... | | |

---

## `GeoModel`

Base wrapper class that adds geospatial utilities to any PyTorch model.

```python
from pygeovision.models.base import GeoModel, GeoModelConfig

config = GeoModelConfig(
    name="my-model",
    task="segmentation",
    num_classes=7,
    in_channels=4,
    input_size=(512, 512),
)

geo_model = GeoModel(pytorch_model, config)

# Export to ONNX
geo_model.export_onnx("model.onnx", input_shape=(1, 4, 512, 512))

# Move device
geo_model.to("cuda")

print(geo_model)
# GeoModel(name=my-model, task=segmentation, classes=7, params=27.5M)
```

---

## Weight Downloader

```python
from pygeovision.models.weights.downloader import WeightDownloader

downloader = WeightDownloader(cache_dir="~/.cache/pygeovision/weights")

# Download a model from HuggingFace Hub
path = downloader.download("facebook/sam-vit-large")

# List cached weights
cached = downloader.list_cached()
for w in cached:
    print(f"{w['path']}  {w['size_mb']:.0f}MB")

# Cache size
print(f"Cache: {downloader.cache_size_gb():.2f} GB")

# Clear specific model
downloader.clear_cache("facebook/sam-vit-large")
```
