# Frequently Asked Questions

---

## General

**Does PyGeoVision require geoai-py?**

No. PyGeoVision v2.0 is 100% independent of geoai-py. All model implementations are self-contained in pure PyTorch + HuggingFace Transformers. You can verify this yourself:

```python
import pygeovision as pgv
pgv.PyGeoVision()  # works without geoai-py installed
```

**Which Python versions are supported?**

Python 3.10, 3.11, and 3.12.

**Does it work on Windows?**

Yes, with minor caveats: GDAL/rasterio on Windows requires installing from the [Christoph Gohlke wheels](https://www.lfd.uci.edu/~gohlke/pythonlibs/) or using conda. Everything else runs natively on Windows.

**Is a GPU required?**

No. All models support CPU inference (slow but functional). A GPU with ≥8GB VRAM is recommended for training. For inference on large GeoTIFFs, tiled processing keeps GPU memory bounded regardless of image size.

---

## Data

**Which satellite providers are supported?**

22 providers including: Planetary Computer, AWS Earth, Copernicus, USGS EarthExplorer, Maxar, Planet, Airbus SPOT/Pléiades, Satellogic, DigitalGlobe, and more.

**What image formats does PyGeoVision read?**

Any format rasterio supports: GeoTIFF, Cloud-Optimized GeoTIFF (COG), NetCDF, HDF5, VRT, JPEG2000, and more.

**How do I handle large GeoTIFFs (>1 GB)?**

Use `TiledInference` — it processes the image in overlapping chips and reassembles the result:

```python
from pygeovision.inference.tiled import TiledInference

inf = TiledInference(model=model, chip_size=512, overlap=64)
result = inf.infer("large_100km.tif", "output.tif")
```

**What CRS does PyGeoVision use?**

All spatial operations preserve the input CRS. Outputs are written with the same CRS as the input unless you explicitly reproject via `post_process=["reproject:EPSG:4326"]`.

---

## Models

**How do I add my own model?**

Register it in the model registry and implement a factory function:

```python
from pygeovision.models.registry import ModelSpec, register_model

register_model(ModelSpec(
    name="my-custom-unet",
    task="segmentation",
    family="unet",
    params_m=12.5,
    description="Custom U-Net for wetland mapping",
))
```

**Can I use models fine-tuned on my own data?**

Yes. Load the checkpoint directly:

```python
import torch
from pygeovision.models import get_model

model = get_model("segformer-b2", num_classes=5)
model.load_state_dict(torch.load("my_checkpoint.pth")["model_state_dict"])
```

**How do I export a model for production deployment?**

```python
from pygeovision.edge.onnx_rt import ONNXRuntimeInference

ONNXRuntimeInference.from_pytorch(model, "model.onnx",
                                   input_shape=(1, 4, 512, 512))
```

---

## Foundation Models

**What is the difference between DINOv3 web and SAT models?**

Web models (`dinov3_vitl16`) are pre-trained on LVD-1689M — a large internet image dataset. SAT models (`dinov3_vitl16_sat`) are additionally fine-tuned on SAT-493M, a curated dataset of 493 million satellite images. SAT models produce better features for geospatial tasks.

Critically, each uses different normalisation statistics — the wrong transform will silently degrade performance:

| Model type | Mean (R, G, B) | Std (R, G, B) |
|------------|---------------|--------------|
| Web (ImageNet) | 0.485, 0.456, 0.406 | 0.229, 0.224, 0.225 |
| SAT-493M | 0.430, 0.411, 0.296 | 0.213, 0.156, 0.143 |

PyGeoVision selects the correct transform automatically via `get_transform(model_name)`.

**What spectral bands does Prithvi expect?**

Prithvi-EO is pre-trained on HLS (Harmonized Landsat Sentinel-2) data with 6 bands in this order: **Blue, Green, Red, NIR, SWIR1, SWIR2**.

PyGeoVision handles the band reordering automatically:

```python
from pygeovision.models.foundation.prithvi import map_bands

# Sentinel-2 → Prithvi HLS order (6 bands)
data_hls = map_bands(sentinel2_data, source="sentinel2", n_prithvi_bands=6)
```

**Can I fine-tune DINOv3 or Prithvi on my own dataset?**

Yes, both have built-in fine-tuning support:

```python
# DINOv3
from pygeovision.models.foundation.dinov3 import finetune_dinov3
result = finetune_dinov3("dinov3_vitl16_sat", task="segmentation", num_classes=5)

# Prithvi
from pygeovision.models.foundation.prithvi import finetune_prithvi
result = finetune_prithvi("prithvi_eo_2_0", task="land_cover", num_classes=10)
```

---

## Training

**How do I enable distributed training?**

```python
from pygeovision.training.distributed import launch_ddp

def train_fn(rank, world_size):
    from pygeovision.training.trainer import GeoTrainer
    trainer = GeoTrainer(model=model, distributed=True)
    trainer.fit(train_dl, val_dl)

launch_ddp(train_fn)
```

**What mixed precision format should I use?**

BF16 is recommended for modern GPUs (Ampere+, A100, H100, RTX 3090+). Use FP16 for older GPUs.

```python
trainer = GeoTrainer(model=model, mixed_precision="bf16")  # or "fp16"
```

**Can I resume training from a checkpoint?**

```python
from pygeovision.training.checkpoint import CheckpointManager

cm = CheckpointManager("./checkpoints/")
state = cm.load_last(model, optimizer, scheduler)
start_epoch = state["epoch"] + 1
```

---

## Deployment

**How do I start the inference server?**

```python
from pygeovision.serving import InferenceServer

server = InferenceServer(auth_keys={"myuser": "my-secret-key"})
server.register("seg_v1", "./model.onnx", task="segmentation", num_classes=7)
server.serve(host="0.0.0.0", port=8080)
```

Then call it from any HTTP client:

```bash
curl -X POST http://localhost:8080/predict \
  -H "X-API-Key: my-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://...", "model_name": "seg_v1"}'
```

**How do I deploy to AWS SageMaker?**

```python
from pygeovision.cloud.deploy import AWSDeployer

result = AWSDeployer(region="us-east-1").deploy(
    "./model.onnx",
    endpoint_name="pygeovision-prod",
    instance_type="ml.g4dn.xlarge",
)
print(result["endpoint_url"])
```

**Does the Jetson Nano / Orin deployment work without internet?**

Yes. Export the ONNX model on an internet-connected machine, copy it to the Jetson, and convert with TensorRT locally:

```bash
pgv edge export-onnx segformer-b2 --output model.onnx --classes 7
# Copy model.onnx to Jetson, then on-device:
pgv edge deploy-jetson model.onnx --output model.trt --fp16
```

---

## Getting Help

- GitHub Issues: [github.com/pygeovision/pygeovision/issues](https://github.com/pygeovision/pygeovision/issues)
- Discussions: [github.com/pygeovision/pygeovision/discussions](https://github.com/pygeovision/pygeovision/discussions)
- Documentation: [pygeovision.org](https://pygeovision.org)
