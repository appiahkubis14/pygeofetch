# Inference Engine

Four inference modes for arbitrarily large GeoTIFFs: tiled, batch, streaming, and ensemble. All preserve geospatial metadata (CRS, transform, nodata) in the output.

---

## Tiled Inference

`TiledInference` processes large images by splitting into overlapping chips, running the model on each, and reassembling with seamless blending. This is the primary inference mode for satellite imagery.

```python
from pygeovision.inference.tiled import TiledInference
from pygeovision.models import get_model

model = get_model("segformer-b2", num_classes=7, in_channels=4)

inf = TiledInference(
    model=model,
    chip_size=512,           # Chip width and height in pixels
    overlap=64,              # Overlap between adjacent chips
    blend_mode="gaussian",   # "gaussian" | "linear" | "constant"
    num_classes=7,
    device="cuda",           # Auto-detected if None
    batch_size=4,            # Chips processed per forward pass
    tta=False,               # Test-time augmentation (4× slower)
)

result = inf.infer(
    input_path="large_scene.tif",
    output_path="./output/prediction.tif",
)

print(f"Chips processed:  {result['n_chips']}")
print(f"Time:             {result['duration_seconds']:.1f}s")
print(f"Throughput:       {result['chips_per_second']:.1f} chips/s")
print(f"Output:           {result['output_path']}")
```

### Blend Modes

| Mode | Description | Best For |
|------|-------------|----------|
| `gaussian` | Smooth Gaussian weight falloff from chip centre | General use — eliminates seam artefacts |
| `linear` | Linear weight from centre to edge | Slightly faster than Gaussian |
| `constant` | Uniform weight, hard boundaries at overlaps | Fast inspection runs |

### Test-Time Augmentation

```python
inf = TiledInference(model=model, tta=True, tta_flips=["h","v","hv"])
```

Each chip is inferred 4× (original + 3 flips) and averaged. Increases accuracy ~1–2 mIoU points at 4× the inference cost.

---

## Batch Inference

`BatchInferenceEngine` processes entire directories of GeoTIFFs in parallel.

```python
from pygeovision.inference.batch import BatchInferenceEngine

engine = BatchInferenceEngine(
    model=model,
    chip_size=512,
    overlap=64,
    n_workers=4,            # Parallel worker processes
    device="cuda",
    output_suffix="_pred",
)

# Process a directory
result = engine.run_directory(
    input_dir="./data/scenes/",
    output_dir="./data/predictions/",
    pattern="*.tif",         # Glob pattern
    recursive=False,
)

print(f"Success:  {result['n_success']}")
print(f"Failed:   {result['n_failed']}")
print(f"Total:    {result['total_time_s']:.0f}s")
print(f"Speed:    {result['throughput_fps']:.2f} frames/s")

# Process a list of specific files
result = engine.run_files(["a.tif", "b.tif", "c.tif"])
```

---

## Streaming Inference

`StreamingInference` is memory-optimised for extremely large images (50+ GB). It reads and writes one row-of-chips at a time, keeping RAM usage constant regardless of image size.

```python
from pygeovision.inference.stream import StreamingInference

stream = StreamingInference(
    model=model,
    chip_size=1024,
    overlap=128,
    device="cuda",
    buffer_rows=2,          # Chip rows buffered in RAM at once
)

result = stream.infer(
    input_path="100km_scene_50gb.tif",
    output_path="./output/stream_pred.tif",
)
```

---

## Ensemble Inference

`EnsembleInference` combines predictions from multiple models into a single output.

```python
from pygeovision.inference.stream import EnsembleInference
from pygeovision.models import get_model

models = [
    get_model("segformer-b2",  num_classes=7),
    get_model("unet-r50",      num_classes=7),
    get_model("deeplab-r101",  num_classes=7),
]

ensemble = EnsembleInference(
    models=models,
    weights=[0.5, 0.3, 0.2],       # Per-model importance weights
    fusion="weighted_mean",          # "mean" | "weighted_mean" | "max_confidence"
    chip_size=512,
    overlap=64,
)

result = ensemble.infer("scene.tif", "ensemble_pred.tif")
print(f"Ensemble of {result['n_models']} models")
```

### Fusion Strategies

| Strategy | Description |
|----------|-------------|
| `mean` | Simple average of softmax probabilities |
| `weighted_mean` | Weighted average — higher weight = more influence |
| `max_confidence` | Take argmax of the most confident model at each pixel |
| `majority_vote` | Pixel-wise majority vote across argmax predictions |

---

## ONNX Inference

For production deployments, use `ONNXRuntimeInference` (see [Edge Deployment](edge.md)):

```python
from pygeovision.edge.onnx_rt import ONNXRuntimeInference

# Export once
ONNXRuntimeInference.from_pytorch(model, "model.onnx")

# Then run — no PyTorch required
eng    = ONNXRuntimeInference("model.onnx", device="cuda")
result = eng.infer_geotiff("scene.tif", "output.tif")
```

---

## Output Format

All inference engines write standard GeoTIFF outputs:

- **dtype:** `uint8` for class predictions, `float32` for probability maps
- **CRS:** Copied from input
- **Transform:** Copied from input (pixel-perfect alignment)
- **Bands:** 1 band (class ID) or C bands (per-class probabilities if `return_probabilities=True`)
- **Compression:** LZW (lossless, ~4× size reduction)
- **Tags:** Model name, inference date, chip_size, overlap written to GeoTIFF metadata
