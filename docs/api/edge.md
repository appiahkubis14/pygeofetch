# Edge Deployment

Deploy PyGeoVision models to edge hardware: ONNX Runtime (CPU/CUDA/TensorRT/CoreML) and NVIDIA Jetson (TensorRT).

---

## ONNX Export

Convert any PyTorch model to ONNX format for portable, framework-independent deployment.

```python
from pygeovision.edge.onnx_rt import ONNXRuntimeInference
from pygeovision.models import get_model

model = get_model("segformer-b2", num_classes=7, in_channels=4)

# Export to ONNX
ONNXRuntimeInference.from_pytorch(
    model=model,
    output_path="./deploy/model.onnx",
    input_shape=(1, 4, 512, 512),   # (batch, channels, height, width)
    opset_version=17,
    simplify=True,                   # Run onnx-simplifier (pip install onnxsim)
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    verbose=False,
)

print("ONNX model exported")
```

CLI equivalent:

```bash
pgv edge export-onnx segformer-b2 \
    --output ./deploy/model.onnx \
    --classes 7 \
    --in-channels 4 \
    --input-size 512
```

---

## ONNX Inference

```python
from pygeovision.edge.onnx_rt import ONNXRuntimeInference

eng = ONNXRuntimeInference(
    model_path="./deploy/model.onnx",
    device="cuda",            # "cpu" | "cuda" | "tensorrt" | "coreml"
    num_threads=4,            # CPU inference thread count
    optimize_level=99,        # ORT optimisation level (0=disabled, 99=full)
)

# Infer a single GeoTIFF (tiled, preserves CRS)
result = eng.infer_geotiff(
    input_path="scene.tif",
    output_path="./output/pred.tif",
    chip_size=512,
    overlap=64,
    blend_mode="gaussian",
)

print(f"Chips: {result['n_chips']}  Time: {result['duration_s']:.1f}s")
print(f"Speed: {result['fps']:.1f} chips/s")
```

---

## Benchmarking

```python
eng = ONNXRuntimeInference("model.onnx", device="cuda")

report = eng.benchmark(
    n_runs=200,
    input_shape=(1, 4, 512, 512),
    warmup_runs=10,
)

print(f"Mean latency:  {report['mean_ms']:.1f} ms")
print(f"P50 latency:   {report['p50_ms']:.1f} ms")
print(f"P95 latency:   {report['p95_ms']:.1f} ms")
print(f"P99 latency:   {report['p99_ms']:.1f} ms")
print(f"Throughput:    {report['fps']:.1f} fps")
print(f"Device:        {report['device']}")
```

CLI:

```bash
pgv edge benchmark-onnx model.onnx --device cuda --runs 200
```

---

## Execution Providers

| Provider | Hardware | Install |
|----------|----------|---------|
| `cpu` | Any CPU | Default |
| `cuda` | NVIDIA GPU (CUDA 11+) | `pip install onnxruntime-gpu` |
| `tensorrt` | NVIDIA GPU (TensorRT 8+) | `pip install onnxruntime-gpu` + TensorRT |
| `coreml` | Apple Silicon / macOS | `pip install onnxruntime-silicon` |
| `openvino` | Intel CPU/GPU | `pip install onnxruntime-openvino` |

```python
# Auto-detect best available provider
eng = ONNXRuntimeInference("model.onnx", device="auto")
print(f"Using: {eng.provider}")
```

---

## NVIDIA Jetson Deployment

Convert ONNX models to TensorRT engine files optimised for Jetson hardware (Orin, Xavier, Nano).

```python
from pygeovision.edge.jetson import JetsonDeployer

deployer = JetsonDeployer(
    device_type="orin",     # "nano" | "xavier" | "orin"
    workspace_gb=4,
)

result = deployer.convert(
    onnx_path="model.onnx",
    engine_path="model_fp16.trt",
    trt_path="./trt_cache/",
    precision="fp16",        # "fp32" | "fp16" | "int8"
    max_batch_size=4,
    input_shape=(1, 4, 512, 512),
    calibration_images=None, # Required for INT8 — list of calibration image paths
)

print(f"Engine: {result['engine_path']}")
print(f"Size:   {result['size_mb']:.1f} MB")
print(f"Time:   {result['build_time_s']:.0f}s")
```

### Jetson Inference

```python
deployer = JetsonDeployer()
result   = deployer.infer(
    engine_path="model_fp16.trt",
    image_path="scene.tif",
    output_path="./output/jetson_pred.tif",
)

print(f"Latency: {result['latency_ms']:.1f}ms per chip")
```

---

## Expected Performance

Approximate inference throughput for `segformer-b2` on 512×512 chips:

| Hardware | Precision | Speed |
|----------|-----------|-------|
| CPU (8-core) | FP32 | ~2 chips/s |
| RTX 3090 (ONNX) | FP16 | ~120 chips/s |
| RTX 3090 (TensorRT) | FP16 | ~200 chips/s |
| Jetson Orin (TRT) | FP16 | ~45 chips/s |
| Jetson Xavier (TRT) | FP16 | ~25 chips/s |
| Jetson Nano (TRT) | FP16 | ~8 chips/s |
