# Deployment

Deploy PyGeoVision models from development to production on edge hardware and cloud platforms.

---

## Export to ONNX

The first step for any deployment is exporting your trained model to ONNX format.

```python
from pygeovision.models import get_model
from pygeovision.edge.onnx_rt import ONNXRuntimeInference
import torch

# Load your trained model
model = get_model("segformer-b2", num_classes=7, in_channels=4)
model.load_state_dict(torch.load("./checkpoints/best.pth")["model_state_dict"])
model.eval()

# Export to ONNX
ONNXRuntimeInference.from_pytorch(
    model=model,
    output_path="./deploy/segformer_b2_lc7.onnx",
    input_shape=(1, 4, 512, 512),
    opset_version=17,
    simplify=True,
)
print("ONNX exported")
```

CLI:

```bash
pgv edge export-onnx segformer-b2 \
  --output ./deploy/model.onnx \
  --classes 7 \
  --in-channels 4 \
  --input-size 512
```

---

## Test ONNX Inference

```python
from pygeovision.edge.onnx_rt import ONNXRuntimeInference

eng = ONNXRuntimeInference("./deploy/model.onnx", device="cuda")

# Benchmark
report = eng.benchmark(n_runs=200)
print(f"Mean: {report['mean_ms']:.1f}ms | P95: {report['p95_ms']:.1f}ms | {report['fps']:.0f} fps")

# Infer a GeoTIFF
result = eng.infer_geotiff("scene.tif", "./output/onnx_pred.tif")
```

---

## REST Inference Server

```python
from pygeovision.serving import InferenceServer

server = InferenceServer(auth_keys={"myuser": "my-api-key-1234"})
server.register("seg_v1", "./deploy/model.onnx",
                 task="segmentation", num_classes=7)
server.serve(host="0.0.0.0", port=8080)
```

Test it:

```bash
curl -X POST http://localhost:8080/predict \
  -H "X-API-Key: my-api-key-1234" \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/scene.tif", "model_name": "seg_v1"}'
```

---

## Docker Container

```dockerfile
FROM python:3.12-slim
RUN apt-get update && apt-get install -y libgdal-dev && rm -rf /var/lib/apt/lists/*
RUN pip install "pygeovision[serve,geo,edge]"

COPY ./deploy/model.onnx /app/model.onnx

ENV API_USER=prod API_KEY=change-me-in-production

EXPOSE 8080
CMD python -c "
from pygeovision.serving import InferenceServer
import os
s = InferenceServer(auth_keys={os.environ['API_USER']: os.environ['API_KEY']})
s.register('seg_v1', '/app/model.onnx', task='segmentation', num_classes=7)
s.serve(host='0.0.0.0', port=8080)
"
```

```bash
docker build -t pygeovision-server .
docker run -p 8080:8080 -e API_USER=user -e API_KEY=secret pygeovision-server
```

---

## NVIDIA Jetson (Edge)

```python
from pygeovision.edge.jetson import JetsonDeployer

deployer = JetsonDeployer(device_type="orin")

result = deployer.convert(
    onnx_path="./deploy/model.onnx",
    engine_path="./deploy/model_fp16.trt",
    trt_path="./trt_cache/",
    precision="fp16",
    input_shape=(1, 4, 512, 512),
)
print(f"TensorRT engine: {result['engine_path']}")
print(f"Build time:      {result['build_time_s']:.0f}s")
```

---

## AWS SageMaker

```python
from pygeovision.cloud.deploy import AWSDeployer

deployer = AWSDeployer(region="us-east-1")

result = deployer.deploy(
    "./deploy/model.onnx",
    endpoint_name="pygeovision-seg-prod",
    instance_type="ml.g4dn.xlarge",
)

print(f"Endpoint: {result['endpoint_url']}")
```

---

## GCP Vertex AI

```python
from pygeovision.cloud.deploy import GCPDeployer

result = GCPDeployer(project_id="my-project", region="us-central1").deploy(
    "./deploy/model.onnx",
    endpoint_name="pygeovision-seg",
    machine_type="n1-standard-8",
    accelerator_type="NVIDIA_TESLA_T4",
)
print(f"Vertex endpoint: {result['endpoint_url']}")
```

---

## Monitoring in Production

```python
from pygeovision.monitoring.drift   import DriftDetector
from pygeovision.monitoring.tracker import ModelPerformanceTracker
from pygeovision.monitoring.alerts  import AlertManager

drift   = DriftDetector().fit(reference_images)
tracker = ModelPerformanceTracker(["val_iou"])
alerts  = AlertManager(channels={"slack": {"webhook_url": "..."}})

# Run daily check
def daily_monitor(new_images, today_metrics):
    drift_report = drift.check(new_images)
    tracker.log(epoch=today_epoch, metrics=today_metrics)
    alerts.check({**today_metrics,
                  "psi": drift_report["data_drift"]["psi_score"]})
```
