# Serving API

FastAPI-based inference server with JWT authentication, WebSocket streaming, Prometheus metrics, and batch endpoints.

---

## Quick Start

```python
from pygeovision.serving import InferenceServer

server = InferenceServer(auth_keys={"prod_user": "my-secret-key"})
server.register("seg_v1", "./model.onnx", task="segmentation", num_classes=7)
server.serve(host="0.0.0.0", port=8080)
```

Then call from any HTTP client:

```bash
curl -X POST http://localhost:8080/predict \
  -H "X-API-Key: my-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"image_url":"https://myhost.com/scene.tif","model_name":"seg_v1"}'
```

---

## `InferenceServer`

```python
from pygeovision.serving import InferenceServer

server = InferenceServer(
    auth_keys={"user1": "key1", "user2": "key2"},  # API key → username mapping
    enable_metrics=True,   # Expose Prometheus /metrics endpoint
)

# Register models
server.register(
    name="seg_v1",
    model_path="./model.onnx",
    task="segmentation",
    num_classes=7,
    in_channels=4,
    version="1.2.0",
    description="Urban land cover segmentation",
)

server.register("detect_v1", "./yolo.onnx", task="detection", num_classes=5)

# Start (blocks)
server.serve(host="0.0.0.0", port=8080, workers=2)

# Or get the FastAPI app for custom deployment
app = server.app
# uvicorn.run(app, host="0.0.0.0", port=8080)
```

---

## `create_app()`

For advanced configuration — returns the raw FastAPI application.

```python
from pygeovision.serving.api import create_app

app = create_app(
    auth_keys={"user": "key"},
    enable_metrics=True,
)

# Serve with uvicorn
import uvicorn
uvicorn.run(app, host="0.0.0.0", port=8080, reload=False, workers=4)
```

---

## REST Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/` | No | API info and version |
| `GET` | `/health` | No | Health status + GPU/RAM |
| `GET` | `/models` | Yes | List registered models |
| `POST` | `/models/register` | Yes | Register a new model |
| `DELETE` | `/models/{name}` | Yes | Deregister a model |
| `POST` | `/predict` | Yes | Single image inference |
| `POST` | `/predict/batch` | Yes | Batch inference (async) |
| `GET` | `/metrics` | No | Prometheus metrics |
| `WS` | `/ws/stream` | No | WebSocket streaming |
| `GET` | `/docs` | No | Swagger UI |
| `GET` | `/redoc` | No | ReDoc documentation |

---

## Request / Response Models

### `PredictRequest`

```json
{
  "image_b64": "base64-encoded-bytes...",
  "image_url": "https://myhost.com/scene.tif",
  "model_name": "seg_v1",
  "task": "segmentation",
  "confidence_threshold": 0.5,
  "chip_size": 512,
  "overlap": 64,
  "return_probabilities": false,
  "output_format": "geotiff"
}
```

### `PredictResponse`

```json
{
  "success": true,
  "model_name": "seg_v1",
  "task": "segmentation",
  "n_classes": 7,
  "output_url": "https://...",
  "statistics": {
    "class_distribution": {"0": 45.2, "1": 22.1, "2": 18.3}
  },
  "inference_time_ms": 312.5,
  "timestamp": "2024-11-12T14:32:00"
}
```

---

## Authentication

### API Key (stateless, recommended for services)

```python
from pygeovision.serving.auth import APIKeyAuth

auth = APIKeyAuth(keys={"alice": "key_abc", "bob": "key_xyz"})

# Add a new user
new_key = auth.generate_key("charlie")  # Returns "pgv_<32 random hex chars>"

# Verify
user = auth.verify(new_key)   # Returns "charlie" or None
```

Pass the key in requests via header: `X-API-Key: <key>`

### JWT (for user-facing applications)

```python
from pygeovision.serving.auth import JWTAuth

jwt = JWTAuth(secret="my-signing-secret", expiry_hours=24)

# Issue token at login
token = jwt.create_token({"user": "alice", "role": "analyst"})

# Verify on each request
payload = jwt.verify_token(token)   # Returns dict or None
```

Pass in requests via header: `Authorization: Bearer <token>`

---

## WebSocket Streaming

Connect to `/ws/stream` for real-time inference on video or streaming data:

```python
import asyncio, websockets, json

async def stream_inference():
    async with websockets.connect("ws://localhost:8080/ws/stream") as ws:
        # Send request
        await ws.send(json.dumps({
            "image_b64": "...",
            "model_name": "seg_v1"
        }))
        # Receive prediction
        result = json.loads(await ws.recv())
        print(result)

asyncio.run(stream_inference())
```

---

## Health Endpoint

```bash
curl http://localhost:8080/health
```

```json
{
  "status": "healthy",
  "version": "2.0.4",
  "uptime_s": 3600.1,
  "models_loaded": 2,
  "gpu": {
    "available": true,
    "name": "NVIDIA A100-SXM4-80GB",
    "total_mb": 81920,
    "allocated_mb": 4200
  },
  "memory": {
    "total_gb": 256.0,
    "available_gb": 201.3,
    "percent_used": 21.4
  }
}
```

---

## Docker Deployment

```dockerfile
FROM python:3.12-slim

RUN pip install "pygeovision[serve,geo]"

COPY model.onnx /app/model.onnx
COPY serve.py   /app/serve.py

EXPOSE 8080
CMD ["python", "/app/serve.py"]
```

```python
# serve.py
from pygeovision.serving import InferenceServer
import os

server = InferenceServer(auth_keys={os.environ["API_USER"]: os.environ["API_KEY"]})
server.register("seg_v1", "/app/model.onnx", task="segmentation", num_classes=7)
server.serve(host="0.0.0.0", port=8080, workers=2)
```
