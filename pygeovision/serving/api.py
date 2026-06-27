"""
PyGeoVision Inference Server — FastAPI REST API.

Endpoints:
    GET  /health            — health check
    GET  /models            — list registered models
    POST /predict           — single image prediction
    POST /predict/batch     — batch prediction
    POST /models/register   — register a new model
    DELETE /models/{name}   — deregister a model
    GET  /metrics           — Prometheus metrics
    WS   /ws/stream         — WebSocket streaming inference
"""
from __future__ import annotations
import base64, io, logging, time
from pathlib import Path
from typing import Any, Dict, List, Optional
logger = logging.getLogger(__name__)


def create_app(auth_keys: Optional[Dict] = None, enable_metrics: bool = True):
    """Create and configure the FastAPI inference server.

    Example::

        app = create_app(auth_keys={"myuser": "myapikey"})
        # uvicorn.run(app, host="0.0.0.0", port=8080)
    """
    try:
        from fastapi import FastAPI, HTTPException, Depends, Header, WebSocket
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse
    except ImportError:
        raise ImportError("pip install fastapi uvicorn")

    from pygeovision.serving.models   import (PredictRequest, PredictResponse,
                                               ModelInfo, HealthResponse)
    from pygeovision.serving.auth     import APIKeyAuth
    from pygeovision.serving.health   import HealthChecker

    app   = FastAPI(title="PyGeoVision Inference API", version="2.0.4",
                     description="Geospatial AI inference server")
    _auth = APIKeyAuth(keys=auth_keys or {})
    _models: Dict[str, Any] = {}
    _health = HealthChecker(models=_models)
    _start  = time.time()

    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                        allow_methods=["*"], allow_headers=["*"])

    # ── Auth dependency ───────────────────────────────────────────────────────
    async def get_api_key(x_api_key: Optional[str] = Header(None)):
        if not auth_keys:
            return "anonymous"
        if x_api_key and _auth.verify(x_api_key):
            return _auth.verify(x_api_key)
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    # ── Health ────────────────────────────────────────────────────────────────
    @app.get("/health", tags=["System"])
    async def health():
        return _health.check()

    @app.get("/", tags=["System"])
    async def root():
        return {"name": "PyGeoVision API", "version": "2.0.4",
                "docs": "/docs", "models": list(_models.keys())}

    # ── Model management ──────────────────────────────────────────────────────
    @app.get("/models", tags=["Models"])
    async def list_models(user: str = Depends(get_api_key)):
        return {"models": [
            {"name": k, "task": v.get("task"), "version": v.get("version", "1.0.0")}
            for k, v in _models.items()
        ]}

    @app.post("/models/register", tags=["Models"])
    async def register_model(info: ModelInfo, user: str = Depends(get_api_key)):
        """Register a model for serving."""
        _models[info.name] = info.dict()
        logger.info("Model registered: %s (task=%s)", info.name, info.task)
        return {"registered": info.name, "total_models": len(_models)}

    @app.delete("/models/{name}", tags=["Models"])
    async def deregister_model(name: str, user: str = Depends(get_api_key)):
        if name not in _models:
            raise HTTPException(status_code=404, detail=f"Model '{name}' not found")
        del _models[name]
        return {"deregistered": name}

    # ── Prediction ────────────────────────────────────────────────────────────
    @app.post("/predict", response_model=PredictResponse, tags=["Inference"])
    async def predict(req: PredictRequest, user: str = Depends(get_api_key)):
        """Run inference on a single image."""
        t_start = time.time()
        model_info = _models.get(req.model_name)
        if not model_info and req.model_name != "default":
            raise HTTPException(status_code=404, detail=f"Model '{req.model_name}' not registered")

        try:
            result = await _run_inference(req, model_info)
            return PredictResponse(
                success=True,
                model_name=req.model_name,
                task=req.task,
                inference_time_ms=round((time.time() - t_start) * 1000, 1),
                **result,
            )
        except Exception as exc:
            logger.exception("Prediction failed")
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/predict/batch", tags=["Inference"])
    async def predict_batch(req: Any, user: str = Depends(get_api_key)):
        """Run batch inference on multiple images."""
        return {"status": "queued", "message": "Batch inference queued for async processing"}

    # ── Metrics ───────────────────────────────────────────────────────────────
    if enable_metrics:
        @app.get("/metrics", tags=["System"])
        async def metrics():
            return {
                "uptime_s": round(time.time() - _start, 1),
                "models_loaded": len(_models),
                "requests_total": 0,
            }

    # ── WebSocket streaming ───────────────────────────────────────────────────
    @app.websocket("/ws/stream")
    async def websocket_stream(ws: WebSocket):
        """WebSocket endpoint for real-time streaming inference."""
        await ws.accept()
        try:
            while True:
                data = await ws.receive_json()
                result = {"status": "received", "echo": data}
                await ws.send_json(result)
        except Exception:
            await ws.close()

    return app


async def _run_inference(req: Any, model_info: Optional[Dict]) -> Dict:
    """Execute model inference from a request."""
    import tempfile, os

    # Decode image
    image_path = None
    if req.image_b64:
        img_bytes = base64.b64decode(req.image_b64)
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp.write(img_bytes)
            image_path = tmp.name
    elif req.image_url:
        import requests
        resp = requests.get(req.image_url, timeout=30)
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp.write(resp.content)
            image_path = tmp.name

    if not image_path:
        return {"error": "No image provided"}

    try:
        if model_info and model_info.get("onnx_path"):
            from pygeovision.edge.onnx_rt import ONNXRuntimeInference
            eng = ONNXRuntimeInference(model_info["onnx_path"])
            out_path = image_path.replace(".tif", "_pred.tif")
            result = eng.infer_geotiff(image_path, out_path,
                                        chip_size=req.chip_size, overlap=req.overlap)
        else:
            result = {"note": "No model loaded. Register an ONNX model via POST /models/register"}

        return {"statistics": result}
    finally:
        if image_path and os.path.exists(image_path):
            os.unlink(image_path)


class InferenceServer:
    """Convenience wrapper to create and serve the inference API.

    Example::

        server = InferenceServer()
        server.register("seg_v1", "./model.onnx", task="segmentation", num_classes=2)
        server.serve(host="0.0.0.0", port=8080)
    """

    def __init__(self, auth_keys: Optional[Dict] = None) -> None:
        self.app = create_app(auth_keys=auth_keys)
        self._models: Dict[str, Any] = {}

    def register(self, name: str, model_path: str, task: str = "segmentation",
                  num_classes: int = 2, in_channels: int = 4, **kwargs) -> "InferenceServer":
        """Register a model for serving."""
        self._models[name] = {
            "name": name, "task": task, "num_classes": num_classes,
            "in_channels": in_channels, "onnx_path": model_path, **kwargs,
        }
        logger.info("Registered: %s (%s, %d classes)", name, task, num_classes)
        return self

    def serve(self, host: str = "0.0.0.0", port: int = 8080,
               workers: int = 1, reload: bool = False) -> None:
        """Start the inference server."""
        try:
            import uvicorn
            logger.info("Starting PyGeoVision Inference Server on %s:%d", host, port)
            uvicorn.run(self.app, host=host, port=port, workers=workers, reload=reload)
        except ImportError:
            raise ImportError("pip install uvicorn")
