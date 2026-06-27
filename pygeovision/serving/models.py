"""Pydantic request/response models for the inference API."""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

try:
    from pydantic import BaseModel, Field
except ImportError:
    # Fallback dataclass
    from dataclasses import dataclass, field as dc_field
    class BaseModel:
        pass

    def Field(default=None, **kw):
        return default


class PredictRequest(BaseModel):
    """Request model for single image prediction."""
    image_b64: Optional[str] = Field(None, description="Base64-encoded image bytes")
    image_url: Optional[str] = Field(None, description="Accessible image URL")
    model_name: str = Field("default", description="Registered model name")
    task: str = Field("segmentation", description="Inference task")
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0)
    chip_size: int = Field(512, ge=32, le=2048)
    overlap: int = Field(64, ge=0, le=512)
    return_probabilities: bool = Field(False)
    output_format: str = Field("geotiff", description="geotiff|png|json")
    extra: Dict[str, Any] = Field(default_factory=dict)


class BatchPredictRequest(BaseModel):
    """Request model for batch prediction."""
    image_urls: List[str] = Field(..., min_length=1, max_length=100)
    model_name: str = "default"
    task: str = "segmentation"
    confidence_threshold: float = 0.5
    async_mode: bool = Field(True, description="Process asynchronously")


class PredictResponse(BaseModel):
    """Response model for prediction results."""
    success: bool
    model_name: str
    task: str
    n_classes: Optional[int] = None
    output_url: Optional[str] = None
    output_b64: Optional[str] = None
    statistics: Dict[str, Any] = Field(default_factory=dict)
    inference_time_ms: float = 0.0
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    error: Optional[str] = None


class ModelInfo(BaseModel):
    """Model registration information."""
    name: str
    task: str
    num_classes: int
    in_channels: int = 4
    description: str = ""
    version: str = "1.0.0"
    onnx_path: Optional[str] = None
    pytorch_path: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: int
    gpu_available: bool
    memory_mb: float
    uptime_s: float
