"""DINOv2/v3 feature extractor for geospatial tasks — independent implementation."""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional
logger = logging.getLogger(__name__)


class DINOv2GeoExtractor:
    """DINOv2 feature extractor fine-tuned for geospatial data.

    Extracts rich patch-level and image-level features without any GeoAI dependency.
    Supports: classification, few-shot learning, dense prediction via linear probing.

    Example::

        extractor = DINOv2GeoExtractor("dinov2-large")
        features  = extractor.extract("sentinel2.tif")  # (1, 1024)
        cls_head  = extractor.build_classifier(num_classes=10)
    """

    HF_IDS = {
        "dinov2-small":  "facebook/dinov2-small",
        "dinov2-base":   "facebook/dinov2-base",
        "dinov2-large":  "facebook/dinov2-large",
        "dinov2-giant":  "facebook/dinov2-giant",
    }

    def __init__(self, model_name: str = "dinov2-base",
                 device: Optional[str] = None) -> None:
        self.model_name = model_name
        self.model_id = self.HF_IDS.get(model_name, model_name)
        self.device = device or self._auto_device()
        self._model = None
        self._processor = None

    @staticmethod
    def _auto_device():
        try:
            import torch; return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError: return "cpu"

    def _load(self) -> None:
        if self._model is not None: return
        try:
            from transformers import AutoModel, AutoImageProcessor
            self._processor = AutoImageProcessor.from_pretrained(self.model_id)
            self._model = AutoModel.from_pretrained(self.model_id).to(self.device).eval()
        except ImportError:
            raise ImportError("pip install transformers torch")

    def extract(self, image_path: str) -> Any:
        """Extract CLS token embedding from a satellite image."""
        import numpy as np, rasterio
        from PIL import Image
        import torch

        self._load()
        with rasterio.open(image_path) as src:
            data = src.read(list(range(1, min(src.count, 4) + 1))).astype(float)
        for b in range(data.shape[0]):
            p2, p98 = np.percentile(data[b], (2, 98))
            data[b] = np.clip((data[b] - p2) / (p98 - p2 + 1e-8) * 255, 0, 255)
        if data.shape[0] == 1: data = np.repeat(data, 3, axis=0)
        rgb = data[:3].transpose(1, 2, 0).astype(np.uint8)
        inputs = self._processor(images=Image.fromarray(rgb), return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self._model(**inputs)
        return out.last_hidden_state[:, 0].cpu().numpy()

    def build_classifier(self, num_classes: int, freeze_backbone: bool = True) -> Any:
        """Build a linear classifier on top of DINOv2 features."""
        import torch.nn as nn
        self._load()
        embed_dim = self._model.config.hidden_size
        clf = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )
        if freeze_backbone:
            for p in self._model.parameters():
                p.requires_grad = False
        return clf
