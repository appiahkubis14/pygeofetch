"""Moondream VLM for satellite image captioning and VQA (G2)."""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Union
logger = logging.getLogger(__name__)


class MoondreamGeo:
    """Moondream2 — lightweight VLM for satellite image understanding (G2).

    Generates natural language descriptions of satellite images and
    answers questions about image content.

    Example::

        vl = MoondreamGeo()
        caption = vl.caption("./sentinel2.tif")
        answer  = vl.vqa("./sentinel2.tif", "How many buildings are visible?")
        results = vl.batch_caption("./data/", sliding_window=True)
    """

    MODEL_ID = "vikhyatk/moondream2"

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device or "cpu"
        self._model = None; self._tokenizer = None

    def _load(self):
        if self._model: return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID, trust_remote_code=True)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.MODEL_ID, trust_remote_code=True, torch_dtype=torch.float16
            ).to(self.device).eval()
        except ImportError:
            raise ImportError("transformers + torch required")

    def _load_pil(self, image_path: str):
        import rasterio, numpy as np
        from PIL import Image
        with rasterio.open(image_path) as src:
            data = src.read(list(range(1, min(src.count, 4)+1))).astype(float)
        for b in range(data.shape[0]):
            p2, p98 = np.percentile(data[b], (2, 98))
            data[b] = np.clip((data[b]-p2)/(p98-p2+1e-8)*255, 0, 255)
        if data.shape[0] == 1: data = np.repeat(data, 3, axis=0)
        return Image.fromarray(data[:3].transpose(1, 2, 0).astype(np.uint8))

    def caption(self, image_path: str) -> str:
        """Generate a natural language caption for a satellite image."""
        self._load()
        img = self._load_pil(image_path)
        enc = self._model.encode_image(img)
        return self._model.answer_question(enc, "Describe this satellite image in detail.", self._tokenizer)

    def vqa(self, image_path: str, question: str) -> str:
        """Answer a natural language question about a satellite image."""
        self._load()
        img = self._load_pil(image_path)
        enc = self._model.encode_image(img)
        return self._model.answer_question(enc, question, self._tokenizer)

    def batch_caption(self, image_dir: str, top_k: int = 10) -> List[Dict]:
        import pathlib
        paths = list(pathlib.Path(image_dir).rglob("*.tif"))[:top_k]
        return [{"path": str(p), "caption": self.caption(str(p))} for p in paths]
