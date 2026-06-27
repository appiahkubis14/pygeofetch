"""CLIP and RemoteCLIP for geospatial zero-shot classification and retrieval (G2)."""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Union
logger = logging.getLogger(__name__)


class CLIPGeo:
    """CLIP/RemoteCLIP for zero-shot geospatial classification and image-text search (G2).

    Remote-sensing-finetuned CLIP understands domain terms like
    "flooded farmland", "urban expansion", "deforestation".

    Example::

        clip = CLIPGeo(model="remoteclip-b32")
        scores = clip.zero_shot(
            image_path="./sentinel2.tif",
            categories=["deforestation", "healthy forest", "agriculture", "urban"],
        )
        similar = clip.search(query="coastal flooding", image_dir="./images/")
    """

    HF_MODELS = {
        "remoteclip-b32": "BAAI/RemoteCLIP-ViT-B-32",
        "remoteclip-l14": "BAAI/RemoteCLIP-ViT-L-14",
        "openclip-b32":   "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
        "openclip-l14":   "openai/clip-vit-large-patch14",
    }

    def __init__(self, model: str = "openclip-b32", device: Optional[str] = None) -> None:
        self.model_name = model
        self.model_id = self.HF_MODELS.get(model, model)
        self.device = device or ("cpu")
        self._model = None; self._processor = None

    def _load(self):
        if self._model: return
        try:
            from transformers import CLIPModel, CLIPProcessor
            self._processor = CLIPProcessor.from_pretrained(self.model_id)
            self._model = CLIPModel.from_pretrained(self.model_id).to(self.device).eval()
        except ImportError:
            raise ImportError("transformers required: pip install transformers")

    def _load_image(self, image_path: str):
        import rasterio, numpy as np
        from PIL import Image
        with rasterio.open(image_path) as src:
            data = src.read(list(range(1, min(src.count, 4)+1))).astype(float)
        for b in range(data.shape[0]):
            p2, p98 = np.percentile(data[b], (2, 98))
            data[b] = np.clip((data[b]-p2)/(p98-p2+1e-8)*255, 0, 255)
        if data.shape[0] == 1: data = np.repeat(data, 3, axis=0)
        return Image.fromarray(data[:3].transpose(1, 2, 0).astype(np.uint8))

    def zero_shot(self, image_path: str, categories: List[str]) -> Dict[str, float]:
        """Zero-shot classify an image against text categories."""
        import torch
        self._load()
        img = self._load_image(image_path)
        inputs = self._processor(text=categories, images=img, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0].cpu().tolist()
        return {cat: round(p, 4) for cat, p in zip(categories, probs)}

    def embed_image(self, image_path: str) -> Any:
        """Get CLIP image embedding for semantic search."""
        import torch
        self._load()
        img = self._load_image(image_path)
        inputs = self._processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            feat = self._model.get_image_features(**inputs)
        import torch.nn.functional as F
        return F.normalize(feat, dim=-1).cpu().numpy().squeeze()

    def embed_text(self, text: str) -> Any:
        """Get CLIP text embedding."""
        import torch
        self._load()
        inputs = self._processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            feat = self._model.get_text_features(**inputs)
        import torch.nn.functional as F
        return F.normalize(feat, dim=-1).cpu().numpy().squeeze()

    def search(self, query: str, image_dir: str, top_k: int = 5) -> List[Dict]:
        """Search a directory of images by text query."""
        import pathlib, numpy as np
        image_paths = list(pathlib.Path(image_dir).rglob("*.tif")) + \
                      list(pathlib.Path(image_dir).rglob("*.png"))
        query_emb = self.embed_text(query)
        results = []
        for p in image_paths:
            try:
                img_emb = self.embed_image(str(p))
                score = float(np.dot(query_emb, img_emb))
                results.append({"path": str(p), "score": round(score, 4)})
            except Exception:
                continue
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
