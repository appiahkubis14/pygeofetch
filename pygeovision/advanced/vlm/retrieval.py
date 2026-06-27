"""Geo image retrieval using CLIP embeddings."""
from __future__ import annotations
import logging, json
from pathlib import Path
from typing import Any, Dict, List, Optional
logger = logging.getLogger(__name__)


class GeoImageRetrieval:
    """Image-to-image and text-to-image retrieval for geospatial archives."""

    def __init__(self, model: str = "openclip-b32") -> None:
        from pygeovision.advanced.vlm.clip_geo import CLIPGeo
        self._clip = CLIPGeo(model=model)
        self._index: Dict[str, Any] = {}

    def build_index(self, image_dir: str, save_path: Optional[str] = None) -> int:
        import pathlib, numpy as np
        paths = list(pathlib.Path(image_dir).rglob("*.tif"))
        for p in paths:
            try:
                emb = self._clip.embed_image(str(p))
                self._index[str(p)] = emb.tolist()
            except Exception as exc:
                logger.debug("Skipping %s: %s", p, exc)
        if save_path:
            with open(save_path, "w") as f:
                json.dump(self._index, f)
        logger.info("Index built: %d images", len(self._index))
        return len(self._index)

    def load_index(self, path: str) -> None:
        with open(path) as f:
            self._index = json.load(f)

    def search_by_text(self, query: str, top_k: int = 10) -> List[Dict]:
        import numpy as np
        query_emb = self._clip.embed_text(query)
        scores = [(p, float(np.dot(query_emb, np.array(emb))))
                  for p, emb in self._index.items()]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [{"path": p, "score": round(s, 4)} for p, s in scores[:top_k]]

    def search_by_image(self, image_path: str, top_k: int = 10) -> List[Dict]:
        import numpy as np
        query_emb = self._clip.embed_image(image_path)
        scores = [(p, float(np.dot(query_emb, np.array(emb))))
                  for p, emb in self._index.items() if p != image_path]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [{"path": p, "score": round(s, 4)} for p, s in scores[:top_k]]
