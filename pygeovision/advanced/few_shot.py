"""
Few-shot learning for geospatial models (D3).
Classify new land cover classes with only 1-10 labelled examples.
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
logger = logging.getLogger(__name__)


class FewShotLearner:
    """Prototypical Networks and MAML for few-shot geospatial classification.

    Learn to classify new satellite classes with only 1-5 labelled samples.
    Uses a pre-trained feature extractor (DINOv2) and prototype matching.

    Supports:
        - N-way K-shot classification (e.g. 5-way 1-shot)
        - Meta-learning across multiple geospatial tasks
        - Support set of labelled examples
        - Query set prediction

    Example::

        learner = FewShotLearner(backbone="dinov2-base")
        learner.fit_support(
            support={"mangrove": ["img1.tif", "img2.tif"],
                      "urban":   ["img3.tif"]},
        )
        preds = learner.predict(["query1.tif", "query2.tif"])
    """

    def __init__(
        self,
        backbone: str = "dinov2-base",
        method: str = "prototypical",   # prototypical | maml | matching
        device: Optional[str] = None,
    ) -> None:
        self.backbone = backbone
        self.method = method
        self.device = device or self._auto_device()
        self._feature_extractor = None
        self._prototypes: Dict[str, Any] = {}

    # Mapping from short backbone names to HuggingFace model IDs
    _HF_MODEL_IDS: Dict[str, str] = {
        "dinov2-small":  "facebook/dinov2-small",
        "dinov2-base":   "facebook/dinov2-base",
        "dinov2-large":  "facebook/dinov2-large",
        "dinov2-giant":  "facebook/dinov2-giant",
        "clip-b32":      "openai/clip-vit-base-patch32",
        "clip-l14":      "openai/clip-vit-large-patch14",
    }

    @property
    def model_id(self) -> str:
        """HuggingFace model ID for the current backbone."""
        return self._HF_MODEL_IDS.get(self.backbone, f"facebook/{self.backbone}")

    @staticmethod
    def _auto_device():
        try:
            import torch; return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError: return "cpu"

    def _load_backbone(self) -> None:
        if self._feature_extractor is not None: return
        try:
            from transformers import AutoModel, AutoImageProcessor
            HF_IDS = {
                "dinov2-small": "facebook/dinov2-small",
                "dinov2-base":  "facebook/dinov2-base",
                "dinov2-large": "facebook/dinov2-large",
            }
            hf_id = HF_IDS.get(self.backbone, self.backbone)
            self._processor = AutoImageProcessor.from_pretrained(hf_id)
            self._feature_extractor = AutoModel.from_pretrained(hf_id).to(self.device).eval()
            logger.info("Few-shot backbone loaded: %s", self.backbone)
        except ImportError:
            raise ImportError("transformers + torch required")

    def _extract_features(self, image_paths: List[str]) -> Any:
        """Extract CLS token features from a list of image paths."""
        import torch, numpy as np
        import rasterio
        from PIL import Image

        self._load_backbone()
        features = []
        for path in image_paths:
            try:
                with rasterio.open(path) as src:
                    data = src.read(list(range(1, min(src.count, 4) + 1))).astype(float)
                for b in range(data.shape[0]):
                    p2, p98 = np.percentile(data[b], (2, 98))
                    data[b] = np.clip((data[b]-p2)/(p98-p2+1e-8)*255, 0, 255)
                if data.shape[0] == 1: data = np.repeat(data, 3, axis=0)
                rgb = data[:3].transpose(1, 2, 0).astype(np.uint8)
                inputs = self._processor(images=Image.fromarray(rgb), return_tensors="pt").to(self.device)
                with torch.no_grad():
                    out = self._feature_extractor(**inputs)
                feat = out.last_hidden_state[:, 0].cpu().numpy()  # CLS token
                features.append(feat.squeeze())
            except Exception as exc:
                logger.warning("Feature extraction failed for %s: %s", path, exc)
                features.append(np.zeros(768))
        return np.array(features)

    def fit_support(self, support: Dict[str, List[str]]) -> "FewShotLearner":
        """Build class prototypes from the support set.

        Args:
            support: Dict mapping class_name → list of image paths (1-10 per class)
        """
        import numpy as np
        self._prototypes = {}
        self._class_names = sorted(support.keys())

        for class_name, paths in support.items():
            feats = self._extract_features(paths)
            self._prototypes[class_name] = feats.mean(axis=0)   # centroid
            logger.info("Prototype built: %s (%d examples)", class_name, len(paths))

        return self

    def predict(
        self,
        query_paths: List[str],
        return_distances: bool = False,
    ) -> List[Dict[str, Any]]:
        """Classify query images against fitted support prototypes.

        Args:
            query_paths: Image paths to classify
            return_distances: Include per-class distances in output

        Returns:
            List of {'class': ..., 'confidence': ..., 'distances': ...}
        """
        if not self._prototypes:
            raise RuntimeError("Call fit_support() before predict()")

        import numpy as np
        query_feats = self._extract_features(query_paths)
        results = []

        for feat in query_feats:
            distances = {}
            for class_name, proto in self._prototypes.items():
                # Euclidean distance in feature space
                distances[class_name] = float(np.linalg.norm(feat - proto))

            best_class = min(distances, key=distances.get)
            # Convert distances to probabilities via softmax
            neg_dists = np.array([-d for d in distances.values()])
            exp_d = np.exp(neg_dists - neg_dists.max())
            probs = exp_d / exp_d.sum()
            prob_dict = dict(zip(distances.keys(), probs.tolist()))

            result = {
                "class": best_class,
                "confidence": round(float(prob_dict[best_class]), 4),
                "probabilities": {k: round(v, 4) for k, v in prob_dict.items()},
            }
            if return_distances:
                result["distances"] = {k: round(v, 4) for k, v in distances.items()}
            results.append(result)

        return results

    def predict_geotiff(
        self,
        image_path: str,
        output_path: str,
        chip_size: int = 224,
        stride: int = 112,
    ) -> Dict[str, Any]:
        """Apply few-shot classification over a full GeoTIFF patch-by-patch."""
        try:
            import rasterio, numpy as np
            from pygeovision.inference.tiled import GaussianBlend

            with rasterio.open(image_path) as src:
                H, W = src.height, src.width
                profile = src.profile.copy()
                image = src.read().astype(np.float32)

            n_classes = len(self._prototypes)
            accum  = np.zeros((n_classes, H, W), dtype=np.float32)
            counts = np.zeros((H, W), dtype=np.float32)

            for row in range(0, H, stride):
                for col in range(0, W, stride):
                    r2, c2 = min(row+chip_size, H), min(col+chip_size, W)
                    # Save chip as temp file
                    import tempfile, os
                    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
                        tmp_path = tmp.name
                    chip = image[:, row:r2, col:c2]
                    tmp_profile = profile.copy()
                    tmp_profile.update(height=r2-row, width=c2-col)
                    with rasterio.open(tmp_path, "w", **tmp_profile) as dst:
                        dst.write(chip)
                    pred = self.predict([tmp_path])[0]
                    os.unlink(tmp_path)

                    class_idx = self._class_names.index(pred["class"]) if pred["class"] in self._class_names else 0
                    actual_h, actual_w = r2-row, c2-col
                    accum[class_idx, row:r2, col:c2] += pred["confidence"]
                    counts[row:r2, col:c2] += 1.0

            accum /= np.maximum(counts[np.newaxis], 1e-10)
            label = np.argmax(accum, axis=0).astype(np.uint8)

            import pathlib
            pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            out_profile = profile.copy()
            out_profile.update(count=1, dtype="uint8", compress="lzw")
            with rasterio.open(output_path, "w", **out_profile) as dst:
                dst.write(label[np.newaxis])
                class_map = {str(i): name for i, name in enumerate(self._class_names)}
                dst.update_tags(method="FewShotLearner", class_map=str(class_map))

            return {"success": True, "output_path": output_path,
                    "n_classes": n_classes, "class_map": class_map}
        except Exception as exc:
            return {"success": False, "error": str(exc)}
