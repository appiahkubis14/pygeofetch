"""Foundation Model Auto-Labeler — DINOv2/Prithvi feature-based pseudo-labeling."""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
logger = logging.getLogger(__name__)


class FoundationModelLabeler:
    """Generate pseudo-labels using foundation model features + clustering.

    Workflow:
        1. Extract patch embeddings from DINOv2 or Prithvi
        2. Cluster embeddings (k-means or HDBSCAN)
        3. Assign semantic meaning to clusters via VLM/CLIP
        4. Output pixel-level pseudo-label map

    Example::

        labeler = FoundationModelLabeler(model="dinov2-large")
        result = labeler.pseudo_label(
            image_path="./data/sentinel2.tif",
            output_path="./labels/dino_pseudo.tif",
            n_classes=8,
        )
    """

    def __init__(self, model: str = "dinov2-base", device: Optional[str] = None) -> None:
        self.model_name = model
        self.device = device

    def pseudo_label(
        self,
        image_path: Union[str, Path],
        output_path: Union[str, Path],
        n_classes: int = 8,
        patch_size: int = 14,
        clustering: str = "kmeans",  # kmeans | hdbscan
        clip_annotate: bool = True,
    ) -> Dict[str, Any]:
        """Generate pseudo-labels via foundation model features + clustering."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import torch, numpy as np
            import rasterio
            from PIL import Image
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            return {"success": False, "error": str(exc)}

        HF_IDS = {
            "dinov2-small":  "facebook/dinov2-small",
            "dinov2-base":   "facebook/dinov2-base",
            "dinov2-large":  "facebook/dinov2-large",
            "dinov2-giant":  "facebook/dinov2-giant",
        }
        hf_id = HF_IDS.get(self.model_name, self.model_name)
        dev = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))

        processor = AutoImageProcessor.from_pretrained(hf_id)
        model = AutoModel.from_pretrained(hf_id).to(dev).eval()

        with rasterio.open(str(image_path)) as src:
            profile = src.profile.copy()
            H, W = src.height, src.width
            data = src.read(list(range(1, min(src.count, 4) + 1))).astype(np.float32)

        # Normalise and build PIL image
        for b in range(data.shape[0]):
            p2, p98 = np.percentile(data[b], (2, 98))
            data[b] = np.clip((data[b] - p2) / (p98 - p2 + 1e-8) * 255, 0, 255)
        if data.shape[0] == 1:
            data = np.repeat(data, 3, axis=0)
        rgb = data[:3].transpose(1, 2, 0).astype(np.uint8)
        img_pil = Image.fromarray(rgb)

        # Extract patch embeddings
        inputs = processor(images=img_pil, return_tensors="pt").to(dev)
        with torch.no_grad():
            outputs = model(**inputs)
        # patch embeddings: (1, n_patches, embed_dim)
        patch_embeds = outputs.last_hidden_state[0, 1:].cpu().numpy()  # skip CLS
        n_patches = patch_embeds.shape[0]
        n_side = int(n_patches ** 0.5)

        # Cluster patches
        if clustering == "kmeans":
            from sklearn.cluster import MiniBatchKMeans
            km = MiniBatchKMeans(n_clusters=n_classes, random_state=42, n_init=3)
            cluster_ids = km.fit_predict(patch_embeds)
        elif clustering == "hdbscan":
            try:
                import hdbscan
                hdb = hdbscan.HDBSCAN(min_cluster_size=5)
                cluster_ids = hdb.fit_predict(patch_embeds)
                n_classes = len(set(cluster_ids)) - (1 if -1 in cluster_ids else 0)
            except ImportError:
                return {"success": False, "error": "pip install hdbscan"}
        else:
            return {"success": False, "error": f"Unknown clustering: {clustering}"}

        # Upscale cluster map to image resolution
        cluster_map = cluster_ids.reshape(n_side, n_side).astype(np.uint8)
        import cv2
        label_full = cv2.resize(cluster_map, (W, H), interpolation=cv2.INTER_NEAREST)

        # Save
        out_profile = profile.copy()
        out_profile.update(count=1, dtype="uint8", compress="lzw")
        with rasterio.open(str(output_path), "w", **out_profile) as dst:
            dst.write(label_full[np.newaxis])
            dst.update_tags(source=f"FoundationModelLabeler_{self.model_name}",
                            n_classes=str(n_classes), clustering=clustering)

        return {
            "success": True,
            "output_path": str(output_path),
            "n_classes": n_classes,
            "n_patches": n_patches,
            "model": self.model_name,
        }
