"""SAM and SAM2 for zero-shot geospatial segmentation."""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Tuple
logger = logging.getLogger(__name__)


def build_sam(variant: str = "vit-large", device: str = "cpu", **kwargs) -> Any:
    """Load SAM (Segment Anything Model) for geospatial use.

    Args:
        variant: "vit-huge" | "vit-large" | "vit-base" | "sam2-hiera-l"

    Example::

        model = build_sam("vit-large")
        masks = model.generate_masks("scene.tif")
    """
    HF_IDS = {
        "vit-huge":     "facebook/sam-vit-huge",
        "vit-large":    "facebook/sam-vit-large",
        "vit-base":     "facebook/sam-vit-base",
        "sam2-hiera-l": "facebook/sam2-hiera-large",
        "sam2-hiera-b": "facebook/sam2-hiera-base-plus",
    }
    return GeoSAM(hf_id=HF_IDS.get(variant, HF_IDS["vit-large"]), device=device)


class GeoSAM:
    """Geospatial SAM wrapper — handles large GeoTIFF, CRS, pixel-to-geo coord conversion."""

    def __init__(self, hf_id: str, device: str = "cpu") -> None:
        self.hf_id = hf_id
        self.device = device
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model: return
        from transformers import SamModel, SamProcessor
        self._processor = SamProcessor.from_pretrained(self.hf_id)
        self._model = SamModel.from_pretrained(self.hf_id).to(self.device).eval()

    def generate_masks(self, image_path: str, output_path: Optional[str] = None,
                        points_per_side: int = 32, min_area_m2: float = 10.0) -> Dict:
        """Generate segmentation masks for a full GeoTIFF."""
        from pygeovision.labeling.sam_auto import SAMAutoLabeler
        labeler = SAMAutoLabeler(device=self.device)
        labeler._model = self._model if self._model else None
        return labeler.auto_label(image_path, output_path or "./output/sam_masks.tif",
                                   points_per_side=points_per_side, min_area_m2=min_area_m2)

    def predict_points(self, image: Any, input_points: List[List[float]],
                        input_labels: Optional[List[int]] = None) -> Any:
        """Predict masks from point prompts."""
        import torch
        self._load()
        from PIL import Image as PILImage
        import numpy as np

        if isinstance(image, str):
            import rasterio
            with rasterio.open(image) as src:
                data = src.read(list(range(1, min(src.count, 4) + 1))).astype(float)
            for b in range(data.shape[0]):
                p2, p98 = np.percentile(data[b], (2, 98))
                data[b] = np.clip((data[b] - p2) / (p98 - p2 + 1e-8) * 255, 0, 255)
            if data.shape[0] == 1: data = np.repeat(data, 3, axis=0)
            image = PILImage.fromarray(data[:3].transpose(1, 2, 0).astype(np.uint8))

        labels = input_labels or [1] * len(input_points)
        inputs = self._processor(
            images=image,
            input_points=[input_points],
            input_labels=[labels],
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        masks = self._processor.post_process_masks(
            outputs.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"]
        )
        return masks
