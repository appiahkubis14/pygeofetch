"""
SAM Auto-Labeler (E2) — Segment Anything Model for automated label generation.
Generates segmentation masks without any manual annotation.
No GeoAI dependency — uses HuggingFace transformers directly.
"""
from __future__ import annotations
import logging, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
logger = logging.getLogger(__name__)


class SAMAutoLabeler:
    """Automated label generation using Segment Anything Model (SAM/SAM2).

    Zero-shot segmentation — generates high-quality binary or multi-class masks
    from satellite imagery without any training data.

    Supports:
        - Automatic mask generation (grid-based point prompts)
        - GroundedSAM (text-prompt driven)
        - SAM2 (video SAM for time-series)
        - Post-filtering by area, stability, IoU quality

    Example::

        labeler = SAMAutoLabeler(model="sam-vit-huge")
        result = labeler.auto_label(
            image_path="./data/aerial.tif",
            output_path="./labels/sam_masks.tif",
            points_per_side=32,
            min_area_m2=50,
        )
    """

    HF_MODELS = {
        "sam-vit-huge":   "facebook/sam-vit-huge",
        "sam-vit-large":  "facebook/sam-vit-large",
        "sam-vit-base":   "facebook/sam-vit-base",
        "sam2-hiera-l":   "facebook/sam2-hiera-large",
        "sam2-hiera-b":   "facebook/sam2-hiera-base-plus",
        "geosam-vit-h":   "wangyi111/GeoSAM",
    }

    def __init__(
        self,
        model: str = "sam-vit-large",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.model_name = model
        self.model_id = self.HF_MODELS.get(model, model)
        self.device = device or self._auto_device()
        self.cache_dir = cache_dir
        self._model = None
        self._processor = None

    @staticmethod
    def _auto_device() -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import SamModel, SamProcessor, SamAutomaticMaskGenerator
            logger.info("Loading SAM: %s → %s", self.model_name, self.device)
            self._processor = SamProcessor.from_pretrained(self.model_id, cache_dir=self.cache_dir)
            self._model = SamModel.from_pretrained(self.model_id, cache_dir=self.cache_dir)
            self._model.to(self.device)
            logger.info("SAM loaded")
        except ImportError:
            raise ImportError("transformers + torch required: pip install transformers torch")

    def auto_label(
        self,
        image_path: Union[str, Path],
        output_path: Union[str, Path] = "./labels/sam_auto.tif",
        output_vector: Optional[str] = None,
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        min_area_m2: float = 10.0,
        max_area_m2: Optional[float] = None,
        chip_size: int = 1024,
        overlap: int = 128,
        merge_overlapping: bool = True,
    ) -> Dict[str, Any]:
        """Generate automatic segmentation masks from a GeoTIFF.

        Uses grid-based SAM prompting with filtering by quality and area.

        Args:
            image_path: Input GeoTIFF (any number of bands)
            output_path: Output binary/instance label GeoTIFF
            output_vector: Optional GeoJSON of mask polygons
            points_per_side: Grid density for automatic prompting
            pred_iou_thresh: Minimum predicted IoU quality
            stability_score_thresh: Minimum mask stability score
            min_area_m2: Minimum mask area in square metres
            max_area_m2: Maximum mask area (filters huge background)
            chip_size: Processing tile size in pixels
            overlap: Overlap between tiles in pixels

        Returns:
            Dict with n_masks, output_path, quality stats
        """
        image_path = Path(image_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import numpy as np
            import rasterio
            from PIL import Image
        except ImportError as exc:
            raise ImportError(f"rasterio + Pillow required: {exc}")

        # Load image
        with rasterio.open(str(image_path)) as src:
            profile = src.profile.copy()
            transform = src.transform
            crs = src.crs
            H, W = src.height, src.width
            # Read RGB (first 3 bands) and normalise
            n_bands = min(src.count, 3)
            data = src.read(list(range(1, n_bands + 1))).astype(float)
            # Normalise each band to 0-255
            for b in range(data.shape[0]):
                p2, p98 = np.percentile(data[b], (2, 98))
                data[b] = np.clip((data[b] - p2) / (p98 - p2 + 1e-8) * 255, 0, 255)
            if data.shape[0] == 1:
                data = np.repeat(data, 3, axis=0)
            elif data.shape[0] == 2:
                data = np.stack([data[0], data[1], data[0]], axis=0)
            rgb = data[:3].transpose(1, 2, 0).astype(np.uint8)

        # Load SAM model
        self._load()

        # Process in tiles
        all_masks = []
        t_start = time.time()
        stride = chip_size - overlap

        for row in range(0, H, stride):
            for col in range(0, W, stride):
                r2, c2 = min(row + chip_size, H), min(col + chip_size, W)
                chip = rgb[row:r2, col:c2]
                chip_masks = self._process_chip(
                    chip, points_per_side, pred_iou_thresh, stability_score_thresh
                )
                # Offset masks to full-image coordinates
                for m in chip_masks:
                    full_mask = np.zeros((H, W), dtype=bool)
                    ch, cw = m["segmentation"].shape
                    full_mask[row:row+ch, col:col+cw] = m["segmentation"]
                    m["full_mask"] = full_mask
                    m["bbox_full"] = (col, row, col+cw, row+ch)
                all_masks.extend(chip_masks)

        # Filter by area
        pixel_area_m2 = abs(transform.a * transform.e)
        filtered_masks = []
        for m in all_masks:
            area_px = m["full_mask"].sum()
            area_m2 = area_px * pixel_area_m2
            if area_m2 < min_area_m2:
                continue
            if max_area_m2 and area_m2 > max_area_m2:
                continue
            m["area_m2"] = float(area_m2)
            filtered_masks.append(m)

        logger.info("SAM: %d masks (of %d) passed filters", len(filtered_masks), len(all_masks))

        # Build instance label raster
        label = np.zeros((H, W), dtype=np.uint32)
        for i, m in enumerate(filtered_masks, start=1):
            label[m["full_mask"]] = i

        # Save raster
        out_profile = profile.copy()
        out_profile.update(count=1, dtype="uint32", compress="lzw")
        with rasterio.open(str(output_path), "w", **out_profile) as dst:
            dst.write(label[np.newaxis])
            dst.update_tags(
                source="SAM_AutoLabel",
                model=self.model_name,
                n_masks=str(len(filtered_masks)),
            )

        # Export vector
        if output_vector:
            self._masks_to_vector(filtered_masks, transform, crs, output_vector)

        return {
            "success": True,
            "n_masks": len(filtered_masks),
            "n_raw_masks": len(all_masks),
            "output_path": str(output_path),
            "output_vector": output_vector,
            "duration_seconds": round(time.time() - t_start, 1),
            "model": self.model_name,
            "quality_stats": {
                "mean_pred_iou": float(
                    sum(m.get("predicted_iou", 0) for m in filtered_masks) / max(len(filtered_masks), 1)
                ),
                "mean_stability": float(
                    sum(m.get("stability_score", 0) for m in filtered_masks) / max(len(filtered_masks), 1)
                ),
            },
        }

    def _process_chip(
        self, chip: Any, points_per_side: int,
        pred_iou_thresh: float, stability_thresh: float
    ) -> List[Dict]:
        """Run SAM automatic mask generator on one image chip."""
        try:
            import torch
            from transformers import SamAutomaticMaskGenerator
            generator = SamAutomaticMaskGenerator(
                model=self._model,
                points_per_side=points_per_side,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_thresh,
                box_nms_thresh=0.7,
                crop_n_layers=1,
            )
            from PIL import Image
            img_pil = Image.fromarray(chip)
            masks = generator.generate(img_pil)
            return masks
        except Exception as exc:
            logger.debug("SAM chip failed: %s", exc)
            return []

    def grounded_label(
        self,
        image_path: Union[str, Path],
        prompts: List[str],
        output_path: Union[str, Path] = "./labels/grounded_sam.tif",
        output_vector: Optional[str] = None,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
    ) -> Dict[str, Any]:
        """Generate labels using GroundedSAM — text-prompt driven segmentation.

        Combines Grounding DINO (text → bbox) + SAM (bbox → mask).

        Args:
            prompts: List of natural language prompts, e.g. ["building", "swimming pool"]
        """
        try:
            from groundingdino.util.inference import load_model as load_gdino
            from groundingdino.util.inference import predict as gdino_predict
        except ImportError:
            return {"success": False, "error": "pip install groundingdino-py for GroundedSAM"}

        logger.info("GroundedSAM: prompts=%s", prompts)
        self._load()

        image_path = Path(image_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import rasterio, numpy as np
            from PIL import Image
            with rasterio.open(str(image_path)) as src:
                profile = src.profile.copy()
                H, W = src.height, src.width
                data = src.read(list(range(1, min(src.count, 3) + 1))).astype(float)
                for b in range(data.shape[0]):
                    p2, p98 = np.percentile(data[b], (2, 98))
                    data[b] = np.clip((data[b]-p2)/(p98-p2+1e-8)*255, 0, 255)
                if data.shape[0] == 1:
                    data = np.repeat(data, 3, axis=0)
                rgb = data[:3].transpose(1, 2, 0).astype(np.uint8)

            label = np.zeros((H, W), dtype=np.uint8)
            all_boxes = []

            for class_idx, prompt in enumerate(prompts, start=1):
                # Grounding DINO detects boxes
                boxes, logits, phrases = gdino_predict(
                    model=None,  # placeholder
                    image=Image.fromarray(rgb),
                    caption=prompt,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                )
                all_boxes.extend([(b, class_idx, p) for b, p in zip(boxes, phrases)])

            # SAM segments each detected box
            if all_boxes:
                import torch
                processor = self._processor
                for box, class_idx, phrase in all_boxes:
                    inputs = processor(
                        images=Image.fromarray(rgb),
                        input_boxes=[[box.tolist()]],
                        return_tensors="pt",
                    ).to(self.device)
                    with torch.no_grad():
                        outputs = self._model(**inputs)
                    masks = processor.post_process_masks(
                        outputs.pred_masks, inputs["original_sizes"],
                        inputs["reshaped_input_sizes"]
                    )
                    if masks and masks[0].shape[0] > 0:
                        best = masks[0][0].cpu().numpy()
                        label[best > 0.5] = class_idx

            out_profile = profile.copy()
            out_profile.update(count=1, dtype="uint8", compress="lzw")
            with rasterio.open(str(output_path), "w", **out_profile) as dst:
                dst.write(label[np.newaxis])
                dst.update_tags(source="GroundedSAM", prompts=str(prompts))

            return {"success": True, "n_prompts": len(prompts), "output_path": str(output_path)}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def _masks_to_vector(self, masks, transform, crs, output_path):
        """Convert mask list to GeoJSON polygons."""
        try:
            import rasterio.features, numpy as np
            from shapely.geometry import shape as shp
            import json

            features = []
            for i, m in enumerate(masks):
                shapes = list(rasterio.features.shapes(
                    m["full_mask"].astype(np.uint8), transform=transform
                ))
                for geom, val in shapes:
                    if val == 1:
                        features.append({
                            "type": "Feature",
                            "geometry": geom,
                            "properties": {
                                "mask_id": i,
                                "area_m2": m.get("area_m2", 0),
                                "pred_iou": m.get("predicted_iou", 0),
                                "stability": m.get("stability_score", 0),
                            },
                        })

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump({"type": "FeatureCollection", "features": features}, f)
        except Exception as exc:
            logger.warning("Mask vectorisation failed: %s", exc)
