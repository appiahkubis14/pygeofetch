"""YOLOv8/v9 for geospatial object detection — independent of GeoAI."""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
logger = logging.getLogger(__name__)


def build_yolo(variant: str = "yolov8-m", num_classes: int = 5, **kwargs) -> Any:
    """Build YOLOv8/v9 for satellite object detection.

    Args:
        variant: "yolov8-n|s|m|l|x" or "yolov9-c|e"
        num_classes: Object classes

    Example::

        model = build_yolo("yolov8-m", num_classes=3)  # ships, planes, vehicles
    """
    try:
        from ultralytics import YOLO
        size_map = {"yolov8-n": "yolov8n", "yolov8-s": "yolov8s", "yolov8-m": "yolov8m",
                    "yolov8-l": "yolov8l", "yolov8-x": "yolov8x",
                    "yolov9-c": "yolov9c", "yolov9-e": "yolov9e"}
        model_str = size_map.get(variant, "yolov8m") + ".pt"
        model = YOLO(model_str)
        return model
    except ImportError:
        raise ImportError("pip install ultralytics")


class GeoYOLO:
    """YOLOv8 wrapper with geospatial pre/post-processing."""

    def __init__(self, variant: str = "yolov8-m", num_classes: int = 5,
                 class_names: Optional[List[str]] = None) -> None:
        self.variant = variant
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self._model = None

    def _load(self) -> None:
        if self._model: return
        self._model = build_yolo(self.variant, self.num_classes)

    def detect(self, image_path: str, conf: float = 0.25, iou: float = 0.45,
                output_path: Optional[str] = None) -> Dict[str, Any]:
        """Detect objects in a GeoTIFF and return geo-referenced results.

        Returns:
            Dict with detections (bbox in pixel + geo coords), class labels, confidence scores
        """
        import numpy as np
        self._load()

        try:
            import rasterio
            with rasterio.open(image_path) as src:
                transform = src.transform
                crs = src.crs
                data = src.read(list(range(1, min(src.count, 4) + 1))).astype(float)
                for b in range(data.shape[0]):
                    p2, p98 = np.percentile(data[b], (2, 98))
                    data[b] = np.clip((data[b] - p2) / (p98 - p2 + 1e-8) * 255, 0, 255)
                if data.shape[0] == 1: data = np.repeat(data, 3, axis=0)
                rgb = data[:3].transpose(1, 2, 0).astype(np.uint8)
        except Exception as exc:
            return {"error": str(exc)}

        results = self._model(rgb, conf=conf, iou=iou)
        detections = []
        for r in results:
            for box, cls, conf_score in zip(
                r.boxes.xyxy.cpu().numpy(),
                r.boxes.cls.cpu().numpy().astype(int),
                r.boxes.conf.cpu().numpy(),
            ):
                x1, y1, x2, y2 = box
                # Convert pixel to geo coordinates
                geo_x1, geo_y1 = rasterio.transform.xy(transform, y1, x1)
                geo_x2, geo_y2 = rasterio.transform.xy(transform, y2, x2)
                detections.append({
                    "class": self.class_names[min(cls, len(self.class_names) - 1)],
                    "class_id": int(cls),
                    "confidence": round(float(conf_score), 4),
                    "bbox_px": [int(x1), int(y1), int(x2), int(y2)],
                    "bbox_geo": [float(geo_x1), float(geo_y1), float(geo_x2), float(geo_y2)],
                })

        return {
            "n_detections": len(detections),
            "detections": detections,
            "model": self.variant,
            "image_path": image_path,
        }

    def train(self, data_yaml: str, epochs: int = 100, imgsz: int = 640,
               batch: int = 16, device: str = "0", **kwargs) -> Any:
        """Fine-tune YOLOv8 on a geospatial dataset (YOLO format)."""
        self._load()
        return self._model.train(data=data_yaml, epochs=epochs, imgsz=imgsz,
                                   batch=batch, device=device, **kwargs)
