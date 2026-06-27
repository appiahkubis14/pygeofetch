"""DINOv3 proxy for the GeoAI Engine — exposes all DINOv3 capabilities."""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Union
logger = logging.getLogger(__name__)


class DINOv3Proxy:
    """DINOv3 subsystem for PyGeoVision GeoAI Engine.

    Access via: client.geoai.dinov3 (lazy-loaded, no import cost)

    Example::

        pgv.geoai.dinov3.load("dinov3_vitl16_sat")
        features = pgv.geoai.dinov3.extract_features("sentinel2.tif")
        mask     = pgv.geoai.dinov3.zero_shot("image.tif", "buildings")
        height   = pgv.geoai.dinov3.canopy_height("forest.tif")
    """

    def __init__(self) -> None:
        self._backbone = None
        self._chm      = None
        self._txt      = None
        self._model_name: Optional[str] = None

    def load(self, model_name: str = "dinov3_vitl16_sat",
              method: str = "hf", device: Optional[str] = None) -> "DINOv3Proxy":
        """Load a DINOv3 model. Returns self for chaining.

        Args:
            model_name: Any of the 12 DINOv3 variants
            method: "hf" (HuggingFace) | "hub" (PyTorch Hub) | "local"
            device: "cuda" | "cpu" | "mps"
        """
        from pygeovision.models.foundation.dinov3 import DINOv3Backbone
        self._backbone   = DINOv3Backbone(model_name, method=method, device=device)
        self._backbone._load()
        self._model_name = model_name
        logger.info("DINOv3 proxy loaded: %s", model_name)
        return self

    def _ensure_backbone(self, model: str = "dinov3_vitl16_sat") -> None:
        if self._backbone is None:
            self.load(model)

    def extract_features(self, image: Union[str, Any],
                          model: str = "dinov3_vitl16_sat") -> Any:
        """Extract dense spatial feature map (H_p, W_p, D)."""
        self._ensure_backbone(model)
        return self._backbone.extract_features(image)

    def extract_embeddings(self, image: Union[str, Any],
                            model: str = "dinov3_vitl16_sat") -> Any:
        """Extract global CLS embedding for retrieval."""
        self._ensure_backbone(model)
        return self._backbone.extract_embeddings(image)

    def extract_patch_features(self, image: Union[str, Any]) -> Any:
        """Extract per-patch features (N, D) for dense prediction."""
        self._ensure_backbone()
        return self._backbone.extract_patch_features(image)

    def get_attention_maps(self, image: Union[str, Any]) -> Any:
        """Extract attention maps for explainability."""
        self._ensure_backbone()
        return self._backbone.get_attention_maps(image)

    def segment(self, image: Union[str, Any], task: str = "building",
                 num_classes: int = 2) -> Any:
        """Segment a satellite image using DINOv3 features + task head."""
        from pygeovision.models.foundation.dinov3 import DINOv3Backbone
        self._ensure_backbone()
        clf = self._backbone.build_classifier(num_classes, freeze_backbone=True)
        return clf

    def detect(self, image: Union[str, Any]) -> Dict:
        """Object detection placeholder — use zero_shot() for text-driven detection."""
        return {"note": "Use client.geoai.dinov3.zero_shot(image, 'object description')"}

    def classify(self, image: Union[str, Any], num_classes: int = 10) -> Any:
        """Build classification head and infer."""
        self._ensure_backbone()
        return self._backbone.build_classifier(num_classes)

    def canopy_height(self, image_path: str,
                       output_path: Optional[str] = None) -> Dict[str, Any]:
        """Predict canopy height using CHMv2 (DINOv3 ViT-L SAT + DPT)."""
        from pygeovision.models.foundation.dinov3 import CHMv2Model
        if self._chm is None:
            self._chm = CHMv2Model()
        return self._chm.predict_canopy_height(image_path, output_path)

    def estimate_biomass(self, image_path: str) -> Dict[str, Any]:
        """Estimate above-ground biomass from canopy height."""
        from pygeovision.models.foundation.dinov3 import CHMv2Model
        if self._chm is None:
            self._chm = CHMv2Model()
        return self._chm.estimate_biomass(image_path)

    def detect_deforestation(self, before: str, after: str,
                               output_path: Optional[str] = None) -> Dict:
        """Detect deforestation between two dates using CHMv2."""
        from pygeovision.models.foundation.dinov3 import CHMv2Model
        if self._chm is None:
            self._chm = CHMv2Model()
        return self._chm.detect_deforestation(before, after, output_path=output_path)

    def zero_shot(self, image: Union[str, Any], text_prompt: str,
                   mode: str = "segment") -> Any:
        """Zero-shot geospatial AI using dino.txt.

        Args:
            image: Satellite image
            text_prompt: What to find ("solar panels", "cargo ships", "water bodies")
            mode: "segment" | "detect" | "classify"
        """
        from pygeovision.models.foundation.dinov3 import DINOv3Text
        if self._txt is None:
            self._txt = DINOv3Text()
        if mode == "segment":
            return self._txt.segment_by_text(image, text_prompt)
        elif mode == "detect":
            return self._txt.detect_by_text(image, text_prompt)
        else:
            return self._txt.classify_by_text(image, [text_prompt])

    def list_models(self) -> List[str]:
        """List all available DINOv3 models."""
        from pygeovision.models.foundation.dinov3 import list_dinov3_models
        return list_dinov3_models()

    def list_satellite_models(self) -> List[str]:
        """List DINOv3 models trained on satellite data (SAT-493M)."""
        from pygeovision.models.foundation.dinov3 import list_satellite_models
        return list_satellite_models()

    def get_info(self, model_name: str) -> Dict:
        """Get detailed spec for a DINOv3 model."""
        from pygeovision.models.foundation.dinov3 import get_dinov3_info
        return get_dinov3_info(model_name)

    def finetune(self, task: str = "segmentation", num_classes: int = 2,
                  model_name: str = "dinov3_vitl16_sat", **kwargs) -> Dict:
        """Fine-tune DINOv3 for a geospatial task."""
        from pygeovision.models.foundation.dinov3 import finetune_dinov3
        return finetune_dinov3(model_name=model_name, task=task,
                                num_classes=num_classes, **kwargs)

    def __repr__(self) -> str:
        loaded = self._model_name or "none"
        return f"DINOv3Proxy(loaded={loaded!r}, 12 variants, 6 heads)"
