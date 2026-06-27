"""DETR / RT-DETR / RF-DETR for satellite object detection."""
from typing import Any, Optional


def build_detr(variant: str = "detr-r50", num_classes: int = 5,
                in_channels: int = 3, pretrained: bool = True, **kwargs) -> Any:
    """Build DETR-family detection model.

    Example::

        model = build_detr("rt-detr-l", num_classes=5)
    """
    HF_IDS = {
        "detr-r50":   "facebook/detr-resnet-50",
        "detr-r101":  "facebook/detr-resnet-101",
        "rt-detr-l":  "PekingU/rtdetr_r50vd",
        "rf-detr-b":  "roboflow/rf-detr-base",
    }
    try:
        from transformers import AutoModelForObjectDetection, AutoConfig
        hf_id = HF_IDS.get(variant, HF_IDS["detr-r50"])
        config = AutoConfig.from_pretrained(hf_id)
        if hasattr(config, "num_labels"):
            config.num_labels = num_classes + 1  # +1 for background
        if pretrained:
            return AutoModelForObjectDetection.from_pretrained(
                hf_id, config=config, ignore_mismatched_sizes=True,
            )
        return AutoModelForObjectDetection.from_config(config)
    except ImportError:
        raise ImportError("pip install transformers")
