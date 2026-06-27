"""Vision Transformer for geospatial classification."""
from __future__ import annotations
from typing import Any, Optional


def build_vit(variant: str = "b16", num_classes: int = 10, in_channels: int = 4,
               pretrained: bool = True, **kwargs) -> Any:
    """Build Vision Transformer for satellite image classification.

    Args:
        variant: "b16" | "l16" | "h14"
        num_classes: Output classes
        in_channels: Input spectral bands (4 for Sentinel-2 BGRN)
        pretrained: Load ImageNet-21k weights

    Example::

        model = build_vit("b16", num_classes=10, in_channels=4)
    """
    HF_IDS = {
        "b16": "google/vit-base-patch16-224",
        "l16": "google/vit-large-patch16-224",
        "h14": "google/vit-huge-patch14-224-in21k",
    }
    try:
        from transformers import ViTForImageClassification, ViTConfig
        config = ViTConfig.from_pretrained(HF_IDS.get(variant, HF_IDS["b16"]))
        config.num_labels = num_classes
        config.num_channels = in_channels
        if pretrained:
            model = ViTForImageClassification.from_pretrained(
                HF_IDS.get(variant, HF_IDS["b16"]),
                config=config, ignore_mismatched_sizes=True,
            )
        else:
            model = ViTForImageClassification(config)
        return model
    except ImportError:
        # Fallback: timm
        try:
            import timm
            return timm.create_model(
                f"vit_{variant}", pretrained=pretrained,
                num_classes=num_classes, in_chans=in_channels, **kwargs,
            )
        except ImportError:
            raise ImportError("pip install transformers timm")
