"""Classification model architectures."""
from __future__ import annotations
import logging
from typing import Any, Optional
logger = logging.getLogger(__name__)


def build_resnet(depth: int = 50, in_channels: int = 3, num_classes: int = 10, pretrained: bool = True, **kwargs: Any) -> Any:
    """Build a ResNet classifier."""
    try:
        import torch.nn as nn
        import torchvision.models as models
        _map = {18: models.resnet18, 34: models.resnet34, 50: models.resnet50,
                101: models.resnet101, 152: models.resnet152}
        if depth not in _map:
            raise ValueError(f"depth must be one of {list(_map.keys())}")
        weights = "DEFAULT" if (pretrained and in_channels == 3) else None
        model = _map[depth](weights=weights)
        if in_channels != 3:
            model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        logger.info("Built ResNet-%d(in_channels=%d, num_classes=%d)", depth, in_channels, num_classes)
        return model
    except ImportError as exc:
        raise ImportError("ResNet requires torchvision. Install: pip install torchvision") from exc


def build_efficientnet(model_size: str = "b3", in_channels: int = 3, num_classes: int = 10, pretrained: bool = True, **kwargs: Any) -> Any:
    """Build an EfficientNet classifier."""
    try:
        import timm
        name = f"efficientnet_{model_size}"
        model = timm.create_model(name, pretrained=pretrained, in_chans=in_channels, num_classes=num_classes)
        logger.info("Built EfficientNet-%s(in_channels=%d, num_classes=%d)", model_size, in_channels, num_classes)
        return model
    except ImportError as exc:
        raise ImportError("EfficientNet requires timm. Install: pip install timm") from exc


def build_vit(model_size: str = "b16", in_channels: int = 3, num_classes: int = 10, pretrained: bool = True, **kwargs: Any) -> Any:
    """Build a Vision Transformer (ViT) classifier."""
    _name_map = {"b16": "vit_base_patch16_224", "b32": "vit_base_patch32_224",
                 "l16": "vit_large_patch16_224", "s16": "vit_small_patch16_224"}
    try:
        import timm
        name = _name_map.get(model_size, f"vit_{model_size}")
        model = timm.create_model(name, pretrained=pretrained, in_chans=in_channels, num_classes=num_classes)
        logger.info("Built ViT-%s(in_channels=%d, num_classes=%d)", model_size, in_channels, num_classes)
        return model
    except ImportError as exc:
        raise ImportError("ViT requires timm. Install: pip install timm") from exc
