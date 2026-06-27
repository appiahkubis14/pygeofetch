"""DeepLabV3+ for geospatial segmentation."""
from typing import Any


def build_deeplab(backbone: str = "resnet50", num_classes: int = 2,
                   in_channels: int = 4, pretrained: bool = True, **kwargs) -> Any:
    """Build DeepLabV3+ with atrous convolution for multi-scale context.

    Example::

        model = build_deeplab("resnet101", num_classes=11)
    """
    try:
        import segmentation_models_pytorch as smp
        return smp.DeepLabV3Plus(
            encoder_name=backbone,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=in_channels,
            classes=num_classes,
        )
    except ImportError:
        # Torchvision fallback
        try:
            import torchvision.models.segmentation as tvseg
            import torch.nn as nn
            model = tvseg.deeplabv3_resnet50(
                pretrained=False, num_classes=num_classes,
            )
            if in_channels != 3:
                model.backbone.conv1 = nn.Conv2d(
                    in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
                )
            return model
        except ImportError:
            raise ImportError("pip install segmentation-models-pytorch OR torch torchvision")
