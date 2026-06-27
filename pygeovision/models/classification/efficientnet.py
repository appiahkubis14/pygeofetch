"""EfficientNet for geospatial classification."""
from typing import Any

def build_efficientnet(variant: str = "b4", num_classes: int = 10,
                        in_channels: int = 4, pretrained: bool = True, **kwargs) -> Any:
    """Build EfficientNet-B0 through B7 for satellite classification.

    Example::

        model = build_efficientnet("b4", num_classes=9, in_channels=4)
    """
    try:
        import timm
        return timm.create_model(
            f"efficientnet_{variant}", pretrained=pretrained,
            num_classes=num_classes, in_chans=in_channels, **kwargs,
        )
    except ImportError:
        raise ImportError("pip install timm")
