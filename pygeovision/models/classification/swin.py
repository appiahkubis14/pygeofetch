"""Swin Transformer for geospatial classification."""
from typing import Any

def build_swin(variant: str = "t", num_classes: int = 10, in_channels: int = 4,
                pretrained: bool = True, **kwargs) -> Any:
    """Build Swin Transformer (Tiny/Base/Large) for satellite classification.

    Example::

        model = build_swin("b", num_classes=20, in_channels=13)  # Sentinel-2 all bands
    """
    TIMM_IDS = {
        "t": "swin_tiny_patch4_window7_224",
        "s": "swin_small_patch4_window7_224",
        "b": "swin_base_patch4_window7_224",
        "l": "swin_large_patch4_window7_224",
    }
    try:
        import timm
        return timm.create_model(
            TIMM_IDS.get(variant, TIMM_IDS["t"]),
            pretrained=pretrained, num_classes=num_classes,
            in_chans=in_channels, **kwargs,
        )
    except ImportError:
        raise ImportError("pip install timm")
