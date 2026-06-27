"""
Segmentation model architectures for PyGeoVision.

Provides factory functions for building pixel-wise segmentation models:
U-Net, DeepLabV3+, SegFormer — all configurable for arbitrary numbers
of input bands and output classes.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def build_unet(
    encoder: str = "resnet50",
    in_channels: int = 3,
    num_classes: int = 2,
    encoder_weights: Optional[str] = "imagenet",
    activation: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Build a U-Net segmentation model.

    Uses segmentation-models-pytorch (smp) as the backend.

    Args:
        encoder: Encoder backbone name (e.g. 'resnet50', 'efficientnet-b4').
        in_channels: Number of input channels (bands).
        num_classes: Number of output classes.
        encoder_weights: Pretrained weight source ('imagenet' or None).
        activation: Output activation ('softmax2d', 'sigmoid', or None).
        **kwargs: Additional args forwarded to smp.Unet.

    Returns:
        smp.Unet model instance.

    Raises:
        ImportError: If segmentation-models-pytorch is not installed.

    Example:
        >>> model = build_unet(encoder="resnet50", in_channels=4, num_classes=10)
    """
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise ImportError(
            "U-Net requires segmentation-models-pytorch. "
            "Install: pip install segmentation-models-pytorch"
        ) from exc

    # When in_channels != 3 and encoder_weights='imagenet', SMP patches
    # the first conv layer to accept arbitrary channels.
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=activation,
        **kwargs,
    )
    logger.info(
        "Built UNet(encoder=%s, in_channels=%d, num_classes=%d)",
        encoder, in_channels, num_classes,
    )
    return model


def build_deeplabv3plus(
    encoder: str = "resnet101",
    in_channels: int = 3,
    num_classes: int = 2,
    encoder_weights: Optional[str] = "imagenet",
    encoder_output_stride: int = 16,
    **kwargs: Any,
) -> Any:
    """Build a DeepLabV3+ segmentation model.

    Args:
        encoder: Encoder backbone (e.g. 'resnet101', 'xception').
        in_channels: Number of input channels.
        num_classes: Number of output classes.
        encoder_weights: Pretrained source ('imagenet' or None).
        encoder_output_stride: Output stride (16 or 8; 8 = higher resolution).
        **kwargs: Additional args for smp.DeepLabV3Plus.

    Returns:
        smp.DeepLabV3Plus model instance.

    Example:
        >>> model = build_deeplabv3plus(encoder="resnet101", num_classes=15)
    """
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise ImportError(
            "DeepLabV3+ requires segmentation-models-pytorch. "
            "Install: pip install segmentation-models-pytorch"
        ) from exc

    model = smp.DeepLabV3Plus(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        encoder_output_stride=encoder_output_stride,
        **kwargs,
    )
    logger.info(
        "Built DeepLabV3+(encoder=%s, in_channels=%d, num_classes=%d)",
        encoder, in_channels, num_classes,
    )
    return model


def build_segformer(
    model_size: str = "b2",
    in_channels: int = 3,
    num_classes: int = 2,
    pretrained: bool = True,
    **kwargs: Any,
) -> Any:
    """Build a SegFormer segmentation model.

    Args:
        model_size: SegFormer variant: 'b0', 'b1', 'b2', 'b3', 'b4', 'b5'.
        in_channels: Number of input channels.
        num_classes: Number of output classes.
        pretrained: Load ImageNet-21k pretrained weights.
        **kwargs: Additional model args.

    Returns:
        SegFormer model (HuggingFace transformers or timm).

    Example:
        >>> model = build_segformer(model_size="b5", num_classes=20)
    """
    _SIZES = {
        "b0": "nvidia/segformer-b0-finetuned-ade-512-512",
        "b1": "nvidia/segformer-b1-finetuned-ade-512-512",
        "b2": "nvidia/segformer-b2-finetuned-ade-512-512",
        "b3": "nvidia/segformer-b3-finetuned-ade-512-512",
        "b4": "nvidia/segformer-b4-finetuned-ade-512-512",
        "b5": "nvidia/segformer-b5-finetuned-ade-640-640",
    }
    if model_size not in _SIZES:
        raise ValueError(
            f"model_size must be one of {list(_SIZES.keys())}, got {model_size!r}"
        )

    try:
        from transformers import SegformerForSemanticSegmentation, SegformerConfig

        if pretrained:
            model = SegformerForSemanticSegmentation.from_pretrained(
                _SIZES[model_size],
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )
        else:
            config = SegformerConfig(
                num_labels=num_classes,
                **kwargs,
            )
            model = SegformerForSemanticSegmentation(config)

        # Patch first convolution for non-RGB inputs
        if in_channels != 3:
            import torch.nn as nn
            first_conv = model.segformer.encoder.patch_embeddings[0].proj
            new_conv = nn.Conv2d(
                in_channels,
                first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
            )
            if pretrained:
                # Initialize new channels from mean of RGB weights
                import torch
                with torch.no_grad():
                    new_conv.weight[:, :3] = first_conv.weight
                    if in_channels > 3:
                        mean_w = first_conv.weight.mean(dim=1, keepdim=True)
                        for c in range(3, in_channels):
                            new_conv.weight[:, c:c+1] = mean_w
            model.segformer.encoder.patch_embeddings[0].proj = new_conv

        logger.info(
            "Built SegFormer-%s(in_channels=%d, num_classes=%d)",
            model_size.upper(), in_channels, num_classes,
        )
        return model

    except ImportError:
        pass

    # Fallback: use timm SegFormer
    try:
        import timm
        timm_name = f"mit_{model_size}"  # e.g. mit_b2
        backbone = timm.create_model(
            timm_name,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=0,
        )
        logger.info(
            "Built SegFormer-%s via timm (in_channels=%d)", model_size, in_channels
        )
        return backbone
    except Exception as exc:
        raise ImportError(
            "SegFormer requires transformers or timm. "
            "Install: pip install transformers  OR  pip install timm"
        ) from exc


def build_fpn(
    encoder: str = "resnet50",
    in_channels: int = 3,
    num_classes: int = 2,
    encoder_weights: Optional[str] = "imagenet",
    **kwargs: Any,
) -> Any:
    """Build an FPN (Feature Pyramid Network) segmentation model.

    Args:
        encoder: Encoder backbone name.
        in_channels: Number of input channels.
        num_classes: Number of output classes.
        encoder_weights: Pretrained weights source.

    Returns:
        smp.FPN model instance.
    """
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise ImportError(
            "FPN requires segmentation-models-pytorch. "
            "Install: pip install segmentation-models-pytorch"
        ) from exc

    return smp.FPN(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        **kwargs,
    )


def build_pan(
    encoder: str = "resnet50",
    in_channels: int = 3,
    num_classes: int = 2,
    encoder_weights: Optional[str] = "imagenet",
    **kwargs: Any,
) -> Any:
    """Build a PAN (Pyramid Attention Network) segmentation model.

    Args:
        encoder: Backbone encoder.
        in_channels: Input channels.
        num_classes: Output classes.
        encoder_weights: Pretrained source.

    Returns:
        smp.PAN model instance.
    """
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise ImportError(
            "PAN requires segmentation-models-pytorch. "
            "Install: pip install segmentation-models-pytorch"
        ) from exc

    return smp.PAN(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        **kwargs,
    )
