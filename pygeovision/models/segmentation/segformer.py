"""SegFormer for geospatial semantic segmentation."""
from typing import Any, Optional


def build_segformer(variant: str = "b2", num_classes: int = 2, in_channels: int = 4,
                     pretrained: bool = True, **kwargs) -> Any:
    """Build SegFormer (Mix Transformer backbone + lightweight head).

    Args:
        variant: "b0" | "b1" | "b2" | "b3" | "b4" | "b5"
        num_classes: Segmentation classes
        in_channels: Input channels (adapted via 1×1 conv if != 3)

    Example::

        model = build_segformer("b2", num_classes=7, in_channels=4)
    """
    HF_IDS = {
        "b0": "nvidia/mit-b0", "b1": "nvidia/mit-b1",
        "b2": "nvidia/mit-b2", "b3": "nvidia/mit-b3",
        "b4": "nvidia/mit-b4", "b5": "nvidia/mit-b5",
    }
    try:
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
        import torch.nn as nn, torch

        hf_id = HF_IDS.get(variant, HF_IDS["b2"])
        config = SegformerConfig.from_pretrained(hf_id)
        config.num_labels = num_classes
        config.id2label = {i: str(i) for i in range(num_classes)}
        config.label2id = {str(i): i for i in range(num_classes)}

        if pretrained:
            model = SegformerForSemanticSegmentation.from_pretrained(
                hf_id, config=config, ignore_mismatched_sizes=True,
            )
        else:
            model = SegformerForSemanticSegmentation(config)

        # Handle non-3-channel input
        if in_channels != 3:
            old_conv = model.segformer.encoder.patch_embeddings[0].proj
            new_conv = nn.Conv2d(in_channels, old_conv.out_channels,
                                  kernel_size=old_conv.kernel_size,
                                  stride=old_conv.stride, padding=old_conv.padding)
            with torch.no_grad():
                # Copy first 3 channel weights, initialise remaining
                new_conv.weight[:, :min(3, in_channels)] = old_conv.weight[:, :min(3, in_channels)]
                if in_channels > 3:
                    nn.init.kaiming_normal_(new_conv.weight[:, 3:])
            model.segformer.encoder.patch_embeddings[0].proj = new_conv

        return model
    except ImportError:
        raise ImportError("pip install transformers torch")
