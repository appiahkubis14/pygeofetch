"""Change detection model architectures."""
from __future__ import annotations
import logging
from typing import Any
logger = logging.getLogger(__name__)


def build_siamese_unet(in_channels: int = 3, num_classes: int = 2, encoder: str = "resnet34", pretrained: bool = True, **kwargs: Any) -> Any:
    """Build a Siamese U-Net for bi-temporal change detection."""
    import torch
    import torch.nn as nn
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise ImportError("Siamese-UNet requires segmentation-models-pytorch. Install: pip install segmentation-models-pytorch") from exc

    class SiameseUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = smp.Unet(encoder_name=encoder, encoder_weights="imagenet" if pretrained else None, in_channels=in_channels, classes=num_classes)
            # Difference-based: concat t1 and t2 then feed through decoder
            self.encoder_t1 = smp.encoders.get_encoder(encoder, in_channels=in_channels, depth=5, weights="imagenet" if pretrained else None)
            self.encoder_t2 = smp.encoders.get_encoder(encoder, in_channels=in_channels, depth=5, weights="imagenet" if pretrained else None)
            enc_ch = self.encoder_t1.out_channels
            double_ch = tuple(c * 2 for c in enc_ch)
            self.decoder = smp.decoders.unet.decoder.UnetDecoder(
                encoder_channels=double_ch, decoder_channels=(256, 128, 64, 32, 16), n_blocks=5
            )
            self.head = nn.Conv2d(16, num_classes, kernel_size=1)

        def forward(self, t1, t2):
            feats1 = self.encoder_t1(t1)
            feats2 = self.encoder_t2(t2)
            diff_feats = [torch.cat([f1, f2], dim=1) for f1, f2 in zip(feats1, feats2)]
            dec = self.decoder(*diff_feats)
            return self.head(dec)

    model = SiameseUNet()
    logger.info("Built SiameseUNet(in_channels=%d, num_classes=%d, encoder=%s)", in_channels, num_classes, encoder)
    return model


def build_changeformer(in_channels: int = 3, num_classes: int = 2, **kwargs: Any) -> Any:
    """Build a ChangeFormer transformer change detection model."""
    try:
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
        import torch.nn as nn

        class ChangeFormer(nn.Module):
            def __init__(self):
                super().__init__()
                cfg = SegformerConfig(num_labels=num_classes, num_channels=in_channels)
                self.encoder_t1 = SegformerForSemanticSegmentation(cfg)
                self.encoder_t2 = SegformerForSemanticSegmentation(cfg)
                self.fusion = nn.Conv2d(num_classes * 2, num_classes, 1)

            def forward(self, t1, t2):
                out1 = self.encoder_t1(pixel_values=t1).logits
                out2 = self.encoder_t2(pixel_values=t2).logits
                import torch.nn.functional as F
                out1 = F.interpolate(out1, size=t1.shape[-2:], mode="bilinear", align_corners=False)
                out2 = F.interpolate(out2, size=t2.shape[-2:], mode="bilinear", align_corners=False)
                import torch
                return self.fusion(torch.cat([out1, out2], dim=1))

        model = ChangeFormer()
        logger.info("Built ChangeFormer(in_channels=%d, num_classes=%d)", in_channels, num_classes)
        return model
    except ImportError as exc:
        raise ImportError("ChangeFormer requires transformers. Install: pip install transformers") from exc
