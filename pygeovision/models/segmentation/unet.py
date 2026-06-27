"""U-Net and variants for geospatial segmentation — independent PyTorch implementation."""
from __future__ import annotations
import logging
from typing import Any, List, Optional
logger = logging.getLogger(__name__)


def build_unet(backbone: str = "resnet50", num_classes: int = 2, in_channels: int = 4,
                pretrained: bool = True, decoder_channels: List[int] = None, **kwargs) -> Any:
    """Build U-Net with configurable encoder backbone.

    Args:
        backbone: Encoder backbone ("resnet18"|"resnet50"|"resnet101"|"efficientnet_b4"|"none")
        num_classes: Number of segmentation classes
        in_channels: Input spectral bands
        pretrained: Load ImageNet pretrained encoder
        decoder_channels: Decoder feature map sizes (default: [256, 128, 64, 32, 16])

    Example::

        model = build_unet("resnet50", num_classes=7, in_channels=4)
    """
    decoder_channels = decoder_channels or [256, 128, 64, 32, 16]
    try:
        import segmentation_models_pytorch as smp
        return smp.Unet(
            encoder_name=backbone,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=in_channels,
            classes=num_classes,
            decoder_channels=decoder_channels,
            **kwargs,
        )
    except ImportError:
        logger.info("smp not installed — using built-in SimpleUNet")
        return _build_simple_unet(in_channels, num_classes)


def _build_simple_unet(in_ch: int, n_classes: int) -> Any:
    """Minimal U-Net that works without SMP."""
    import torch, torch.nn as nn

    def _block(ci, co, p=1):
        return nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=p), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
            nn.Conv2d(co, co, 3, padding=p), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
        )

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.e1 = _block(in_ch, 64);   self.p1 = nn.MaxPool2d(2)
            self.e2 = _block(64, 128);     self.p2 = nn.MaxPool2d(2)
            self.e3 = _block(128, 256);    self.p3 = nn.MaxPool2d(2)
            self.e4 = _block(256, 512);    self.p4 = nn.MaxPool2d(2)
            self.bn = _block(512, 1024)
            self.u4 = nn.ConvTranspose2d(1024, 512, 2, stride=2); self.d4 = _block(1024, 512)
            self.u3 = nn.ConvTranspose2d(512, 256, 2, stride=2);  self.d3 = _block(512, 256)
            self.u2 = nn.ConvTranspose2d(256, 128, 2, stride=2);  self.d2 = _block(256, 128)
            self.u1 = nn.ConvTranspose2d(128, 64, 2, stride=2);   self.d1 = _block(128, 64)
            self.out = nn.Conv2d(64, n_classes, 1)

        def forward(self, x):
            e1 = self.e1(x); e2 = self.e2(self.p1(e1))
            e3 = self.e3(self.p2(e2)); e4 = self.e4(self.p3(e3))
            b  = self.bn(self.p4(e4))
            d4 = self.d4(torch.cat([self.u4(b), e4], 1))
            d3 = self.d3(torch.cat([self.u3(d4), e3], 1))
            d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
            d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
            return self.out(d1)
    return UNet()
