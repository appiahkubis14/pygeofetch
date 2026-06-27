"""Super-resolution model architectures for satellite imagery."""
from __future__ import annotations
import logging
from typing import Any
logger = logging.getLogger(__name__)


def build_srcnn(in_channels: int = 3, scale_factor: int = 4, **kwargs: Any) -> Any:
    """Build a Super-Resolution CNN (SRCNN)."""
    import torch.nn as nn

    class SRCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.upsample = nn.Upsample(scale_factor=scale_factor, mode="bicubic", align_corners=False)
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, 64, 9, padding=4), nn.ReLU(True),
                nn.Conv2d(64, 32, 1, padding=0), nn.ReLU(True),
                nn.Conv2d(32, in_channels, 5, padding=2),
            )

        def forward(self, x):
            return self.net(self.upsample(x))

    model = SRCNN()
    logger.info("Built SRCNN(in_channels=%d, scale_factor=%d)", in_channels, scale_factor)
    return model


def build_esrgan_geo(in_channels: int = 3, scale_factor: int = 4, num_rrdb: int = 12, **kwargs: Any) -> Any:
    """Build an ESRGAN generator adapted for satellite imagery."""
    import torch
    import torch.nn as nn

    class ResidualDenseBlock(nn.Module):
        def __init__(self, nf: int = 64, gc: int = 32):
            super().__init__()
            self.conv1 = nn.Conv2d(nf, gc, 3, padding=1)
            self.conv2 = nn.Conv2d(nf + gc, gc, 3, padding=1)
            self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, padding=1)
            self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, padding=1)
            self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, padding=1)
            self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, x):
            x1 = self.lrelu(self.conv1(x))
            x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
            x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], 1)))
            x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
            x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
            return x5 * 0.2 + x

    class RRDB(nn.Module):
        def __init__(self, nf: int = 64):
            super().__init__()
            self.rdb1 = ResidualDenseBlock(nf)
            self.rdb2 = ResidualDenseBlock(nf)
            self.rdb3 = ResidualDenseBlock(nf)

        def forward(self, x):
            return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x

    class ESRGANGeo(nn.Module):
        def __init__(self):
            super().__init__()
            nf = 64
            self.conv_first = nn.Conv2d(in_channels, nf, 3, padding=1)
            self.body = nn.Sequential(*[RRDB(nf) for _ in range(num_rrdb)])
            self.conv_body = nn.Conv2d(nf, nf, 3, padding=1)
            # Pixel-shuffle upsampling
            ups = []
            for _ in range(scale_factor // 2):
                ups += [nn.Conv2d(nf, nf * 4, 3, padding=1), nn.PixelShuffle(2), nn.LeakyReLU(0.2, True)]
            self.upsample = nn.Sequential(*ups)
            self.conv_last = nn.Conv2d(nf, in_channels, 3, padding=1)

        def forward(self, x):
            feat = self.conv_first(x)
            feat = self.conv_body(self.body(feat)) + feat
            feat = self.upsample(feat)
            return self.conv_last(feat)

    model = ESRGANGeo()
    logger.info("Built ESRGAN-Geo(in_channels=%d, scale=%dx, rrdb=%d)", in_channels, scale_factor, num_rrdb)
    return model
