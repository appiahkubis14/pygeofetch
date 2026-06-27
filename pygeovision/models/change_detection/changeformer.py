"""ChangeFormer for bi-temporal satellite change detection."""
from __future__ import annotations
import logging
from typing import Any, Optional
logger = logging.getLogger(__name__)


class ChangeFormer:
    """ChangeFormer — transformer-based bi-temporal change detection.

    Processes two co-registered GeoTIFFs (before/after) and outputs
    a binary or multi-class change mask.

    Example::

        model = ChangeFormer(num_classes=2, in_channels=4)
        change_map = model.detect("before.tif", "after.tif", "change.tif")
    """

    def __init__(self, num_classes: int = 2, in_channels: int = 4,
                 backbone: str = "mit-b0", device: Optional[str] = None) -> None:
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.backbone = backbone
        self.device = device or self._auto_device()
        self._model = None

    @staticmethod
    def _auto_device():
        try:
            import torch; return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError: return "cpu"

    def build(self) -> Any:
        """Build the ChangeFormer model architecture."""
        try:
            import torch, torch.nn as nn
        except ImportError:
            raise ImportError("torch required")

        class SiameseEncoder(nn.Module):
            def __init__(self, in_ch, embed_dim=64):
                super().__init__()
                self.shared = nn.Sequential(
                    nn.Conv2d(in_ch, embed_dim, 3, padding=1), nn.BatchNorm2d(embed_dim), nn.ReLU(),
                    nn.Conv2d(embed_dim, embed_dim*2, 3, stride=2, padding=1), nn.BatchNorm2d(embed_dim*2), nn.ReLU(),
                    nn.Conv2d(embed_dim*2, embed_dim*4, 3, stride=2, padding=1), nn.BatchNorm2d(embed_dim*4), nn.ReLU(),
                )
                # Cross-attention for temporal interaction
                self.attn = nn.MultiheadAttention(embed_dim*4, num_heads=4, batch_first=True)

            def forward(self, x1, x2):
                f1 = self.shared(x1)  # (B, C, H/4, W/4)
                f2 = self.shared(x2)
                B, C, H, W = f1.shape
                # Flatten for attention
                f1_flat = f1.view(B, C, -1).permute(0, 2, 1)
                f2_flat = f2.view(B, C, -1).permute(0, 2, 1)
                # Cross-attention: f1 queries f2
                f1_attn, _ = self.attn(f1_flat, f2_flat, f2_flat)
                diff = (f1_attn - f2_flat).permute(0, 2, 1).view(B, C, H, W)
                return diff

        class ChangeFormerModel(nn.Module):
            def __init__(self, in_ch, n_classes):
                super().__init__()
                self.encoder = SiameseEncoder(in_ch)
                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(256, 128, 2, stride=2),
                    nn.ReLU(),
                    nn.ConvTranspose2d(128, 64, 2, stride=2),
                    nn.ReLU(),
                    nn.Conv2d(64, n_classes, 1),
                )

            def forward(self, x1, x2=None):
                if x2 is None:
                    # Assume x1 is stacked: (B, 2*C, H, W)
                    C = x1.shape[1] // 2
                    x2, x1 = x1[:, :C], x1[:, C:]
                diff = self.encoder(x1, x2)
                return self.decoder(diff)

        self._model = ChangeFormerModel(self.in_channels, self.num_classes).to(self.device)
        return self._model

    def detect(self, before_path: str, after_path: str,
                output_path: str = "./output/change.tif") -> dict:
        """Detect changes between two co-registered GeoTIFFs.

        Handles common channel mismatches:
        - If the raster has more channels than in_channels, the first
          in_channels bands are selected.
        - If the raster has fewer channels, bands are repeated cyclically
          to reach in_channels (e.g. a greyscale image repeated 4×).
        """
        if self._model is None:
            self.build()

        try:
            import torch, numpy as np, rasterio
        except ImportError as exc:
            return {"error": str(exc)}

        self._model.eval()
        with rasterio.open(before_path) as s1:
            before = s1.read().astype(np.float32)
            profile = s1.profile.copy()
        with rasterio.open(after_path) as s2:
            after = s2.read().astype(np.float32)

        def _fix_channels(arr: "np.ndarray", target_c: int) -> "np.ndarray":
            """Return arr with exactly target_c channels (C, H, W)."""
            c = arr.shape[0]
            if c == target_c:
                return arr
            if c > target_c:
                # Take first target_c bands (e.g. RGB+NIR from a 12-band scene)
                return arr[:target_c]
            # Repeat bands cyclically until we reach target_c
            repeats = (target_c + c - 1) // c        # ceil division
            arr_rep = np.concatenate([arr] * repeats, axis=0)
            return arr_rep[:target_c]

        before = _fix_channels(before, self.in_channels)
        after  = _fix_channels(after,  self.in_channels)

        # Normalise per-band to [0, 1]
        for arr in [before, after]:
            for b in range(arr.shape[0]):
                mn, mx = arr[b].min(), arr[b].max()
                if mx - mn > 1e-8:
                    arr[b] = (arr[b] - mn) / (mx - mn)
                else:
                    arr[b] = 0.0

        # Pad spatial dims to a multiple of 4
        H, W = before.shape[1], before.shape[2]
        pH = (4 - H % 4) % 4
        pW = (4 - W % 4) % 4
        if pH or pW:
            before = np.pad(before, ((0, 0), (0, pH), (0, pW)))
            after  = np.pad(after,  ((0, 0), (0, pH), (0, pW)))

        t1 = torch.tensor(before).unsqueeze(0).to(self.device)
        t2 = torch.tensor(after).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self._model(t1, t2)

        pred = logits.argmax(dim=1)[0, :H, :W].cpu().numpy().astype(np.uint8)
        change_pct = float(pred.mean()) * 100

        import pathlib
        pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        profile.update(count=1, dtype="uint8", compress="lzw")
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(pred[np.newaxis])
            dst.update_tags(method="ChangeFormer", change_pct=f"{change_pct:.2f}")

        return {"output_path": output_path, "change_pct": round(change_pct, 3), "model": "ChangeFormer"}
