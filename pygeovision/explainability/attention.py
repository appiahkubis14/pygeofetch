"""Attention map extraction for transformer-based geospatial models (G6)."""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional
logger = logging.getLogger(__name__)


class AttentionMapExtractor:
    """Extract self-attention maps from Vision Transformer (ViT) based models.

    Works with: SegFormer, DINOv2, Prithvi, SAM (ViT backbone)

    Example::

        extractor = AttentionMapExtractor(model)
        attn = extractor.extract(image_tensor, layer_idx=-1)
        extractor.visualise(image_path, attn, "./results/attention.png")
    """

    def __init__(self, model: Any, device: Optional[str] = None) -> None:
        self.model = model
        self.device = device or ("cpu")
        self._attention_maps: List = []

    def _register_attention_hooks(self) -> None:
        """Register hooks on all attention layers."""
        import torch.nn as nn
        self._attention_maps = []

        def _hook(_, __, output):
            if isinstance(output, tuple):
                self._attention_maps.append(output[1].detach().cpu())
            elif hasattr(output, "attentions") and output.attentions:
                self._attention_maps.extend([a.detach().cpu() for a in output.attentions])

        for name, module in self.model.named_modules():
            if "attention" in name.lower() or "attn" in name.lower():
                module.register_forward_hook(_hook)

    def extract(self, image: Any, layer_idx: int = -1,
                head_idx: Optional[int] = None) -> Any:
        """Extract attention maps for an image."""
        try:
            import torch, numpy as np
            self._register_attention_hooks()
            if isinstance(image, np.ndarray):
                image = torch.tensor(image, dtype=torch.float32)
            if image.ndim == 3: image = image.unsqueeze(0)
            image = image.to(self.device)
            self.model.eval()
            with torch.no_grad():
                _ = self.model(image)

            if not self._attention_maps:
                return np.zeros((image.shape[-2], image.shape[-1]))

            attn = self._attention_maps[layer_idx]
            if head_idx is not None:
                attn = attn[:, head_idx]
            else:
                attn = attn.mean(dim=1)  # average over heads

            # Reshape to spatial
            attn = attn.squeeze(0)
            if attn.ndim == 2:
                n = int(attn.shape[0] ** 0.5)
                if n * n == attn.shape[0]:
                    attn = attn.mean(0).reshape(n, n).numpy()
                else:
                    attn = attn.mean(0).numpy()
            return attn
        except Exception as exc:
            import numpy as np
            logger.warning("Attention extraction failed: %s", exc)
            return np.zeros((64, 64))
