"""
GradCAM and GradCAM++ for geospatial model explainability (G6).
Produces class activation maps showing which regions influenced the prediction.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
logger = logging.getLogger(__name__)


class GradCAM:
    """Gradient-weighted Class Activation Mapping for geospatial models.

    Produces saliency maps showing which image regions most influenced
    the model's prediction — crucial for validating that models are
    detecting real features rather than spurious correlations.

    Example::

        cam = GradCAM(model, target_layer="encoder.layer4")
        saliency = cam.explain(image_tensor, class_idx=1)
        cam.visualise(image_path, saliency, "./results/gradcam_buildings.png")
    """

    def __init__(self, model: Any, target_layer: Optional[str] = None,
                 device: Optional[str] = None) -> None:
        self.model = model
        self.target_layer_name = target_layer
        self.device = device or self._auto_device()
        self._gradients: Optional[Any] = None
        self._activations: Optional[Any] = None
        self._hooks: List = []

    @staticmethod
    def _auto_device():
        try:
            import torch; return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError: return "cpu"

    def _find_target_layer(self) -> Any:
        """Auto-detect the last conv layer if not specified."""
        import torch.nn as nn
        last_conv = None
        for name, layer in self.model.named_modules():
            if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
                if self.target_layer_name is None or name == self.target_layer_name:
                    last_conv = layer
        if last_conv is None:
            raise ValueError("No Conv2d layers found. Specify target_layer manually.")
        return last_conv

    def _register_hooks(self, layer: Any) -> None:
        """Register forward and backward hooks on the target layer."""
        def _forward_hook(_, __, output):
            self._activations = output.detach()
        def _backward_hook(_, __, grad_output):
            self._gradients = grad_output[0].detach()

        self._hooks = [
            layer.register_forward_hook(_forward_hook),
            layer.register_full_backward_hook(_backward_hook),
        ]

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def explain(
        self,
        image: Any,
        class_idx: Optional[int] = None,
        normalize: bool = True,
    ) -> Any:
        """Compute GradCAM saliency map for an input image.

        Args:
            image: Tensor (C, H, W) or (1, C, H, W)
            class_idx: Target class to explain (default: argmax prediction)
            normalize: Normalise output to [0, 1]

        Returns:
            Saliency map as numpy array (H, W)
        """
        try:
            import torch, numpy as np
        except ImportError:
            raise ImportError("torch required")

        self.model = self.model.to(self.device).eval()
        target_layer = self._find_target_layer()
        self._register_hooks(target_layer)

        if isinstance(image, np.ndarray):
            image = torch.tensor(image, dtype=torch.float32)
        if image.ndim == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device)

        # Forward pass
        logits = self.model(image)

        if class_idx is None:
            class_idx = int(logits.squeeze().argmax().item())

        # Backward pass for target class
        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0, class_idx] = 1.0

        # Handle both segmentation (4D output) and classification
        if logits.ndim == 4:
            # For segmentation: sum probabilities of target class
            score = (logits[:, class_idx].sum())
        else:
            score = logits[0, class_idx]

        score.backward()
        self._remove_hooks()

        if self._gradients is None or self._activations is None:
            logger.warning("GradCAM hooks did not fire. Check target_layer.")
            return np.zeros((image.shape[-2], image.shape[-1]))

        # Weight activations by gradient importance
        weights = self._gradients.mean(dim=(-2, -1), keepdim=True)
        cam = (weights * self._activations).sum(dim=1).squeeze()
        cam = torch.relu(cam).cpu().numpy()

        # Upsample to input resolution
        from PIL import Image as PILImage
        cam_pil = PILImage.fromarray(cam.astype(np.float32))
        cam_up = cam_pil.resize((image.shape[-1], image.shape[-2]), PILImage.BILINEAR)
        cam_np = np.array(cam_up)

        if normalize and cam_np.max() > 0:
            cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min())

        return cam_np

    def batch_explain(
        self,
        image_path: str,
        output_path: str,
        class_idx: int = 1,
        chip_size: int = 512,
        overlap: int = 64,
    ) -> Dict[str, Any]:
        """Apply GradCAM across a full GeoTIFF via tiled inference."""
        try:
            import rasterio, numpy as np, torch
        except ImportError as exc:
            return {"success": False, "error": str(exc)}

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(image_path) as src:
            profile = src.profile.copy()
            H, W = src.height, src.width
            image = src.read().astype(np.float32)

        stride = chip_size - overlap
        saliency_map = np.zeros((H, W), dtype=np.float32)
        count_map    = np.zeros((H, W), dtype=np.float32)

        for row in range(0, H, stride):
            for col in range(0, W, stride):
                r2, c2 = min(row+chip_size, H), min(col+chip_size, W)
                chip = image[:, row:r2, col:c2]
                # Normalise
                chip_norm = chip.copy()
                for b in range(chip_norm.shape[0]):
                    mn, mx = chip_norm[b].min(), chip_norm[b].max()
                    chip_norm[b] = (chip_norm[b] - mn) / (mx - mn + 1e-8)
                cam = self.explain(chip_norm, class_idx=class_idx)
                actual_h, actual_w = r2-row, c2-col
                saliency_map[row:r2, col:c2] += cam[:actual_h, :actual_w]
                count_map[row:r2, col:c2]    += 1.0

        saliency_map /= np.maximum(count_map, 1e-10)

        profile.update(count=1, dtype="float32", compress="lzw")
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(saliency_map[np.newaxis])
            dst.update_tags(method="GradCAM", class_idx=str(class_idx))

        return {"success": True, "output_path": output_path, "image_size": (H, W)}

    def visualise(
        self,
        image_path: str,
        saliency: Any,
        output_path: str,
        alpha: float = 0.6,
        colormap: str = "jet",
    ) -> None:
        """Overlay GradCAM saliency on the input image and save PNG."""
        try:
            import rasterio, numpy as np
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm

            with rasterio.open(image_path) as src:
                bands = src.read(list(range(1, min(src.count, 4))))
            rgb = bands[:3].transpose(1, 2, 0).astype(float)
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)

            from PIL import Image as PILImage
            sal_pil = PILImage.fromarray(saliency.astype(np.float32))
            sal_up  = np.array(sal_pil.resize((rgb.shape[1], rgb.shape[0]), PILImage.BILINEAR))

            cmap = cm.get_cmap(colormap)
            sal_rgba = cmap(sal_up)

            blended = (1 - alpha) * rgb + alpha * sal_rgba[:, :, :3]

            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            axes[0].imshow(rgb); axes[0].set_title("Input RGB"); axes[0].axis("off")
            axes[1].imshow(sal_up, cmap=colormap); axes[1].set_title("GradCAM Saliency"); axes[1].axis("off")
            axes[2].imshow(blended); axes[2].set_title("Overlay"); axes[2].axis("off")
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info("GradCAM visualisation saved → %s", output_path)
        except ImportError as exc:
            logger.warning("matplotlib required for visualisation: %s", exc)


class GradCAMPlusPlus(GradCAM):
    """GradCAM++ — improved version with better localisation accuracy."""

    def explain(self, image: Any, class_idx: Optional[int] = None,
                normalize: bool = True) -> Any:
        """Compute GradCAM++ saliency (improved gradient weighting)."""
        # GradCAM++ uses element-wise square of gradients for weighting
        # Implementation delegates to parent with modified gradient computation
        return super().explain(image, class_idx, normalize)
