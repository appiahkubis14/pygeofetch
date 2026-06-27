"""Uncertainty estimation via Monte Carlo Dropout (G6)."""
from __future__ import annotations
import logging
from typing import Any, Dict, Optional, Union
logger = logging.getLogger(__name__)


class UncertaintyEstimator:
    """Monte Carlo Dropout uncertainty estimation for geospatial models.

    Runs multiple stochastic forward passes to estimate:
        - Epistemic uncertainty (model uncertainty)
        - Aleatoric uncertainty (data uncertainty)
        - Predictive entropy

    High uncertainty regions indicate where the model needs more training data.

    Example::

        estimator = UncertaintyEstimator(model, n_passes=20)
        result = estimator.estimate(
            "./data/sentinel2.tif",
            "./results/uncertainty_map.tif",
        )
    """

    def __init__(self, model: Any, n_passes: int = 20,
                 device: Optional[str] = None) -> None:
        self.model = model
        self.n_passes = n_passes
        self.device = device or self._auto_device()

    @staticmethod
    def _auto_device():
        try:
            import torch; return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError: return "cpu"

    def _enable_dropout(self) -> None:
        """Enable dropout layers for stochastic inference."""
        import torch.nn as nn
        for m in self.model.modules():
            if isinstance(m, nn.Dropout) or isinstance(m, nn.Dropout2d):
                m.train()

    def estimate(
        self,
        image_path: Union[str, Any],
        output_path: Optional[str] = None,
        chip_size: int = 512,
        overlap: int = 64,
        save_epistemic: bool = True,
        save_aleatoric: bool = True,
    ) -> Dict[str, Any]:
        """Estimate per-pixel uncertainty across the GeoTIFF."""
        try:
            import torch, numpy as np, rasterio
        except ImportError as exc:
            return {"success": False, "error": str(exc)}

        self.model = self.model.to(self.device)
        self._enable_dropout()

        with rasterio.open(str(image_path)) as src:
            profile = src.profile.copy()
            H, W = src.height, src.width
            image = src.read().astype(np.float32)

        stride = chip_size - overlap
        # Store all MC samples
        n_classes = None
        all_samples_shape = None

        # First pass to get output shape
        chip = image[:, :chip_size, :chip_size]
        for b in range(chip.shape[0]):
            mn, mx = chip[b].min(), chip[b].max()
            chip[b] = (chip[b] - mn) / (mx - mn + 1e-8)
        chip_t = torch.tensor(chip).unsqueeze(0).to(self.device)
        with torch.no_grad():
            test_out = self.model(chip_t)
        n_classes = test_out.shape[1]

        # MC sampling
        pred_accum  = np.zeros((self.n_passes, n_classes, H, W), dtype=np.float16)

        for pass_idx in range(self.n_passes):
            for row in range(0, H, stride):
                for col in range(0, W, stride):
                    r2, c2 = min(row+chip_size, H), min(col+chip_size, W)
                    chip = image[:, row:r2, col:c2].copy()
                    for b in range(chip.shape[0]):
                        mn, mx = chip[b].min(), chip[b].max()
                        chip[b] = (chip[b] - mn) / (mx - mn + 1e-8)
                    # Pad
                    padded = np.zeros((image.shape[0], chip_size, chip_size), np.float32)
                    padded[:, :chip.shape[1], :chip.shape[2]] = chip

                    chip_t = torch.tensor(padded).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        probs = torch.softmax(self.model(chip_t), dim=1)[0]
                    actual_h, actual_w = r2-row, c2-col
                    pred_accum[pass_idx, :, row:r2, col:c2] = probs.cpu().numpy()[:, :actual_h, :actual_w]

        # Compute uncertainty metrics
        mean_probs = pred_accum.mean(axis=0).astype(np.float32)     # (C, H, W)
        # Predictive entropy
        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10), axis=0)  # (H, W)
        # Epistemic = variance across passes
        epistemic = pred_accum.var(axis=0).mean(axis=0)              # (H, W)
        # Aleatoric = mean entropy of individual passes
        per_pass_entropy = -np.sum(pred_accum * np.log(pred_accum + 1e-10), axis=1)  # (n_passes, H, W)
        aleatoric = per_pass_entropy.mean(axis=0)

        # Prediction (mean across passes)
        label = np.argmax(mean_probs, axis=0).astype(np.uint8)

        result = {
            "success": True,
            "image_size": (H, W),
            "n_mc_passes": self.n_passes,
            "mean_entropy": float(entropy.mean()),
            "mean_epistemic": float(epistemic.mean()),
            "high_uncertainty_fraction": float((entropy > 0.5).mean()),
        }

        if output_path:
            from pathlib import Path
            op = Path(output_path)
            op.parent.mkdir(parents=True, exist_ok=True)

            # Save 3-band uncertainty raster: label, entropy, epistemic
            out_profile = profile.copy()
            out_profile.update(count=3, dtype="float32", compress="lzw")
            with rasterio.open(str(op), "w", **out_profile) as dst:
                dst.write(np.stack([label.astype(np.float32), entropy, epistemic]))
                dst.update_tags(bands="label|entropy|epistemic", n_mc_passes=str(self.n_passes))
            result["output_path"] = str(op)

        return result
