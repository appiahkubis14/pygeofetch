"""Streaming inference for ultra-large GeoTIFFs (B3, B4) — never loads full image."""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
logger = logging.getLogger(__name__)


class StreamingInference:
    """Memory-efficient streaming inference for ultra-large GeoTIFFs (B4).

    Reads only the current tile from disk, processes it, and writes results
    directly to output. Peak RAM = chip_size * chip_size * bands * dtype.

    Supports GeoTIFFs larger than available RAM.
    """

    def __init__(self, model: Any, chip_size: int = 1024,
                 overlap: int = 128, num_classes: int = 2,
                 device: Optional[str] = None) -> None:
        self.model = model
        self.chip_size = chip_size
        self.overlap = overlap
        self.num_classes = num_classes
        self.device = device or ("cuda" if self._has_cuda() else "cpu")

    @staticmethod
    def _has_cuda():
        try:
            import torch; return torch.cuda.is_available()
        except ImportError: return False

    def stream_chips(self, image_path: str) -> Generator[Dict, None, None]:
        """Generator that yields image chips one at a time (never loads full image)."""
        try:
            import rasterio, numpy as np
            from rasterio.windows import Window
        except ImportError:
            raise ImportError("rasterio required")

        with rasterio.open(image_path) as src:
            H, W = src.height, src.width
            stride = self.chip_size - self.overlap

            for row in range(0, H, stride):
                for col in range(0, W, stride):
                    r2 = min(row + self.chip_size, H)
                    c2 = min(col + self.chip_size, W)
                    window = Window(col, row, c2 - col, r2 - row)
                    chip = src.read(window=window).astype(np.float32)
                    # Normalise
                    for b in range(chip.shape[0]):
                        mn, mx = chip[b].min(), chip[b].max()
                        chip[b] = (chip[b] - mn) / (mx - mn + 1e-8)
                    yield {"chip": chip, "row": row, "col": col,
                            "height": r2 - row, "width": c2 - col,
                            "window": (col, row, c2 - col, r2 - row)}

    def infer(self, image_path: Union[str, Path],
               output_path: Union[str, Path]) -> Dict[str, Any]:
        """Stream-infer a large GeoTIFF with minimal memory footprint."""
        import numpy as np, rasterio
        from pygeovision.inference.tiled import TiledInference
        # Delegate to TiledInference with batch_tiles=1 (minimal memory)
        inf = TiledInference(
            model=self.model, chip_size=self.chip_size,
            overlap=self.overlap, num_classes=self.num_classes,
            device=self.device, batch_tiles=1,
        )
        return inf.infer(image_path, output_path)


class EnsembleInference:
    """Ensemble of multiple models for robust predictions."""

    def __init__(self, models: List[Any], weights: Optional[List[float]] = None,
                 fusion: str = "mean", num_classes: int = 2,
                 device: Optional[str] = None) -> None:
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        self.fusion = fusion
        self.num_classes = num_classes
        self.device = device or ("cuda" if StreamingInference._has_cuda() else "cpu")

    def infer(self, image_path: Union[str, Path],
               output_path: Union[str, Path]) -> Dict[str, Any]:
        from pygeovision.inference.tiled import TiledInference
        import numpy as np, rasterio

        all_probs = []
        for model, weight in zip(self.models, self.weights):
            inf = TiledInference(model=model, num_classes=self.num_classes, device=self.device)
            tmp = str(output_path) + f".tmp{len(all_probs)}.tif"
            inf.infer(image_path, tmp, return_probabilities=True)
            with rasterio.open(tmp) as src:
                probs = src.read().astype(np.float32) / 255.0 * weight
                all_probs.append(probs)
                if len(all_probs) == 1:
                    profile = src.profile.copy()

        if self.fusion == "mean":
            fused = np.stack(all_probs).mean(0)
        elif self.fusion == "max":
            fused = np.stack(all_probs).max(0)
        else:
            fused = np.stack(all_probs).mean(0)

        label = np.argmax(fused, axis=0).astype(np.uint8)
        profile.update(count=1, dtype="uint8")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(str(output_path), "w", **profile) as dst:
            dst.write(label[np.newaxis])

        return {"success": True, "output_path": str(output_path), "n_models": len(self.models)}
