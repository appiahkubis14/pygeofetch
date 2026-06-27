"""Base classes for all PyGeoVision models."""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
logger = logging.getLogger(__name__)


@dataclass
class GeoModelConfig:
    """Configuration for any PyGeoVision model."""
    name: str
    task: str                        # segmentation|detection|classification|change|foundation
    backbone: str = "resnet50"
    num_classes: int = 2
    in_channels: int = 4             # Multispectral default
    pretrained: bool = True
    pretrained_source: str = "imagenet"   # imagenet|sentinel2|prithvi|none
    input_size: Tuple[int, int] = (512, 512)
    freeze_backbone: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}


class GeoModel:
    """Thin wrapper around any PyTorch model with geospatial utilities."""

    def __init__(self, model: Any, config: GeoModelConfig) -> None:
        self.model = model
        self.config = config

    def to(self, device: str) -> "GeoModel":
        self.model = self.model.to(device)
        return self

    def eval(self) -> "GeoModel":
        self.model.eval()
        return self

    def train(self) -> "GeoModel":
        self.model.train()
        return self

    def parameters(self):
        return self.model.parameters()

    def named_parameters(self):
        return self.model.named_parameters()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict: Dict, strict: bool = True) -> None:
        self.model.load_state_dict(state_dict, strict=strict)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def export_onnx(self, output_path: str, input_shape: Tuple = (1, 4, 512, 512)) -> str:
        """Export model to ONNX."""
        try:
            import torch
            self.model.eval()
            dummy = torch.randn(*input_shape)
            torch.onnx.export(self.model, dummy, output_path,
                               opset_version=17, input_names=["input"],
                               output_names=["output"],
                               dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
            logger.info("ONNX exported: %s", output_path)
            return output_path
        except Exception as exc:
            raise RuntimeError(f"ONNX export failed: {exc}")

    def __repr__(self) -> str:
        try:
            import torch.nn as nn
            n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        except Exception:
            n_params = 0
        return (f"GeoModel(name={self.config.name}, task={self.config.task}, "
                f"classes={self.config.num_classes}, params={n_params:.1f}M)")
