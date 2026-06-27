"""
PyGeoVision Model Registry.

Central registry for all built-in and user-registered geospatial AI models.
Provides a unified interface for model discovery, instantiation, and metadata.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Metadata entry for a registered model.

    Attributes:
        name: Unique model identifier.
        task: Primary task the model was designed for.
        architecture: Base architecture class name.
        input_bands: Number of expected input bands (-1 = variable).
        num_classes: Default number of output classes (-1 = variable).
        pretrained_available: Whether pretrained weights are available.
        paper_url: Link to the original paper, if applicable.
        description: Human-readable model description.
        tags: Searchable tags (e.g. ['multispectral', 'sentinel-2']).
        factory_fn: Callable that builds the model given kwargs.
    """

    name: str
    task: str
    architecture: str
    input_bands: int = 3
    num_classes: int = -1
    pretrained_available: bool = False
    paper_url: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)
    factory_fn: Optional[Callable[..., Any]] = None


class ModelRegistry:
    """Central registry for PyGeoVision model architectures.

    Supports registration, lookup, and instantiation of geospatial AI models.
    Users can register custom models alongside built-in ones.

    Example:
        >>> from pygeovision.ai.models.registry import registry
        >>> print(registry.list_models(task="segmentation"))
        >>> model = registry.build("unet_resnet50", num_classes=10, in_channels=4)
    """

    def __init__(self) -> None:
        self._models: Dict[str, ModelInfo] = {}

    def register(self, info: ModelInfo) -> None:
        """Register a model with the registry.

        Args:
            info: ModelInfo describing the model.

        Raises:
            ValueError: If a model with the same name is already registered.
        """
        if info.name in self._models:
            logger.warning(
                "Model '%s' is already registered; overwriting.", info.name
            )
        self._models[info.name] = info
        logger.debug("Registered model: %s (task=%s)", info.name, info.task)

    def get(self, name: str) -> ModelInfo:
        """Retrieve model info by name.

        Args:
            name: Model identifier.

        Returns:
            ModelInfo for the requested model.

        Raises:
            KeyError: If the model is not registered.
        """
        if name not in self._models:
            available = ", ".join(sorted(self._models.keys()))
            raise KeyError(
                f"Model '{name}' not found in registry. "
                f"Available models: {available}"
            )
        return self._models[name]

    def build(self, name: str, **kwargs: Any) -> Any:
        """Build a model instance by name.

        Args:
            name: Model identifier.
            **kwargs: Additional arguments passed to the model factory.

        Returns:
            Instantiated model.

        Raises:
            KeyError: If the model is not registered.
            RuntimeError: If the model has no factory function.
        """
        info = self.get(name)
        if info.factory_fn is None:
            raise RuntimeError(
                f"Model '{name}' has no factory_fn. "
                "Register with a factory_fn or use ModelHub.load() instead."
            )
        return info.factory_fn(**kwargs)

    def list_models(
        self,
        task: Optional[str] = None,
        tags: Optional[List[str]] = None,
        pretrained_only: bool = False,
    ) -> List[ModelInfo]:
        """List registered models with optional filtering.

        Args:
            task: Filter by task name (e.g. 'segmentation', 'detection').
            tags: Filter by tags (model must have ALL listed tags).
            pretrained_only: If True, only return models with pretrained weights.

        Returns:
            List of matching ModelInfo objects.
        """
        results = list(self._models.values())

        if task:
            results = [m for m in results if m.task == task]
        if tags:
            results = [m for m in results if all(t in m.tags for t in tags)]
        if pretrained_only:
            results = [m for m in results if m.pretrained_available]

        return sorted(results, key=lambda m: m.name)

    def summary(self) -> str:
        """Return a formatted summary of all registered models.

        Returns:
            Multi-line string table of model names, tasks, and architectures.
        """
        lines = [
            f"{'Name':<35} {'Task':<22} {'Architecture':<30} {'Pretrained'}",
            "-" * 100,
        ]
        for m in sorted(self._models.values(), key=lambda x: (x.task, x.name)):
            pretrained = "✓" if m.pretrained_available else ""
            lines.append(
                f"{m.name:<35} {m.task:<22} {m.architecture:<30} {pretrained}"
            )
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._models)

    def __contains__(self, name: str) -> bool:
        return name in self._models


# -----------------------------------------------------------------------
# Global singleton registry
# -----------------------------------------------------------------------
registry = ModelRegistry()


def _register_builtin_models() -> None:
    """Register all built-in PyGeoVision model architectures."""
    from pygeovision.ai.models.architectures.segmentation import (
        build_unet,
        build_deeplabv3plus,
        build_segformer,
    )
    from pygeovision.ai.models.architectures.detection import (
        build_fcos,
        build_retinanet,
    )
    from pygeovision.ai.models.architectures.classification import (
        build_resnet,
        build_efficientnet,
        build_vit,
    )
    from pygeovision.ai.models.architectures.change_detection import (
        build_siamese_unet,
        build_changeformer,
    )
    from pygeovision.ai.models.architectures.super_resolution import (
        build_srcnn,
        build_esrgan_geo,
    )

    _BUILTIN = [
        # ---- Segmentation ----
        ModelInfo(
            name="unet_resnet50",
            task="segmentation",
            architecture="UNet",
            input_bands=3,
            pretrained_available=True,
            description="U-Net with ResNet-50 encoder. Workhorse for pixel-wise segmentation.",
            tags=["segmentation", "rgb", "encoder-decoder"],
            factory_fn=lambda **kw: build_unet(encoder="resnet50", **kw),
        ),
        ModelInfo(
            name="unet_efficientnet_b4",
            task="segmentation",
            architecture="UNet",
            input_bands=3,
            pretrained_available=True,
            description="U-Net with EfficientNet-B4 encoder. Better accuracy at similar speed.",
            tags=["segmentation", "rgb", "encoder-decoder"],
            factory_fn=lambda **kw: build_unet(encoder="efficientnet-b4", **kw),
        ),
        ModelInfo(
            name="deeplabv3plus_resnet101",
            task="segmentation",
            architecture="DeepLabV3+",
            input_bands=3,
            pretrained_available=True,
            description="DeepLabV3+ with ResNet-101 — strong baseline for land cover mapping.",
            tags=["segmentation", "rgb", "atrous-convolution"],
            factory_fn=lambda **kw: build_deeplabv3plus(encoder="resnet101", **kw),
        ),
        ModelInfo(
            name="segformer_b2",
            task="segmentation",
            architecture="SegFormer",
            input_bands=3,
            pretrained_available=True,
            description="SegFormer-B2 transformer segmentation model.",
            tags=["segmentation", "rgb", "transformer", "attention"],
            factory_fn=lambda **kw: build_segformer(model_size="b2", **kw),
        ),
        ModelInfo(
            name="segformer_b5",
            task="segmentation",
            architecture="SegFormer",
            input_bands=3,
            pretrained_available=True,
            description="SegFormer-B5 — largest SegFormer variant, highest accuracy.",
            tags=["segmentation", "rgb", "transformer", "attention"],
            factory_fn=lambda **kw: build_segformer(model_size="b5", **kw),
        ),
        # ---- Detection ----
        ModelInfo(
            name="fcos_resnet50",
            task="detection",
            architecture="FCOS",
            input_bands=3,
            pretrained_available=True,
            description="FCOS anchor-free object detector with ResNet-50 backbone.",
            tags=["detection", "rgb", "anchor-free"],
            factory_fn=lambda **kw: build_fcos(backbone="resnet50", **kw),
        ),
        ModelInfo(
            name="retinanet_resnet50",
            task="detection",
            architecture="RetinaNet",
            input_bands=3,
            pretrained_available=True,
            description="RetinaNet with focal loss for dense small object detection.",
            tags=["detection", "rgb", "anchor-based"],
            factory_fn=lambda **kw: build_retinanet(backbone="resnet50", **kw),
        ),
        # ---- Classification ----
        ModelInfo(
            name="resnet50_cls",
            task="classification",
            architecture="ResNet-50",
            input_bands=3,
            pretrained_available=True,
            description="ResNet-50 scene classifier.",
            tags=["classification", "rgb", "scene"],
            factory_fn=lambda **kw: build_resnet(depth=50, **kw),
        ),
        ModelInfo(
            name="efficientnet_b3_cls",
            task="classification",
            architecture="EfficientNet-B3",
            input_bands=3,
            pretrained_available=True,
            description="EfficientNet-B3 scene classifier — compact and accurate.",
            tags=["classification", "rgb", "scene"],
            factory_fn=lambda **kw: build_efficientnet(model_size="b3", **kw),
        ),
        ModelInfo(
            name="vit_b16_cls",
            task="classification",
            architecture="ViT-B/16",
            input_bands=3,
            pretrained_available=True,
            description="Vision Transformer (ViT-B/16) scene classifier.",
            tags=["classification", "rgb", "transformer"],
            factory_fn=lambda **kw: build_vit(model_size="b16", **kw),
        ),
        # ---- Change Detection ----
        ModelInfo(
            name="siamese_unet",
            task="change_detection",
            architecture="Siamese-UNet",
            input_bands=3,
            pretrained_available=False,
            description="Siamese U-Net for bi-temporal change detection.",
            tags=["change-detection", "rgb", "siamese"],
            factory_fn=lambda **kw: build_siamese_unet(**kw),
        ),
        ModelInfo(
            name="changeformer",
            task="change_detection",
            architecture="ChangeFormer",
            input_bands=3,
            pretrained_available=True,
            description="Transformer-based change detection (ChangeFormer).",
            tags=["change-detection", "rgb", "transformer"],
            factory_fn=lambda **kw: build_changeformer(**kw),
        ),
        # ---- Super Resolution ----
        ModelInfo(
            name="srcnn",
            task="super_resolution",
            architecture="SRCNN",
            input_bands=-1,
            pretrained_available=False,
            description="Super-Resolution CNN (SRCNN) — lightweight baseline.",
            tags=["super-resolution"],
            factory_fn=lambda **kw: build_srcnn(**kw),
        ),
        ModelInfo(
            name="esrgan_geo",
            task="super_resolution",
            architecture="ESRGAN-Geo",
            input_bands=-1,
            pretrained_available=False,
            description="ESRGAN adapted for satellite imagery super-resolution.",
            tags=["super-resolution", "gan"],
            factory_fn=lambda **kw: build_esrgan_geo(**kw),
        ),
    ]

    for info in _BUILTIN:
        registry.register(info)

    logger.debug("Registered %d built-in PyGeoVision models.", len(_BUILTIN))


# Register built-in models at import time
try:
    _register_builtin_models()
except Exception as _exc:  # pragma: no cover
    logger.warning("Could not register all built-in models: %s", _exc)
