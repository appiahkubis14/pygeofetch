"""Detection model architectures."""
from __future__ import annotations
import logging
from typing import Any, Optional
logger = logging.getLogger(__name__)


def build_fcos(backbone: str = "resnet50", num_classes: int = 2, in_channels: int = 3, pretrained: bool = True, **kwargs: Any) -> Any:
    """Build an FCOS anchor-free object detector."""
    try:
        import torchvision.models.detection as det
        if backbone == "resnet50" and in_channels == 3:
            weights = "DEFAULT" if pretrained else None
            model = det.fcos_resnet50_fpn(weights=weights, num_classes=num_classes, **kwargs)
            logger.info("Built FCOS(backbone=%s, num_classes=%d)", backbone, num_classes)
            return model
    except Exception:
        pass

    try:
        from mmdet.models import build_detector  # type: ignore
        cfg = dict(type="FCOS", backbone=dict(type="ResNet", depth=50), num_classes=num_classes)
        return build_detector(cfg)
    except ImportError:
        pass

    raise ImportError(
        "FCOS requires torchvision>=0.12 or mmdetection. "
        "Install: pip install torchvision"
    )


def build_retinanet(backbone: str = "resnet50", num_classes: int = 2, in_channels: int = 3, pretrained: bool = True, **kwargs: Any) -> Any:
    """Build a RetinaNet object detector with focal loss."""
    try:
        import torchvision.models.detection as det
        weights = "DEFAULT" if (pretrained and in_channels == 3) else None
        model = det.retinanet_resnet50_fpn(weights=weights, num_classes=num_classes, **kwargs)
        logger.info("Built RetinaNet(backbone=%s, num_classes=%d)", backbone, num_classes)
        return model
    except Exception as exc:
        raise ImportError(
            "RetinaNet requires torchvision. Install: pip install torchvision"
        ) from exc
