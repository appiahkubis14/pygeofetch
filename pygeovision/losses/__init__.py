"""
PyGeoVision Geospatial Loss Functions (D1) — beyond generic cross-entropy.
All losses are independent of GeoAI, implemented in pure PyTorch.
"""
from pygeovision.losses.segmentation import (
    DiceLoss, FocalLoss, TverskyLoss, ComboLoss,
    BoundaryAwareLoss, LovaszLoss, OhemCrossEntropy,
    GeospatialMixedLoss,
)
from pygeovision.losses.detection import (
    CIoULoss, DIoULoss, GIoULoss, SIoULoss,
)
from pygeovision.losses.class_balance import (
    ClassBalancedCrossEntropy, LabelSmoothingCrossEntropy,
    FocalCrossEntropy,
)

__all__ = [
    # Segmentation
    "DiceLoss", "FocalLoss", "TverskyLoss", "ComboLoss",
    "BoundaryAwareLoss", "LovaszLoss", "OhemCrossEntropy",
    "GeospatialMixedLoss",
    # Detection
    "CIoULoss", "DIoULoss", "GIoULoss", "SIoULoss",
    # Class balance
    "ClassBalancedCrossEntropy", "LabelSmoothingCrossEntropy", "FocalCrossEntropy",
]
