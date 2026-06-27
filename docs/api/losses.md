# Loss Functions

10 geospatial-specific loss functions for segmentation, detection, and classification. All are `torch.nn.Module` subclasses that integrate directly into any PyTorch training loop.

---

## Import

```python
from pygeovision.losses.segmentation import (
    DiceLoss, FocalLoss, TverskyLoss, ComboLoss,
    BoundaryAwareLoss, LovaszLoss, OhemCrossEntropy,
    GeospatialMixedLoss,
)
from pygeovision.losses.detection   import CIoULoss, DIoULoss
from pygeovision.losses.class_balance import ClassBalancedCrossEntropy
```

---

## Segmentation Losses

### `DiceLoss`

Overlap-based loss well-suited for imbalanced classes (e.g. buildings on a large background).

```python
loss_fn = DiceLoss(
    smooth=1.0,             # Laplace smoothing constant
    from_logits=True,       # Input is raw logits (recommended)
    reduction="mean",       # "mean" | "sum" | "none"
)

loss = loss_fn(predictions, targets)   # predictions: (B, C, H, W), targets: (B, H, W)
```

**When to use:** Small object segmentation, class-imbalanced datasets, building/road extraction.

---

### `FocalLoss`

Addresses class imbalance by down-weighting easy (well-classified) examples.

```python
loss_fn = FocalLoss(
    alpha=0.25,     # Class weight (0.25 standard for binary)
    gamma=2.0,      # Focusing parameter (2.0 standard)
    from_logits=True,
)
```

**When to use:** Very sparse foreground (e.g. ships, vehicles in aerial images). Excellent for object detection training.

---

### `TverskyLoss`

Generalisation of Dice — independently control precision and recall via α/β.

```python
loss_fn = TverskyLoss(
    alpha=0.3,    # False positive weight (lower = higher precision)
    beta=0.7,     # False negative weight (higher = higher recall)
    smooth=1.0,
)
```

**When to use:** When missing foreground (false negatives) is more costly than false alarms. Useful for flood detection, damage assessment.

---

### `ComboLoss`

Weighted combination of Dice and Binary Cross-Entropy.

```python
loss_fn = ComboLoss(
    dice_weight=0.5,
    bce_weight=0.5,
    from_logits=True,
)
```

**When to use:** General-purpose default. Balances boundary accuracy (BCE) with region overlap (Dice).

---

### `BoundaryAwareLoss`

Adds extra weight to pixels near class boundaries — improves edge sharpness.

```python
loss_fn = BoundaryAwareLoss(
    base_loss="combo",        # "dice" | "focal" | "combo"
    boundary_weight=5.0,      # Weight multiplier at boundaries
    boundary_dilation=3,      # Dilation radius in pixels
)
```

**When to use:** Building footprint extraction, road segmentation, land-cover mapping where crisp boundaries matter.

---

### `LovaszLoss`

Directly optimises the Jaccard index (IoU) via a convex surrogate.

```python
from pygeovision.losses.segmentation import LovaszLoss

loss_fn = LovaszLoss(
    per_image=True,
    ignore_index=255,
)
```

**When to use:** When validation metric is mIoU — this loss directly optimises what you measure.

---

### `OhemCrossEntropy`

Online Hard Example Mining — focuses training on the hardest pixels.

```python
loss_fn = OhemCrossEntropy(
    threshold=0.7,         # Only mine pixels with CE loss > this value
    min_kept=100000,       # Minimum pixels to keep per batch
    ignore_index=255,
)
```

**When to use:** Complex multi-class scenes (urban land cover, vegetation types) where easy classes dominate.

---

### `GeospatialMixedLoss`

Weighted combination of any subset of the above losses — the recommended default for training from scratch.

```python
loss_fn = GeospatialMixedLoss(
    weights={
        "combo":    0.5,
        "boundary": 0.3,
        "ohem":     0.2,
    }
)

# Or use the preset for building extraction
loss_fn = GeospatialMixedLoss.building_extraction()

# Preset for land cover mapping
loss_fn = GeospatialMixedLoss.land_cover()
```

---

## Detection Losses

### `CIoULoss`

Complete Intersection-over-Union — accounts for overlap, distance, and aspect ratio.

```python
from pygeovision.losses.detection import CIoULoss

loss_fn = CIoULoss()
loss = loss_fn(pred_boxes, target_boxes)  # (N, 4) xyxy format
```

### `DIoULoss`

Distance-IoU — adds centre-point distance to IoU.

```python
from pygeovision.losses.detection import DIoULoss
loss_fn = DIoULoss()
```

---

## Class Imbalance

### `ClassBalancedCrossEntropy`

Inverse-frequency class weighting for highly imbalanced multi-class problems.

```python
from pygeovision.losses.class_balance import ClassBalancedCrossEntropy

loss_fn = ClassBalancedCrossEntropy(
    num_classes=11,
    beta=0.9999,           # Smoothing (0 = uniform, 1 = full inverse freq)
    samples_per_class=None,  # Auto-compute from dataset if None
)
```

---

## Loss Selection Guide

| Scenario | Recommended Loss |
|----------|-----------------|
| Building / road extraction | `BoundaryAwareLoss` or `GeospatialMixedLoss.building_extraction()` |
| Flood / damage detection | `TverskyLoss(alpha=0.3, beta=0.7)` |
| Multi-class land cover | `GeospatialMixedLoss.land_cover()` |
| Ship / vehicle detection | `FocalLoss` |
| Training from scratch | `ComboLoss` → switch to `LovaszLoss` after warmup |
| Very sparse objects | `FocalLoss(alpha=0.1, gamma=3.0)` |

---

## GeoTrainer Integration

```python
from pygeovision.training.trainer    import GeoTrainer
from pygeovision.losses.segmentation import GeospatialMixedLoss

trainer = GeoTrainer(
    model=model,
    loss_fn=GeospatialMixedLoss(weights={"combo": 0.5, "boundary": 0.5}),
    num_classes=7,
)
trainer.fit(train_dl, val_dl)
```
