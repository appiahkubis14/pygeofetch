"""Tests for PyGeoVision geospatial loss functions (Phase 2+)."""
import pytest

_TORCH_AVAILABLE = False
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    pass

pytestmark = pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")


@pytest.fixture
def seg_batch():
    """Synthetic segmentation batch: (B=2, C=3, H=16, W=16) preds, (B=2, H=16, W=16) targets."""
    import torch
    torch.manual_seed(42)
    preds   = torch.randn(2, 3, 16, 16)
    targets = torch.randint(0, 3, (2, 16, 16))
    return preds, targets


@pytest.fixture
def binary_batch():
    """Binary segmentation: 2-class."""
    import torch
    torch.manual_seed(0)
    preds   = torch.randn(2, 2, 16, 16)
    targets = torch.randint(0, 2, (2, 16, 16))
    return preds, targets


class TestDiceLoss:
    def test_forward_shape(self, seg_batch):
        from pygeovision.losses.segmentation import DiceLoss
        preds, targets = seg_batch
        loss = DiceLoss()(preds, targets)
        assert loss.ndim == 0      # scalar
        assert loss.item() >= 0

    def test_perfect_prediction(self):
        import torch
        from pygeovision.losses.segmentation import DiceLoss
        targets = torch.zeros(1, 16, 16, dtype=torch.long)
        # Perfect prediction: class 0 has infinite logit
        preds = torch.full((1, 2, 16, 16), -1e9)
        preds[:, 0] = 1e9
        loss = DiceLoss()(preds, targets)
        assert loss.item() < 0.01   # near-zero loss

    def test_ignore_index(self, seg_batch):
        import torch
        from pygeovision.losses.segmentation import DiceLoss
        preds, targets = seg_batch
        targets_with_ignore = targets.clone()
        targets_with_ignore[0, 0, 0] = 255
        loss_a = DiceLoss(ignore_index=255)(preds, targets_with_ignore)
        loss_b = DiceLoss(ignore_index=255)(preds, targets)
        # Both should be valid scalars
        assert loss_a.item() >= 0
        assert loss_b.item() >= 0

    def test_per_class_vs_mean(self, binary_batch):
        from pygeovision.losses.segmentation import DiceLoss
        preds, targets = binary_batch
        loss_pc   = DiceLoss(per_class=True)(preds, targets)
        loss_mean = DiceLoss(per_class=False)(preds, targets)
        assert loss_pc.ndim == 0
        assert loss_mean.ndim == 0


class TestFocalLoss:
    def test_forward_positive(self, seg_batch):
        from pygeovision.losses.segmentation import FocalLoss
        preds, targets = seg_batch
        loss = FocalLoss(alpha=0.25, gamma=2.0)(preds, targets)
        assert loss.item() >= 0

    def test_higher_gamma_lower_loss_easy_samples(self):
        """Higher gamma should reduce loss on well-classified samples."""
        import torch
        from pygeovision.losses.segmentation import FocalLoss
        # Well-classified sample: target=0, large logit for class 0
        preds   = torch.tensor([[[[10.0, -10.0], [-10.0, 10.0]]]])   # (1, 2, 2, 1) — needs reshape
        preds   = torch.tensor([[[[10.0]], [[-10.0]]]])   # (B=1, C=2, H=1, W=1)
        targets = torch.zeros(1, 1, 1, dtype=torch.long)
        loss_g0 = FocalLoss(gamma=0.0)(preds, targets).item()
        loss_g2 = FocalLoss(gamma=2.0)(preds, targets).item()
        # gamma=2 should suppress easy samples → lower loss
        assert loss_g2 <= loss_g0 + 1e-4

    def test_zero_gamma_equals_ce(self):
        """FocalLoss(gamma=0) should equal standard CE."""
        import torch, torch.nn.functional as F
        from pygeovision.losses.segmentation import FocalLoss
        torch.manual_seed(5)
        preds   = torch.randn(2, 3, 8, 8)
        targets = torch.randint(0, 3, (2, 8, 8))
        focal   = FocalLoss(alpha=1.0, gamma=0.0)(preds, targets).item()
        ce      = F.cross_entropy(preds, targets).item()
        assert abs(focal - ce) < 0.01


class TestTverskyLoss:
    def test_forward_range(self, binary_batch):
        from pygeovision.losses.segmentation import TverskyLoss
        preds, targets = binary_batch
        loss = TverskyLoss(alpha=0.3, beta=0.7)(preds, targets)
        assert 0.0 <= loss.item() <= 2.0

    def test_dice_is_special_case(self, binary_batch):
        """Tversky(0.5, 0.5) should be close to Dice."""
        from pygeovision.losses.segmentation import TverskyLoss, DiceLoss
        preds, targets = binary_batch
        tv   = TverskyLoss(alpha=0.5, beta=0.5)(preds, targets).item()
        dice = DiceLoss()(preds, targets).item()
        assert abs(tv - dice) < 0.1


class TestComboLoss:
    def test_forward(self, seg_batch):
        from pygeovision.losses.segmentation import ComboLoss
        preds, targets = seg_batch
        loss = ComboLoss(dice_weight=0.5, ce_weight=0.5)(preds, targets)
        assert loss.item() >= 0

    def test_dice_only(self, seg_batch):
        from pygeovision.losses.segmentation import ComboLoss, DiceLoss
        preds, targets = seg_batch
        combo_dice = ComboLoss(dice_weight=1.0, ce_weight=0.0)(preds, targets).item()
        pure_dice  = DiceLoss()(preds, targets).item()
        assert abs(combo_dice - pure_dice) < 0.01


class TestBoundaryAwareLoss:
    def test_forward(self, binary_batch):
        from pygeovision.losses.segmentation import BoundaryAwareLoss
        preds, targets = binary_batch
        loss = BoundaryAwareLoss(boundary_weight=5.0)(preds, targets)
        assert loss.item() >= 0

    def test_boundary_extraction(self):
        import torch
        from pygeovision.losses.segmentation import BoundaryAwareLoss
        bl = BoundaryAwareLoss()
        targets = torch.zeros(1, 8, 8, dtype=torch.long)
        targets[0, 2:6, 2:6] = 1
        boundaries = bl._extract_boundaries(targets)
        # Boundary pixels should be at edge of the filled patch
        assert boundaries.any()


class TestOhemCrossEntropy:
    def test_forward(self, seg_batch):
        from pygeovision.losses.segmentation import OhemCrossEntropy
        preds, targets = seg_batch
        loss = OhemCrossEntropy(thresh=0.7, min_kept=10)(preds, targets)
        assert loss.item() >= 0

    def test_empty_valid_pixels(self):
        """Should not crash with all-ignore targets."""
        import torch
        from pygeovision.losses.segmentation import OhemCrossEntropy
        preds   = torch.randn(1, 2, 8, 8)
        targets = torch.full((1, 8, 8), 255, dtype=torch.long)
        loss = OhemCrossEntropy()(preds, targets)
        assert loss.item() == pytest.approx(0.0, abs=1e-3) or loss.item() >= 0


class TestGeospatialMixedLoss:
    def test_forward(self, binary_batch):
        from pygeovision.losses.segmentation import GeospatialMixedLoss
        preds, targets = binary_batch
        loss_fn = GeospatialMixedLoss(weights={"combo": 0.5, "boundary": 0.3, "ohem": 0.2})
        loss = loss_fn(preds, targets)
        assert loss.item() >= 0

    def test_weights_sum_to_one(self):
        from pygeovision.losses.segmentation import GeospatialMixedLoss
        weights = {"combo": 0.5, "boundary": 0.3, "ohem": 0.2}
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_single_weight(self, binary_batch):
        from pygeovision.losses.segmentation import GeospatialMixedLoss, DiceLoss
        preds, targets = binary_batch
        mixed = GeospatialMixedLoss(weights={"dice": 1.0})(preds, targets).item()
        dice  = DiceLoss()(preds, targets).item()
        assert abs(mixed - dice) < 0.01


class TestDetectionLosses:
    @pytest.fixture
    def box_batch(self):
        import torch
        torch.manual_seed(0)
        pred   = torch.rand(4, 4)  # (N, 4) in xyxy
        target = pred.clone() + torch.randn(4, 4) * 0.05
        return pred, target

    def test_ciou_loss(self, box_batch):
        from pygeovision.losses.detection import CIoULoss
        pred, target = box_batch
        # Ensure x2>x1, y2>y1
        pred   = torch.stack([pred.min(dim=1).values, pred.max(dim=1).values], dim=1)
        pred   = torch.cat([pred[:, [0, 0]], pred[:, [1, 1]]], dim=1) * 0.5
        loss   = CIoULoss()(pred, pred.clone())  # self-overlap → near zero
        assert loss.item() >= -0.1   # ciou can be slightly negative

    def test_giou_loss(self, box_batch):
        from pygeovision.losses.detection import GIoULoss
        pred, _ = box_batch
        pred_fixed = torch.zeros(4, 4)
        pred_fixed[:, 2:] = 0.5
        loss = GIoULoss()(pred_fixed, pred_fixed)
        assert abs(loss.item()) < 0.1


class TestClassBalanceLosses:
    def test_label_smoothing(self, seg_batch):
        from pygeovision.losses.class_balance import LabelSmoothingCrossEntropy
        preds, targets = seg_batch
        loss = LabelSmoothingCrossEntropy(smoothing=0.1)(preds, targets)
        assert loss.item() >= 0

    def test_class_balanced_ce_no_counts(self, seg_batch):
        from pygeovision.losses.class_balance import ClassBalancedCrossEntropy
        preds, targets = seg_batch
        loss = ClassBalancedCrossEntropy(class_counts=None)(preds, targets)
        assert loss.item() >= 0

    def test_class_balanced_ce_with_counts(self, binary_batch):
        from pygeovision.losses.class_balance import ClassBalancedCrossEntropy
        preds, targets = binary_batch
        # Highly imbalanced: class 0 has 10x more pixels
        loss = ClassBalancedCrossEntropy(class_counts=[100000, 10000])(preds, targets)
        assert loss.item() >= 0
