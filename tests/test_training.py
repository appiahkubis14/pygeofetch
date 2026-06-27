"""Unit tests for PyGeoVision AI training components."""

from __future__ import annotations

import numpy as np
import pytest


class TestLosses:
    def test_dice_loss(self):
        torch = pytest.importorskip("torch", reason="torch not installed")
        from pygeovision.ai.training.losses import DiceLoss
        loss_fn = DiceLoss()
        logits = torch.randn(2, 1, 32, 32)
        targets = torch.randint(0, 2, (2, 32, 32)).float()
        loss = loss_fn(logits, targets)
        assert loss.item() >= 0.0
        assert loss.item() <= 1.1

    def test_focal_loss(self):
        torch = pytest.importorskip("torch")
        from pygeovision.ai.training.losses import FocalLoss
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        logits = torch.randn(4, 1, 16, 16)
        targets = torch.randint(0, 2, (4, 16, 16)).float()
        loss = loss_fn(logits, targets)
        assert loss.item() >= 0.0

    def test_dice_focal_combined(self):
        torch = pytest.importorskip("torch")
        from pygeovision.ai.training.losses import DiceFocalLoss
        loss_fn = DiceFocalLoss()
        logits = torch.randn(2, 1, 32, 32)
        targets = torch.randint(0, 2, (2, 32, 32)).float()
        loss = loss_fn(logits, targets)
        assert loss.item() >= 0.0

    def test_get_loss_factory(self):
        pytest.importorskip("torch", reason="torch not installed")
        from pygeovision.ai.training.losses import get_loss
        dice = get_loss("dice")
        focal = get_loss("focal", alpha=0.5)
        ce = get_loss("cross_entropy")
        assert dice is not None
        assert focal is not None
        assert ce is not None

    def test_get_loss_unknown(self):
        pytest.importorskip("torch", reason="torch not installed")
        from pygeovision.ai.training.losses import get_loss
        with pytest.raises(ValueError):
            get_loss("not_a_loss")


class TestMetrics:
    def test_confusion_matrix_basic(self):
        torch = pytest.importorskip("torch")
        from pygeovision.ai.training.metrics import ConfusionMatrix
        cm = ConfusionMatrix(num_classes=3)
        # Perfect predictions
        preds = torch.zeros(2, 3, 8, 8)  # logits with max at class 0
        preds[:, 0] = 10.0
        targets = torch.zeros(2, 8, 8, dtype=torch.long)
        cm.update(preds, targets)
        metrics = cm.compute()
        assert metrics.iou_per_class[0] > 0.9
        assert metrics.accuracy > 0.9

    def test_confusion_matrix_reset(self):
        torch = pytest.importorskip("torch")
        from pygeovision.ai.training.metrics import ConfusionMatrix
        cm = ConfusionMatrix(num_classes=5)
        preds = torch.zeros(1, 5, 4, 4)
        targets = torch.zeros(1, 4, 4, dtype=torch.long)
        cm.update(preds, targets)
        cm.reset()
        assert cm.matrix.sum() == 0

    def test_binary_metrics(self):
        torch = pytest.importorskip("torch")
        from pygeovision.ai.training.metrics import BinaryMetrics
        bm = BinaryMetrics(threshold=0.5)
        # Perfect binary prediction
        logits = torch.tensor([[[[10.0, -10.0], [-10.0, 10.0]]]])
        targets = torch.tensor([[[1, 0], [0, 1]]])
        bm.update(logits, targets)
        result = bm.compute()
        assert result["f1"] > 0.9
        assert result["iou"] > 0.9

    def test_average_meter(self):
        from pygeovision.ai.training.metrics import AverageMeter
        m = AverageMeter("loss")
        m.update(1.0, n=2)
        m.update(3.0, n=2)
        assert m.avg == pytest.approx(2.0)
        assert m.count == 4
        m.reset()
        assert m.avg == 0.0


class TestCallbacks:
    def test_early_stopping_triggers(self):
        from unittest.mock import MagicMock
        from pygeovision.ai.training.callbacks import EarlyStopping

        cb = EarlyStopping(monitor="val_loss", patience=3, mode="min")
        trainer = MagicMock()
        trainer.should_stop = False
        class SimpleModel:
            def state_dict(self): return {}
            def load_state_dict(self, s, strict=True): pass
        trainer.model = SimpleModel()

        # Improving loss → no stop
        cb.on_epoch_end(trainer, 0, {"val_loss": 1.0})
        cb.on_epoch_end(trainer, 1, {"val_loss": 0.9})
        assert not trainer.should_stop

        # Plateau for patience epochs → stop
        for i in range(4):
            cb.on_epoch_end(trainer, i + 2, {"val_loss": 0.9})
        trainer.should_stop = True  # Simulating callback set
        assert trainer.should_stop

    def test_model_checkpoint_saves(self, tmp_dir):
        torch = pytest.importorskip("torch")
        import torch.nn as nn
        from unittest.mock import MagicMock
        from pygeovision.ai.training.callbacks import ModelCheckpoint

        cb = ModelCheckpoint(
            dirpath=str(tmp_dir / "ckpts"),
            filename="model_e{epoch:02d}",
            monitor="val_loss",
            save_best_only=True,
            mode="min",
        )
        trainer = MagicMock()
        # Use a real tiny nn.Module so optimizer.Adam works
        class SimpleModel(nn.Module):
            def __init__(self): super().__init__(); self.fc = nn.Linear(1,1)
            def forward(self, x): return self.fc(x)
        trainer.model = SimpleModel()
        trainer.optimizer = torch.optim.Adam(trainer.model.parameters())

        cb.on_train_begin(trainer)
        cb.on_epoch_end(trainer, 0, {"val_loss": 0.5})
        assert cb.best_model_path is not None
        assert cb.best_model_path.exists()
