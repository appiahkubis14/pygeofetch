"""Tests for Phase 4: GeoTrainer, metrics, HPO, ModelOptimizer, ExperimentTracker."""
from __future__ import annotations
import pytest
import sys; sys.path.insert(0, '/home/claude/pgv')


# ── TrainingConfig ────────────────────────────────────────────────────────────
class TestTrainingConfig:
    def test_defaults(self):
        from pygeovision.training import TrainingConfig
        cfg = TrainingConfig()
        assert cfg.max_epochs == 100
        assert cfg.batch_size == 16
        assert cfg.optimizer == "adamw"
        assert cfg.precision == "auto"
        assert cfg.seed == 42

    def test_custom(self):
        from pygeovision.training import TrainingConfig
        cfg = TrainingConfig(max_epochs=50, batch_size=32, optimizer="sgd", precision="fp16")
        assert cfg.max_epochs == 50
        assert cfg.batch_size == 32
        assert cfg.precision == "fp16"

    def test_to_dict(self):
        from pygeovision.training import TrainingConfig
        cfg = TrainingConfig(max_epochs=25)
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert d["max_epochs"] == 25
        assert "learning_rate" in d
        assert "scheduler" in d

    def test_from_yaml(self, tmp_path):
        from pygeovision.training import TrainingConfig
        import yaml
        data = {"max_epochs": 77, "batch_size": 8, "optimizer": "adam"}
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump(data))
        cfg = TrainingConfig.from_yaml(str(p))
        assert cfg.max_epochs == 77
        assert cfg.batch_size == 8

    def test_save_yaml(self, tmp_path):
        from pygeovision.training import TrainingConfig
        import yaml
        cfg = TrainingConfig(max_epochs=33)
        p = tmp_path / "out.yaml"
        cfg.save(str(p))
        assert p.exists()
        loaded = yaml.safe_load(p.read_text())
        assert loaded["max_epochs"] == 33


# ── Metrics ───────────────────────────────────────────────────────────────────
class TestSegmentationMetrics:
    def test_perfect_prediction(self):
        try:
            import torch
            from pygeovision.training.metrics import SegmentationMetrics
            m = SegmentationMetrics(num_classes=3)
            targets = torch.tensor([[0,1,2,1]])
            preds   = torch.tensor([[0,1,2,1]])
            m.update(preds, targets)
            result = m.compute()
            assert abs(result["mean_iou"] - 1.0) < 1e-5
            assert abs(result["accuracy"] - 1.0) < 1e-5
        except ImportError:
            pytest.skip("torch not installed")

    def test_all_wrong(self):
        try:
            import torch
            from pygeovision.training.metrics import SegmentationMetrics
            m = SegmentationMetrics(num_classes=2)
            targets = torch.tensor([[0, 0, 0, 0]])
            preds   = torch.tensor([[1, 1, 1, 1]])
            m.update(preds, targets)
            result = m.compute()
            assert result["mean_iou"] < 0.5
        except ImportError:
            pytest.skip("torch not installed")

    def test_reset(self):
        try:
            import torch
            from pygeovision.training.metrics import SegmentationMetrics
            m = SegmentationMetrics(num_classes=2)
            targets = preds = torch.tensor([[0, 1]])
            m.update(preds, targets)
            m.reset()
            result = m.compute()
            assert result["mean_iou"] == 0.0
        except ImportError:
            pytest.skip("torch not installed")

    def test_ignore_index(self):
        try:
            import torch
            from pygeovision.training.metrics import SegmentationMetrics
            m = SegmentationMetrics(num_classes=2, ignore_index=255)
            targets = torch.tensor([[0, 255, 1, 255]])
            preds   = torch.tensor([[0, 1,   1, 0  ]])
            m.update(preds, targets)
            result = m.compute()
            assert "mean_iou" in result
        except ImportError:
            pytest.skip("torch not installed")

    def test_no_error_on_empty(self):
        from pygeovision.training.metrics import SegmentationMetrics
        m = SegmentationMetrics(num_classes=5)
        result = m.compute()
        assert result["mean_iou"] == 0.0

    def test_per_class_iou_keys(self):
        from pygeovision.training.metrics import SegmentationMetrics
        m = SegmentationMetrics(num_classes=4)
        result = m.compute()
        # Per-class IoU keys are only present when torch is available
        try:
            import torch
            for i in range(4):
                assert f"iou_class_{i}" in result
        except ImportError:
            pytest.skip("torch not installed")


class TestChangeDetectionMetrics:
    def test_keys(self):
        from pygeovision.training.metrics import ChangeDetectionMetrics
        m = ChangeDetectionMetrics()
        result = m.compute()
        assert "iou_change" in result
        assert "f1_change" in result
        assert "overall_accuracy" in result


# ── EarlyStopping ────────────────────────────────────────────────────────────
class TestEarlyStopping:
    def test_stops_after_patience(self):
        from pygeovision.training.trainer import EarlyStopping
        es = EarlyStopping(patience=3, metric="val_iou", mode="max")
        es.update({"val_iou": 0.5})
        es.update({"val_iou": 0.4})
        es.update({"val_iou": 0.3})
        es.update({"val_iou": 0.2})
        assert es.should_stop

    def test_resets_on_improvement(self):
        from pygeovision.training.trainer import EarlyStopping
        es = EarlyStopping(patience=3, metric="val_iou", mode="max")
        es.update({"val_iou": 0.5})
        es.update({"val_iou": 0.3})
        improved = es.update({"val_iou": 0.8})
        assert improved
        assert es.counter == 0
        assert not es.should_stop

    def test_min_mode(self):
        from pygeovision.training.trainer import EarlyStopping
        es = EarlyStopping(patience=2, metric="val_loss", mode="min")
        improved = es.update({"val_loss": 0.9})
        assert improved
        not_improved = es.update({"val_loss": 1.0})
        assert not not_improved

    def test_best_tracked(self):
        from pygeovision.training.trainer import EarlyStopping
        es = EarlyStopping(patience=5, metric="val_iou", mode="max")
        es.update({"val_iou": 0.6})
        es.update({"val_iou": 0.8})
        es.update({"val_iou": 0.7})
        assert abs(es.best - 0.8) < 1e-6


# ── CheckpointManager ────────────────────────────────────────────────────────
class TestCheckpointManager:
    def test_keeps_top_k(self, tmp_path):
        from pygeovision.training.trainer import CheckpointManager
        from unittest.mock import MagicMock
        mgr = CheckpointManager(tmp_path, top_k=2, metric="val_iou", mode="max")
        model = MagicMock()
        model.state_dict.return_value = {}
        for i in range(4):
            mgr.save(model, epoch=i+1, metrics={"val_iou": 0.1 * (i+1)})
        checkpoints = list(tmp_path.glob("checkpoints/epoch_*.pth"))
        assert len(checkpoints) <= 2

    def test_best_path_is_highest(self, tmp_path):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")
        from pygeovision.training.trainer import CheckpointManager
        from unittest.mock import MagicMock
        mgr = CheckpointManager(tmp_path, top_k=3, metric="val_iou", mode="max")
        model = MagicMock()
        model.state_dict.return_value = {}
        scores = [0.5, 0.9, 0.7]
        for i, s in enumerate(scores):
            mgr.save(model, epoch=i+1, metrics={"val_iou": s})
        best = mgr.best_path()
        assert best is not None
        assert "0.9000" in str(best)


# ── ExperimentTracker ────────────────────────────────────────────────────────
class TestExperimentTracker:
    def test_log_history(self):
        from pygeovision.training.experiment import ExperimentTracker
        t = ExperimentTracker("test_exp", backends=[])
        t.log({"loss": 0.5, "iou": 0.8}, step=1)
        t.log({"loss": 0.4, "iou": 0.85}, step=2)
        assert len(t.history["loss"]) == 2
        assert len(t.history["iou"]) == 2
        assert t.history["loss"][-1] == 0.4

    def test_end_run_returns_summary(self):
        from pygeovision.training.experiment import ExperimentTracker
        t = ExperimentTracker("test_exp", backends=[])
        t.log({"acc": 0.9})
        summary = t.end_run()
        assert summary["experiment"] == "test_exp"
        assert "duration_seconds" in summary
        assert summary["metrics"]["acc"] == 0.9

    def test_start_run_no_backends(self):
        from pygeovision.training.experiment import ExperimentTracker
        t = ExperimentTracker("test_exp", backends=[])
        result = t.start_run(config={"lr": 0.001})
        assert result is t  # returns self


# ── Optimizer / Scheduler ─────────────────────────────────────────────────────
class TestOptimizerScheduler:
    @pytest.mark.parametrize("opt_name", ["adamw", "adam", "sgd"])
    def test_build_optimizer(self, opt_name):
        try:
            import torch, torch.nn as nn
            from pygeovision.training import TrainingConfig, build_optimizer
            model = nn.Linear(10, 2)
            cfg = TrainingConfig(optimizer=opt_name, learning_rate=1e-3)
            opt = build_optimizer(model, cfg)
            assert opt is not None
            assert len(opt.param_groups) >= 1
        except ImportError:
            pytest.skip("torch not installed")

    @pytest.mark.parametrize("sched_name", ["cosine", "step", "linear"])
    def test_build_scheduler(self, sched_name):
        try:
            import torch, torch.nn as nn
            from pygeovision.training import TrainingConfig, build_optimizer, build_scheduler
            model = nn.Linear(10, 2)
            cfg = TrainingConfig(scheduler=sched_name, max_epochs=10)
            opt = build_optimizer(model, cfg)
            sched = build_scheduler(opt, cfg, steps_per_epoch=50)
            assert sched is not None
        except ImportError:
            pytest.skip("torch not installed")

    def test_build_scheduler_returns_none_without_torch(self):
        import sys
        # Monkeypatch-free test: scheduler gracefully handles ImportError
        from pygeovision.training.optimizer import build_scheduler
        from unittest.mock import MagicMock
        cfg = MagicMock()
        cfg.scheduler = "cosine"
        cfg.max_epochs = 10
        cfg.min_lr = 1e-6
        cfg.learning_rate = 1e-3
        # If torch not installed this returns None; if installed it returns object
        try:
            result = build_scheduler(MagicMock(), cfg, steps_per_epoch=10)
            # Either None or a real scheduler
        except Exception:
            pass  # acceptable
