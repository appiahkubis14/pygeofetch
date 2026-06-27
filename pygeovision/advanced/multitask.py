"""
Multi-task learning for geospatial models (G5).
Jointly optimise segmentation + detection + classification from one backbone.
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Tuple
logger = logging.getLogger(__name__)


class MultiTaskLearner:
    """Joint multi-task training with shared geospatial backbone (G5).

    Learns to simultaneously perform:
        - Semantic segmentation (e.g. land cover)
        - Object detection (e.g. buildings, cars)
        - Scene classification (e.g. urban/rural/forest)

    Using a shared encoder reduces total parameters and leverages
    shared geospatial representations across tasks.

    Example::

        learner = MultiTaskLearner(
            backbone="resnet50",
            tasks=["segmentation", "detection", "classification"],
            n_classes={"segmentation": 8, "detection": 3, "classification": 5}
        )
        model = learner.build()
        results = learner.train(train_loader, val_loader, epochs=100)
    """

    TASK_HEADS = {
        "segmentation": "DeepLabV3PlusHead",
        "detection":    "FPNDetectionHead",
        "classification": "GlobalAvgPoolHead",
        "change_detection": "SiameseChangeHead",
    }

    def __init__(
        self,
        backbone: str = "resnet50",
        tasks: Optional[List[str]] = None,
        n_classes: Optional[Dict[str, int]] = None,
        task_weights: Optional[Dict[str, float]] = None,
        pretrained: bool = True,
        device: Optional[str] = None,
    ) -> None:
        self.backbone = backbone
        self.tasks = tasks or ["segmentation", "classification"]
        self.n_classes = n_classes or {"segmentation": 2, "classification": 4}
        self.task_weights = task_weights or {t: 1.0 / len(self.tasks) for t in self.tasks}
        self.pretrained = pretrained
        self.device = device or self._auto_device()
        self._model = None

    @staticmethod
    def _auto_device():
        try:
            import torch; return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError: return "cpu"

    def build(self) -> Any:
        """Build the multi-task model with shared backbone and task-specific heads."""
        try:
            import torch.nn as nn
            import torchvision.models as models
        except ImportError:
            raise ImportError("torch + torchvision required")

        class MultiTaskModel(nn.Module):
            def __init__(self, backbone_name, tasks, n_classes):
                super().__init__()
                # Shared encoder
                backbone_map = {
                    "resnet50":  (models.resnet50,  2048),
                    "resnet101": (models.resnet101, 2048),
                    "vgg16":     (models.vgg16,     512),
                }
                backbone_fn, feat_dim = backbone_map.get(backbone_name, (models.resnet50, 2048))
                base = backbone_fn(pretrained=True)
                self.encoder = nn.Sequential(*list(base.children())[:-2])
                self.feat_dim = feat_dim
                self.tasks = tasks
                self.heads = nn.ModuleDict()

                for task in tasks:
                    nc = n_classes.get(task, 2)
                    if task == "segmentation":
                        self.heads[task] = nn.Sequential(
                            nn.ConvTranspose2d(feat_dim, 256, 4, stride=4),
                            nn.ReLU(), nn.BatchNorm2d(256),
                            nn.ConvTranspose2d(256, 64, 4, stride=4),
                            nn.ReLU(), nn.BatchNorm2d(64),
                            nn.Conv2d(64, nc, 1),
                        )
                    elif task in ("classification", "change_detection"):
                        self.heads[task] = nn.Sequential(
                            nn.AdaptiveAvgPool2d(1),
                            nn.Flatten(),
                            nn.Linear(feat_dim, 256), nn.ReLU(), nn.Dropout(0.3),
                            nn.Linear(256, nc),
                        )
                    elif task == "detection":
                        self.heads[task] = nn.Sequential(
                            nn.AdaptiveAvgPool2d(7),
                            nn.Flatten(),
                            nn.Linear(feat_dim * 49, 1024), nn.ReLU(),
                            nn.Linear(1024, nc * 5),  # nc × (x, y, w, h, conf)
                        )

            def forward(self, x, tasks=None):
                features = self.encoder(x)
                tasks = tasks or self.tasks
                return {task: self.heads[task](features) for task in tasks if task in self.heads}

        self._model = MultiTaskModel(self.backbone, self.tasks, self.n_classes)
        self._model = self._model.to(self.device)
        logger.info("MultiTaskModel built: backbone=%s tasks=%s", self.backbone, self.tasks)
        return self._model

    def compute_loss(self, outputs: Dict[str, Any], targets: Dict[str, Any]) -> Any:
        """Compute weighted multi-task loss."""
        try:
            import torch
        except ImportError:
            raise ImportError("torch required")

        from pygeovision.losses.segmentation import ComboLoss
        from pygeovision.losses.class_balance import LabelSmoothingCrossEntropy
        import torch.nn.functional as F

        task_losses = {}
        total_loss = torch.tensor(0.0, requires_grad=True)

        for task in self.tasks:
            if task not in outputs or task not in targets:
                continue
            pred, target = outputs[task], targets[task]
            weight = self.task_weights.get(task, 1.0)

            if task == "segmentation":
                loss = ComboLoss()(pred, target)
            elif task in ("classification",):
                loss = LabelSmoothingCrossEntropy()(pred, target)
            elif task == "detection":
                loss = F.smooth_l1_loss(pred, target.float())
            else:
                loss = F.cross_entropy(pred, target.long())

            task_losses[task] = loss.item()
            total_loss = total_loss + weight * loss

        return total_loss, task_losses

    def train(
        self,
        train_loader: Any,
        val_loader: Any,
        epochs: int = 100,
        lr: float = 1e-4,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Train the multi-task model."""
        if self._model is None:
            self.build()

        try:
            import torch
        except ImportError:
            return {"success": False, "error": "torch required"}

        optimizer = torch.optim.AdamW(self._model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        history = {"train_loss": [], "val_loss": [], "task_losses": []}
        best_val_loss = float("inf")

        for epoch in range(1, epochs + 1):
            self._model.train()
            train_losses = []
            for batch in train_loader:
                inputs   = batch["inputs"].to(self.device)
                t_targets = {k: v.to(self.device) for k, v in batch.items()
                              if k in self.tasks}
                optimizer.zero_grad()
                outputs = self._model(inputs)
                loss, task_losses = self.compute_loss(outputs, t_targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())

            scheduler.step()
            avg_train = sum(train_losses) / max(len(train_losses), 1)

            # Validation
            self._model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch["inputs"].to(self.device)
                    t_targets = {k: v.to(self.device) for k, v in batch.items()
                                  if k in self.tasks}
                    outputs = self._model(inputs)
                    loss, _ = self.compute_loss(outputs, t_targets)
                    val_losses.append(loss.item())
            avg_val = sum(val_losses) / max(len(val_losses), 1)

            history["train_loss"].append(avg_train)
            history["val_loss"].append(avg_val)

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                if save_path:
                    torch.save(self._model.state_dict(), save_path)

            if epoch % 10 == 0:
                logger.info("Epoch %d/%d | train=%.4f | val=%.4f",
                             epoch, epochs, avg_train, avg_val)

        return {
            "success": True,
            "best_val_loss": round(best_val_loss, 6),
            "history": history,
            "epochs_trained": epochs,
        }
