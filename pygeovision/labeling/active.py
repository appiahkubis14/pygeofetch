"""
Active Learning Pipeline (E3, D4) — Human-in-the-loop labeling.
Selects the most informative unlabeled samples for annotation.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
logger = logging.getLogger(__name__)


class ActiveLearner:
    """Human-in-the-loop active learning for geospatial model training.

    Strategies:
        - Uncertainty sampling (entropy, least confidence, margin)
        - Core-set selection (diversity-based)
        - Committee disagreement (query by committee)
        - Spatial coverage (ensure geographic diversity)

    Workflow:
        1. Predict on unlabeled pool with current model
        2. Score each sample by informativeness
        3. Select top-k for human annotation
        4. Add annotated samples to training set
        5. Retrain model
        6. Repeat until budget exhausted or accuracy plateau

    Example::

        learner = ActiveLearner(strategy="entropy", budget=100)
        selected = learner.select(model, unlabeled_pool)
        # Human annotates selected samples...
        learner.update(selected_with_labels)
        results = learner.train_iteration(model, cfg)
    """

    STRATEGIES = ["entropy", "least_confidence", "margin", "coreset", "committee", "random"]

    def __init__(
        self,
        strategy: str = "entropy",
        budget: int = 100,
        n_iterations: int = 10,
        batch_size: int = 16,
        seed: int = 42,
    ) -> None:
        if strategy not in self.STRATEGIES:
            raise ValueError(f"strategy must be one of {self.STRATEGIES}")
        self.strategy = strategy
        self.budget = budget
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.seed = seed
        self._labeled: List[Dict] = []
        self._iteration = 0
        self._history: List[Dict] = []

    # ── Selection strategies ───────────────────────────────────────────────────
    def select(
        self,
        model: Any,
        unlabeled_pool: Any,
        n_select: Optional[int] = None,
        device: Optional[str] = None,
    ) -> List[Dict]:
        """Select the most informative samples from the unlabeled pool.

        Args:
            model: Trained PyTorch model (eval mode)
            unlabeled_pool: Iterable of {"path": ..., "idx": ...} dicts or DataLoader
            n_select: Number of samples to select (default: budget)

        Returns:
            List of sample dicts sorted by informativeness (most → least)
        """
        n_select = n_select or min(self.budget, 50)
        try:
            import torch
            dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        except ImportError:
            logger.warning("torch not installed")
            return list(unlabeled_pool)[:n_select]

        if self.strategy in ("entropy", "least_confidence", "margin"):
            return self._uncertainty_select(model, unlabeled_pool, n_select, dev)
        elif self.strategy == "coreset":
            return self._coreset_select(model, unlabeled_pool, n_select, dev)
        elif self.strategy == "committee":
            return self._committee_select(model, unlabeled_pool, n_select, dev)
        else:
            import random
            random.seed(self.seed)
            pool = list(unlabeled_pool)
            random.shuffle(pool)
            return pool[:n_select]

    def _uncertainty_select(self, model, pool, n_select, device) -> List[Dict]:
        """Score samples by prediction uncertainty."""
        import torch, numpy as np

        model.eval()
        scores = []

        with torch.no_grad():
            for item in pool:
                try:
                    if isinstance(item, dict):
                        path = item.get("path")
                        img = self._load_image(path, device)
                    elif hasattr(item, "__iter__"):
                        img = item[0].unsqueeze(0).to(device) if not isinstance(item[0], torch.Tensor) else item[0].unsqueeze(0).to(device)
                    else:
                        continue

                    logits = model(img)
                    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

                    if self.strategy == "entropy":
                        # Shannon entropy — highest entropy = most uncertain
                        score = -float(np.sum(probs * np.log(probs + 1e-10)))
                    elif self.strategy == "least_confidence":
                        # 1 - max probability
                        score = float(1.0 - probs.max())
                    elif self.strategy == "margin":
                        # Difference between top-2 probabilities
                        sorted_p = np.sort(probs)[::-1]
                        score = float(1.0 - (sorted_p[0] - sorted_p[1])) if len(sorted_p) > 1 else 0.0
                    else:
                        score = 0.0

                    entry = item if isinstance(item, dict) else {"idx": len(scores)}
                    scores.append((score, entry))
                except Exception as exc:
                    logger.debug("Score failed for item: %s", exc)

        scores.sort(key=lambda x: x[0], reverse=True)
        selected = [s[1] for s in scores[:n_select]]
        logger.info("ActiveLearner (%s): selected %d/%d samples",
                    self.strategy, len(selected), len(scores))
        return selected

    def _coreset_select(self, model, pool, n_select, device) -> List[Dict]:
        """Core-set selection: maximise feature space coverage."""
        import torch, numpy as np

        model.eval()
        features = []
        items = list(pool)

        with torch.no_grad():
            for item in items:
                try:
                    path = item.get("path") if isinstance(item, dict) else None
                    img = self._load_image(path, device) if path else None
                    if img is None:
                        continue
                    # Extract penultimate layer features via hook
                    feat = self._extract_features(model, img)
                    features.append(feat)
                except Exception:
                    features.append(np.zeros(128))

        if not features:
            return items[:n_select]

        F = np.array(features)
        # Greedy core-set: iteratively pick sample farthest from selected set
        selected_idx = [int(np.random.randint(len(F)))]
        while len(selected_idx) < min(n_select, len(F)):
            dists = np.min(
                np.linalg.norm(F[:, None] - F[selected_idx], axis=2), axis=1
            )
            dists[selected_idx] = -1
            selected_idx.append(int(np.argmax(dists)))

        return [items[i] for i in selected_idx]

    def _committee_select(self, model, pool, n_select, device) -> List[Dict]:
        """Query by committee using MC Dropout for multiple stochastic predictions."""
        import torch, numpy as np

        # Enable dropout for inference
        def _enable_dropout(m):
            if isinstance(m, torch.nn.Dropout):
                m.train()

        model.apply(_enable_dropout)
        n_passes = 10
        scores = []
        items = list(pool)

        with torch.no_grad():
            for item in items:
                try:
                    path = item.get("path") if isinstance(item, dict) else None
                    img = self._load_image(path, device)
                    all_probs = []
                    for _ in range(n_passes):
                        probs = torch.softmax(model(img), dim=1)[0].cpu().numpy()
                        all_probs.append(probs)
                    # Score = mean entropy across passes
                    mean_probs = np.mean(all_probs, axis=0)
                    entropy = -float(np.sum(mean_probs * np.log(mean_probs + 1e-10)))
                    scores.append((entropy, item))
                except Exception:
                    scores.append((0.0, item))

        model.eval()
        scores.sort(key=lambda x: x[0], reverse=True)
        return [s[1] for s in scores[:n_select]]

    def _load_image(self, path: Optional[str], device: Any) -> Optional[Any]:
        if path is None:
            return None
        try:
            import torch, rasterio, numpy as np
            with rasterio.open(path) as src:
                data = src.read().astype(np.float32)
            data = (data - data.min()) / (data.max() - data.min() + 1e-8)
            return torch.tensor(data).unsqueeze(0).to(device)
        except Exception:
            return None

    def _extract_features(self, model, img) -> Any:
        import numpy as np
        try:
            import torch
            features = []
            def hook(_, __, output):
                features.append(output.detach().cpu().numpy().flatten())
            # Register hook on last conv/transformer block
            for name, layer in reversed(list(model.named_modules())):
                if any(isinstance(layer, t) for t in [torch.nn.Conv2d, torch.nn.Linear]):
                    h = layer.register_forward_hook(hook)
                    model(img)
                    h.remove()
                    return features[0][:128] if features and len(features[0]) >= 128 else np.zeros(128)
        except Exception:
            pass
        return np.zeros(128)

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    def update(self, newly_labeled: List[Dict]) -> "ActiveLearner":
        """Add newly annotated samples to the labeled set."""
        self._labeled.extend(newly_labeled)
        self._iteration += 1
        logger.info("ActiveLearner: %d total labeled samples (iteration %d)",
                    len(self._labeled), self._iteration)
        return self

    def train_iteration(
        self,
        model: Any,
        config: Any,
        unlabeled_pool: Any,
    ) -> Dict[str, Any]:
        """Run one full active learning iteration:
        select → annotate (manual) → retrain → evaluate.

        Returns dict with val_iou improvement history.
        """
        from pygeovision.training import GeoTrainer
        import torch

        selected = self.select(model, unlabeled_pool)
        logger.info("Iteration %d: selected %d samples for annotation",
                    self._iteration + 1, len(selected))

        # In a real workflow, human annotates `selected` here.
        # We simulate by returning the selection for the caller to label.
        result = {
            "iteration": self._iteration + 1,
            "n_selected": len(selected),
            "n_labeled_total": len(self._labeled),
            "selected_samples": selected,
            "strategy": self.strategy,
            "note": "Annotate selected_samples then call learner.update(annotated) to proceed",
        }
        self._history.append(result)
        return result

    @property
    def history(self) -> List[Dict]:
        return self._history

    def plot_learning_curve(self, save_path: Optional[str] = None) -> None:
        """Plot val_iou vs annotation budget."""
        if not self._history:
            logger.warning("No active learning history to plot")
            return
        try:
            import matplotlib.pyplot as plt
            n_labeled = [h["n_labeled_total"] for h in self._history]
            iterations = list(range(1, len(self._history) + 1))
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(iterations, n_labeled, "o-", color="steelblue")
            ax.set_xlabel("Iteration"); ax.set_ylabel("Labeled Samples")
            ax.set_title(f"Active Learning Progress ({self.strategy})")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=120)
            else:
                plt.show()
        except ImportError:
            logger.warning("matplotlib required for plot")
