"""SHAP values for geospatial feature importance (G6)."""
from __future__ import annotations
import logging
from typing import Any, Dict, Optional, Union
logger = logging.getLogger(__name__)


class GeospatialSHAP:
    """SHAP-based feature importance for satellite imagery models.

    Computes band-level and spatial SHAP values to explain
    which spectral bands are most important for predictions.

    Example::

        shap_exp = GeospatialSHAP(model)
        values = shap_exp.band_importance(image_tensor, n_samples=100)
        shap_exp.plot_band_importance(values, band_names=["B02","B03","B04","B08"])
    """

    def __init__(self, model: Any, device: Optional[str] = None,
                 background_samples: int = 50) -> None:
        self.model = model
        self.device = device or "cpu"
        self.background_samples = background_samples

    def band_importance(self, image: Any, n_samples: int = 100) -> Dict[str, Any]:
        """Compute spectral band importance using SHAP."""
        try:
            import shap, torch, numpy as np
        except ImportError:
            return {"error": "pip install shap"}

        self.model.eval()
        if isinstance(image, np.ndarray):
            image = torch.tensor(image, dtype=torch.float32)
        if image.ndim == 3:
            image = image.unsqueeze(0)

        def predict_fn(x):
            with torch.no_grad():
                out = self.model(torch.tensor(x, dtype=torch.float32).to(self.device))
                return torch.softmax(out, dim=1).cpu().numpy()

        background = torch.zeros_like(image).numpy()
        explainer = shap.DeepExplainer(self.model, torch.tensor(background).to(self.device))
        shap_values = explainer.shap_values(image.to(self.device))

        band_importance = {}
        for b in range(image.shape[1]):
            if isinstance(shap_values, list):
                importance = float(abs(shap_values[0][:, b]).mean())
            else:
                importance = float(abs(shap_values[:, b]).mean())
            band_importance[f"band_{b+1}"] = round(importance, 6)

        return {"band_importance": band_importance, "method": "shap"}

    def plot_band_importance(self, values: Dict, band_names: Optional[list] = None,
                              save_path: Optional[str] = None) -> None:
        try:
            import matplotlib.pyplot as plt
            bi = values.get("band_importance", {})
            labels = band_names or list(bi.keys())
            importances = list(bi.values())
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(labels, importances, color="steelblue")
            ax.set_xlabel("Mean |SHAP value|")
            ax.set_title("Spectral Band Importance (SHAP)")
            ax.grid(True, alpha=0.3, axis="x")
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=120)
            else:
                plt.show()
        except ImportError:
            logger.warning("matplotlib required for plot")
