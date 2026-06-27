"""
Foundation Model Labeler.

Generates pseudo-labels for satellite imagery using geospatial foundation
models (Prithvi, Clay, SatMAE, GeoFM) via zero-shot or few-shot inference.
Foundation models provide rich geospatial representations that can be used
as label proxies when manual annotation is unavailable.

Supported models:
    - Prithvi (NASA/IBM): 6-band Harmonized Landsat Sentinel-2 foundation model
    - Clay: Multi-source geospatial foundation model
    - SatMAE: Satellite Masked Autoencoder
    - RemoteCLIP: Vision-language model for remote sensing

Example:
    >>> from pygeovision.ai.labeling.foundation_labeler import FoundationModelLabeler
    >>> labeler = FoundationModelLabeler(model="prithvi", task="land_cover")
    >>> results = labeler.label_tiles(tiles, output_dir="./labels/")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pygeovision.ai.labeling.base_labeler import BaseLabeler, LabelingResult
from pygeovision.core.exceptions import LabelingError

logger = logging.getLogger(__name__)

# Registry of supported foundation models with HuggingFace Hub IDs
_FOUNDATION_MODELS: Dict[str, Dict[str, Any]] = {
    "prithvi": {
        "hub_id": "ibm-nasa-geospatial/Prithvi-100M",
        "input_bands": 6,  # Blue, Green, Red, Narrow NIR, SWIR-1, SWIR-2
        "patch_size": 16,
        "image_size": 224,
        "description": "NASA/IBM Prithvi 100M geospatial foundation model (HLS imagery)",
    },
    "prithvi_600": {
        "hub_id": "ibm-nasa-geospatial/Prithvi-EO-2.0-600M",
        "input_bands": 6,
        "patch_size": 16,
        "image_size": 512,
        "description": "NASA/IBM Prithvi 2.0 600M parameter model",
    },
    "clay": {
        "hub_id": "made-with-clay/Clay",
        "input_bands": -1,  # Variable
        "patch_size": 8,
        "image_size": 256,
        "description": "Clay Foundation Model — multi-source geospatial",
    },
    "satmae": {
        "hub_id": "MVRL/satmae-vitl-pretrain",
        "input_bands": 3,
        "patch_size": 16,
        "image_size": 224,
        "description": "Satellite Masked Autoencoder (SatMAE)",
    },
    "remote_clip": {
        "hub_id": "chendelong/RemoteCLIP-ViT-L-14",
        "input_bands": 3,
        "patch_size": 14,
        "image_size": 224,
        "description": "RemoteCLIP vision-language model for remote sensing",
    },
}


@dataclass
class FoundationLabelConfig:
    """Configuration for the Foundation Model Labeler.

    Attributes:
        model: Foundation model name (key in _FOUNDATION_MODELS).
        task: Downstream task for pseudo-labeling.
            'land_cover', 'change_detection', 'object_detection', 'embedding'.
        text_prompts: Class text prompts for vision-language models.
        num_classes: Number of output classes for k-means clustering.
        cluster_method: Clustering method for embedding-based labels.
        device: Compute device.
        checkpoint_dir: Local model cache directory.
        batch_size: Inference batch size (patches).
        input_bands: Band indices (0-based) to feed to the model.
    """

    model: str = "prithvi"
    task: str = "land_cover"
    text_prompts: Optional[List[str]] = None
    num_classes: int = 10
    cluster_method: str = "kmeans"  # 'kmeans' | 'spectral'
    device: str = "cpu"
    checkpoint_dir: Optional[Path] = None
    batch_size: int = 16
    input_bands: Optional[List[int]] = None


class FoundationModelLabeler(BaseLabeler):
    """Pseudo-label generator using geospatial foundation models.

    Extracts rich embeddings from foundation models and uses clustering
    or zero-shot classification to generate pseudo-labels for satellite tiles.
    Useful for bootstrapping annotation workflows or active learning.

    Workflow:
        1. Load foundation model (lazy-loaded on first use).
        2. Tile the image into patches.
        3. Extract embeddings for each patch.
        4. Cluster embeddings (k-means) or classify via text prompts (CLIP).
        5. Upsample cluster IDs back to pixel resolution.

    Args:
        model: Foundation model to use. One of: 'prithvi', 'prithvi_600',
            'clay', 'satmae', 'remote_clip'.
        task: Labeling task type. 'land_cover' uses k-means clustering;
            'embedding' returns raw PCA-compressed embeddings as pseudo-labels.
        text_prompts: For RemoteCLIP — list of class text descriptions.
            Example: ["forest", "water body", "urban area", "cropland"].
        num_classes: Number of classes for k-means clustering.
        device: Compute device. Auto-detected if None.
        input_bands: 0-based band indices to pass to the model.
            Defaults to first N bands where N = model's input_bands.
        num_workers: Number of parallel workers.
        skip_existing: Skip tiles with existing labels.

    Example:
        >>> # Prithvi for land cover clustering
        >>> labeler = FoundationModelLabeler(model="prithvi", num_classes=8)
        >>> # RemoteCLIP for text-guided classification
        >>> labeler = FoundationModelLabeler(
        ...     model="remote_clip",
        ...     text_prompts=["tree cover", "water", "buildings", "bare soil"],
        ... )
    """

    def __init__(
        self,
        model: str = "prithvi",
        task: str = "land_cover",
        text_prompts: Optional[List[str]] = None,
        num_classes: int = 10,
        device: Optional[str] = None,
        input_bands: Optional[List[int]] = None,
        num_workers: int = 1,
        skip_existing: bool = True,
    ) -> None:
        super().__init__(
            labeler_name=f"foundation_{model}",
            num_workers=num_workers,
            skip_existing=skip_existing,
        )
        if model not in _FOUNDATION_MODELS:
            raise ValueError(
                f"model must be one of {list(_FOUNDATION_MODELS.keys())}, got {model!r}"
            )

        resolved_device = device or self._detect_device()
        self.config = FoundationLabelConfig(
            model=model,
            task=task,
            text_prompts=text_prompts,
            num_classes=num_classes,
            device=resolved_device,
            checkpoint_dir=Path.home() / ".pygeovision" / "cache" / "foundation_models",
            input_bands=input_bands,
        )
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._model_info = _FOUNDATION_MODELS[model]
        self._model: Optional[Any] = None
        self._processor: Optional[Any] = None

    # ------------------------------------------------------------------
    # BaseLabeler abstract properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return 'foundation'

    @property
    def supported_tasks(self) -> list:
        return ['segmentation', 'classification']

    # ------------------------------------------------------------------
    # BaseLabeler interface
    # ------------------------------------------------------------------

    def label_tile(
        self,
        tile_path: Path,
        tile_metadata: Any,
        output_path: Path,
    ) -> LabelingResult:
        """Generate pseudo-labels for a single satellite tile.

        Args:
            tile_path: Path to the GeoTIFF tile.
            tile_metadata: TileMetadata with bounds, CRS, shape.
            output_path: Destination path for the pseudo-label GeoTIFF.

        Returns:
            LabelingResult with cluster statistics.
        """
        try:
            import rasterio
        except ImportError as exc:
            raise LabelingError(
                "foundation_labeler requires rasterio. Install: pip install rasterio"
            ) from exc

        try:
            self._ensure_model_loaded()

            with rasterio.open(tile_path) as src:
                height, width = src.height, src.width
                crs = src.crs
                transform = src.transform
                band_count = src.count

            # Load imagery bands
            image = self._load_bands(tile_path, band_count)

            # Extract embeddings
            embeddings, patch_grid = self._extract_embeddings(image)

            # Generate pseudo-labels
            if self.config.model == "remote_clip" and self.config.text_prompts:
                patch_labels = self._classify_with_text(embeddings)
            else:
                patch_labels = self._cluster_embeddings(embeddings)

            # Upsample to pixel resolution
            mask = self._upsample_patch_labels(
                patch_labels=patch_labels,
                patch_grid=patch_grid,
                height=height,
                width=width,
            )

            meta = {
                "driver": "GTiff",
                "dtype": "uint8",
                "width": width,
                "height": height,
                "count": 1,
                "crs": crs,
                "transform": transform,
                "compress": "lzw",
            }
            self._write_label_geotiff(mask, output_path, meta)

            unique_classes = np.unique(mask)
            stats = {f"class_{c}": float(np.sum(mask == c)) / mask.size for c in unique_classes}

            return LabelingResult(
                tile_path=tile_path,
                label_path=output_path,
                success=True,
                labeler=f"foundation_{self.config.model}",
                class_distribution=stats,
                metadata={
                    "model": self.config.model,
                    "task": self.config.task,
                    "num_classes": len(unique_classes),
                    "embedding_dim": embeddings.shape[-1] if embeddings.ndim > 1 else 0,
                    "device": self.config.device,
                },
            )

        except Exception as exc:
            logger.error(
                "FoundationModelLabeler failed for %s: %s", tile_path, exc
            )
            return LabelingResult(
                tile_path=tile_path,
                label_path=output_path,
                success=False,
                labeler=f"foundation_{self.config.model}",
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_device() -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def _ensure_model_loaded(self) -> None:
        """Lazy-load the foundation model from HuggingFace Hub."""
        if self._model is not None:
            return

        hub_id = self._model_info["hub_id"]
        logger.info("Loading foundation model '%s' from %s…", self.config.model, hub_id)

        try:
            if self.config.model == "remote_clip":
                self._load_remote_clip(hub_id)
            elif self.config.model.startswith("prithvi"):
                self._load_prithvi(hub_id)
            elif self.config.model == "clay":
                self._load_clay(hub_id)
            else:
                self._load_satmae(hub_id)
            logger.info("Foundation model '%s' loaded.", self.config.model)
        except Exception as exc:
            raise LabelingError(
                f"Failed to load foundation model '{self.config.model}': {exc}"
            ) from exc

    def _load_prithvi(self, hub_id: str) -> None:
        """Load Prithvi model from HuggingFace Hub."""
        try:
            from transformers import AutoModel, AutoConfig
            config = AutoConfig.from_pretrained(hub_id, trust_remote_code=True)
            self._model = AutoModel.from_pretrained(
                hub_id, config=config, trust_remote_code=True
            )
            self._model.to(self.config.device)
            self._model.eval()
        except Exception:
            # Fallback: use timm MAE ViT as proxy
            try:
                import timm
                self._model = timm.create_model(
                    "vit_base_patch16_224", pretrained=True, num_classes=0
                )
                self._model.to(self.config.device)
                self._model.eval()
                logger.warning(
                    "Prithvi not available; using timm ViT-B/16 as embedding proxy."
                )
            except ImportError as exc:
                raise LabelingError(
                    "Prithvi requires transformers>=4.30 or timm. "
                    "Install: pip install transformers timm"
                ) from exc

    def _load_remote_clip(self, hub_id: str) -> None:
        """Load RemoteCLIP model."""
        try:
            import open_clip  # type: ignore
            self._model, _, self._processor = open_clip.create_model_and_transforms(
                "ViT-L-14",
                pretrained=hub_id,
                cache_dir=str(self.config.checkpoint_dir),
            )
            self._model.to(self.config.device)
            self._model.eval()
        except ImportError:
            # Fallback to standard CLIP
            try:
                from transformers import CLIPModel, CLIPProcessor
                self._model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
                self._processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
                self._model.to(self.config.device)
                self._model.eval()
                logger.warning("RemoteCLIP not available; using OpenAI CLIP as fallback.")
            except ImportError as exc:
                raise LabelingError(
                    "RemoteCLIP requires open-clip-torch or transformers. "
                    "Install: pip install open-clip-torch"
                ) from exc

    def _load_clay(self, hub_id: str) -> None:
        """Load Clay foundation model."""
        try:
            from transformers import AutoModel
            self._model = AutoModel.from_pretrained(hub_id, trust_remote_code=True)
            self._model.to(self.config.device)
            self._model.eval()
        except Exception:
            self._load_prithvi("ibm-nasa-geospatial/Prithvi-100M")

    def _load_satmae(self, hub_id: str) -> None:
        """Load SatMAE model."""
        try:
            from transformers import AutoModel
            self._model = AutoModel.from_pretrained(hub_id, trust_remote_code=True)
            self._model.to(self.config.device)
            self._model.eval()
        except Exception:
            self._load_prithvi("ibm-nasa-geospatial/Prithvi-100M")

    def _load_bands(self, tile_path: Path, band_count: int) -> np.ndarray:
        """Load imagery bands into a float32 array.

        Args:
            tile_path: Path to GeoTIFF.
            band_count: Total number of bands in the file.

        Returns:
            float32 array of shape (C, H, W) normalized to [0, 1].
        """
        import rasterio

        n_bands = self._model_info["input_bands"]
        if n_bands == -1:
            n_bands = min(band_count, 6)  # Clay accepts variable bands

        if self.config.input_bands:
            band_indices = [b + 1 for b in self.config.input_bands[:n_bands]]
        else:
            band_indices = list(range(1, min(n_bands + 1, band_count + 1)))

        with rasterio.open(tile_path) as src:
            data = src.read(band_indices).astype(np.float32)

        # Normalize per-band to [0, 1]
        for i in range(data.shape[0]):
            band = data[i]
            valid = band[band > 0]
            if valid.size > 0:
                p2, p98 = np.percentile(valid, (2, 98))
                data[i] = np.clip((band - p2) / max(p98 - p2, 1e-6), 0, 1)

        return data

    def _extract_embeddings(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Extract patch embeddings from the foundation model.

        Args:
            image: float32 (C, H, W) image array in [0, 1].

        Returns:
            Tuple of (embeddings array (N, D), patch_grid (rows, cols)).
        """
        import torch

        patch_size = self._model_info.get("patch_size", 16)
        img_size = self._model_info.get("image_size", 224)

        C, H, W = image.shape

        # Tile image into img_size x img_size patches
        patches: List[np.ndarray] = []
        positions: List[Tuple[int, int]] = []

        stride = img_size - patch_size  # Small overlap
        stride = max(stride, img_size // 2)

        for row in range(0, H, stride):
            for col in range(0, W, stride):
                patch = image[:, row:row + img_size, col:col + img_size]
                if patch.shape[1] < img_size or patch.shape[2] < img_size:
                    # Pad to img_size
                    padded = np.zeros((C, img_size, img_size), dtype=np.float32)
                    padded[:, :patch.shape[1], :patch.shape[2]] = patch
                    patch = padded
                patches.append(patch)
                positions.append((row, col))

        if not patches:
            return np.zeros((1, 768)), (1, 1)

        # Run in batches
        all_embeddings: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(patches), self.config.batch_size):
                batch = np.stack(patches[i:i + self.config.batch_size])
                batch_tensor = torch.from_numpy(batch).to(self.config.device)

                # Handle different model interfaces
                try:
                    output = self._model(pixel_values=batch_tensor)
                    if hasattr(output, "last_hidden_state"):
                        emb = output.last_hidden_state[:, 0].cpu().numpy()  # CLS token
                    else:
                        emb = output.cpu().numpy().reshape(batch_tensor.shape[0], -1)
                except Exception:
                    try:
                        emb = self._model(batch_tensor).cpu().numpy()
                        if emb.ndim > 2:
                            emb = emb.reshape(emb.shape[0], -1)
                    except Exception:
                        emb = np.zeros((batch_tensor.shape[0], 768), dtype=np.float32)

                all_embeddings.append(emb)

        embeddings = np.concatenate(all_embeddings, axis=0)

        # Determine patch grid dimensions
        rows = len(range(0, H, stride))
        cols = len(range(0, W, stride))

        return embeddings, (rows, cols)

    def _cluster_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Cluster patch embeddings with k-means.

        Args:
            embeddings: (N, D) embedding array.

        Returns:
            (N,) int array of cluster IDs.
        """
        from sklearn.cluster import KMeans, MiniBatchKMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        # PCA dimensionality reduction for clustering stability
        n_components = min(64, embeddings.shape[0] - 1, embeddings.shape[1])
        if n_components > 1:
            pca = PCA(n_components=n_components)
            reduced = pca.fit_transform(embeddings)
        else:
            reduced = embeddings

        scaled = StandardScaler().fit_transform(reduced)

        n_clusters = min(self.config.num_classes, embeddings.shape[0])
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=3)
        labels = kmeans.fit_predict(scaled).astype(np.uint8)
        return labels

    def _classify_with_text(self, embeddings: np.ndarray) -> np.ndarray:
        """Zero-shot classify patches using text prompts (RemoteCLIP).

        Args:
            embeddings: (N, D) vision embeddings.

        Returns:
            (N,) int array of class IDs (index into text_prompts).
        """
        import torch

        if not self.config.text_prompts:
            return self._cluster_embeddings(embeddings)

        try:
            # Encode text prompts
            if hasattr(self._processor, "tokenize"):
                # open_clip interface
                tokens = self._processor(self.config.text_prompts)
                with torch.no_grad():
                    text_emb = self._model.encode_text(
                        tokens.to(self.config.device)
                    ).cpu().numpy()
            else:
                # HuggingFace CLIP interface
                inputs = self._processor(
                    text=self.config.text_prompts,
                    return_tensors="pt",
                    padding=True,
                ).to(self.config.device)
                with torch.no_grad():
                    text_emb = self._model.get_text_features(**inputs).cpu().numpy()

            # Cosine similarity: (N, D) @ (D, K) → (N, K)
            emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-6)
            text_norm = text_emb / (np.linalg.norm(text_emb, axis=1, keepdims=True) + 1e-6)
            sim = emb_norm @ text_norm.T
            labels = np.argmax(sim, axis=1).astype(np.uint8)
            return labels

        except Exception as exc:
            logger.warning("Text classification failed, falling back to clustering: %s", exc)
            return self._cluster_embeddings(embeddings)

    def _upsample_patch_labels(
        self,
        patch_labels: np.ndarray,
        patch_grid: Tuple[int, int],
        height: int,
        width: int,
    ) -> np.ndarray:
        """Upsample patch-level labels to pixel resolution.

        Args:
            patch_labels: (N,) patch label array.
            patch_grid: (num_rows, num_cols) grid dimensions.
            height: Target image height in pixels.
            width: Target image width in pixels.

        Returns:
            uint8 label array of shape (H, W).
        """
        from PIL import Image

        rows, cols = patch_grid
        grid = patch_labels[:rows * cols].reshape(rows, cols).astype(np.uint8)

        # Nearest-neighbor upsample
        img = Image.fromarray(grid, mode="L")
        upsampled = img.resize((width, height), Image.NEAREST)
        return np.array(upsampled, dtype=np.uint8)
