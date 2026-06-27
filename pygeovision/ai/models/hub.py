"""
PyGeoVision Model Hub.

Central hub for downloading, caching, and loading pretrained geospatial AI models.
Supports models from HuggingFace Hub, PyGeoVision's own model registry, and local paths.

Example:
    >>> from pygeovision.ai.models.hub import ModelHub
    >>> hub = ModelHub()
    >>> model = hub.load("unet_resnet50", num_classes=10, in_channels=4)
    >>> hub.list_available()
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_PYGEOVISION_HUB_URL = "https://huggingface.co/pygeovision"


@dataclass
class CachedModel:
    """Metadata for a locally cached model checkpoint.

    Attributes:
        name: Model identifier.
        local_path: Path to the checkpoint file.
        source_url: Download source URL.
        checksum: SHA-256 hash of the checkpoint (if verified).
        metadata: Additional model info (task, classes, etc.).
    """
    name: str
    local_path: Path
    source_url: str = ""
    checksum: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelHub:
    """Hub for managing PyGeoVision pretrained model checkpoints.

    Handles downloading, caching, integrity checking, and loading of
    pretrained weights for all PyGeoVision model architectures.

    Args:
        cache_dir: Local directory for model checkpoints.
            Defaults to ~/.pygeovision/models/.

    Example:
        >>> hub = ModelHub()
        >>> model = hub.load("unet_resnet50", num_classes=5, in_channels=3)
        >>> hub.list_cached()
        >>> hub.clear_cache(model_name="unet_resnet50")
    """

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self.cache_dir = cache_dir or Path.home() / ".pygeovision" / "models"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.cache_dir / "manifest.json"
        self._manifest: Dict[str, Dict[str, Any]] = self._load_manifest()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(
        self,
        model_name: str,
        checkpoint: Optional[str] = None,
        device: str = "cpu",
        **model_kwargs: Any,
    ) -> Any:
        """Build and optionally load pretrained weights for a model.

        Args:
            model_name: Model identifier from the PyGeoVision registry.
            checkpoint: Path to a local checkpoint, HuggingFace Hub ID,
                or URL. If None, uses pretrained ImageNet weights where
                available.
            device: Compute device to load the model on.
            **model_kwargs: Architecture kwargs (e.g. num_classes, in_channels).

        Returns:
            Instantiated (and optionally weight-loaded) PyTorch model.

        Example:
            >>> model = hub.load("segformer_b2", num_classes=10, in_channels=6)
            >>> model = hub.load("unet_resnet50", checkpoint="/path/to/best.pth")
        """
        from pygeovision.ai.models.registry import registry
        import torch

        logger.info("Building model: %s", model_name)
        model = registry.build(model_name, **model_kwargs)

        if checkpoint is not None:
            ckpt_path = self._resolve_checkpoint(checkpoint, model_name)
            logger.info("Loading checkpoint from %s", ckpt_path)
            state = torch.load(ckpt_path, map_location=device)

            # Handle common checkpoint formats
            if isinstance(state, dict):
                for key in ("model", "model_state_dict", "state_dict", "network"):
                    if key in state:
                        state = state[key]
                        break

            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                logger.warning("Missing keys when loading checkpoint: %s", missing[:5])
            if unexpected:
                logger.warning("Unexpected keys in checkpoint: %s", unexpected[:5])
            logger.info("Checkpoint loaded successfully.")

        model.to(device)
        return model

    def save(
        self,
        model: Any,
        model_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save a model's state dict to the hub cache.

        Args:
            model: PyTorch model to save.
            model_name: Identifier for the saved checkpoint.
            metadata: Optional metadata dict (task, num_classes, etc.).

        Returns:
            Path to the saved checkpoint file.
        """
        import torch

        out_path = self.cache_dir / f"{model_name}.pth"
        payload = {
            "model_state_dict": model.state_dict(),
            "metadata": metadata or {},
        }
        torch.save(payload, out_path)

        checksum = self._compute_checksum(out_path)
        self._manifest[model_name] = {
            "local_path": str(out_path),
            "checksum": checksum,
            "metadata": metadata or {},
        }
        self._save_manifest()
        logger.info("Saved model '%s' to %s", model_name, out_path)
        return out_path

    def list_cached(self) -> List[CachedModel]:
        """List all locally cached model checkpoints.

        Returns:
            List of CachedModel objects.
        """
        cached = []
        for name, info in self._manifest.items():
            path = Path(info.get("local_path", ""))
            if path.exists():
                cached.append(CachedModel(
                    name=name,
                    local_path=path,
                    source_url=info.get("source_url", ""),
                    checksum=info.get("checksum", ""),
                    metadata=info.get("metadata", {}),
                ))
        return cached

    def list_available(self) -> None:
        """Print a summary of all models available in the PyGeoVision registry."""
        from pygeovision.ai.models.registry import registry
        print(registry.summary())

    def clear_cache(self, model_name: Optional[str] = None) -> None:
        """Remove cached model checkpoints.

        Args:
            model_name: If provided, remove only this model's checkpoint.
                If None, clear the entire model cache.
        """
        if model_name is not None:
            info = self._manifest.pop(model_name, None)
            if info:
                path = Path(info.get("local_path", ""))
                if path.exists():
                    path.unlink()
                    logger.info("Removed cached checkpoint: %s", path)
            self._save_manifest()
        else:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._manifest = {}
            self._save_manifest()
            logger.info("Cleared all cached models from %s", self.cache_dir)

    def verify_checkpoint(self, model_name: str) -> bool:
        """Verify the integrity of a cached checkpoint.

        Args:
            model_name: Model identifier to check.

        Returns:
            True if the checksum matches, False otherwise.
        """
        info = self._manifest.get(model_name)
        if not info:
            logger.warning("No manifest entry for '%s'", model_name)
            return False

        path = Path(info.get("local_path", ""))
        if not path.exists():
            return False

        expected = info.get("checksum", "")
        if not expected:
            logger.warning("No checksum on record for '%s'", model_name)
            return True

        actual = self._compute_checksum(path)
        if actual != expected:
            logger.error(
                "Checksum mismatch for '%s': expected %s, got %s",
                model_name, expected[:8], actual[:8],
            )
            return False
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_checkpoint(self, checkpoint: str, model_name: str) -> Path:
        """Resolve a checkpoint string to a local file path.

        Handles: local paths, URLs (http/https), and HuggingFace Hub IDs.

        Args:
            checkpoint: Local path, URL, or HuggingFace Hub model ID.
            model_name: Model name for cache file naming.

        Returns:
            Local Path to the checkpoint file.
        """
        ckpt_path = Path(checkpoint)
        if ckpt_path.exists():
            return ckpt_path

        if checkpoint.startswith(("http://", "https://")):
            return self._download_checkpoint(checkpoint, model_name)

        # Assume HuggingFace Hub ID
        try:
            from huggingface_hub import hf_hub_download
            local = hf_hub_download(
                repo_id=checkpoint,
                filename="model.pth",
                cache_dir=str(self.cache_dir),
            )
            return Path(local)
        except Exception as exc:
            raise FileNotFoundError(
                f"Could not resolve checkpoint '{checkpoint}'. "
                "Provide a local path, URL, or HuggingFace Hub ID."
            ) from exc

    def _download_checkpoint(self, url: str, model_name: str) -> Path:
        """Download a checkpoint from a URL.

        Args:
            url: Download URL.
            model_name: Used for cache file naming.

        Returns:
            Local path to the downloaded checkpoint.
        """
        import requests

        local_path = self.cache_dir / f"{model_name}.pth"
        if local_path.exists():
            logger.info("Using cached checkpoint at %s", local_path)
            return local_path

        logger.info("Downloading checkpoint from %s…", url)
        resp = requests.get(url, stream=True, timeout=300)
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(local_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                fh.write(chunk)
                downloaded += len(chunk)
                if total:
                    logger.debug("Download: %.1f%%", 100 * downloaded / total)

        checksum = self._compute_checksum(local_path)
        self._manifest[model_name] = {
            "local_path": str(local_path),
            "source_url": url,
            "checksum": checksum,
        }
        self._save_manifest()
        logger.info("Checkpoint saved to %s", local_path)
        return local_path

    def _load_manifest(self) -> Dict[str, Any]:
        """Load the local model manifest JSON."""
        if self._manifest_path.exists():
            try:
                return json.loads(self._manifest_path.read_text())
            except json.JSONDecodeError:
                logger.warning("Corrupt model manifest; resetting.")
        return {}

    def _save_manifest(self) -> None:
        """Persist the model manifest to disk."""
        self._manifest_path.write_text(json.dumps(self._manifest, indent=2))

    @staticmethod
    def _compute_checksum(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
        """Compute the SHA-256 checksum of a file.

        Args:
            path: File path.
            chunk_size: Read chunk size in bytes.

        Returns:
            Hex-encoded SHA-256 digest.
        """
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            while chunk := fh.read(chunk_size):
                h.update(chunk)
        return h.hexdigest()
