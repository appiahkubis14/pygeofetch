"""HuggingFace Hub weight downloader and manager."""
from __future__ import annotations
import logging, json
from pathlib import Path
from typing import Any, Dict, List, Optional
logger = logging.getLogger(__name__)


class WeightDownloader:
    """Download and manage model weights from HuggingFace Hub."""

    DEFAULT_CACHE = Path.home() / ".cache" / "pygeovision" / "weights"

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        self.cache_dir = Path(cache_dir or self.DEFAULT_CACHE)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download(self, hf_id: str, filename: Optional[str] = None,
                  revision: str = "main") -> str:
        """Download model weights from HuggingFace Hub.

        Args:
            hf_id: HuggingFace model ID (e.g. "facebook/sam-vit-large")
            filename: Specific file to download (None = full model)
            revision: Git revision

        Returns:
            Local path to downloaded weights
        """
        try:
            from huggingface_hub import hf_hub_download, snapshot_download
            if filename:
                path = hf_hub_download(repo_id=hf_id, filename=filename,
                                         cache_dir=str(self.cache_dir), revision=revision)
            else:
                path = snapshot_download(repo_id=hf_id, cache_dir=str(self.cache_dir),
                                          revision=revision)
            logger.info("Downloaded: %s → %s", hf_id, path)
            return path
        except ImportError:
            raise ImportError("pip install huggingface_hub")
        except Exception as exc:
            raise RuntimeError(f"Download failed for {hf_id}: {exc}")

    def list_cached(self) -> List[Dict[str, Any]]:
        """List all downloaded model weights in the cache."""
        results = []
        for p in self.cache_dir.rglob("*.bin"):
            results.append({"path": str(p), "size_mb": p.stat().st_size / 1024**2})
        for p in self.cache_dir.rglob("*.safetensors"):
            results.append({"path": str(p), "size_mb": p.stat().st_size / 1024**2})
        return sorted(results, key=lambda x: x["size_mb"], reverse=True)

    def cache_size_gb(self) -> float:
        total = sum(p.stat().st_size for p in self.cache_dir.rglob("*") if p.is_file())
        return round(total / 1024**3, 2)

    def clear_cache(self, model_name: Optional[str] = None) -> int:
        """Clear cached weights. Returns bytes freed."""
        import shutil
        if model_name:
            target = self.cache_dir / model_name.replace("/", "--")
            if target.exists():
                size = sum(p.stat().st_size for p in target.rglob("*") if p.is_file())
                shutil.rmtree(target)
                return size
            return 0
        size = sum(p.stat().st_size for p in self.cache_dir.rglob("*") if p.is_file())
        shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return size
