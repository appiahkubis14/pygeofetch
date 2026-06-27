"""DatasetLoader — unified interface for downloading and loading datasets."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Optional

class DatasetLoader:
    """Unified loader for any dataset in the registry."""
    def __init__(self, data_root: str = "~/.pygeovision/datasets"):
        self.data_root = Path(data_root).expanduser()

    def info(self, name: str) -> None:
        from pygeovision.datasets.registry import dataset_registry
        d = dataset_registry[name]
        print(f"\n{'─'*60}")
        print(f"  {d.name}")
        print(f"{'─'*60}")
        for k, v in d.to_dict().items():
            if v: print(f"  {k:<18}: {v}")

    def download(self, name: str, output_dir: Optional[str] = None) -> Path:
        from pygeovision.datasets.registry import dataset_registry
        d = dataset_registry[name]
        if not d.download_url:
            raise ValueError(f"No download URL for '{name}'. Visit: {d.paper_url}")
        out = Path(output_dir or self.data_root / name)
        out.mkdir(parents=True, exist_ok=True)
        print(f"  Download '{name}' → {out}")
        print(f"  URL: {d.download_url}")
        print(f"  Size: ~{d.volume_gb:.1f} GB")
        return out
