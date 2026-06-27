"""
PyGeoVision Configuration.

Hierarchical config system supporting:
1. Built-in defaults
2. User config (~/.pygeovision/config.yaml)
3. Project config (.pygeovision.yaml)
4. Environment variables (PYGEOVISION_*)
5. Programmatic overrides
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class GPUConfig(BaseModel):
    """GPU/compute device configuration."""
    device: str = Field("auto", description="'auto', 'cuda', 'mps', or 'cpu'")
    mixed_precision: Any = Field("auto", description="True/False/'auto'")
    memory_fraction: float = Field(0.9, gt=0, le=1.0)

    @field_validator("mixed_precision", mode="before")
    @classmethod
    def parse_mixed_precision(cls, v):
        if isinstance(v, str) and v.lower() == "true":
            return True
        if isinstance(v, str) and v.lower() == "false":
            return False
        return v


class TrainingDefaultsConfig(BaseModel):
    """Default training hyperparameters."""
    batch_size: int = Field(16, gt=0)
    learning_rate: float = Field(1e-4, gt=0)
    max_epochs: int = Field(100, gt=0)
    warmup_epochs: int = Field(5, ge=0)
    gradient_accumulation_steps: int = Field(1, gt=0)
    max_grad_norm: float = Field(1.0, ge=0)


class ModelHubConfig(BaseModel):
    """Model hub / checkpoint cache configuration."""
    cache_dir: Path = Field(
        default_factory=lambda: Path.home() / ".pygeovision" / "models"
    )
    checksum_verify: bool = True


class PyGeoFetchConfig(BaseModel):
    """PyGeoFetch pass-through configuration."""
    default_providers: List[str] = Field(
        default_factory=lambda: ["planetary_computer", "aws_earth", "element84"],
        description="Default providers for open-access searches",
    )
    cache_ttl_seconds: int = Field(3600, ge=0)
    download_parallel: int = Field(4, gt=0)
    download_retry_attempts: int = Field(5, gt=0)
    verify_checksum: bool = False
    timeout_seconds: int = Field(120, gt=0)


class PyGeoVisionConfig(BaseModel):
    """Complete PyGeoVision configuration.

    Loaded from (in priority order):
    1. Built-in defaults
    2. ~/.pygeovision/config.yaml
    3. .pygeovision.yaml (project-level)
    4. Environment variables: PYGEOVISION_GPU_DEVICE, PYGEOVISION_MODEL_HUB_CACHE_DIR, etc.
    5. Programmatic overrides

    Example config YAML::

        gpu:
          device: cuda
          mixed_precision: true
        training:
          batch_size: 32
          learning_rate: 5.0e-5
        model_hub:
          cache_dir: ~/.pygeovision/models
        pygeofetch:
          default_providers: [planetary_computer, copernicus, aws_earth]
          cache_ttl_seconds: 7200
          download_parallel: 8
    """

    gpu: GPUConfig = Field(default_factory=GPUConfig)
    training: TrainingDefaultsConfig = Field(default_factory=TrainingDefaultsConfig)
    model_hub: ModelHubConfig = Field(default_factory=ModelHubConfig)
    pygeofetch: PyGeoFetchConfig = Field(default_factory=PyGeoFetchConfig)
    log_level: str = Field("INFO")

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "PyGeoVisionConfig":
        """Load config from YAML file(s) and environment variables."""
        data: Dict[str, Any] = {}

        # Load user config
        user_config = Path.home() / ".pygeovision" / "config.yaml"
        for path in [user_config, Path(".pygeovision.yaml"), config_path]:
            if path and Path(path).exists():
                try:
                    with open(path) as f:
                        file_data = yaml.safe_load(f) or {}
                    # Deep merge
                    for k, v in file_data.items():
                        if isinstance(v, dict) and isinstance(data.get(k), dict):
                            data[k].update(v)
                        else:
                            data[k] = v
                except Exception:
                    pass

        # Apply environment overrides
        env_map = {
            "PYGEOVISION_GPU_DEVICE": ("gpu", "device"),
            "PYGEOVISION_GPU_MIXED_PRECISION": ("gpu", "mixed_precision"),
            "PYGEOVISION_MODEL_HUB_CACHE_DIR": ("model_hub", "cache_dir"),
            "PYGEOVISION_LOG_LEVEL": ("log_level",),
            "PYGEOVISION_TRAINING_BATCH_SIZE": ("training", "batch_size"),
            "PYGEOVISION_TRAINING_LR": ("training", "learning_rate"),
        }
        for env_key, config_path_tuple in env_map.items():
            val = os.environ.get(env_key)
            if val is not None:
                if len(config_path_tuple) == 2:
                    section, key = config_path_tuple
                    data.setdefault(section, {})[key] = val
                else:
                    data[config_path_tuple[0]] = val

        return cls(**data)

    def as_pygeofetch_config(self) -> Dict[str, Any]:
        """Return the PyGeoFetch-compatible config subset."""
        return {
            "providers": self.pygeofetch.default_providers,
            "cache_ttl": self.pygeofetch.cache_ttl_seconds,
            "parallel": self.pygeofetch.download_parallel,
            "retry_attempts": self.pygeofetch.download_retry_attempts,
            "verify_checksum": self.pygeofetch.verify_checksum,
            "timeout": self.pygeofetch.timeout_seconds,
        }

    def save(self, path: Optional[Path] = None) -> Path:
        """Save config to YAML."""
        target = path or (Path.home() / ".pygeovision" / "config.yaml")
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)
        return target
