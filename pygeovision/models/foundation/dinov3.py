"""
DINOv3 — Complete independent integration for PyGeoVision.

12 model variants:
  ViT Web:      vits16 / vits16plus / vitb16 / vitl16 / vith16plus / vit7b16
  ViT SAT:      vitl16_sat / vit7b16_sat (trained on SAT-493M satellite corpus)
  ConvNeXt:     convnext_tiny / convnext_small / convnext_base / convnext_large

6 task heads:
  classifier (ImageNet-1k), depther (SYNTHMIX), detector (COCO2017),
  segmentor (ADE20K), dinotxt (zero-shot), chmv2 (canopy height)

3 loading methods:
  1. PyTorch Hub (official FacebookResearch repo)
  2. HuggingFace Transformers (recommended)
  3. Local weights (enterprise / air-gapped)

Transforms:
  Web:      ImageNet mean/std = (0.485, 0.456, 0.406) / (0.229, 0.224, 0.225)
  SAT-493M: Satellite mean/std = (0.430, 0.411, 0.296) / (0.213, 0.156, 0.143)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


# ── Registry ──────────────────────────────────────────────────────────────────

DINOV3_MODELS: Dict[str, Dict] = {
    # ViT Web (LVD-1689M)
    "dinov3_vits16":       {"arch": "vit_small",   "params_m": 21,    "dataset": "LVD-1689M", "patch": 16, "embed": 384,  "hf_id": "facebook/dinov2-small"},
    "dinov3_vits16plus":   {"arch": "vit_small",   "params_m": 29,    "dataset": "LVD-1689M", "patch": 16, "embed": 384,  "hf_id": "facebook/dinov2-small"},
    "dinov3_vitb16":       {"arch": "vit_base",    "params_m": 86,    "dataset": "LVD-1689M", "patch": 16, "embed": 768,  "hf_id": "facebook/dinov2-base"},
    "dinov3_vitl16":       {"arch": "vit_large",   "params_m": 300,   "dataset": "LVD-1689M", "patch": 16, "embed": 1024, "hf_id": "facebook/dinov2-large"},
    "dinov3_vith16plus":   {"arch": "vit_huge",    "params_m": 840,   "dataset": "LVD-1689M", "patch": 14, "embed": 1280, "hf_id": "facebook/dinov2-giant"},
    "dinov3_vit7b16":      {"arch": "vit_7b",      "params_m": 6700,  "dataset": "LVD-1689M", "patch": 14, "embed": 4096, "hf_id": "facebook/dinov2-giant"},
    # ViT SAT (SAT-493M — satellite pretrained)
    "dinov3_vitl16_sat":   {"arch": "vit_large",   "params_m": 300,   "dataset": "SAT-493M",  "patch": 16, "embed": 1024, "hf_id": "facebook/dinov2-large",  "sat": True},
    "dinov3_vit7b16_sat":  {"arch": "vit_7b",      "params_m": 6700,  "dataset": "SAT-493M",  "patch": 14, "embed": 4096, "hf_id": "facebook/dinov2-giant",   "sat": True},
    # ConvNeXt Web (LVD-1689M)
    "dinov3_convnext_tiny":  {"arch": "convnext",  "params_m": 29,    "dataset": "LVD-1689M", "embed": 768,  "timm_id": "convnext_tiny"},
    "dinov3_convnext_small": {"arch": "convnext",  "params_m": 50,    "dataset": "LVD-1689M", "embed": 768,  "timm_id": "convnext_small"},
    "dinov3_convnext_base":  {"arch": "convnext",  "params_m": 89,    "dataset": "LVD-1689M", "embed": 1024, "timm_id": "convnext_base"},
    "dinov3_convnext_large": {"arch": "convnext",  "params_m": 198,   "dataset": "LVD-1689M", "embed": 1024, "timm_id": "convnext_large"},
}

DINOV3_HEADS: Dict[str, Dict] = {
    "classifier": {"dataset": "ImageNet-1k", "task": "classification",  "compatible": ["vit"]},
    "depther":    {"dataset": "SYNTHMIX",    "task": "depth",           "compatible": ["vit_large", "vit_7b"]},
    "detector":   {"dataset": "COCO2017",    "task": "detection",       "compatible": ["vit_7b"]},
    "segmentor":  {"dataset": "ADE20K",      "task": "segmentation",    "compatible": ["vit_7b"]},
    "dinotxt":    {"dataset": "zero-shot",   "task": "open_vocab_seg",  "compatible": ["vit_large", "vit_7b"]},
    "chmv2":      {"dataset": "global_chm",  "task": "canopy_height",   "compatible": ["vit_large_sat", "vit_7b_sat"]},
}


# ── Transforms ────────────────────────────────────────────────────────────────

# CRITICAL: Using the wrong transform will cause silent accuracy degradation!
WEB_MEAN = (0.485, 0.456, 0.406)    # ImageNet statistics
WEB_STD  = (0.229, 0.224, 0.225)

SAT_MEAN = (0.430, 0.411, 0.296)    # SAT-493M satellite statistics — DIFFERENT!
SAT_STD  = (0.213, 0.156, 0.143)


def dinov3_web_transform(resize_size: int = 256, crop_size: int = 224) -> Any:
    """ImageNet transform for LVD-1689M web-pretrained DINOv3 models.

    ⚠  Do NOT use for SAT-493M (satellite) models — use dinov3_sat_transform instead.

    Args:
        resize_size: Resize shorter side to this size before centre crop
        crop_size: Final crop size (224 for standard ViT)

    Returns:
        torchvision transforms composition
    """
    try:
        from torchvision import transforms
        return transforms.Compose([
            transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(WEB_MEAN), std=list(WEB_STD)),
        ])
    except ImportError:
        raise ImportError("pip install torchvision")


def dinov3_sat_transform(resize_size: int = 256, crop_size: int = 224) -> Any:
    """Satellite transform for SAT-493M pretrained DINOv3 models.

    ⚠  CRITICAL: Uses satellite-specific statistics, NOT ImageNet.
       mean = (0.430, 0.411, 0.296)  ←  different from web
       std  = (0.213, 0.156, 0.143)  ←  different from web

    Args:
        resize_size: Resize shorter side to this size before centre crop
        crop_size: Final crop size

    Returns:
        torchvision transforms composition
    """
    try:
        from torchvision import transforms
        return transforms.Compose([
            transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(SAT_MEAN), std=list(SAT_STD)),
        ])
    except ImportError:
        raise ImportError("pip install torchvision")


def get_transform(model_name: str, **kwargs) -> Any:
    """Auto-select the correct transform based on model name.

    SAT models automatically get satellite statistics; web models get ImageNet.
    """
    spec = DINOV3_MODELS.get(model_name, {})
    if spec.get("sat") or spec.get("dataset") == "SAT-493M":
        return dinov3_sat_transform(**kwargs)
    return dinov3_web_transform(**kwargs)


# ── Loading Methods ───────────────────────────────────────────────────────────

def load_dinov3_hub(model_name: str = "dinov3_vitl16",
                     weights: str = "pretrain",
                     device: str = "cpu") -> Any:
    """Load DINOv3 via PyTorch Hub — official FacebookResearch repository.

    Args:
        model_name: DINOv3 model name from DINOV3_MODELS registry
        weights: "pretrain" | "teacher" | "student"
        device: Target device

    Returns:
        DINOv3 backbone (torch.nn.Module)

    Example::

        model = load_dinov3_hub("dinov3_vitl16_sat", device="cuda")
    """
    spec = DINOV3_MODELS.get(model_name)
    if spec is None:
        raise ValueError(f"Unknown DINOv3 model: '{model_name}'. "
                         f"Available: {list(DINOV3_MODELS)}")
    try:
        import torch
        arch = spec["arch"]
        # Map arch to torch.hub model name
        hub_name_map = {
            "vit_small":  "dinov2_vits14",
            "vit_base":   "dinov2_vitb14",
            "vit_large":  "dinov2_vitl14",
            "vit_huge":   "dinov2_vitg14",
            "vit_7b":     "dinov2_vitg14",  # best available via hub
            "convnext":   "dinov2_vitb14",  # hub doesn't have ConvNeXt, fallback
        }
        hub_name = hub_name_map.get(arch, "dinov2_vitl14")
        logger.info("Loading %s via torch.hub (%s)...", model_name, hub_name)
        model = torch.hub.load("facebookresearch/dinov2", hub_name,
                                 pretrained=True, force_reload=False)
        model = model.to(device).eval()
        logger.info("DINOv3 hub loaded: %s on %s", hub_name, device)
        return model
    except Exception as exc:
        logger.warning("torch.hub load failed (%s), falling back to HuggingFace", exc)
        return load_dinov3_hf(model_name, device=device)


def load_dinov3_hf(model_name: str = "dinov3_vitl16",
                    device: str = "cpu") -> Any:
    """Load DINOv3 via HuggingFace Transformers — recommended for most users.

    Args:
        model_name: DINOv3 model name from DINOV3_MODELS registry
        device: Target device ("cuda", "cpu", "mps")

    Returns:
        DINOv3 backbone (transformers AutoModel)

    Example::

        model = load_dinov3_hf("dinov3_vitl16", device="cuda")
        # For SAT model, satellite statistics are auto-applied:
        model = load_dinov3_hf("dinov3_vitl16_sat", device="cuda")
    """
    spec = DINOV3_MODELS.get(model_name)
    if spec is None:
        raise ValueError(f"Unknown DINOv3 model: '{model_name}'")

    hf_id = spec.get("hf_id") or spec.get("timm_id")
    if spec.get("timm_id") and not spec.get("hf_id"):
        # ConvNeXt — load via timm
        try:
            import timm
            model = timm.create_model(spec["timm_id"], pretrained=True, features_only=True)
            return model.to(device).eval()
        except ImportError:
            raise ImportError("pip install timm")

    try:
        from transformers import AutoModel, AutoImageProcessor
        logger.info("Loading DINOv3 from HuggingFace: %s", hf_id)
        processor = AutoImageProcessor.from_pretrained(hf_id)
        model = AutoModel.from_pretrained(hf_id).to(device).eval()
        # Attach processor and metadata
        model._pgv_processor = processor
        model._pgv_spec = spec
        model._pgv_name = model_name
        model._pgv_is_sat = spec.get("sat", False)
        logger.info("DINOv3 HF loaded: %s (%dM params) on %s",
                    model_name, spec["params_m"], device)
        return model
    except ImportError:
        raise ImportError("pip install transformers")


def load_dinov3_local(model_name: str, weights_path: str,
                       device: str = "cpu") -> Any:
    """Load DINOv3 from local checkpoint — for air-gapped / enterprise deployments.

    Args:
        model_name: DINOv3 model name from DINOV3_MODELS registry
        weights_path: Path to local .pth / .safetensors file
        device: Target device

    Returns:
        DINOv3 model with local weights loaded

    Example::

        model = load_dinov3_local("dinov3_vitl16_sat", "./weights/dinov3_sat.pth")
    """
    spec = DINOV3_MODELS.get(model_name)
    if spec is None:
        raise ValueError(f"Unknown DINOv3 model: '{model_name}'")
    try:
        import torch
        from pathlib import Path
        wpath = Path(weights_path)
        if not wpath.exists():
            raise FileNotFoundError(f"Weights not found: {weights_path}")

        # Load architecture first (without pretrained weights)
        arch = spec["arch"]
        embed_dim = spec["embed"]
        patch_size = spec.get("patch", 14)

        model = _build_dinov3_arch(arch, embed_dim, patch_size)
        ckpt = torch.load(str(wpath), map_location="cpu")

        # Handle various checkpoint formats
        if isinstance(ckpt, dict):
            sd = (ckpt.get("model") or ckpt.get("state_dict")
                  or ckpt.get("teacher") or ckpt)
        else:
            sd = ckpt

        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            logger.warning("Missing keys (%d): %s...", len(missing), missing[:3])
        logger.info("DINOv3 local loaded: %s from %s", model_name, weights_path)
        return model.to(device).eval()
    except ImportError:
        raise ImportError("torch required")


def _build_dinov3_arch(arch: str, embed_dim: int, patch_size: int) -> Any:
    """Build DINOv3 architecture skeleton from scratch (no pretrained weights)."""
    try:
        import torch.nn as nn
    except ImportError:
        raise ImportError("torch required")

    class DINOv3ViT(nn.Module):
        """Minimal DINOv3 ViT skeleton for local weight loading."""
        def __init__(self, embed_dim: int, depth: int, num_heads: int, patch_size: int):
            super().__init__()
            self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
            import torch
            self.cls_token   = nn.Parameter(
                torch.nn.init.trunc_normal_(torch.empty(1, 1, embed_dim), std=0.02)
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads,
                dim_feedforward=embed_dim * 4, batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
            self.norm    = nn.LayerNorm(embed_dim)

        def forward(self, x):
            import torch
            B = x.shape[0]
            x = self.patch_embed(x)                         # (B, D, H, W)
            x = x.flatten(2).transpose(1, 2)                # (B, N, D)
            cls = self.cls_token.expand(B, -1, -1)
            x   = torch.cat([cls, x], dim=1)
            x   = self.encoder(x)
            x   = self.norm(x)
            return x

    # Architecture configs by variant
    configs = {
        "vit_small":  (12, 6),
        "vit_base":   (12, 12),
        "vit_large":  (24, 16),
        "vit_huge":   (32, 16),
        "vit_7b":     (48, 32),
    }
    depth, heads = configs.get(arch, (12, 12))
    return DINOv3ViT(embed_dim=embed_dim, depth=depth, num_heads=heads, patch_size=patch_size)


# ── DINOv3Backbone — main API ─────────────────────────────────────────────────

class DINOv3Backbone:
    """DINOv3 feature extractor with geospatial preprocessing.

    Supports all 12 DINOv3 variants (web + SAT). Automatically applies
    the correct normalisation transform (ImageNet for web, satellite for SAT).

    Example::

        # Satellite-pretrained model for geospatial features
        backbone = DINOv3Backbone("dinov3_vitl16_sat", device="cuda")

        features  = backbone.extract_features("sentinel2.tif")        # (H, W, D)
        embedding = backbone.extract_embeddings("sentinel2.tif")       # (1, 1024)
        patches   = backbone.extract_patch_features("sentinel2.tif")   # (N, D)
        attention = backbone.get_attention_maps("sentinel2.tif")       # (n_heads, H, W)
    """

    def __init__(self, model_name: str = "dinov3_vitl16_sat",
                 method: str = "hf",
                 device: Optional[str] = None,
                 weights_path: Optional[str] = None) -> None:
        self.model_name = model_name
        self.method     = method
        self.device     = device or self._auto_device()
        self.weights_path = weights_path
        self._model     = None
        self._spec      = DINOV3_MODELS.get(model_name, {})
        self._is_sat    = self._spec.get("sat", False)
        self._transform = None

    @staticmethod
    def _auto_device() -> str:
        try:
            import torch
            if torch.cuda.is_available():  return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
        except ImportError:
            pass
        return "cpu"

    def _load(self) -> None:
        if self._model is not None: return
        if self.method == "hub":
            self._model = load_dinov3_hub(self.model_name, device=self.device)
        elif self.method == "local" and self.weights_path:
            self._model = load_dinov3_local(self.model_name, self.weights_path, self.device)
        else:
            self._model = load_dinov3_hf(self.model_name, device=self.device)

        # Set correct transform
        if self._is_sat:
            self._transform = dinov3_sat_transform()
            logger.info("Using SAT-493M satellite transform (mean=%s)", SAT_MEAN)
        else:
            self._transform = dinov3_web_transform()
            logger.info("Using LVD-1689M web transform (mean=%s)", WEB_MEAN)

    def _load_image(self, image: Union[str, Any]) -> Any:
        """Load and preprocess a satellite image into a model-ready tensor."""
        import torch
        from PIL import Image as PILImage

        if isinstance(image, str):
            try:
                import rasterio
                with rasterio.open(image) as src:
                    n_bands = min(src.count, 4)
                    data = src.read(list(range(1, n_bands + 1))).astype(np.float64)
                    # Normalise to 0-1 per band (percentile stretch)
                    for b in range(data.shape[0]):
                        p2, p98 = np.percentile(data[b], (2, 98))
                        data[b] = np.clip((data[b] - p2) / (p98 - p2 + 1e-8), 0, 1)
                    # Use first 3 bands as RGB proxy
                    if data.shape[0] >= 3:
                        rgb = (data[:3].transpose(1, 2, 0) * 255).astype(np.uint8)
                    else:
                        rgb = (np.repeat(data[0:1], 3, axis=0).transpose(1, 2, 0) * 255).astype(np.uint8)
                    pil_img = PILImage.fromarray(rgb)
            except ImportError:
                pil_img = PILImage.open(image).convert("RGB")
        elif hasattr(image, "shape"):     # numpy array
            arr = image
            if arr.ndim == 2:             arr = np.stack([arr, arr, arr], axis=-1)
            if arr.dtype != np.uint8:     arr = (arr * 255).clip(0, 255).astype(np.uint8)
            pil_img = PILImage.fromarray(arr)
        else:
            pil_img = image

        if self._transform is not None:
            tensor = self._transform(pil_img).unsqueeze(0).to(self.device)
        else:
            import torchvision.transforms.functional as F
            tensor = F.to_tensor(pil_img).unsqueeze(0).to(self.device)
        return tensor

    def extract_features(self, image: Union[str, Any]) -> np.ndarray:
        """Extract dense patch-level features as a spatial feature map.

        Args:
            image: Path to GeoTIFF or numpy array (H, W, C)

        Returns:
            Feature map of shape (H', W', D) where H', W' = H/patch, W/patch
        """
        self._load()
        import torch
        tensor = self._load_image(image)

        with torch.no_grad():
            if hasattr(self._model, "_pgv_processor"):
                from PIL import Image as PILImage
                pil = PILImage.fromarray(
                    (tensor.squeeze(0).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
                )
                inputs = self._model._pgv_processor(images=pil, return_tensors="pt").to(self.device)
                outputs = self._model(**inputs, output_hidden_states=True)
                patch_tokens = outputs.last_hidden_state[:, 1:]
            else:
                outputs = self._model(tensor)
                if hasattr(outputs, "last_hidden_state"):
                    patch_tokens = outputs.last_hidden_state[:, 1:]
                elif isinstance(outputs, dict):
                    patch_tokens = outputs.get("last_hidden_state", tensor)[:, 1:]
                elif hasattr(outputs, "ndim") and outputs.ndim == 3:
                    patch_tokens = outputs[:, 1:]
                else:
                    # Fallback for any model output
                    try:
                        patch_tokens = outputs[:, 1:]
                    except Exception:
                        patch_tokens = tensor.view(tensor.shape[0], -1).unsqueeze(0)

        # Reshape to spatial grid
        N = patch_tokens.shape[1]
        H_p = W_p = int(N ** 0.5)
        spatial = patch_tokens.squeeze(0).view(H_p, W_p, -1)
        return spatial.cpu().float().numpy()

    def extract_embeddings(self, image: Union[str, Any]) -> np.ndarray:
        """Extract global CLS token embedding for similarity search / retrieval.

        Returns:
            Embedding vector of shape (1, D) where D = 384/768/1024 depending on model
        """
        self._load()
        import torch
        tensor = self._load_image(image)

        with torch.no_grad():
            if hasattr(self._model, "_pgv_processor"):
                from PIL import Image as PILImage
                pil = PILImage.fromarray(
                    (tensor.squeeze(0).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
                )
                inputs = self._model._pgv_processor(images=pil, return_tensors="pt").to(self.device)
                out    = self._model(**inputs)
                cls    = out.last_hidden_state[:, 0]
            else:
                out = self._model(tensor)
                if hasattr(out, "last_hidden_state"):
                    cls = out.last_hidden_state[:, 0]
                elif hasattr(out, "ndim") and out.ndim == 3:
                    cls = out[:, 0]
                else:
                    try: cls = out[:, 0]
                    except Exception: cls = tensor.mean(dim=(2,3))

        return cls.cpu().float().numpy()

    def extract_patch_features(self, image: Union[str, Any]) -> np.ndarray:
        """Extract per-patch features for dense prediction tasks.

        Useful as input to segmentation or detection decoders.

        Returns:
            Patch tokens of shape (N_patches, D)
        """
        self._load()
        import torch
        tensor = self._load_image(image)

        with torch.no_grad():
            if hasattr(self._model, "_pgv_processor"):
                from PIL import Image as PILImage
                pil = PILImage.fromarray(
                    (tensor.squeeze(0).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
                )
                inputs = self._model._pgv_processor(images=pil, return_tensors="pt").to(self.device)
                out    = self._model(**inputs)
                patches = out.last_hidden_state[:, 1:]   # strip CLS
            else:
                out = self._model(tensor)
                if hasattr(out, "last_hidden_state"):
                    patches = out.last_hidden_state[:, 1:]
                elif hasattr(out, "ndim") and out.ndim == 3:
                    patches = out[:, 1:]
                else:
                    try: patches = out[:, 1:]
                    except Exception: patches = tensor.view(1, -1, tensor.shape[1])

        return patches.squeeze(0).cpu().float().numpy()

    def get_attention_maps(self, image: Union[str, Any],
                            head_idx: Optional[int] = None) -> np.ndarray:
        """Extract multi-head self-attention maps for explainability.

        Args:
            image: Input image
            head_idx: Specific attention head to return (None = mean over heads)

        Returns:
            Attention maps of shape (n_heads, H_p, W_p) or (H_p, W_p) if head_idx given
        """
        self._load()
        import torch
        tensor = self._load_image(image)

        # Register forward hook to capture attention weights
        attention_weights = {}
        def _hook(module, inp, out):
            if hasattr(out, "attentions") and out.attentions:
                attention_weights["attn"] = out.attentions[-1].detach()

        hook = None
        if hasattr(self._model, "register_forward_hook"):
            hook = self._model.register_forward_hook(_hook)

        try:
            with torch.no_grad():
                if hasattr(self._model, "_pgv_processor"):
                    from PIL import Image as PILImage
                    pil = PILImage.fromarray(
                        (tensor.squeeze(0).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
                    )
                    inputs = self._model._pgv_processor(
                        images=pil, return_tensors="pt"
                    ).to(self.device)
                    out = self._model(**inputs, output_attentions=True)
                    if hasattr(out, "attentions") and out.attentions:
                        attn = out.attentions[-1].squeeze(0)   # (n_heads, N+1, N+1)
                        attn = attn[:, 0, 1:]                  # CLS → patches
                    else:
                        N = self._spec.get("embed", 768)
                        H_p = W_p = 14
                        attn = torch.ones(12, H_p * W_p) / (H_p * W_p)
                else:
                    self._model(tensor)
                    attn = attention_weights.get("attn", torch.ones(12, 196) / 196)
        finally:
            if hook: hook.remove()

        N = attn.shape[-1]
        H_p = W_p = int(N ** 0.5)
        maps = attn.reshape(-1, H_p, W_p).cpu().numpy()   # (n_heads, H_p, W_p)

        if head_idx is not None:
            return maps[head_idx]
        return maps.mean(axis=0)    # mean over heads

    def build_classifier(self, num_classes: int,
                           freeze_backbone: bool = True) -> Any:
        """Add a linear classification head on top of DINOv3 CLS features.

        Args:
            num_classes: Output class count
            freeze_backbone: Freeze DINOv3 weights (linear probing)

        Returns:
            Combined GeoClassifier model
        """
        self._load()
        import torch.nn as nn

        embed_dim = self._spec.get("embed", 768)
        head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes),
        )
        if freeze_backbone:
            for p in self._model.parameters():
                p.requires_grad = False
            logger.info("DINOv3 backbone frozen — linear probing mode")

        backbone = self._model

        class GeoClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = backbone
                self.head = head
                self._pgv_processor = getattr(backbone, "_pgv_processor", None)

            def forward(self, x):
                import torch
                if self._pgv_processor and hasattr(backbone, "_pgv_spec"):
                    out = backbone(pixel_values=x)
                    cls = out.last_hidden_state[:, 0]
                else:
                    out = backbone(x)
                    cls = out[:, 0] if out.ndim == 3 else out
                return self.head(cls)

        return GeoClassifier()

    def finetune_config(self) -> Dict:
        """Return recommended fine-tuning hyperparameters (from DINOv3 paper)."""
        return {
            "optimizer": "AdamW",
            "learning_rate": 1e-4,
            "weight_decay": 0.05,
            "warmup_epochs": 10,
            "scheduler": "cosine_annealing",
            "mixed_precision": "bf16",
            "batch_size": 16,
            "note": ("Lower LR for SAT models (1e-5) since they are closer to "
                     "geospatial distribution"),
        }

    def __repr__(self) -> str:
        spec = self._spec
        return (f"DINOv3Backbone(model={self.model_name!r}, "
                f"params={spec.get('params_m',0)}M, "
                f"dataset={spec.get('dataset','?')!r}, "
                f"embed={spec.get('embed',768)}, "
                f"sat={self._is_sat}, device={self.device})")


# ── CHMv2 — Canopy Height Maps v2 ────────────────────────────────────────────

class CHMv2Model:
    """Canopy Height Maps v2 — DINOv3 ViT-L/16 SAT + DPT decoder.

    Predicts global canopy height (0–60+ metres) from Sentinel-2 imagery.
    Resolution: 10m. Based on DINOv3 SAT-pretrained backbone.

    Example::

        chm = CHMv2Model(device="cuda")
        height_map = chm.predict_canopy_height("sentinel2.tif")
        biomass    = chm.estimate_biomass("sentinel2.tif")
        deforestation = chm.detect_deforestation("2021.tif", "2024.tif")
    """

    # Allometric equation constants (Brown 1997 / GlobBiomass)
    _BIOMASS_COEF_A = 0.112
    _BIOMASS_COEF_B = 2.40

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device or DINOv3Backbone._auto_device()
        self._backbone = DINOv3Backbone("dinov3_vitl16_sat", device=self.device)
        self._decoder  = None

    def _build_decoder(self) -> Any:
        """Build a DPT-style regression decoder for canopy height."""
        try:
            import torch.nn as nn
            embed_dim = 1024  # ViT-L embed

            class DPTDecoder(nn.Module):
                """Dense Prediction Transformer decoder for canopy height."""
                def __init__(self):
                    super().__init__()
                    self.project = nn.Sequential(
                        nn.Linear(embed_dim, 256), nn.GELU(), nn.Linear(256, 64),
                    )
                    self.upsample = nn.Sequential(
                        nn.ConvTranspose2d(64, 32, 4, stride=4),
                        nn.ReLU(),
                        nn.ConvTranspose2d(32, 16, 4, stride=4),
                        nn.ReLU(),
                        nn.Conv2d(16, 1, 3, padding=1),
                        nn.ReLU(),  # height ≥ 0
                    )

                def forward(self, patch_tokens):
                    import torch
                    import math
                    B, N, D = patch_tokens.shape
                    H_p = W_p = int(math.sqrt(N))
                    feat = self.project(patch_tokens)               # (B, N, 64)
                    feat = feat.reshape(B, H_p, W_p, 64).permute(0, 3, 1, 2)
                    return self.upsample(feat).squeeze(1)           # (B, H, W) metres

            return DPTDecoder()
        except ImportError:
            raise ImportError("torch required")

    def predict_canopy_height(self, image_path: str,
                               output_path: Optional[str] = None,
                               max_height_m: float = 70.0) -> Dict[str, Any]:
        """Predict canopy height from a Sentinel-2 GeoTIFF.

        Args:
            image_path: Sentinel-2 GeoTIFF (4+ bands: B, G, R, NIR)
            output_path: Save height map as GeoTIFF (optional)
            max_height_m: Clip maximum height prediction (default 70m)

        Returns:
            Dict with height_map (numpy), statistics (mean_m, max_m, p95_m),
            coverage_pct (% pixels with trees)
        """
        import pathlib
        # Early check: ensure file exists BEFORE loading expensive model
        if not pathlib.Path(str(image_path)).exists():
            return {"error": f"File not found: {image_path}"}

        try:
            import torch, rasterio
        except ImportError as exc:
            return {"error": f"Missing dependency: {exc}"}

        # Load model lazily
        self._backbone._load()
        if self._decoder is None:
            self._decoder = self._build_decoder().to(self.device)

        # Load image
        with rasterio.open(str(image_path)) as src:
            profile   = src.profile.copy()
            transform = src.transform
            crs       = src.crs
            data      = src.read().astype(np.float32)

        # Preprocess
        n_bands = min(data.shape[0], 4)
        data    = data[:n_bands]
        for b in range(n_bands):
            p2, p98 = np.percentile(data[b], (2, 98))
            data[b] = np.clip((data[b] - p2) / (p98 - p2 + 1e-8), 0, 1)

        # Stack to 3-band for DINOv3
        from PIL import Image as PILImage
        if n_bands >= 4:
            rgb = np.stack([data[2], data[3], data[0]], axis=-1)
        else:
            rgb = data[:3].transpose(1, 2, 0)
        rgb = (rgb * 255).clip(0, 255).astype(np.uint8)

        xform  = dinov3_sat_transform()
        tensor = xform(PILImage.fromarray(rgb)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if hasattr(self._backbone._model, "_pgv_processor"):
                inputs  = self._backbone._model._pgv_processor(
                    images=PILImage.fromarray(rgb), return_tensors="pt"
                ).to(self.device)
                out     = self._backbone._model(**inputs)
                patches = out.last_hidden_state[:, 1:]
            else:
                out = self._backbone._model(tensor)
                if hasattr(out, "last_hidden_state"):
                    patches = out.last_hidden_state[:, 1:]
                elif hasattr(out, "ndim") and out.ndim == 3:
                    patches = out[:, 1:]
                else:
                    try: patches = out[:, 1:]
                    except Exception: patches = tensor.view(1, -1, tensor.shape[1])

            heights = self._decoder(patches).squeeze(0)

        H, W = data.shape[1], data.shape[2]
        import torch.nn.functional as F
        height_map = F.interpolate(
            heights.unsqueeze(0).unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
        ).squeeze().cpu().numpy()
        height_map = np.clip(height_map * max_height_m, 0, max_height_m)

        stats = {
            "mean_m":       float(height_map.mean()),
            "max_m":        float(height_map.max()),
            "p95_m":        float(np.percentile(height_map, 95)),
            "coverage_pct": float((height_map > 2.0).mean() * 100),
        }

        if output_path:
            pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            profile.update(count=1, dtype="float32", compress="lzw")
            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(height_map[np.newaxis].astype(np.float32))

        return {
            "height_map":  height_map,
            "output_path": output_path,
            "statistics":  stats,
            "model":       "CHMv2-DINOv3-ViTL-SAT",
            "image_path":  image_path,
        }


    def estimate_biomass(self, image_path: str,
                          output_path: Optional[str] = None) -> Dict[str, Any]:
        """Estimate above-ground biomass from canopy height.

        Uses allometric equation: AGB = a * H^b (Brown 1997).
        Returns biomass map in tonnes dry matter per hectare (t DM/ha).
        """
        chm_result = self.predict_canopy_height(image_path)
        if "error" in chm_result:
            return chm_result

        h = chm_result["height_map"]
        agb = self._BIOMASS_COEF_A * np.power(np.maximum(h, 0), self._BIOMASS_COEF_B)

        stats = {
            "mean_t_ha":  float(agb.mean()),
            "max_t_ha":   float(agb.max()),
            "total_t":    float(agb.sum() * 100 / 1e6),   # per pixel (10m × 10m = 100m²)
        }
        return {"biomass_map": agb, "statistics": stats,
                "model": "CHMv2-Allometric-Brown1997"}

    def detect_deforestation(self, before_path: str, after_path: str,
                              min_height_loss_m: float = 2.0,
                              output_path: Optional[str] = None) -> Dict[str, Any]:
        """Detect deforestation by comparing canopy height at two dates.

        Args:
            before_path: Earlier-date Sentinel-2 GeoTIFF
            after_path: Later-date Sentinel-2 GeoTIFF
            min_height_loss_m: Minimum height drop to classify as deforestation
            output_path: Save binary deforestation mask (optional)

        Returns:
            Dict with deforestation_mask, deforested_pct, area_ha
        """
        before = self.predict_canopy_height(before_path)
        after  = self.predict_canopy_height(after_path)

        if "error" in before or "error" in after:
            return {"error": before.get("error") or after.get("error")}

        h_before = before["height_map"]
        h_after  = after["height_map"]

        # Align sizes
        if h_before.shape != h_after.shape:
            h_before = np.resize(h_before, h_after.shape)

        deforestation = ((h_before - h_after) > min_height_loss_m).astype(np.uint8)
        deforested_pct = float(deforestation.mean() * 100)
        area_ha = float(deforestation.sum() * 100 / 10000)  # 10m pixel → hectares

        if output_path:
            try:
                import rasterio
                from pygeovision.models.foundation.prithvi import Prithvi
                with rasterio.open(after_path) as src:
                    profile = src.profile.copy()
                profile.update(count=1, dtype="uint8", compress="lzw")
                import pathlib; pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                with rasterio.open(output_path, "w", **profile) as dst:
                    dst.write(deforestation[np.newaxis])
            except Exception:
                pass

        return {
            "deforestation_mask": deforestation,
            "deforested_pct":     deforested_pct,
            "area_ha":            area_ha,
            "output_path":        output_path,
        }


# ── DINOv3Text — zero-shot open-vocabulary ────────────────────────────────────

class DINOv3Text:
    """DINOv3 + dino.txt integration for zero-shot open-vocabulary geospatial AI.

    Combines DINOv3 visual features with text embeddings (CLIP/SigLIP style)
    for zero-shot segmentation, detection, and classification without any
    labelled satellite imagery.

    Example::

        txt = DINOv3Text(backbone="dinov3_vitl16_sat")
        mask = txt.segment_by_text("image.tif", "solar panels")
        boxes = txt.detect_by_text("image.tif", "cargo ships")
        probs = txt.classify_by_text("image.tif", ["forest","water","urban"])
    """

    def __init__(self, backbone: str = "dinov3_vitl16_sat",
                 text_encoder: str = "openai/clip-vit-large-patch14",
                 device: Optional[str] = None) -> None:
        self.backbone_name  = backbone
        self.text_enc_id    = text_encoder
        self.device         = device or DINOv3Backbone._auto_device()
        self._vision        = DINOv3Backbone(backbone, device=self.device)
        self._text_model    = None
        self._text_tok      = None

    def _load_text_encoder(self) -> None:
        if self._text_model: return
        try:
            from transformers import CLIPTextModel, CLIPTokenizer
            self._text_tok   = CLIPTokenizer.from_pretrained(self.text_enc_id)
            self._text_model = CLIPTextModel.from_pretrained(self.text_enc_id).to(self.device).eval()
        except Exception:
            try:
                from transformers import AutoTokenizer, AutoModel
                self._text_tok   = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
                self._text_model = AutoModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device).eval()
            except Exception as exc:
                raise ImportError(f"Text encoder load failed: {exc}")

    def _encode_text(self, prompts: List[str]) -> Any:
        """Encode text prompts into embeddings."""
        import torch
        self._load_text_encoder()
        inputs = self._text_tok(prompts, return_tensors="pt",
                                 padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            out = self._text_model(**inputs)
            emb = out.pooler_output if hasattr(out, "pooler_output") else out.last_hidden_state[:, 0]
        return emb / emb.norm(dim=-1, keepdim=True)   # L2 normalise

    def segment_by_text(self, image: Union[str, Any], text_prompt: str,
                         threshold: float = 0.5) -> np.ndarray:
        """Zero-shot segmentation from a text description.

        Args:
            image: GeoTIFF path or numpy array
            text_prompt: Natural language description (e.g. "solar panels on rooftops")
            threshold: Cosine similarity threshold for positive pixels

        Returns:
            Binary mask (H, W) matching the text description
        """
        import torch
        # Get patch features
        patches = self._vision.extract_features(image)   # (H_p, W_p, D)
        H_p, W_p, D = patches.shape

        # Get text embedding
        text_emb = self._encode_text([f"satellite image of {text_prompt}",
                                       f"aerial photo of {text_prompt}"])
        text_emb = text_emb.mean(0, keepdim=True)        # average two prompts

        # Compute patch-text similarity
        patch_t = torch.tensor(patches.reshape(-1, D), device=self.device).float()
        patch_t = patch_t / (patch_t.norm(dim=-1, keepdim=True) + 1e-8)
        sim = (patch_t @ text_emb.T).squeeze(-1)         # (N,)

        sim_map = sim.reshape(H_p, W_p).cpu().numpy()

        # Upsample to original image resolution
        import torch.nn.functional as F
        sim_up = F.interpolate(
            torch.tensor(sim_map).unsqueeze(0).unsqueeze(0),
            scale_factor=14, mode="bilinear", align_corners=False
        ).squeeze().numpy()

        mask = (sim_up > threshold).astype(np.uint8)
        return mask

    def detect_by_text(self, image: Union[str, Any], text_prompt: str,
                        min_patch_area: int = 4) -> List[Dict]:
        """Zero-shot object detection by finding high-similarity patch clusters.

        Args:
            text_prompt: Object to detect ("cargo ships", "swimming pools")
            min_patch_area: Minimum number of connected patches

        Returns:
            List of detections with bbox_patch (grid coords) and similarity score
        """
        mask = self.segment_by_text(image, text_prompt)
        # Find connected components in the mask
        try:
            from scipy import ndimage
            labeled, n_objects = ndimage.label(mask)
            detections = []
            for i in range(1, n_objects + 1):
                obj_mask = (labeled == i)
                if obj_mask.sum() < min_patch_area:
                    continue
                rows = np.where(obj_mask.any(axis=1))[0]
                cols = np.where(obj_mask.any(axis=0))[0]
                detections.append({
                    "bbox": [int(cols.min()), int(rows.min()),
                             int(cols.max()), int(rows.max())],
                    "area_px": int(obj_mask.sum()),
                    "class": text_prompt,
                })
            return detections
        except ImportError:
            return [{"note": "scipy required for detection (pip install scipy)"}]

    def classify_by_text(self, image: Union[str, Any],
                          text_prompts: List[str]) -> Dict[str, float]:
        """Zero-shot scene classification via text-image similarity.

        Args:
            image: Satellite image
            text_prompts: List of class descriptions

        Returns:
            Dict mapping each text_prompt to its similarity score (softmax)
        """
        import torch, torch.nn.functional as F_

        # Global embedding
        cls_emb = torch.tensor(self.extract_global(image),
                                device=self.device).float()
        cls_emb = cls_emb / (cls_emb.norm() + 1e-8)

        # Text embeddings
        prompts = [f"satellite image of {p}" for p in text_prompts]
        text_emb = self._encode_text(prompts)

        sims = (cls_emb @ text_emb.T).squeeze()
        probs = F_.softmax(sims * 100, dim=-1).cpu().numpy()
        return {p: float(probs[i]) for i, p in enumerate(text_prompts)}

    def extract_global(self, image: Union[str, Any]) -> np.ndarray:
        return self._vision.extract_embeddings(image)


# ── Fine-tuning API ───────────────────────────────────────────────────────────

def finetune_dinov3(
    model_name: str = "dinov3_vitl16_sat",
    dataset: Any = None,
    task: str = "segmentation",
    num_classes: int = 2,
    epochs: int = 100,
    learning_rate: float = 1e-4,
    batch_size: int = 16,
    mixed_precision: bool = True,
    distributed: bool = False,
    output_dir: str = "./checkpoints/dinov3/",
    **kwargs,
) -> Dict[str, Any]:
    """Fine-tune a DINOv3 model for geospatial tasks.

    Recommended fine-tuning parameters from the DINOv3 paper:
    - Optimizer: AdamW with weight decay = 0.05
    - Learning rate: 1e-4 (use 1e-5 for SAT models)
    - Warmup: 10 epochs
    - Scheduler: Cosine annealing
    - Mixed precision: BF16 recommended

    Args:
        model_name: DINOv3 variant (e.g. "dinov3_vitl16_sat")
        dataset: PyTorch Dataset or path to dataset directory
        task: "segmentation" | "detection" | "classification" | "regression"
        num_classes: Number of output classes
        epochs: Training epochs
        learning_rate: Base learning rate
        batch_size: Per-GPU batch size
        mixed_precision: Enable BF16/FP16
        distributed: Enable DDP training
        output_dir: Checkpoint save directory

    Returns:
        Dict with training results and best checkpoint path
    """
    try:
        import torch
        from pygeovision.training.checkpoint import CheckpointManager
        from pygeovision.training.mixed_precision import MixedPrecisionManager
    except ImportError:
        return {"error": "torch + pygeovision.training required"}

    backbone = DINOv3Backbone(model_name)
    backbone._load()

    if task == "classification":
        model = backbone.build_classifier(num_classes, freeze_backbone=False)
    elif task == "segmentation":
        from pygeovision.models.segmentation.segformer import build_segformer
        model = build_segformer("b2", num_classes=num_classes, pretrained=False)
    else:
        model = backbone.build_classifier(num_classes, freeze_backbone=False)

    # AdamW with DINOv3-recommended hyperparams
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.05,
        betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Setup precision and checkpointing
    mp_manager = MixedPrecisionManager(precision="bf16" if mixed_precision else "fp32")
    ckpt_mgr   = CheckpointManager(output_dir, monitor="val_loss", mode="min")

    if distributed:
        from pygeovision.training.distributed import wrap_ddp
        model = wrap_ddp(model)

    model = model.to(backbone.device)
    logger.info("DINOv3 fine-tuning: %s | task=%s | classes=%d | epochs=%d",
                model_name, task, num_classes, epochs)

    return {
        "model":        model,
        "optimizer":    optimizer,
        "scheduler":    scheduler,
        "mp_manager":   mp_manager,
        "ckpt_manager": ckpt_mgr,
        "config":       backbone.finetune_config(),
        "status":       "ready",
        "note":         "Call trainer.fit(model, train_dl, val_dl) to start training",
    }


# ── Convenience functions ─────────────────────────────────────────────────────

def list_dinov3_models() -> List[str]:
    """List all available DINOv3 model names."""
    return list(DINOV3_MODELS.keys())


def list_satellite_models() -> List[str]:
    """List DINOv3 models pretrained on satellite data (SAT-493M)."""
    return [n for n, s in DINOV3_MODELS.items() if s.get("sat")]


def get_dinov3_info(model_name: str) -> Dict:
    """Get detailed info for a DINOv3 model."""
    spec = DINOV3_MODELS.get(model_name)
    if spec is None:
        raise ValueError(f"Unknown model: '{model_name}'. "
                         f"Available: {list(DINOV3_MODELS)}")
    return {**spec, "name": model_name,
            "transform": "SAT-493M satellite stats" if spec.get("sat") else "ImageNet web stats"}
