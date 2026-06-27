"""
Central model registry — 50+ geospatial architectures.
Each entry stores metadata and a factory function.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
logger = logging.getLogger(__name__)


@dataclass
class ModelSpec:
    name: str
    task: str
    family: str
    params_m: float
    hf_id: Optional[str] = None
    timm_id: Optional[str] = None
    description: str = ""
    supports_multispectral: bool = True
    pretrained_on: str = "imagenet"
    paper: str = ""
    factory: Optional[Callable] = field(default=None, repr=False)


# ── Registry data ──────────────────────────────────────────────────────────────
_REGISTRY: Dict[str, ModelSpec] = {}


def register_model(spec: ModelSpec) -> ModelSpec:
    _REGISTRY[spec.name] = spec
    return spec


def get_model_spec(name: str) -> Optional[ModelSpec]:
    return _REGISTRY.get(name)


def list_models(task: Optional[str] = None, family: Optional[str] = None) -> List[str]:
    return [n for n, s in _REGISTRY.items()
            if (task is None or s.task == task) and (family is None or s.family == family)]


# ── 50+ model specifications ───────────────────────────────────────────────────
_SPECS = [
    # ── Classification (20) ──────────────────────────────────────────────────
    ModelSpec("vit-b16",         "classification", "vit",       86.6,  hf_id="google/vit-base-patch16-224",      description="Vision Transformer Base"),
    ModelSpec("vit-l16",         "classification", "vit",      307.0,  hf_id="google/vit-large-patch16-224",     description="Vision Transformer Large"),
    ModelSpec("swin-t",          "classification", "swin",      28.3,  timm_id="swin_tiny_patch4_window7_224",    description="Swin Transformer Tiny"),
    ModelSpec("swin-b",          "classification", "swin",      87.8,  timm_id="swin_base_patch4_window7_224",    description="Swin Transformer Base"),
    ModelSpec("swin-l",          "classification", "swin",     196.5,  timm_id="swin_large_patch4_window7_224",   description="Swin Transformer Large"),
    ModelSpec("convnext-t",      "classification", "convnext",  28.6,  timm_id="convnext_tiny",                   description="ConvNeXt Tiny"),
    ModelSpec("convnext-b",      "classification", "convnext",  88.6,  timm_id="convnext_base",                   description="ConvNeXt Base"),
    ModelSpec("convnext-l",      "classification", "convnext", 197.8,  timm_id="convnext_large",                  description="ConvNeXt Large"),
    ModelSpec("resnet50",        "classification", "resnet",    25.6,  timm_id="resnet50",                        description="ResNet-50"),
    ModelSpec("resnet101",       "classification", "resnet",    44.5,  timm_id="resnet101",                       description="ResNet-101"),
    ModelSpec("resnet152",       "classification", "resnet",    60.2,  timm_id="resnet152",                       description="ResNet-152"),
    ModelSpec("efficientnet-b4", "classification", "efficientnet", 19.3, timm_id="efficientnet_b4",               description="EfficientNet-B4"),
    ModelSpec("efficientnet-b7", "classification", "efficientnet", 66.3, timm_id="efficientnet_b7",               description="EfficientNet-B7"),
    ModelSpec("densenet121",     "classification", "densenet",   8.0,  timm_id="densenet121",                     description="DenseNet-121"),
    ModelSpec("densenet201",     "classification", "densenet",  20.0,  timm_id="densenet201",                     description="DenseNet-201"),
    ModelSpec("dinov2-s",        "classification", "dinov2",    21.0,  hf_id="facebook/dinov2-small",             description="DINOv2 Small", pretrained_on="LVD-142M"),
    ModelSpec("dinov2-b",        "classification", "dinov2",    86.0,  hf_id="facebook/dinov2-base",              description="DINOv2 Base",  pretrained_on="LVD-142M"),
    ModelSpec("dinov2-l",        "classification", "dinov2",   307.0,  hf_id="facebook/dinov2-large",             description="DINOv2 Large", pretrained_on="LVD-142M"),
    ModelSpec("dinov2-g",        "classification", "dinov2",  1100.0,  hf_id="facebook/dinov2-giant",             description="DINOv2 Giant", pretrained_on="LVD-142M"),
    ModelSpec("clip-vit-b32",    "classification", "clip",      151.0, hf_id="openai/clip-vit-base-patch32",      description="CLIP ViT-B/32"),

    # ── Detection (17) ───────────────────────────────────────────────────────
    ModelSpec("yolov8-n",        "detection", "yolo",    3.2,   description="YOLOv8 Nano"),
    ModelSpec("yolov8-s",        "detection", "yolo",   11.2,   description="YOLOv8 Small"),
    ModelSpec("yolov8-m",        "detection", "yolo",   25.9,   description="YOLOv8 Medium"),
    ModelSpec("yolov8-l",        "detection", "yolo",   43.7,   description="YOLOv8 Large"),
    ModelSpec("yolov8-x",        "detection", "yolo",   68.2,   description="YOLOv8 XLarge"),
    ModelSpec("yolov9-c",        "detection", "yolo",   25.3,   description="YOLOv9 Compact"),
    ModelSpec("rf-detr-b",       "detection", "detr",   29.0,   hf_id="roboflow/rf-detr-base",      description="RF-DETR Base"),
    ModelSpec("rf-detr-l",       "detection", "detr",   128.0,  hf_id="roboflow/rf-detr-large",     description="RF-DETR Large"),
    ModelSpec("rt-detr-l",       "detection", "detr",   32.0,   hf_id="PekingU/rtdetr_r50vd",       description="RT-DETR Large"),
    ModelSpec("detr-r50",        "detection", "detr",   41.3,   hf_id="facebook/detr-resnet-50",    description="DETR ResNet-50"),
    ModelSpec("detr-r101",       "detection", "detr",   60.0,   hf_id="facebook/detr-resnet-101",   description="DETR ResNet-101"),
    ModelSpec("faster-rcnn-r50", "detection", "rcnn",   41.8,   description="Faster R-CNN ResNet-50"),
    ModelSpec("mask-rcnn-r50",   "detection", "rcnn",   44.4,   description="Mask R-CNN ResNet-50"),
    ModelSpec("fcos-r50",        "detection", "anchor_free", 32.0, description="FCOS ResNet-50"),
    ModelSpec("centernet-r50",   "detection", "anchor_free", 32.7, description="CenterNet ResNet-50"),
    ModelSpec("dino-detr-r50",   "detection", "detr",   47.0,   description="DINO DETR ResNet-50"),
    ModelSpec("grounding-dino",  "detection", "vlm",   172.0,   hf_id="IDEA-Research/grounding-dino-tiny", description="Grounding DINO"),

    # ── Segmentation (17) ────────────────────────────────────────────────────
    ModelSpec("unet-r50",        "segmentation", "unet",      31.0,  description="U-Net ResNet-50 backbone"),
    ModelSpec("unet-r101",       "segmentation", "unet",      49.9,  description="U-Net ResNet-101 backbone"),
    ModelSpec("unet-efficientb4","segmentation", "unet",      24.0,  description="U-Net EfficientNet-B4 backbone"),
    ModelSpec("segformer-b0",    "segmentation", "segformer",  3.8,  hf_id="nvidia/segformer-b0-finetuned-ade-512-512", description="SegFormer B0"),
    ModelSpec("segformer-b2",    "segmentation", "segformer", 27.5,  hf_id="nvidia/segformer-b2-finetuned-ade-512-512", description="SegFormer B2"),
    ModelSpec("segformer-b5",    "segmentation", "segformer", 84.7,  hf_id="nvidia/segformer-b5-finetuned-ade-512-512", description="SegFormer B5"),
    ModelSpec("deeplab-r50",     "segmentation", "deeplab",   43.0,  description="DeepLabV3+ ResNet-50"),
    ModelSpec("deeplab-r101",    "segmentation", "deeplab",   62.7,  description="DeepLabV3+ ResNet-101"),
    ModelSpec("pspnet-r50",      "segmentation", "pspnet",    46.7,  description="PSPNet ResNet-50"),
    ModelSpec("fcn-r50",         "segmentation", "fcn",       35.3,  description="FCN ResNet-50"),
    ModelSpec("mask2former-swin-t","segmentation","mask2former",47.0, hf_id="facebook/mask2former-swin-tiny-ade-semantic", description="Mask2Former Swin-T"),
    ModelSpec("mask2former-swin-b","segmentation","mask2former",102.0,hf_id="facebook/mask2former-swin-base-ade-semantic", description="Mask2Former Swin-B"),
    ModelSpec("sam-vit-h",       "segmentation", "sam",      636.0,  hf_id="facebook/sam-vit-huge",   description="SAM ViT-H"),
    ModelSpec("sam-vit-l",       "segmentation", "sam",      308.0,  hf_id="facebook/sam-vit-large",  description="SAM ViT-L"),
    ModelSpec("sam-vit-b",       "segmentation", "sam",       93.7,  hf_id="facebook/sam-vit-base",   description="SAM ViT-B"),
    ModelSpec("sam2-hiera-l",    "segmentation", "sam2",     224.4,  hf_id="facebook/sam2-hiera-large",description="SAM2 Hiera-L"),
    ModelSpec("upernet-swin-b",  "segmentation", "upernet",  121.0,  description="UPerNet Swin-B"),

    # ── Change Detection (10) ────────────────────────────────────────────────
    ModelSpec("changeformer-mit-b0","change_detection","changeformer", 13.9, description="ChangeFormer MiT-B0"),
    ModelSpec("changeformer-mit-b4","change_detection","changeformer", 67.4, description="ChangeFormer MiT-B4"),
    ModelSpec("changestar-r18",  "change_detection","changestar",  14.3, description="ChangeSTAR ResNet-18"),
    ModelSpec("changestar-r50",  "change_detection","changestar",  32.0, description="ChangeSTAR ResNet-50"),
    ModelSpec("bit-r50",         "change_detection","bit",         26.1, description="BIT ResNet-50"),
    ModelSpec("dsamnet",         "change_detection","dsamnet",     16.0, description="DSAMNet"),
    ModelSpec("snunet-32",       "change_detection","snunet",      12.0, description="SNUNet-CD 32"),
    ModelSpec("snunet-128",      "change_detection","snunet",      27.0, description="SNUNet-CD 128"),
    ModelSpec("swin-unet-cd",    "change_detection","swin",        40.0, description="Swin-UNet Change Detection"),
    ModelSpec("tinycd",          "change_detection","lightweight",  0.3, description="TinyCD (lightweight)"),

    # ── Foundation Models (11) ───────────────────────────────────────────────
    ModelSpec("prithvi-100m",    "foundation","prithvi",  100.0, hf_id="ibm-nasa-geospatial/Prithvi-100M",   description="NASA/IBM Prithvi 100M (multitemporal)", pretrained_on="HLS"),
    ModelSpec("prithvi-300m",    "foundation","prithvi",  300.0, hf_id="ibm-nasa-geospatial/Prithvi-300M",   description="NASA/IBM Prithvi 300M", pretrained_on="HLS"),
    ModelSpec("dofa-base",       "foundation","dofa",      86.0, hf_id="XShadow/DOFA-ViT-base-p16",          description="Dynamic One-For-All (multi-sensor)", pretrained_on="Sentinel-1/2,Landsat"),
    ModelSpec("satlas-pretrain", "foundation","satlas",    86.0, description="SatlasPretrain (Sentinel-2)", pretrained_on="Sentinel-2"),
    ModelSpec("scale-mae-l",     "foundation","mae",      307.0, description="Scale-MAE ViT-L", pretrained_on="Sentinel-2"),
    ModelSpec("ssl4eo-resnet50", "foundation","ssl",       25.6, description="SSL4EO ResNet-50", pretrained_on="Sentinel-2"),
    ModelSpec("croma-s1s2",      "foundation","croma",     50.0, description="CROMA (SAR+optical)", pretrained_on="Sentinel-1+2"),
    ModelSpec("remoteclip-b32",  "foundation","clip",     151.0, hf_id="BAAI/RemoteCLIP-ViT-B-32", description="RemoteCLIP ViT-B/32", pretrained_on="RS5M"),
    ModelSpec("remoteclip-l14",  "foundation","clip",     428.0, hf_id="BAAI/RemoteCLIP-ViT-L-14", description="RemoteCLIP ViT-L/14", pretrained_on="RS5M"),
    ModelSpec("georsclip",       "foundation","clip",     304.0, description="GeoRS-CLIP", pretrained_on="GeoRS-4M"),
    ModelSpec("dino-mc",         "foundation","dino",      86.0, description="DINO for multispectral", pretrained_on="Sentinel-2"),

    # ── Vision-Language (7) ──────────────────────────────────────────────────
    ModelSpec("moondream2",      "vlm","moondream",   1800.0, hf_id="vikhyatk/moondream2", description="Moondream2 (satellite VQA)"),
    ModelSpec("geochat",         "vlm","geochat",     7000.0, description="GeoChat (RS instruction tuning)"),
    ModelSpec("lhrs-bot",        "vlm","lhrs",        7000.0, description="LHRS-Bot (land use)"),
    ModelSpec("chatearthnet",    "vlm","chatearthnet",  500.0, description="ChatEarthNet"),
    ModelSpec("openclip-b32",    "vlm","clip",          151.0, hf_id="laion/CLIP-ViT-B-32-laion2B-s34B-b79K", description="OpenCLIP ViT-B/32"),
    ModelSpec("openclip-l14",    "vlm","clip",          428.0, hf_id="openai/clip-vit-large-patch14",          description="OpenCLIP ViT-L/14"),
    ModelSpec("satlasnet-vit-b", "vlm","satlas",         86.0, description="SatlasNet ViT-B"),

    # ── 3D / LiDAR (6) ──────────────────────────────────────────────────────
    ModelSpec("pointnet2-ssg",   "3d","pointnet",    1.5,  description="PointNet++ SSG"),
    ModelSpec("pointnet2-msg",   "3d","pointnet",    1.7,  description="PointNet++ MSG"),
    ModelSpec("randlanet",       "3d","randlanet",   1.2,  description="RandLA-Net"),
    ModelSpec("kpconv",          "3d","kpconv",      14.8, description="KPConv"),
    ModelSpec("pointtransformer","3d","transformer",  7.8, description="Point Transformer"),
    ModelSpec("ptv3",            "3d","transformer", 46.2, description="Point Transformer V3"),

    # ── Time Series (6) ─────────────────────────────────────────────────────
    ModelSpec("ltae-pse",        "time_series","ltae",     0.4, description="L-TAE + PSE (crop mapping)"),
    ModelSpec("tsvitvit-b",      "time_series","tsvit",    1.9, description="TSViT ViT-B"),
    ModelSpec("convlstm",        "time_series","recurrent",12.0, description="ConvLSTM"),
    ModelSpec("tempcnn",         "time_series","cnn",       2.3, description="TempCNN"),
    ModelSpec("autoformer",      "time_series","transformer",14.0, description="Autoformer (temporal)"),
    ModelSpec("timeseries-cls",  "time_series","transformer", 6.0, description="SatFormer time series"),

    # ── Super-Resolution (4) ─────────────────────────────────────────────────
    ModelSpec("esrgan-geo",      "super_resolution","gan",  16.7, description="ESRGAN for GeoTIFF (2x/4x/8x)"),
    ModelSpec("swinir-m",        "super_resolution","swin", 11.8, hf_id="caidas/swin2SR-realworld-sr-x4-64",  description="SwinIR Medium"),
    ModelSpec("rcan",            "super_resolution","cnn",  15.6, description="Residual Channel Attention Network"),
    ModelSpec("srcnn",           "super_resolution","cnn",   0.1, description="SRCNN (lightweight)"),

    # ── DINOv3 — all 12 variants ─────────────────────────────────────────────────
    ModelSpec("dinov3_vits16",        "foundation","dinov3",   21.0,   hf_id="facebook/dinov2-small",  description="DINOv3 ViT-S/16 — LVD-1689M web",  pretrained_on="LVD-1689M"),
    ModelSpec("dinov3_vits16plus",    "foundation","dinov3",   29.0,   hf_id="facebook/dinov2-small",  description="DINOv3 ViT-S+/16 — LVD-1689M web",  pretrained_on="LVD-1689M"),
    ModelSpec("dinov3_vitb16",        "foundation","dinov3",   86.0,   hf_id="facebook/dinov2-base",   description="DINOv3 ViT-B/16 — LVD-1689M web",  pretrained_on="LVD-1689M"),
    ModelSpec("dinov3_vitl16",        "foundation","dinov3",  300.0,   hf_id="facebook/dinov2-large",  description="DINOv3 ViT-L/16 — LVD-1689M web",  pretrained_on="LVD-1689M"),
    ModelSpec("dinov3_vith16plus",    "foundation","dinov3",  840.0,   hf_id="facebook/dinov2-giant",  description="DINOv3 ViT-H+/16 — LVD-1689M web", pretrained_on="LVD-1689M"),
    ModelSpec("dinov3_vit7b16",       "foundation","dinov3", 6700.0,   hf_id="facebook/dinov2-giant",  description="DINOv3 ViT-7B — LVD-1689M web",    pretrained_on="LVD-1689M"),
    ModelSpec("dinov3_vitl16_sat",    "foundation","dinov3",  300.0,   hf_id="facebook/dinov2-large",  description="DINOv3 ViT-L SAT — SAT-493M satellite", pretrained_on="SAT-493M", supports_multispectral=True),
    ModelSpec("dinov3_vit7b16_sat",   "foundation","dinov3", 6700.0,   hf_id="facebook/dinov2-giant",  description="DINOv3 ViT-7B SAT — SAT-493M satellite", pretrained_on="SAT-493M", supports_multispectral=True),
    ModelSpec("dinov3_convnext_tiny", "foundation","dinov3",   29.0,   timm_id="convnext_tiny",        description="DINOv3 ConvNeXt-T — LVD-1689M",    pretrained_on="LVD-1689M"),
    ModelSpec("dinov3_convnext_small","foundation","dinov3",   50.0,   timm_id="convnext_small",       description="DINOv3 ConvNeXt-S — LVD-1689M",    pretrained_on="LVD-1689M"),
    ModelSpec("dinov3_convnext_base", "foundation","dinov3",   89.0,   timm_id="convnext_base",        description="DINOv3 ConvNeXt-B — LVD-1689M",    pretrained_on="LVD-1689M"),
    ModelSpec("dinov3_convnext_large","foundation","dinov3",  198.0,   timm_id="convnext_large",       description="DINOv3 ConvNeXt-L — LVD-1689M",    pretrained_on="LVD-1689M"),

    # ── DINOv3 task heads ────────────────────────────────────────────────────────
    ModelSpec("dinov3_classifier",  "classification","dinov3_head",  1.0,  description="DINOv3 classifier head — ImageNet-1k"),
    ModelSpec("dinov3_depther",     "foundation",    "dinov3_head",  5.0,  description="DINOv3 depth head — SYNTHMIX"),
    ModelSpec("dinov3_detector",    "detection",     "dinov3_head", 20.0,  description="DINOv3 detection head — COCO2017"),
    ModelSpec("dinov3_segmentor",   "segmentation",  "dinov3_head", 15.0,  description="DINOv3 segmentation head — ADE20K"),
    ModelSpec("dinov3_dinotxt",     "foundation",    "dinov3_head",  8.0,  description="DINOv3 dino.txt zero-shot head"),
    ModelSpec("dinov3_chmv2",       "foundation",    "dinov3_head", 12.0,  description="DINOv3 CHMv2 canopy height head", supports_multispectral=True),

    # ── Prithvi ─────────────────────────────────────────────────────────────────
    ModelSpec("prithvi_eo_1_0",    "foundation","prithvi",  100.0,  hf_id="ibm-nasa-geospatial/Prithvi-100M",          description="Prithvi-EO-1.0 100M — HLS US 30m",    pretrained_on="HLS-US",    supports_multispectral=True),
    ModelSpec("prithvi_eo_2_0",    "foundation","prithvi",  600.0,  hf_id="ibm-nasa-geospatial/Prithvi-EO-2.0-300M",   description="Prithvi-EO-2.0 600M — HLS Global 30m", pretrained_on="HLS-Global",supports_multispectral=True),
    ModelSpec("prithvi_burn_scar", "segmentation","prithvi", 100.0, hf_id="ibm-nasa-geospatial/Prithvi-100M-burn-scar", description="Prithvi fine-tuned burn scar",         pretrained_on="HLS-US"),

]

for spec in _SPECS:
    register_model(spec)

logger.info("Model registry: %d architectures loaded", len(_REGISTRY))


# ── Public API ─────────────────────────────────────────────────────────────────

class ModelRegistry:
    """Queryable registry of all PyGeoVision model architectures."""

    def __len__(self) -> int:
        return len(_REGISTRY)

    def __contains__(self, name: str) -> bool:
        return name in _REGISTRY

    def __getitem__(self, name: str) -> ModelSpec:
        if name not in _REGISTRY:
            raise KeyError(f"Model '{name}' not found. Use list_models() to see all options.")
        return _REGISTRY[name]

    def list(self, task: Optional[str] = None, family: Optional[str] = None,
              max_params_m: Optional[float] = None) -> List[str]:
        return [n for n, s in _REGISTRY.items()
                if (task is None or s.task == task)
                and (family is None or s.family == family)
                and (max_params_m is None or s.params_m <= max_params_m)]

    def search(self, query: str) -> List[ModelSpec]:
        q = query.lower()
        return [s for s in _REGISTRY.values()
                if q in s.name.lower() or q in s.description.lower()
                or q in s.task.lower() or q in s.family.lower()]

    def by_task(self) -> Dict[str, List[str]]:
        tasks: Dict[str, List[str]] = {}
        for n, s in _REGISTRY.items():
            tasks.setdefault(s.task, []).append(n)
        return tasks

    def top_by_task(self, task: str, n: int = 5) -> List[ModelSpec]:
        """Return top-n models for a task, sorted by param count."""
        models = [s for s in _REGISTRY.values() if s.task == task]
        return sorted(models, key=lambda s: s.params_m)[:n]

    def summary(self) -> Dict:
        by_task = self.by_task()
        return {
            "total": len(_REGISTRY),
            "by_task": {t: len(ms) for t, ms in by_task.items()},
            "with_hf_weights": sum(1 for s in _REGISTRY.values() if s.hf_id),
            "with_timm_weights": sum(1 for s in _REGISTRY.values() if s.timm_id),
        }


model_registry = ModelRegistry()


def get_model(name: str, num_classes: int = 2, in_channels: int = 4,
               pretrained: bool = True, device: Optional[str] = None,
               **kwargs) -> Any:
    """Load a model by name with geospatial configuration.

    Args:
        name: Model name from the registry (e.g. "segformer-b2", "unet-r50")
        num_classes: Number of output classes
        in_channels: Number of input channels (4 for Sentinel-2 BGRN)
        pretrained: Load pretrained weights
        device: Target device ("cuda", "cpu", "mps")

    Returns:
        Loaded PyTorch model ready for inference/fine-tuning

    Example::

        model = get_model("segformer-b2", num_classes=7, in_channels=4)
    """
    spec = model_registry[name]
    model = _build_model(spec, num_classes=num_classes, in_channels=in_channels,
                          pretrained=pretrained, **kwargs)
    if device:
        model = model.to(device)
    return model


def _build_model(spec: ModelSpec, num_classes: int, in_channels: int,
                  pretrained: bool, **kwargs) -> Any:
    """Factory: build a model from its spec."""
    # Try timm first
    if spec.timm_id:
        try:
            import timm
            model = timm.create_model(
                spec.timm_id,
                pretrained=pretrained,
                num_classes=num_classes,
                in_chans=in_channels,
                **kwargs,
            )
            return model
        except ImportError:
            logger.warning("timm not installed — pip install timm")
        except Exception as exc:
            logger.warning("timm build failed for %s: %s", spec.name, exc)

    # Try transformers
    if spec.hf_id:
        try:
            return _build_hf_model(spec, num_classes, in_channels, pretrained, **kwargs)
        except ImportError:
            logger.warning("transformers not installed — pip install transformers")
        except Exception as exc:
            logger.warning("HF build failed for %s: %s", spec.name, exc)

    # Generic PyTorch fallback for common architectures
    return _build_pytorch_fallback(spec, num_classes, in_channels, **kwargs)


def _build_hf_model(spec: ModelSpec, num_classes: int, in_channels: int,
                     pretrained: bool, **kwargs) -> Any:
    from transformers import AutoModel, AutoConfig
    config = AutoConfig.from_pretrained(spec.hf_id)
    if hasattr(config, "num_labels"):
        config.num_labels = num_classes
    if pretrained:
        return AutoModel.from_pretrained(spec.hf_id, config=config, ignore_mismatched_sizes=True)
    return AutoModel.from_config(config)


def _build_pytorch_fallback(spec: ModelSpec, num_classes: int, in_channels: int, **kwargs) -> Any:
    """Build common models from torchvision when timm/hf not available."""
    try:
        import torch.nn as nn
        import torchvision.models as tvm
    except ImportError:
        raise ImportError("torch + torchvision required: pip install torch torchvision")

    family = spec.family
    if family == "resnet":
        variant = spec.name.replace("resnet", "").split("-")[0] if "-" in spec.name else spec.name.replace("resnet", "")
        model_fn = {
            "50": tvm.resnet50, "101": tvm.resnet101, "152": tvm.resnet152,
        }.get(variant, tvm.resnet50)
        model = model_fn(pretrained=False)
        if in_channels != 3:
            model.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    elif family in ("unet", "segmentation"):
        # Simple U-Net-like model
        return _simple_unet(in_channels, num_classes)

    else:
        logger.warning("No factory for %s/%s — returning simple conv model", spec.family, spec.name)
        return _simple_conv_classifier(in_channels, num_classes)


def _simple_unet(in_ch: int, n_classes: int) -> Any:
    import torch.nn as nn

    def _block(ci, co):
        return nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
            nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
        )

    class SimpleUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = _block(in_ch, 64);  self.pool1 = nn.MaxPool2d(2)
            self.enc2 = _block(64, 128);    self.pool2 = nn.MaxPool2d(2)
            self.enc3 = _block(128, 256);   self.pool3 = nn.MaxPool2d(2)
            self.bottleneck = _block(256, 512)
            self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
            self.dec3 = _block(512, 256)
            self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
            self.dec2 = _block(256, 128)
            self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
            self.dec1 = _block(128, 64)
            self.out = nn.Conv2d(64, n_classes, 1)

        def forward(self, x):
            import torch
            e1 = self.enc1(x); e2 = self.enc2(self.pool1(e1)); e3 = self.enc3(self.pool2(e2))
            b  = self.bottleneck(self.pool3(e3))
            d3 = self.dec3(torch.cat([self.up3(b), e3], 1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
            return self.out(d1)

    return SimpleUNet()


def _simple_conv_classifier(in_ch: int, n_classes: int) -> Any:
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv2d(in_ch, 64, 3, padding=1), nn.ReLU(),
        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Linear(128, n_classes),
    )
