"""
AI Model Zoo — 50+ architectures across all geospatial tasks (Phase 2).
Extends the base ModelRegistry with comprehensive coverage.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModelSpec:
    """Complete specification for one AI model architecture."""
    name: str
    task: str                    # segmentation|detection|classification|change|foundation|vlm|3d|timeseries
    architecture: str
    backbone: str = ""
    input_bands: int = 3
    pretrained_available: bool = True
    description: str = ""
    paper_url: str = ""
    hf_model_id: str = ""       # HuggingFace Hub ID for auto-download
    input_size: int = 224
    params_m: float = 0.0       # Million parameters
    tags: List[str] = field(default_factory=list)
    domains: List[str] = field(default_factory=list)


def _build_zoo() -> List[ModelSpec]:
    M: List[ModelSpec] = []

    # ── Classification (Phase 2.1) ─────────────────────────────────────
    M += [
        ModelSpec("vit_b16", "classification", "ViT-B/16", "vit_b_16", 3, True, "Vision Transformer B/16", params_m=86.0, tags=["transformer", "classification"]),
        ModelSpec("vit_l16", "classification", "ViT-L/16", "vit_l_16", 3, True, "Vision Transformer L/16", params_m=307.0),
        ModelSpec("swin_t", "classification", "Swin-T", "swin_tiny", 3, True, "Swin Transformer Tiny", params_m=28.0, tags=["transformer"]),
        ModelSpec("swin_s", "classification", "Swin-S", "swin_small", 3, True, "Swin Transformer Small", params_m=50.0),
        ModelSpec("swin_b", "classification", "Swin-B", "swin_base", 3, True, "Swin Transformer Base", params_m=88.0),
        ModelSpec("swin_l", "classification", "Swin-L", "swin_large", 3, True, "Swin Transformer Large", params_m=197.0),
        ModelSpec("resnet50_cls", "classification", "ResNet-50", "resnet50", 3, True, "ResNet-50 classifier", params_m=25.0, tags=["cnn"]),
        ModelSpec("resnet101_cls", "classification", "ResNet-101", "resnet101", 3, True, params_m=44.0),
        ModelSpec("resnet152_cls", "classification", "ResNet-152", "resnet152", 3, True, params_m=60.0),
        ModelSpec("efficientnet_b0_cls", "classification", "EfficientNet-B0", "efficientnet_b0", 3, True, params_m=5.3),
        ModelSpec("efficientnet_b3_cls", "classification", "EfficientNet-B3", "efficientnet_b3", 3, True, params_m=12.0),
        ModelSpec("efficientnet_b7_cls", "classification", "EfficientNet-B7", "efficientnet_b7", 3, True, params_m=66.0),
        ModelSpec("convnext_t", "classification", "ConvNeXt-T", "convnext_tiny", 3, True, "ConvNeXt Tiny", params_m=28.0),
        ModelSpec("convnext_b", "classification", "ConvNeXt-B", "convnext_base", 3, True, params_m=89.0),
        ModelSpec("densenet121_cls", "classification", "DenseNet-121", "densenet121", 3, True, params_m=8.0),
        ModelSpec("mlp_mixer_b16", "classification", "MLP-Mixer-B/16", "mixer_b16", 3, True, "MLP-Mixer B/16", params_m=59.0, tags=["mlp"]),
        ModelSpec("resmlp_s36", "classification", "ResMLP-S36", "resmlp_s36", 3, True, "ResMLP S36", params_m=44.0),
        ModelSpec("clip_cls", "classification", "CLIP Zero-Shot", "openai/clip-vit-large-patch14", 3, True, "CLIP zero-shot land cover", hf_model_id="openai/clip-vit-large-patch14", tags=["vlm", "zero_shot"]),
        ModelSpec("rs_clip", "classification", "RS-CLIP", "RS-CLIP-ViT-L/14", 3, True, "Remote sensing CLIP", tags=["vlm", "zero_shot", "remote_sensing"]),
        ModelSpec("dinov3_cls", "classification", "DINOv3-Cls", "dinov2_vitl14", 3, True, "DINOv3 linear probe", hf_model_id="facebook/dinov2-large", params_m=307.0, tags=["self_supervised"]),
    ]

    # ── Detection (Phase 2.2) ──────────────────────────────────────────
    M += [
        ModelSpec("yolov8_s", "detection", "YOLOv8-S", "yolov8s", 3, True, "YOLOv8 Small", params_m=11.0, tags=["real_time"]),
        ModelSpec("yolov8_m", "detection", "YOLOv8-M", "yolov8m", 3, True, params_m=26.0),
        ModelSpec("yolov8_l", "detection", "YOLOv8-L", "yolov8l", 3, True, params_m=44.0),
        ModelSpec("yolov8_x", "detection", "YOLOv8-X", "yolov8x", 3, True, params_m=68.0),
        ModelSpec("yolov9_m", "detection", "YOLOv9-M", "yolov9m", 3, True, "YOLOv9 Medium", params_m=20.0),
        ModelSpec("retinanet_resnet50", "detection", "RetinaNet", "resnet50", 3, True, "RetinaNet ResNet-50 FPN", params_m=36.0, tags=["anchor_based"]),
        ModelSpec("fcos_resnet50", "detection", "FCOS", "resnet50", 3, True, "FCOS ResNet-50", params_m=32.0, tags=["anchor_free"]),
        ModelSpec("centernet_r50", "detection", "CenterNet", "resnet50", 3, True, params_m=32.0, tags=["anchor_free"]),
        ModelSpec("detr_r50", "detection", "DETR", "resnet50", 3, True, "DEtection TRansformer", params_m=41.0, tags=["transformer"]),
        ModelSpec("deformable_detr", "detection", "Deformable-DETR", "resnet50", 3, True, params_m=40.0, tags=["transformer"]),
        ModelSpec("rt_detr_r50", "detection", "RT-DETR", "resnet50", 3, True, "Real-Time DETR", params_m=42.0, tags=["transformer", "real_time"]),
        ModelSpec("rfdetr_base", "detection", "RF-DETR-Base", "rf_detr_base", 3, True, "RF-DETR for geospatial", hf_model_id="roboflow/rf-detr-base", params_m=30.0, tags=["real_time", "remote_sensing"]),
        ModelSpec("faster_rcnn_r50", "detection", "Faster R-CNN", "resnet50", 3, True, params_m=41.0, tags=["two_stage"]),
        ModelSpec("mask_rcnn_r50", "detection", "Mask R-CNN", "resnet50", 3, True, "Instance segmentation", params_m=44.0, tags=["instance_segmentation"]),
        ModelSpec("yolo_nas_m", "detection", "YOLO-NAS-M", "yolo_nas_m", 3, True, "Neural Architecture Search YOLO", params_m=20.0, tags=["real_time"]),
        ModelSpec("oriented_rcnn", "detection", "Oriented R-CNN", "resnet50", 3, True, "Rotated bbox detection", tags=["rotated_bbox"]),
        ModelSpec("sar_det_r50", "detection", "SAR-Det", "resnet50", 1, True, "SAR-specific object detection", tags=["sar"]),
    ]

    # ── Segmentation (Phase 2.3) ───────────────────────────────────────
    M += [
        ModelSpec("unet_resnet50", "segmentation", "UNet", "resnet50", 3, True, "UNet + ResNet-50", params_m=32.0, tags=["cnn"]),
        ModelSpec("unet_efficientnet_b4", "segmentation", "UNet", "efficientnet_b4", 3, True, "UNet + EfficientNet-B4", params_m=18.0),
        ModelSpec("unetpp_resnet50", "segmentation", "UNet++", "resnet50", 3, True, "UNet++ ResNet-50", params_m=34.0),
        ModelSpec("deeplabv3plus_resnet101", "segmentation", "DeepLabV3+", "resnet101", 3, True, params_m=59.0, tags=["cnn"]),
        ModelSpec("pspnet_resnet50", "segmentation", "PSPNet", "resnet50", 3, True, "Pyramid Scene Parsing Network", params_m=46.0),
        ModelSpec("fcn_resnet101", "segmentation", "FCN", "resnet101", 3, True, "Fully Convolutional Network", params_m=51.0),
        ModelSpec("segformer_b0", "segmentation", "SegFormer-B0", "mit_b0", 3, True, params_m=3.7, tags=["transformer"]),
        ModelSpec("segformer_b2", "segmentation", "SegFormer-B2", "mit_b2", 3, True, params_m=27.0, tags=["transformer"]),
        ModelSpec("segformer_b5", "segmentation", "SegFormer-B5", "mit_b5", 3, True, params_m=82.0, tags=["transformer"]),
        ModelSpec("upernet_swin_t", "segmentation", "UPerNet-Swin-T", "swin_tiny", 3, True, params_m=60.0, tags=["transformer"]),
        ModelSpec("upernet_swin_b", "segmentation", "UPerNet-Swin-B", "swin_base", 3, True, params_m=121.0),
        ModelSpec("mask2former_swin_t", "segmentation", "Mask2Former-Swin-T", "swin_tiny", 3, True, "Panoptic segmentation", params_m=47.0, tags=["panoptic"]),
        ModelSpec("sam_vit_h", "segmentation", "SAM-ViT-H", "vit_huge", 3, True, "Segment Anything Model ViT-H", hf_model_id="facebook/sam-vit-huge", params_m=636.0, tags=["foundation"]),
        ModelSpec("sam2_hiera_l", "segmentation", "SAM2-Hiera-L", "hiera_large", 3, True, "SAM2 Hiera Large", hf_model_id="facebook/sam2-hiera-large", params_m=224.0, tags=["foundation"]),
        ModelSpec("bisenet_v2", "segmentation", "BiSeNetV2", "BiSeNetV2", 3, True, "Real-time segmentation", params_m=3.4, tags=["real_time"]),
        ModelSpec("fast_scnn", "segmentation", "Fast-SCNN", "fast_scnn", 3, True, "Fast Semantic Segmentation CNN", params_m=1.1, tags=["real_time"]),
        ModelSpec("prithvi_seg", "segmentation", "Prithvi-Seg", "prithvi_eo_v2", 6, True, "NASA Prithvi for segmentation", hf_model_id="ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL", params_m=300.0, tags=["foundation", "multispectral"]),
    ]

    # ── Change Detection (Phase 2.4) ───────────────────────────────────
    M += [
        ModelSpec("changeformer", "change_detection", "ChangeFormer", "mit_b4", 3, True, "Transformer-based change detection", params_m=41.0, tags=["transformer"]),
        ModelSpec("bit_resnet50", "change_detection", "BIT", "resnet50", 3, True, "Binary change detection BIT", params_m=31.0, tags=["transformer"]),
        ModelSpec("dsamnet", "change_detection", "DSAMNet", "resnet18", 3, True, "Dense Self-Attention MixNet", params_m=16.0),
        ModelSpec("changestar", "change_detection", "ChangeSTAR", "farseg", 3, True, "Any-time change detection", params_m=28.0),
        ModelSpec("siamese_unet", "change_detection", "Siamese-UNet", "resnet50", 3, True, params_m=32.0),
        ModelSpec("convlstm_cd", "change_detection", "ConvLSTM-CD", "convlstm", 3, True, "ConvLSTM change detection", params_m=8.0, tags=["rnn"]),
        ModelSpec("swin_unet_cd", "change_detection", "Swin-UNet-CD", "swin_tiny", 3, True, params_m=28.0),
        ModelSpec("tinycd", "change_detection", "TinyCD", "tinycd", 3, True, "Lightweight change detection", params_m=0.27, tags=["lightweight"]),
        ModelSpec("snunet", "change_detection", "SNUNet", "resnet18", 3, True, "CBAM-enhanced Siamese Net", params_m=12.0),
        ModelSpec("hanet_cd", "change_detection", "HANet", "resnet50", 3, True, "Hierarchical Attention Network", params_m=35.0),
    ]

    # ── Foundation Models (Phase 2.5) ──────────────────────────────────
    M += [
        ModelSpec("prithvi_100m", "foundation", "Prithvi-100M", "vit_large", 6, True, "NASA Prithvi HLS 100M MAE", hf_model_id="ibm-nasa-geospatial/Prithvi-EO-1.0-100M", params_m=100.0, tags=["multispectral", "foundation"]),
        ModelSpec("prithvi_300m", "foundation", "Prithvi-300M", "vit_large", 6, True, "NASA Prithvi EO 2.0 300M", hf_model_id="ibm-nasa-geospatial/Prithvi-EO-2.0-300M", params_m=300.0, tags=["multispectral", "foundation"]),
        ModelSpec("dinov2_s", "foundation", "DINOv2-S", "vit_small", 3, True, "DINOv2 ViT-S/14", hf_model_id="facebook/dinov2-small", params_m=21.0, tags=["self_supervised"]),
        ModelSpec("dinov2_b", "foundation", "DINOv2-B", "vit_base", 3, True, "DINOv2 ViT-B/14", hf_model_id="facebook/dinov2-base", params_m=86.0, tags=["self_supervised"]),
        ModelSpec("dinov2_l", "foundation", "DINOv2-L", "vit_large", 3, True, "DINOv2 ViT-L/14", hf_model_id="facebook/dinov2-large", params_m=307.0, tags=["self_supervised"]),
        ModelSpec("satlas_swin_b", "foundation", "SatlasPretrain-Swin-B", "swin_base", 3, True, "SatlasPretrain multi-task", hf_model_id="allenai/satlas-pretrain", params_m=88.0, tags=["foundation", "multi_task"]),
        ModelSpec("ssl4eo_moco", "foundation", "SSL4EO-MoCo", "resnet50", 13, True, "SSL4EO MoCo-v3 Sentinel-2", hf_model_id="wangyi111/SSL4EO-S12-MoCo", params_m=25.0, tags=["self_supervised", "multispectral"]),
        ModelSpec("dofa_vit_b", "foundation", "DOFA-ViT-B", "vit_base", 0, True, "Dynamic One-For-All Foundation", hf_model_id="XShadow/DOFA", params_m=86.0, tags=["foundation", "multi_sensor"]),
        ModelSpec("geosam_vit_h", "foundation", "GeoSAM-ViT-H", "vit_huge", 3, True, "SAM fine-tuned for geospatial", hf_model_id="wangyi111/GeoSAM", params_m=636.0, tags=["foundation", "segmentation"]),
        ModelSpec("ringmo_swin_b", "foundation", "RingMo-Swin-B", "swin_base", 3, True, "Ring Modality Foundation Model", params_m=88.0, tags=["foundation"]),
        ModelSpec("skysense", "foundation", "SkySense", "vit_large", 3, True, "Global-scale RS pretraining", params_m=307.0, tags=["foundation", "multi_modal"]),
    ]

    # ── Vision-Language Models (Phase 2.6) ────────────────────────────
    M += [
        ModelSpec("clip_vit_b32", "vlm", "CLIP-ViT-B/32", "vit_base_32", 3, True, "OpenAI CLIP B/32", hf_model_id="openai/clip-vit-base-patch32", params_m=151.0, tags=["zero_shot"]),
        ModelSpec("clip_vit_l14", "vlm", "CLIP-ViT-L/14", "vit_large_14", 3, True, "OpenAI CLIP L/14", hf_model_id="openai/clip-vit-large-patch14", params_m=427.0, tags=["zero_shot"]),
        ModelSpec("openclip_b32", "vlm", "OpenCLIP-B/32", "vit_base_32", 3, True, "OpenCLIP LAION-2B", hf_model_id="laion/CLIP-ViT-B-32-laion2B-s34B-b79K", params_m=151.0),
        ModelSpec("remoteclip", "vlm", "RemoteCLIP", "vit_large_14", 3, True, "RS image-text pretraining", hf_model_id="chendelong/RemoteCLIP", params_m=427.0, tags=["remote_sensing", "zero_shot"]),
        ModelSpec("geochat", "vlm", "GeoChat", "vicuna_7b", 3, True, "Geospatial conversational VLM", hf_model_id="MBZUAI/geochat-7B", params_m=7000.0, tags=["llm", "conversational"]),
        ModelSpec("moondream2", "vlm", "Moondream2", "phi_1_5", 3, True, "Efficient satellite VLM", hf_model_id="vikhyatk/moondream2", params_m=1870.0, tags=["captioning", "vqa"]),
        ModelSpec("rs5m_clip", "vlm", "RS5M-CLIP", "vit_large", 3, True, "5M RS image-text pairs", params_m=427.0, tags=["remote_sensing"]),
    ]

    # ── 3D / Point Cloud (Phase 2.7) ──────────────────────────────────
    M += [
        ModelSpec("pointnet2_cls", "3d", "PointNet++", "pointnet2", 0, True, "PointNet++ classification", params_m=1.5, tags=["point_cloud"]),
        ModelSpec("pointnet2_seg", "3d", "PointNet++-Seg", "pointnet2", 0, True, "PointNet++ segmentation", params_m=1.7, tags=["point_cloud"]),
        ModelSpec("randla_net", "3d", "RandLA-Net", "randlanet", 0, True, "Efficient large-scale 3D", params_m=1.2, tags=["point_cloud", "efficient"]),
        ModelSpec("kpconv", "3d", "KPConv", "kpconv_rigid", 0, True, "Kernel Point Convolution", params_m=14.9, tags=["point_cloud"]),
        ModelSpec("minkowski_res16", "3d", "MinkowskiNet-34", "minkowski_res16", 0, True, "Sparse tensor segmentation", params_m=37.8, tags=["point_cloud", "sparse"]),
        ModelSpec("pointtransformer_v3", "3d", "PointTransformer-V3", "ptv3", 0, True, "PointTransformer V3", params_m=46.2, tags=["point_cloud", "transformer"]),
    ]

    # ── Time Series (Phase 2.8) ────────────────────────────────────────
    M += [
        ModelSpec("ltae", "timeseries", "L-TAE", "ltae", 13, True, "Lightweight Temporal Attention", params_m=0.26, tags=["transformer", "satellite_ts"]),
        ModelSpec("tsvit", "timeseries", "TSViT", "tsvit", 11, True, "Time Series ViT", params_m=1.8, tags=["transformer"]),
        ModelSpec("convlstm_ts", "timeseries", "ConvLSTM-TS", "convlstm", 13, True, "Convolutional LSTM for time series", params_m=5.0, tags=["rnn"]),
        ModelSpec("transformer_ts", "timeseries", "Informer", "informer", 13, True, "Informer for long sequence prediction", params_m=11.0, tags=["transformer", "prediction"]),
        ModelSpec("autoformer", "timeseries", "Autoformer", "autoformer", 13, True, "Autoformer series forecasting", params_m=14.0, tags=["transformer", "prediction"]),
        ModelSpec("tempformer", "timeseries", "TempFormer", "tempformer", 10, True, "Temporal Transformer for SITS", params_m=6.0, tags=["transformer", "sits"]),
    ]

    # ── Super Resolution ───────────────────────────────────────────────
    M += [
        ModelSpec("esrgan_geo", "super_resolution", "ESRGAN-Geo", "esrgan", 3, True, "ESRGAN for satellite imagery", params_m=16.7),
        ModelSpec("srcnn", "super_resolution", "SRCNN", "srcnn", 3, True, "SRCNN baseline", params_m=0.057),
        ModelSpec("rcan", "super_resolution", "RCAN", "rcan", 3, True, "Residual Channel Attention Network", params_m=15.4),
        ModelSpec("swinir_sr", "super_resolution", "SwinIR-SR", "swin_tiny", 3, True, "SwinIR super-resolution", params_m=11.9, tags=["transformer"]),
    ]

    return M


class ModelZoo:
    """AI Model Zoo — 50+ architectures across all geospatial tasks.

    Usage:
        >>> zoo = ModelZoo()
        >>> print(f"Total: {len(zoo)} model specs")
        >>> seg_models = zoo.filter(task="segmentation")
        >>> top = zoo.top_for_task("detection", n=5)
        >>> spec = zoo["segformer_b2"]
        >>> zoo.print_table(seg_models)
    """
    def __init__(self) -> None:
        self._models = _build_zoo()
        self._by_name = {m.name: m for m in self._models}

    def __len__(self) -> int:
        return len(self._models)

    def __getitem__(self, name: str) -> ModelSpec:
        if name not in self._by_name:
            raise KeyError(f"Model '{name}' not in zoo. Use zoo.search() to discover models.")
        return self._by_name[name]

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def all(self) -> List[ModelSpec]:
        return list(self._models)

    def filter(
        self,
        task: Optional[str] = None,
        backbone: Optional[str] = None,
        tag: Optional[str] = None,
        max_params_m: Optional[float] = None,
        pretrained_only: bool = False,
    ) -> List[ModelSpec]:
        out = []
        for m in self._models:
            if task and m.task != task: continue
            if backbone and backbone.lower() not in m.backbone.lower(): continue
            if tag and tag not in m.tags: continue
            if max_params_m and m.params_m > max_params_m: continue
            if pretrained_only and not m.pretrained_available: continue
            out.append(m)
        return out

    def search(self, query: str) -> List[ModelSpec]:
        q = query.lower()
        return [m for m in self._models if q in (m.name + m.architecture + m.description + " ".join(m.tags)).lower()]

    def tasks(self) -> List[str]:
        return sorted(set(m.task for m in self._models))

    def top_for_task(self, task: str, n: int = 5) -> List[ModelSpec]:
        candidates = self.filter(task=task, pretrained_only=True)
        # Score: recency proxy (more params generally = newer) + smaller models preferred for deployability
        def _score(m: ModelSpec) -> float:
            param_score = min(1.0, m.params_m / 300.0) * 0.4
            pretrained = 0.6 if m.pretrained_available else 0.0
            has_hf = 0.3 if m.hf_model_id else 0.0
            return pretrained + has_hf + param_score
        return sorted(candidates, key=_score, reverse=True)[:n]

    def summary(self) -> Dict[str, Any]:
        tasks: Dict[str, int] = {}
        for m in self._models:
            tasks[m.task] = tasks.get(m.task, 0) + 1
        return {
            "total_models": len(self._models),
            "tasks": tasks,
            "with_hf_weights": sum(1 for m in self._models if m.hf_model_id),
            "pretrained": sum(1 for m in self._models if m.pretrained_available),
        }

    def print_table(self, models: Optional[List[ModelSpec]] = None) -> None:
        items = models or self._models
        print(f"\n{'Name':<28} {'Task':<18} {'Architecture':<20} {'Params(M)':>10} {'HF':>4}")
        print("─" * 85)
        for m in items:
            hf = "✓" if m.hf_model_id else ""
            print(f"{m.name:<28} {m.task:<18} {m.architecture:<20} {m.params_m:>10.1f} {hf:>4}")
        print(f"\n{len(items)} models")


# Singleton
model_zoo = ModelZoo()
