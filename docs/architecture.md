# Architecture

PyGeoVision v2.0 is built on a clean layered architecture with zero dependency on geoai-py. Every layer is independently importable and testable.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                       PyGeoVision Client                            │
│   client.search()  client.download()  client.labeling  client.geoai │
├──────────────────┬──────────────────┬───────────────────────────────┤
│   Data Layer     │   AI Layer       │   Support Layers              │
│   (PyGeoFetch)   │   Models+Tasks   │   XAI · Monitor · Serve       │
├──────────────────┴──────────────────┴───────────────────────────────┤
│   Model Registry (119 specs)   ·   Dataset Registry (503 entries)   │
│   Training Framework           ·   Pipeline Orchestrator            │
│   Edge / Cloud Deployment      ·   CLI (15 command groups)          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Module Map

```
pygeovision/
│
├── __init__.py              Main client — PyGeoVision class
├── version.py               Semantic version (2.0.4)
│
├── data/                    Data acquisition layer
│   └── fetch.py             PyGeoFetch — STAC search, 22 providers, COG download
│
├── models/                  Model layer — 119 architectures
│   ├── registry.py          Central ModelSpec registry
│   ├── base.py              GeoModel + GeoModelConfig base classes
│   ├── classification/      ViT, Swin, ConvNeXt, ResNet, EfficientNet, DINOv2
│   ├── detection/           YOLOv8/v9, RF-DETR, RT-DETR, DETR, Faster/Mask R-CNN
│   ├── segmentation/        U-Net, SegFormer, DeepLabV3+, SAM, SAM2, Mask2Former
│   ├── change_detection/    ChangeFormer, ChangeSTAR, BIT, DSAMNet, SNUNet
│   ├── foundation/          DINOv3 (12 variants), Prithvi-EO (100M/600M)
│   ├── vlm/                 CLIP, OpenCLIP, RemoteCLIP, Moondream
│   ├── _3d/                 PointNet++, RandLA-Net, KPConv
│   └── weights/             HuggingFace Hub downloader + cache manager
│
├── labeling/                Auto-labeling layer
│   ├── osm.py               OpenStreetMap (buildings, roads, water, parks)
│   ├── microsoft.py         Microsoft Building Footprints (~1.4B global)
│   ├── google.py            Google Open Buildings (~1.8B global)
│   ├── esa.py               ESA WorldCover 2021 (11 classes, 10m)
│   ├── dynamic_world.py     Google Dynamic World (9 classes, real-time)
│   ├── sam_auto.py          SAM automatic mask generation
│   ├── foundation.py        DINOv2 + k-means unsupervised labels
│   ├── active.py            Active learning — entropy/diversity sampling
│   ├── quality.py           Label quality assessment (completeness, noise, coverage)
│   └── pipeline.py          Multi-source label fusion (majority vote)
│
├── losses/                  Geospatial loss functions
│   ├── segmentation.py      DiceLoss, FocalLoss, TverskyLoss, ComboLoss,
│   │                        BoundaryAwareLoss, LovászLoss, OhemCrossEntropy
│   ├── detection.py         CIoULoss, DIoULoss, GIoULoss
│   └── class_balance.py     ClassBalancedCrossEntropy
│
├── inference/               Inference engine
│   ├── tiled.py             TiledInference — Gaussian/linear blend, TTA
│   ├── batch.py             BatchInferenceEngine — multi-worker directory processing
│   ├── stream.py            StreamingInference + EnsembleInference
│   └── __init__.py          Public API
│
├── explainability/          XAI layer
│   ├── gradcam.py           GradCAM, GradCAM++, EigenCAM
│   ├── uncertainty.py       MC Dropout uncertainty maps
│   ├── attention.py         Transformer attention map extraction
│   └── shap_geo.py          SHAP for geospatial models
│
├── monitoring/              Production monitoring
│   ├── drift.py             DriftDetector — PSI + KL divergence
│   ├── tracker.py           ModelPerformanceTracker — mIoU, mAP over time
│   └── alerts.py            Alert system — thresholds, webhooks, email
│
├── training/                Training framework
│   ├── trainer.py           GeoTrainer — single/multi-GPU training loop
│   ├── callbacks.py         EarlyStopping, ModelCheckpoint, LearningRateMonitor
│   ├── metrics.py           IoU, F1, mAP, Accuracy, Precision, Recall
│   ├── optimizers.py        AdamW, Lion, SGD, RMSprop with schedulers
│   ├── schedulers.py        CosineAnnealing, OneCycleLR, WarmupScheduler
│   ├── distributed.py       DDP, FSDP, gradient accumulation
│   ├── mixed_precision.py   FP16 / BF16 autocast + gradient scaling
│   ├── checkpoint.py        CheckpointManager — top-k saving, resume
│   ├── data.py              GeoDataset, augmentation, collation
│   └── validation.py        Validation loop with early stopping integration
│
├── serving/                 Inference server
│   ├── api.py               FastAPI app + InferenceServer class
│   ├── auth.py              APIKeyAuth + JWTAuth
│   ├── health.py            HealthChecker — GPU, RAM, uptime
│   └── models.py            Pydantic request/response schemas
│
├── pipelines/               YAML pipeline orchestration
│   ├── orchestrator.py      Pipeline + PipelineOrchestrator
│   ├── yaml_parser.py       PipelineYAMLParser — load, validate
│   ├── scheduler.py         PipelineScheduler — cron expressions
│   ├── steps.py             Step, SearchStep, DownloadStep, InferStep, ExportStep
│   └── templates/           6 ready-to-run YAML templates
│
├── datasets/                Dataset registry
│   ├── registry.py          503-entry DatasetInfo catalog
│   ├── loader.py            Standardised dataset loading
│   ├── catalog.py           Search, filter, rank datasets
│   ├── benchmark.py         Top-5 per task selection
│   └── analysis.py          Correlation matrix, domain statistics
│
├── edge/                    Edge deployment
│   ├── onnx_rt.py           ONNXRuntimeInference — CPU/CUDA/TensorRT/CoreML
│   └── jetson.py            JetsonDeployer — TensorRT conversion
│
├── cloud/                   Cloud deployment
│   └── deploy.py            AWSDeployer, AzureDeployer, GCPDeployer
│
├── advanced/                Advanced AI capabilities
│   ├── few_shot.py          FewShotLearner — prototypical networks
│   ├── multitask.py         MultiTaskModel — shared backbone
│   ├── automl.py            AutoML — Optuna hyperparameter search
│   ├── vlm/                 CLIPGeo, MoondreamGeo, GeoRetrieval
│   ├── timeseries/          GeoTimeSeries — NDVI/EVI trends, anomaly detection
│   └── pointcloud/          LiDARProcessor — CHM, density, segmentation
│
├── cli/                     Command-line interface
│   ├── main.py              15 command groups — 1,800+ lines
│   ├── commands/            Per-module command files
│   └── utils/               Colors, formatting, input validation
│
└── ai/                      GeoAI Engine
    └── geoai/
        ├── __init__.py      GeoAIEngine — 18 subsystems
        ├── dinov3_proxy.py  DINOv3Proxy — 12 variants, CHMv2, dino.txt
        └── prithvi_proxy.py PrithviProxy — EO-1.0, EO-2.0, multi-temporal
```

---

## Design Principles

**1. Zero GeoAI dependency**
Every module is self-contained. `geoai-py` is never imported — not even as an optional fallback. All model implementations use pure PyTorch + HuggingFace Transformers + timm.

**2. Lazy loading**
Heavy dependencies (torch, rasterio, transformers) are imported only when the relevant function is called. `import pygeovision` is fast.

**3. Graceful degradation**
If an optional dependency is missing, PyGeoVision returns a clear error message with the install command rather than crashing silently.

**4. Composability**
Every layer can be used standalone:

```python
# Only need inference — skip everything else
from pygeovision.inference.tiled import TiledInference
from pygeovision.models import get_model

model = get_model("segformer-b2", num_classes=2)
inf   = TiledInference(model=model)
```

**5. Type-annotated throughout**
All public APIs have full Python type hints, enabling IDE autocompletion and static analysis with mypy.

---

## Data Flow

```
Satellite Provider (22)
        │
        ▼
   PyGeoFetch (STAC + CQL2 search)
        │
        ▼
   Download + Post-process (reproject, COG, cloud mask)
        │
        ├──▶ Auto-Labeling (OSM / MS / ESA / SAM / DINOv2)
        │
        ├──▶ Training (GeoTrainer → GeoModel)
        │
        ├──▶ Inference (TiledInference → prediction GeoTIFF)
        │         │
        │         ├──▶ Explainability (GradCAM / SHAP)
        │         └──▶ Monitoring (drift + performance)
        │
        └──▶ Pipeline (YAML orchestration of all the above)
                  │
                  └──▶ Export (GeoJSON / GeoTIFF / COG / Cloud)
```

---

## Model Loading Priority

When `get_model(name)` is called:

1. Check `timm` registry → `timm.create_model(timm_id, ...)`
2. Check HuggingFace Hub → `AutoModel.from_pretrained(hf_id, ...)`
3. Built-in PyTorch fallback → torchvision + custom U-Net
4. Error with helpful install message

```python
from pygeovision.models import get_model

# All three paths work transparently:
model = get_model("segformer-b2")        # → HuggingFace
model = get_model("resnet50")            # → timm
model = get_model("unet-r50")            # → SMP or built-in
```
