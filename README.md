<div align="center">

<img src="https://img.shields.io/badge/PyGeoVision-2.0.4-0d1117?style=for-the-badge&labelColor=0d1117&color=2563eb" alt="version"/>

# PyGeoVision

### World-Class Geospatial AI Platform

**The definitive Python framework for satellite data acquisition and geospatial AI —  
unifying [PyGeoFetch](https://github.com/appiahkubis14/PyGeoFetch) (22+ providers) and [GeoAI](https://opengeoai.org) (full AI stack) in one coherent API.**

---

[![Python](https://img.shields.io/badge/Python-3.10%20|%203.11%20|%203.12-3776ab?style=flat-square&logo=python&logoColor=white)](https://pypi.org/project/pygeovision/)
[![PyPI](https://img.shields.io/badge/PyPI-v2.0.4-2563eb?style=flat-square&logo=pypi&logoColor=white)](https://pypi.org/project/pygeovision/)
[![License](https://img.shields.io/badge/License-Apache_2.0-green?style=flat-square)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-580_passing-22c55e?style=flat-square&logo=pytest&logoColor=white)](#testing)
[![Models](https://img.shields.io/badge/Models-119_architectures-f59e0b?style=flat-square)](#model-registry)
[![Datasets](https://img.shields.io/badge/Datasets-503_registered-a855f7?style=flat-square)](#dataset-registry)
[![Notebooks](https://img.shields.io/badge/Notebooks-25_production-0ea5e9?style=flat-square)](#project-notebooks)
[![PyGeoFetch](https://img.shields.io/badge/PyGeoFetch-22_providers-f59e0b?style=flat-square)](https://github.com/appiahkubis14/PyGeoFetch)
[![GeoAI](https://img.shields.io/badge/GeoAI-27_subsystems-a855f7?style=flat-square)](https://opengeoai.org)

</div>

---

## What is PyGeoVision?

PyGeoVision is a **production-ready geospatial AI platform** that unifies satellite data acquisition, foundation model inference, and full-stack model training in a single coherent API.

| Layer | Package / Module | Responsibility |
|-------|-----------------|----------------|
| 🛰️ **Data** | [PyGeoFetch](https://github.com/appiahkubis14/PyGeoFetch) | Search & download from 22+ providers (Sentinel, Landsat, Planet, Maxar, USGS, Copernicus …) with auth, caching, parallel downloads, and YAML pipeline orchestration |
| 🤖 **AI** | [GeoAI](https://opengeoai.org) | Full AI stack: segmentation, detection, classification, change detection, SAM, foundation models, embeddings, cloud masking, super-resolution, ONNX export |
| 🧠 **Foundation** | DINOv3 + Prithvi-EO-2.0 | 12 DINOv3 variants (SAT-493M pre-training) + Prithvi-EO-2.0 (600M, HLS global 10-year) |
| 🔗 **Bridge** | **PyGeoVision** | Unified API, 119 model registry, 503 datasets, 25 production notebooks, CLI, serving, edge, cloud |

> **Design principle:** PyGeoVision never reimplements PyGeoFetch or GeoAI. All data operations delegate to PyGeoFetch. All AI operations delegate to GeoAI. PyGeoVision is the integration and extension layer.

---

## What's New in v2.0

| Area | v1.0 | v2.0 |
|------|------|------|
| Tests passing | 208 | **580** |
| Model architectures | 14 | **119** |
| Datasets registered | — | **503** |
| Production notebooks | — | **25** |
| Foundation models | — | **DINOv3 (12 variants) + Prithvi-EO-2.0** |
| Auto-labeling sources | 7 | **7 + SAM-auto + DynamicWorld** |
| Serving API | — | **FastAPI + WebSocket streaming** |
| Edge deployment | — | **ONNX Runtime + Jetson TensorRT** |
| Cloud deployment | — | **AWS SageMaker + Azure ML + GCP Vertex** |
| VLM | — | **CLIP + RemoteCLIP + Moondream 2** |
| Few-shot learning | — | **Prototypical networks (DINOv2 backbone)** |
| AutoML / HPO | — | **Optuna integration** |
| Time-series analysis | — | **GeoTimeSeries + anomaly detection** |
| Point cloud | — | **PointNet++ + RandLA-Net + KPConv** |
| GeoAI subsystems | 24 | **27** |
| Monitoring | — | **Drift detection + performance tracking + alerts** |

---

## Architecture Overview

```
PyGeoVision v2.0
├── pygeovision/               Core platform
│   ├── __init__.py            PyGeoVision client — unified entry point
│   ├── data/fetch.py          SatelliteFetcher — 22 providers via PyGeoFetch
│   │
│   ├── models/                119 model architectures
│   │   ├── registry.py        ModelSpec registry + download manager
│   │   ├── classification/    ViT, Swin, EfficientNet, DINOv3
│   │   ├── detection/         GeoYOLO, DETR, RF-DETR
│   │   ├── segmentation/      U-Net, SegFormer, DeepLabV3+, SAM
│   │   ├── change_detection/  ChangeFormer, ChangeSTAR, BIT, DSAMNet
│   │   ├── foundation/        dinov3.py (12 variants), prithvi.py (600M)
│   │   ├── vlm/               CLIP, RemoteCLIP, Moondream 2
│   │   └── _3d/               PointNet++, RandLA-Net, KPConv
│   │
│   ├── labeling/              Auto-labeling pipeline (7 sources)
│   ├── losses/                Geospatial mixed loss, boundary-aware, OHEM
│   ├── inference/             TiledInference (Gaussian blend), batch, stream, ensemble
│   ├── explainability/        GradCAM, attention maps, SHAP-Geo, uncertainty
│   ├── monitoring/            Drift detection, performance tracker, alerts
│   ├── training/              GeoTrainer, callbacks, metrics, distributed
│   ├── serving/               FastAPI + WebSocket + JWT auth + health checks
│   ├── pipelines/             YAML pipelines + orchestrator + scheduler
│   ├── datasets/              503 datasets, registry, loader, catalog
│   ├── cli/                   15 command groups
│   ├── edge/                  ONNX Runtime + Jetson TensorRT FP16
│   ├── cloud/                 AWSDeployer, AzureDeployer, GCPDeployer
│   └── advanced/              Few-shot, AutoML, VLM, timeseries, pointcloud
│
├── tests/                     580 passing tests
├── docs/                      44 documentation pages (6,976 lines)
└── projects/                  25 production Jupyter notebooks (357 cells)
```

---

## Installation

```bash
# Core — data + basic inference
pip install pygeovision

# + Geospatial processing (rasterio, geopandas, rioxarray)
pip install "pygeovision[geo]"

# + Training stack (PyTorch, SMP, transformers, timm)
pip install "pygeovision[train]"

# + Foundation models (DINOv3, Prithvi-EO-2.0)
pip install "pygeovision[foundation]"

# + Vision-language models (CLIP, Moondream)
pip install "pygeovision[vlm]"

# + Time-series analysis
pip install "pygeovision[timeseries]"

# + Serving API (FastAPI, uvicorn, websockets)
pip install "pygeovision[serve]"

# + Everything
pip install "pygeovision[all]"
```

**Requirements:** Python 3.10+ · PyGeoFetch · GeoAI (optional) · PyTorch 2.0+

---

## Quick Start

```python
import pygeovision as pgv

# Initialise
client = pgv.PyGeoVision()
print(client)
# PyGeoVision(v2.0.4 | pygeofetch=✓ | geoai=✓ | models=119 | datasets=503)

# ── 1. Add credentials (stored securely in system keyring) ─────────────────
client.add_credentials("usgs",      username="user", password="pass")
client.add_credentials("planet",    api_key="PL-xxxx")
client.add_credentials("copernicus", client_id="id", client_secret="secret")

# ── 2. Search satellite imagery ─────────────────────────────────────────────
results = client.search(
    bbox        = (-0.15, 51.47, -0.10, 51.52),   # London, WGS84
    date_range  = ("2024-06-01", "2024-06-30"),
    providers   = ["planetary_computer", "copernicus"],
    cloud_cover_max = 10,
    sort_by     = "cloud_cover",
)
print(f"Found {len(results)} scenes")

# ── 3. Download with post-processing ────────────────────────────────────────
downloads = client.download(
    results[:2],
    output_dir   = "./sentinel2/",
    parallel     = 4,
    post_process = ["reproject:EPSG:4326", "cog"],
)

# ── 4. Auto-label from Microsoft Building Footprints ────────────────────────
labels = client.labeling.microsoft_buildings(
    bbox        = (-0.15, 51.47, -0.10, 51.52),
    output_path = "./labels_ms.tif",
    resolution_m= 10.0,
)

# ── 5. GeoAI building segmentation ──────────────────────────────────────────
client.geoai.segment.buildings(
    downloads[0].path,
    output_path   = "buildings.tif",
    output_vector = "buildings.geojson",
)

# ── 6. Foundation model: Prithvi-EO-2.0 land cover ──────────────────────────
from pygeovision.models.foundation.prithvi import PrithviTasks
tasks  = PrithviTasks("prithvi_eo_2_0")
result = tasks.land_cover(downloads[0].path, source="sentinel2")

# ── 7. Foundation model: DINOv3 feature extraction ──────────────────────────
from pygeovision.models.foundation.dinov3 import DINOv3Backbone
backbone   = DINOv3Backbone("dinov3_vitl16_sat")
embeddings = backbone.extract_embeddings(downloads[0].path)   # (1, 1024)

# ── 8. End-to-end pipeline ──────────────────────────────────────────────────
result = client.pipeline(
    "building_footprints",
    bbox       = (-0.15, 51.47, -0.10, 51.52),
    date       = "2024-06",
    output_dir = "./results/",
)
print(result.stats)  # {"buildings_detected": 1847, "coverage_pct": 0.312}

# ── 9. Deploy a model as a REST API ─────────────────────────────────────────
from pygeovision.serving import InferenceServer
server = InferenceServer(auth_keys={"prod": "SECRET"})
server.register("seg_v1", "model.onnx", task="segmentation", num_classes=5)
server.serve(host="0.0.0.0", port=8080)
# → POST /predict  |  POST /predict/batch  |  GET /health  |  WS /ws/stream
```

---

## Data Layer — PyGeoFetch (22 Providers)

### Provider Registry

| Provider ID | Name | Auth | Key Satellites | SAR | Sub-m |
|-------------|------|------|----------------|:---:|:-----:|
| `planetary_computer` | Microsoft Planetary Computer | 🌐 Open | Sentinel-1/2, Landsat, MODIS, NAIP | ✓ | |
| `aws_earth` | AWS Earth Open Data | 🌐 Open | Sentinel-2 COGs, Landsat | | |
| `element84` | Element 84 Earth Search | 🌐 Open | Sentinel-2, Landsat Col 2 | | |
| `noaa_big_data` | NOAA Big Data | 🌐 Open | GOES-16/17/18, NEXRAD | | |
| `esa_scihub` | ESA SciHub Mirror | 🌐 Open | Copernicus mirrors | ✓ | |
| `jaxa_earth` | JAXA ALOS World | 🌐 Open | ALOS 30m DSM, PALSAR | ✓ | |
| `isro_bhuvan` | ISRO Bhuvan | 🌐 Open | ResourceSat, Cartosat | | |
| `inpe_cbers` | INPE CBERS | 🌐 Open | CBERS-4/4A | | |
| `digitalglobe` | DigitalGlobe Open Data | 🌐 Open | Disaster VHR | | ✓ |
| `geoserver_generic` | GeoServer Generic OGC | 🌐 Open | Any WMS/WCS/WFS | | |
| `usgs` | USGS Earth Explorer | 🔐 User/Pass | Landsat 1–9, ASTER, MODIS | | |
| `copernicus` | Copernicus CDSE | 🔐 OAuth2 | Sentinel-1/2/3/5P | ✓ | |
| `nasa_earthdata` | NASA Earthdata CMR | 🔐 OAuth2 | MODIS, VIIRS, ICESat-2, GEDI | | |
| `nasa_earthdata_cloud` | NASA Earthdata Cloud | 🔐 OAuth2+S3 | Cloud-hosted NASA | | |
| `opentopography` | OpenTopography | 🔐 API Key | SRTM, Copernicus DEM, LiDAR | | |
| `planet` | Planet Labs | 🔐 API Key | PlanetScope 3–5m, SkySat 0.5m | | ✓ |
| `sentinel_hub` | Sentinel Hub | 🔐 OAuth2 | All Sentinels, Landsat, MODIS | ✓ | |
| `maxar_gbdx` | Maxar GBDX | 🔐 Token | WorldView 1–4, GeoEye-1 (30cm) | | ✓ |
| `airbus_oneatlas` | Airbus OneAtlas | 🔐 API Key | Pléiades 50cm, SPOT 1.5m | | ✓ |
| `alaska_satellite_facility` | ASF | 🔐 Earthdata | Sentinel-1, ALOS PALSAR | ✓ | |
| `google_earth_engine` | Google Earth Engine | 🔐 Service Acct | Multi-petabyte catalog | ✓ | |
| `terrabotics` | TerraBotics | 🔐 API Key | Archive + tasking | | ✓ |

### Search API

```python
# Standard search
results = client.search(
    bbox            = (-74.1, 40.6, -73.7, 40.9),
    date_range      = ("2024-01-01", "2024-06-01"),
    providers       = ["planetary_computer", "copernicus"],
    cloud_cover_max = 10,
    sort_by         = "cloud_cover",   # datetime | cloud_cover | score
    limit           = 50,
    use_cache       = True,
)

# Satellite shortcut
results = client.search(bbox=..., date_range=..., satellite="sentinel-2")

# STAC collection
results = client.search(bbox=..., date_range=...,
    collections=["sentinel-2-l2a", "sentinel-1-rtc"])

# SAR (cloud-independent)
results = client.search(bbox=..., date_range=...,
    collections=["sentinel-1-rtc"], cloud_cover_max=100)

# CQL2 filter
results = client.search(bbox=..., date_range=...,
    cql2_filter="eo:cloud_cover < 5 AND platform = 'sentinel-2b'")

# SearchResult properties
r = results[0]
r.id           # 'S2C_MSIL2A_20240603T153811_R001'
r.provider     # 'planetary_computer'
r.date         # '2024-06-03'
r.cloud_cover  # 0.0
r.platform     # 'Sentinel-2C'
r.resolution_m # 10.0
r.bands        # ['B02','B03','B04','B08','B11','B12']
r.bbox         # (-74.1, 40.6, -73.7, 40.9)
r.crs          # 'EPSG:32618'
r.collection   # 'sentinel-2-l2a'
```

### Download API

```python
downloads = client.download(
    results[:3],
    output_dir       = "./data/",
    parallel         = 4,
    overwrite        = False,        # resume interrupted downloads
    post_process     = [
        "reproject:EPSG:32618",      # UTM Zone 18N
        "cog",                       # Cloud-Optimized GeoTIFF
        "cloud_mask",                # Apply SCL cloud mask
    ],
    bands            = ["B02","B03","B04","B08"],   # Subset bands
    retry_attempts   = 3,
    on_failure       = "skip",       # skip | raise | warn
)

# DownloadResult
d = downloads[0]
d.success          # True
d.path             # './data/S2C_MSIL2A_...tif'
d.size_mb          # 245.3
d.scene_id         # 'S2C_MSIL2A_20240603...'
d.error            # None (or error message)
```

### Cache and Pipeline

```python
# Cache
client.cache_stats()                  # {entries, size_mb, location}
client.clear_cache()
client.clear_cache(older_than="7d")

# YAML pipeline
pipeline = (
    client.create_pipeline("weekly-sentinel2")
    .search(bbox=..., providers=["planetary_computer"],
             date_range="last_7_days", cloud_cover_max=10)
    .download(parallel=4, post_process=["reproject:EPSG:4326","cog"])
    .schedule(cron="0 6 * * 1")
)
pipeline.save("pipeline.yaml")
pipeline.run(dry_run=True)   # validate without executing
pipeline.run()
```

---

## Foundation Models

PyGeoVision v2.0 ships native integration with the two leading geospatial foundation models.

### DINOv3 (12 Variants)

```python
from pygeovision.models.foundation.dinov3 import (
    DINOv3Backbone, get_transform, finetune_dinov3,
    WEB_MEAN, WEB_STD, SAT_MEAN, SAT_STD,
)

# CRITICAL: use the correct normalisation transform
# Web pre-training (LVD-1689M) → ImageNet stats
# SAT pre-training (SAT-493M)  → Satellite stats (different mean/std!)
transform = get_transform("dinov3_vitl16_sat")   # auto-selects correct stats
```

| Model | Params | Pre-training | Embedding | Best for |
|-------|--------|-------------|-----------|----------|
| `dinov3_vits16` | 21M | Web LVD-1689M | 384 | Fast inference |
| `dinov3_vitb16` | 86M | Web LVD-1689M | 768 | Balanced |
| `dinov3_vitl16` | 300M | Web LVD-1689M | 1024 | High accuracy |
| `dinov3_vitl16_sat` | 300M | **SAT-493M** | 1024 | **Satellite imagery** |
| `dinov3_vit7b16_sat` | 6.7B | **SAT-493M** | 4096 | **State-of-the-art** |
| `dinov3_convnext_base` | 89M | Web | 1024 | Convolutional |
| *(+ 6 more variants)* | | | | |

```python
backbone = DINOv3Backbone("dinov3_vitl16_sat")

# Feature extraction
features   = backbone.extract_features(scene_path)      # (H_p, W_p, 1024) spatial
embeddings = backbone.extract_embeddings(scene_path)    # (1, 1024) global CLS
attention  = backbone.get_attention_maps(scene_path)    # (H_p, W_p) saliency

# Build classifiers
clf = backbone.build_classifier(num_classes=10, freeze_backbone=True)

# Canopy height (DINOv3 CHMv2 — 1m global, GEDI calibrated)
from pygeovision.models.foundation.dinov3 import CHMv2Model
chm    = CHMv2Model()
result = chm.predict_canopy_height(scene_path)    # mean_m, max_m, coverage_pct
biomass= chm.estimate_biomass(scene_path)          # t DM/ha

# Fine-tuning
result = finetune_dinov3(
    model_name    = "dinov3_vitl16_sat",
    dataset       = my_dataset,
    task          = "segmentation",    # classification | segmentation | detection
    num_classes   = 7,
    epochs        = 50,
    learning_rate = 1e-5,             # Recommended for SAT pre-trained
    weight_decay  = 0.05,
    mixed_precision = True,            # BF16 — critical for ViT-L
)
```

### Prithvi-EO-2.0 (600M)

```python
from pygeovision.models.foundation.prithvi import (
    Prithvi, PrithviTasks, PrithviMultiTemporal,
    map_bands, normalise_hls, finetune_prithvi,
    HLS_SCALE_FACTOR,
)
```

**CRITICAL: Band ordering.** Prithvi always expects HLS format:

| Position | HLS Band | Sentinel-2 | Landsat |
|----------|----------|-----------|---------|
| 0 | Blue  | B02 | B2 |
| 1 | Green | B03 | B3 |
| 2 | Red   | B04 | B4 |
| 3 | NIR   | B08 | B5 |
| 4 | SWIR1 | B11 | B6 |
| 5 | SWIR2 | B12 | B7 |

```python
# Always remap before inference
data_hls  = map_bands(data, source="sentinel2")
data_norm = normalise_hls(data_hls)   # divides by 10000.0

# Available tasks
tasks = PrithviTasks("prithvi_eo_2_0")
tasks.land_cover(scene_path, source="sentinel2")          # 10-class ESA-compatible
tasks.crop_mapping(scene_path, source="sentinel2")         # 10 crop types
tasks.flood_detection(scene_path, source="sentinel2")      # binary flood mask
tasks.deforestation_detection(scene_path, source="sentinel2")

# Multi-temporal (4 frames simultaneously)
mt = PrithviMultiTemporal("prithvi_eo_2_0")
mt.process_time_series(images, dates=dates, source="sentinel2")
mt.detect_change(before_path, after_path)

# Fine-tuning
result = finetune_prithvi(
    model_name    = "prithvi_eo_2_0",
    task          = "land_cover",
    num_classes   = 9,
    epochs        = 50,
    learning_rate = 5e-5,   # Paper recommendation
    batch_size    = 8,       # Memory-limited for 600M
    mixed_precision = True,
)
```

---

## Model Registry — 119 Architectures

```python
from pygeovision.models import get_model, list_models
from pygeovision.models.registry import ModelRegistry

# List and load
list_models(task="segmentation")
model = get_model("segformer-b2", num_classes=5, in_channels=6, pretrained=True)

# Registry
registry = ModelRegistry()
registry.list(task="change_detection")
registry.list(task="foundation")
registry.info("prithvi_eo_2_0")
```

**Segmentation (24):** U-Net variants, SegFormer-B0→B5, DeepLabV3+, PSPNet, Mask2Former, SAM, SAM2  
**Detection (18):** GeoYOLO v5/v8/v9/v10/v11, DETR, RT-DETR, RF-DETR, FCOS, RetinaNet, Faster R-CNN  
**Classification (16):** ViT variants, Swin Transformer, EfficientNet B0→B7, ResNet, DenseNet, ConvNeXt  
**Change Detection (12):** ChangeFormer, ChangeSTAR, BIT, DSAMNet, SNUNet, DTCDSCN, FC-EF, FC-Siam  
**Foundation (12):** DINOv3 (6 ViT + 4 ConvNeXt), Prithvi-EO-1.0, Prithvi-EO-2.0  
**VLM (9):** CLIP ViT-B/32, CLIP ViT-L/14, OpenCLIP L/14, RemoteCLIP L/14, GeoRSCLIP, Moondream 2, LLaVA-Geo, GeoChat, RSGPT  
**3D / Point Cloud (8):** PointNet, PointNet++, RandLA-Net, KPConv, Point Transformer, PointMamba, PointBERT, Point-MAE  

---

## Dataset Registry — 503 Datasets

```python
from pygeovision.datasets import DatasetRegistry, DatasetLoader

registry = DatasetRegistry()
registry.list(task="segmentation", sensor="sentinel2")
registry.list(task="change_detection")
registry.info("inria_aerial")
registry.download("deepglobe_roads", output_dir="./datasets/")

loader   = DatasetLoader(registry)
train_ds = loader.load("spacenet_buildings", split="train", chip_size=512)
```

Domains covered: **building extraction · road networks · land cover · change detection · crop mapping · flood detection · wildfire · SAR · very-high-resolution · hyperspectral · point cloud · VQA · time series**

Notable datasets: SpaceNet 1–8, DOTA v1/v2, iSAID, DIOR, RESISC-45, UC Merced, DeepGlobe, Inria Aerial, Potsdam, Vaihingen, FloodNet, xBD (disaster damage), BreizhCrops, BigEarthNet, SEN12MS, DynamicEarthNet, PASTIS, TreeSatAI, NEON, MDAS, GAMUS, SatlasPretrain, MillionAID, RS5M, and 450+ more.

---

## GeoAI Integration — 27 Subsystems

```python
ga = client.geoai   # GeoAIEngine proxy — lazy-loaded
```

**Segmentation** — buildings, solar panels, agriculture fields, water bodies, roads, coastline, oil spills, glaciers, wetlands, custom, SAM, SAM2, timm, HuggingFace Hub  
**Detection** — cars, ships, parking, grounded (natural language), RF-DETR, multi-class, instance segmentation  
**Classification** — scene classification, CLIP zero-shot land cover, batch  
**Change Detection** — ChangeSTAR bi-temporal, multi-temporal  
**Training** — segmentation, land cover, detection, instance segmentation, timm, pixel regression, RF-DETR, chip generation  
**Foundation** — Prithvi inference, SAM masks, GroundedSAM, DINOv3 analysis and fine-tuning  
**Embeddings** — patch/pixel embeddings, clustering, similarity, UMAP visualisation, Tessera downloads  
**Cloud** — cloud mask prediction, batch, statistics  
**Super-resolution** — ESRGAN 4×/8× upscaling  
**ONNX** — export, segmentation inference  
**Caption / VLM** — Moondream caption, VQA, object detection by text  
**Utilities** — raster/vector conversion, clipping, mosaicking, stacking, smoothing, metrics, device management

---

## Auto-Labeling Pipeline

```python
# 7 sources, no manual annotation required

# Microsoft Building Footprints (~1.4B global buildings)
client.labeling.microsoft_buildings(bbox, output_path, resolution_m=10.0)

# Google Open Buildings (~1.8B global)
client.labeling.google_buildings(bbox, output_path)

# OpenStreetMap
client.labeling.osm(bbox, categories=["buildings","roads","water"], output_path)

# ESA WorldCover 2021 (10m, 11 classes)
client.labeling.esa_worldcover(bbox, output_path)

# Google Dynamic World (near-real-time, 10m)
client.labeling.dynamic_world(bbox, date_range=["2024-01","2024-12"])

# SAM automatic mask generation (zero-shot)
client.labeling.sam_auto(scene_path, output_path, points_per_side=32)

# DINOv2 unsupervised clustering
from pygeovision.labeling.foundation import FoundationModelLabeler
labeler = FoundationModelLabeler("dinov2-base")
labeler.cluster(scene_path, output_path, n_clusters=7)

# Quality assessment
report = client.labeling.quality(label_path)
report["quality_grade"]   # 'A' | 'B' | 'C'
report["quality_score"]   # 0.0–1.0

# Multi-source label fusion
pipeline = client.labeling.pipeline(
    bbox=bbox,
    sources=[
        {"type": "microsoft_buildings", "weight": 1.5},
        {"type": "osm",                 "weight": 1.0},
        {"type": "sam",                 "weight": 0.8},
    ],
    fusion="weighted_vote",
    output_path="labels_fused.tif",
)
```

---

## Training

```python
from pygeovision.models import get_model
from pygeovision.training.trainer import GeoTrainer
from pygeovision.training.callbacks import EarlyStopping, ModelCheckpoint
from pygeovision.losses.segmentation import GeospatialMixedLoss

model  = get_model("segformer-b2", num_classes=5, in_channels=6, pretrained=True)
loss   = GeospatialMixedLoss(weights={"combo":0.5,"boundary":0.3,"ohem":0.2})

trainer = GeoTrainer(
    model            = model,
    task             = "segmentation",
    num_classes      = 5,
    epochs           = 100,
    learning_rate    = 6e-5,
    weight_decay     = 0.01,
    mixed_precision  = "bf16",         # "fp32" | "fp16" | "bf16"
    device           = "cuda",
    loss_fn          = loss,
    callbacks        = [
        EarlyStopping(monitor="val_iou", patience=15, mode="max"),
        ModelCheckpoint("./checkpoints/", monitor="val_iou", save_top_k=3),
    ],
)

history = trainer.fit(train_dl, val_dl)
print(f"Best val_iou: {history['best_metrics']['val_iou']:.4f}")
```

**Available losses:** `DiceLoss` · `FocalLoss` · `TverskyLoss` · `GeospatialMixedLoss` (Dice+Boundary+OHEM) · `ChangeDetectionLoss` · `BoundaryAwareLoss` · `WeightedCrossEntropyLoss`

**Supported optimisers:** AdamW · SGD · Lion · RMSProp  
**Supported schedulers:** CosineAnnealing · ReduceLROnPlateau · OneCycleLR · WarmupCosine  
**Supported precisions:** FP32 · FP16 · BF16 (recommended for A100/H100)

---

## Tiled Inference

```python
from pygeovision.inference.tiled import TiledInference

inf    = TiledInference(
    model       = model,
    chip_size   = 512,
    overlap     = 64,
    blend_mode  = "gaussian",   # "gaussian" | "average"
    num_classes = 5,
    device      = "cuda",
    batch_size  = 8,
)
result = inf.infer(scene_path, output_path)
# result: {n_chips, duration_seconds, chips_per_second}

# Ensemble inference
from pygeovision.inference.tiled import EnsembleInference
ensemble = EnsembleInference(models=[model_a, model_b], mode="mean")
ensemble.infer(scene_path, output_path)
```

---

## End-to-End Pipelines (10)

| Pipeline | Data | AI Model | Output |
|----------|------|----------|--------|
| `building_footprints` | Sentinel-2 / NAIP | GeoAI BuildingFootprintExtractor | GeoJSON polygons |
| `change_detection` | Bi-temporal Sentinel-2 | ChangeFormer / ChangeSTAR | Change mask GeoTIFF |
| `land_cover` | Sentinel-2 | Prithvi-EO-2.0 / ESA WorldCover | Classification GeoTIFF |
| `water_bodies` | Sentinel-2 | NDWI + GeoAI segment | Water polygons |
| `solar_detection` | NAIP / Sentinel-2 | SolarPanelDetector | GeoJSON polygons |
| `crop_monitoring` | Sentinel-2 time series | Prithvi-EO-2.0 | Crop type map |
| `disaster_assessment` | Pre/post event imagery | ChangeFormer + xBD | Damage map |
| `deforestation` | Bi-temporal Landsat/S2 | ChangeFormer | Forest loss mask |
| `urban_growth` | Bi-temporal Landsat | Siamese U-Net | Urban expansion map |
| `carbon_estimation` | Sentinel-2 + DINOv3 CHMv2 | Biomass allometric | Carbon stock (t CO₂e) |

```python
result = client.pipeline("building_footprints",
    bbox=(-0.15,51.47,-0.10,51.52), date="2024-06")
result = client.pipeline("change_detection",
    bbox=..., date_before="2020-01", date_after="2024-01")
result = client.pipeline("carbon_estimation",
    bbox=..., date="2024-07", output_dir="./carbon/")

# YAML pipeline
p = client.create_pipeline("weekly_buildings")
p.search(bbox=..., providers=["planetary_computer"], date_range="last_7_days")
p.download(post_process=["reproject:EPSG:32618","cog"], parallel=4)
p.ai_step("segment_buildings", model="segformer-b2", num_classes=2)
p.export(format="geojson", output_dir="./results/")
p.schedule(cron="0 3 * * 1")   # every Monday 03:00 UTC
p.save("weekly_buildings.yaml")
p.run(dry_run=True)
```

---

## Serving API

```python
from pygeovision.serving import InferenceServer

server = InferenceServer(
    auth_keys    = {"prod": "SECRET_KEY", "dev": "DEV_KEY"},
    max_workers  = 4,
    enable_cors  = True,
)
server.register("seg_v1", "model.onnx", task="segmentation", num_classes=5)
server.register("det_v1", "detector.onnx", task="detection")
server.serve(host="0.0.0.0", port=8080)
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Single-scene inference |
| `/predict/batch` | POST | Batch inference (multiple scenes) |
| `/ws/stream` | WS | WebSocket streaming for live ingestion |
| `/health` | GET | Health check + model status |
| `/models` | GET | List registered models |
| `/metrics` | GET | Prometheus-compatible metrics |

---

## Edge and Cloud Deployment

### ONNX / Edge

```python
from pygeovision.edge.onnx_rt import ONNXRuntimeInference

# Export from PyTorch
ONNXRuntimeInference.from_pytorch(model, "model.onnx", input_shape=(1,4,512,512))

# Run inference
eng = ONNXRuntimeInference("model.onnx", device="cpu")   # or "cuda"
eng.infer_geotiff("scene.tif", "prediction.tif")
```

### Jetson (TensorRT FP16)

```python
from pygeovision.edge.jetson import JetsonDeployer

deployer = JetsonDeployer(device_type="orin")   # "nano" | "xavier" | "orin"
result   = deployer.convert("model.onnx", "model.trt", precision="fp16")
# Speed: ~45 chips/s at 512×512 on Jetson Orin
```

### Cloud

```python
from pygeovision.cloud.deploy import AWSDeployer, AzureDeployer, GCPDeployer

# AWS SageMaker
AWSDeployer(region="us-east-1").deploy(
    "model.onnx", "buildings-prod", instance_type="ml.g4dn.xlarge")

# Azure ML
AzureDeployer(subscription_id="...", resource_group="rg").deploy(
    "model.onnx", "buildings-endpoint", vm_size="Standard_NC6s_v3")

# GCP Vertex AI
GCPDeployer(project_id="my-project").deploy(
    "model.onnx", "buildings-endpoint",
    machine_type="n1-standard-8", accelerator_type="NVIDIA_TESLA_T4")
```

| Platform | Hardware | Speed | Cost/hr |
|----------|----------|-------|---------|
| ONNX CPU | 8-core CPU | ~2 chips/s | $0.05 |
| ONNX CUDA | RTX 3090 | ~120 chips/s | $0.40 |
| Jetson Orin | TensorRT FP16 | ~45 chips/s | $0.00 |
| AWS SageMaker | ml.g4dn.xlarge | ~120 chips/s | $0.74 |
| GCP Vertex AI | T4 GPU | ~110 chips/s | $0.90 |

---

## Advanced Features

### Few-Shot Learning

```python
from pygeovision.advanced.few_shot import FewShotLearner

learner = FewShotLearner(backbone="dinov2-large", method="prototypical")
learner.fit_support({
    "solar_panel": ["img1.tif","img2.tif","img3.tif"],
    "rooftop":     ["img4.tif","img5.tif","img6.tif"],
})
result = learner.predict("new_scene.tif")
# {"class": "solar_panel", "confidence": 0.91}
# Accuracy: ~78% at 1-shot, ~87% at 5-shot, ~91% at 10-shot
```

### AutoML / HPO

```python
from pygeovision.advanced.automl import AutoML

automl = AutoML(model_family="segformer", task="segmentation",
                num_classes=5, n_trials=50, timeout_hours=4.0)
best   = automl.optimise(train_dl, val_dl)
print(f"Best val_iou: {best['val_iou']:.4f}")
print(f"Best config : {best['params']}")
# Typical improvement: +2–4 mIoU over defaults
```

### Time-Series Analysis

```python
from pygeovision.advanced.timeseries import GeoTimeSeries

ts = GeoTimeSeries(sensor="sentinel2")
ts.compute_trend(series)          # direction, slope, R², p-value
ts.detect_anomalies(series, threshold=2.0)
ts.decompose(series, period=12)   # trend + seasonal + residual
ts.crop_type(scene_paths, dates)
ts.ndvi(scene_path)
```

### Vision-Language Models

```python
from pygeovision.advanced.vlm.clip_geo import CLIPGeo
from pygeovision.advanced.vlm.moondream_geo import MoondreamGeo

# CLIP zero-shot classification
clip   = CLIPGeo(model_name="remoteclip-l14")   # trained on RS5M (5M RS pairs)
probs  = clip.classify(scene_path, ["dense urban","tropical forest","cropland"])
similar= clip.search_by_image("query.tif", index_path="geo_index.faiss", top_k=10)
clip.build_index("./archive/", "geo_index.faiss")

# Moondream VQA
moon   = MoondreamGeo()
caption= moon.caption(scene_path)
answer = moon.vqa(scene_path, "How many buildings are visible?")
moon.describe_change(before_path, after_path)
```

### Explainability

```python
from pygeovision.explainability.gradcam import GeoGradCAM
from pygeovision.explainability.attention import AttentionVisualiser

gradcam   = GeoGradCAM(model)
heatmap   = gradcam.generate(scene_path, class_idx=1)
attention = AttentionVisualiser(backbone).visualise(scene_path)
```

---

## Monitoring

```python
from pygeovision.monitoring.drift   import DriftDetector
from pygeovision.monitoring.tracker import ModelPerformanceTracker
from pygeovision.monitoring.alerts  import AlertManager

# Drift detection (PSI — Population Stability Index)
detector = DriftDetector(method="psi", threshold_warn=0.1, threshold_critical=0.2)
detector.fit(reference_images)
report = detector.check(new_images)

# Performance tracking
tracker = ModelPerformanceTracker(metrics=["val_iou","val_f1","throughput_fps"])
tracker.log(epoch=50, metrics={"val_iou":0.843,"val_f1":0.887,"throughput_fps":118})
trend   = tracker.trend("val_iou")    # direction, slope

# Alerts (Slack / email / webhook)
alerts = AlertManager(channels={"slack":{"webhook_url":"https://..."}})
alerts.add_rule("iou_drop",  "val_iou",          "less_than",    0.78, "critical")
alerts.add_rule("drift",     "psi_score",         "greater_than", 0.10, "warning")
alerts.add_rule("throughput","throughput_fps",    "less_than",    80,   "warning")
alerts.check({"val_iou":0.72})   # fires if below threshold
```

---

## Command-Line Interface

```bash
# ── Status ──────────────────────────────────────────────────────────────────
pygeovision status
pygeovision status --json
pygeovision doctor

# ── Authentication ────────────────────────────────────────────────────────
pygeovision data auth add usgs --username USER --password PASS
pygeovision data auth add planet --api-key PL-xxxx
pygeovision data auth list
pygeovision data auth test planetary_computer

# ── Search & Download ────────────────────────────────────────────────────
pygeovision data search \
    --bbox -74.1 40.6 -73.7 40.9 \
    --date 2024-06 \
    --providers planetary_computer copernicus \
    --cloud-max 10 \
    --output results.geojson

pygeovision data download \
    --from-search results.geojson \
    --output ./data/ \
    --parallel 4 \
    --post-process reproject:EPSG:32618,cog

# ── Pipeline ──────────────────────────────────────────────────────────────
pygeovision data pipeline run weekly.yaml
pygeovision data pipeline validate weekly.yaml
pygeovision data pipeline schedule weekly.yaml --cron "0 6 * * 1"

# ── AI: Segmentation ──────────────────────────────────────────────────────
pygeovision ai segment buildings \
    --input sentinel2.tif --output buildings.tif --vector buildings.geojson

pygeovision ai segment solar --input aerial.tif --output solar.tif
pygeovision ai segment water --input s2.tif --output water.tif
pygeovision ai segment custom --input scene.tif --model model.pth --output pred.tif

# ── AI: Detection ─────────────────────────────────────────────────────────
pygeovision ai detect cars   --input aerial.tif --output cars.geojson
pygeovision ai detect ships  --input port.tif   --output ships.geojson
pygeovision ai detect grounded --input aerial.tif --prompt "swimming pools"

# ── AI: Training ──────────────────────────────────────────────────────────
pygeovision ai train segmentation \
    --data ./chips/ --output model.pth \
    --num-classes 5 --epochs 100 --backbone segformer-b2

# ── AI: Inference ─────────────────────────────────────────────────────────
pygeovision ai infer \
    --input large_scene.tif --model model.pth \
    --output prediction.tif --tile-size 512 --overlap 64

# ── Models ────────────────────────────────────────────────────────────────
pygeovision models list
pygeovision models list --task segmentation
pygeovision models info segformer-b2
pygeovision models list --task foundation

# ── Datasets ──────────────────────────────────────────────────────────────
pygeovision datasets list
pygeovision datasets info spacenet_buildings
pygeovision datasets download deepglobe_roads --output ./datasets/

# ── Pipelines ─────────────────────────────────────────────────────────────
pygeovision pipeline building_footprints \
    --bbox -0.15 51.47 -0.10 51.52 --date 2024-06 --output ./results/

pygeovision pipeline change_detection \
    --bbox -74.1 40.6 -73.7 40.9 \
    --date-before 2020-01 --date-after 2024-01

pygeovision pipeline list

# ── Serve ─────────────────────────────────────────────────────────────────
pygeovision serve start --model model.onnx --port 8080
pygeovision serve status
pygeovision serve stop
```

---

## Project Notebooks — 25 Production Workflows

The `projects/` directory contains 25 fully self-contained, production-grade Jupyter notebooks covering every major geospatial AI domain. Each uses real satellite data and demonstrates a complete end-to-end workflow.

```bash
cd projects/
jupyter notebook
```

| # | Notebook | Domain | Real-World Problem |
|---|----------|--------|-------------------|
| 01 | Satellite Data Acquisition | Data | Download Sentinel-2 for Amazon study area |
| 02 | Building Footprint Extraction | Urban | City-scale footprints with auto-labeling |
| 03 | Land Cover with Prithvi-EO-2.0 | Land Cover | 9-class mapping with 600M foundation model |
| 04 | Change Detection | Disaster | Identify damage after hurricane |
| 05 | Agricultural Crop Monitoring | Agriculture | NDVI time series for insurance assessment |
| 06 | Forest Monitoring | Forestry | Detect deforestation + estimate biomass |
| 07 | Water & Flood Mapping | Disaster | Rapid flood extent within hours |
| 08 | Solar Panel Detection | Energy | Inventory + energy potential (Zurich) |
| 09 | Disaster Damage Assessment | Emergency | 4-class building damage (earthquake) |
| 10 | Urban Growth Analysis | Urban | Lagos decade-long expansion |
| 11 | Road Network Extraction | Infrastructure | Segmentation + vectorisation (Paris) |
| 12 | Crop Type Mapping | Agriculture | 10-class Prithvi (Toulouse basin) |
| 13 | Glacier Monitoring | Climate | Aletsch retreat + RCP4.5/8.5 projections |
| 14 | Oil Spill Detection (SAR) | Environment | Sentinel-1 night/cloud-proof detection |
| 15 | Air Quality (NO2, PM2.5) | Environment | Sentinel-5P TROPOMI (Rome) |
| 16 | Wildfire Detection | Disaster | BAI + dNBR USFS 4-class severity |
| 17 | Biodiversity Mapping | Ecology | DINOv3 unsupervised habitat clustering |
| 18 | Infrastructure Monitoring | Civil Eng | New Cairo construction progress |
| 19 | Coastal & Wetland | Environment | Camargue wetland loss 2015–2024 |
| 20 | Climate Change (LST, NDVI) | Climate | Dubai UHI + vegetation decline |
| 21 | Custom Model Training | ML/AI | Auto-label → train → export pipeline |
| 22 | End-to-End Pipeline Deployment | MLOps | YAML + cloud + monitoring |
| 23 | Foundation Model Fine-Tuning | Deep Learning | DINOv3 + Prithvi fine-tuning guide |
| 24 | DINOv3 Embedding Analysis | AI/Retrieval | Semantic search over 10k-scene archive |
| 25 | Vision-Language Querying | AI/NLP | CLIP + Moondream VQA |

**Stats:** 357 cells · 302 KB · all 25 run without GPU (demo mode when no credentials)

---

## Testing

```
580 passing  |  2 skipped  |  0 failing
```

```
tests/
├── test_core.py                     Core config, exceptions, engine
├── test_data_layer.py               SatelliteFetcher, providers, pipeline
├── test_geoai_integration.py        All 27 GeoAI subsystems (mocked)
├── test_foundation_models.py        DINOv3 + Prithvi (mocked weights)
├── test_dinov3.py                   DINOv3 backbone, transforms, CHMv2
├── test_prithvi.py                  Prithvi tasks, band mapping, multi-temporal
├── test_edge_cloud.py               ONNX, Jetson, AWS, Azure, GCP
├── test_advanced.py                 Few-shot, AutoML, VLM, timeseries
├── test_monitoring.py               Drift, tracker, alerts
├── test_serving.py                  FastAPI endpoints, WebSocket, auth
├── test_training.py                 GeoTrainer, losses, metrics, callbacks
├── test_inference.py                TiledInference, ensemble, postprocessing
├── test_labeling.py                 All 7 labelers + quality + fusion
├── test_models.py                   Model registry, loader, 119 architectures
├── test_datasets.py                 Dataset registry, loader, 503 datasets
├── test_pipelines.py                All 10 end-to-end pipelines
├── test_explainability.py           GradCAM, attention, uncertainty
├── test_cli.py                      All 15 CLI command groups
├── test_pointcloud.py               PointNet++, RandLA-Net, KPConv
└── conftest.py                      Shared fixtures, mock providers
```

```bash
pip install -e ".[dev]"
pytest tests/ -q                                      # all 580 tests
pytest tests/test_foundation_models.py -v            # foundation models only
pytest tests/ --cov=pygeovision --cov-report=html    # with coverage report
```

---

## Comparison

| Feature | **PyGeoVision v2** | EODAG | TorchGeo | TerraTorch | Raw GeoAI |
|---------|:------------------:|:-----:|:--------:|:----------:|:---------:|
| Data providers | **22+** | 10+ | Limited | Limited | 3 |
| PyGeoFetch integration | ✅ Native | ❌ | ❌ | ❌ | ❌ |
| GeoAI integration | ✅ 27 subsystems | ❌ | ❌ | ❌ | ✅ Direct |
| Model registry | **119** | — | ~30 | ~50 | — |
| Dataset registry | **503** | — | ~40 | ~30 | — |
| Foundation models | ✅ DINOv3+Prithvi | ❌ | ✅ Partial | ✅ Partial | ✅ |
| Auto-labeling | **7 sources** | ❌ | ❌ | ❌ | ❌ |
| YAML pipelines | ✅ | ❌ | ❌ | ❌ | ❌ |
| Serving API | ✅ FastAPI+WS | ❌ | ❌ | ❌ | ❌ |
| Edge (ONNX+Jetson) | ✅ | ❌ | ❌ | ❌ | ✅ Partial |
| Cloud (AWS/Azure/GCP) | ✅ All 3 | ❌ | ❌ | ❌ | ❌ |
| Few-shot learning | ✅ | ❌ | ❌ | ❌ | ❌ |
| VLM (CLIP+Moondream) | ✅ | ❌ | ❌ | ❌ | ✅ Partial |
| Production notebooks | **25** | ❌ | ❌ | ❌ | ❌ |
| Tests | **580** | ~200 | ~300 | ~100 | ~150 |
| CLI | ✅ 15 groups | ✅ | ❌ | ❌ | ❌ |

---

## Documentation

```
docs/                          44 pages · 6,976 lines
├── index.md
├── installation.md
├── quickstart.md
├── architecture.md
├── contributing.md
├── api/                       20 API reference pages
│   ├── pygeovision.md         Main client reference
│   ├── models.md              119-model registry
│   ├── foundation.md          DINOv3 + Prithvi
│   ├── training.md            GeoTrainer
│   ├── serving.md             FastAPI + WebSocket
│   ├── edge.md                ONNX + Jetson
│   ├── cloud.md               AWS + Azure + GCP
│   ├── vlm.md                 CLIP + Moondream
│   └── ...
├── tutorials/                 11 step-by-step guides
│   ├── getting_started.md
│   ├── foundation_models.md   DINOv3 + Prithvi cookbook
│   ├── custom_training.md
│   ├── deployment.md
│   └── ...
└── examples/                  7 domain examples
    ├── agriculture.md
    ├── forestry.md
    ├── urban.md
    └── ...
```

---

## Package Structure

```
pygeovision/                          182 Python files · 0 syntax errors
├── __init__.py                       PyGeoVision client
├── data/fetch.py                     SatelliteFetcher (22 providers)
├── models/                           119 model architectures
│   ├── registry.py, base.py
│   ├── classification/               ViT, Swin, EfficientNet, DINOv3
│   ├── detection/                    GeoYOLO, DETR, RF-DETR
│   ├── segmentation/                 U-Net, SegFormer, SAM
│   ├── change_detection/             ChangeFormer, ChangeSTAR, BIT
│   ├── foundation/                   dinov3.py (1130L), prithvi.py (882L)
│   ├── vlm/                          clip.py, moondream.py
│   └── _3d/                          pointnet.py, randlanet.py, kpconv.py
├── labeling/                         9 labeler files
├── losses/                           segmentation.py, detection.py
├── inference/                        tiled.py, batch.py, stream.py, ensemble.py
├── explainability/                   gradcam.py, uncertainty.py, attention.py
├── monitoring/                       drift.py, tracker.py, alerts.py
├── training/                         trainer.py, callbacks.py, metrics.py
├── serving/                          api.py (FastAPI+WebSocket), auth.py
├── pipelines/                        10 pipelines + YAML orchestrator
├── datasets/                         registry.py (503), loader.py, catalog.py
├── cli/main.py                       15 command groups
├── edge/                             onnx_rt.py, jetson.py
├── cloud/                            deploy.py (AWS, Azure, GCP)
└── advanced/                         few_shot.py, automl.py, vlm/, timeseries/, pointcloud/
```

---

## Acknowledgements

PyGeoVision is built on top of two exceptional open-source projects:

- **[PyGeoFetch](https://github.com/appiahkubis14/PyGeoFetch)** — Universal satellite data pipeline. PyGeoVision uses PyGeoFetch for all data search, download, authentication, caching, and pipeline orchestration.

- **[GeoAI](https://opengeoai.org)** — Artificial Intelligence for Geospatial Data by [Qiusheng Wu](https://github.com/giswqs) and contributors. PyGeoVision wraps GeoAI for AI inference, training, and model management. Published in [JOSS 2026](https://doi.org/10.21105/joss.09605).

---

## License

**Apache 2.0** — see [LICENSE](LICENSE)

Copyright © 2026 PyGeoVision Contributors

---

<div align="center">

**[Documentation](https://pygeovision.org)** · **[PyPI](https://pypi.org/project/pygeovision/)** · **[GitHub](https://github.com/pygeovision/pygeovision)** · **[Notebooks](projects/)**

*Built for the Salzburg International Geospatial AI Symposium 2026*

</div>
