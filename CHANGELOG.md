# Changelog

All notable changes to PyGeoVision are documented here.
This project adheres to [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Added
- Initial public release of PyGeoVision geospatial AI platform
- `PyGeoVision` core class wrapping PyGeoFetch for all data operations
- `AIEngine` with lazy-loaded subsystems (data, labeling, models, training, inference, pipelines)

### AI Data Layer
- `TilingEngine` — streaming tile generator for arbitrarily large rasters
- `GeoDataset` / `TileDataset` — PyTorch Dataset wrappers for GeoTIFFs
- `GeoDataLoader` — spatial train/val/test splits (no data leakage)
- `GeoAugmentationPipeline` — geospatially-safe augmentations (90° rotations only)
- `GeoPreprocessor` — per-band normalization, NDVI, pansharpening
- `ClassBalancedSampler`, `GeographicBlockSampler`, `StratifiedTileSampler`

### Automated Labelers (7 sources)
- `OSMLabeler` — OpenStreetMap buildings, roads, water via Overpass API
- `MicrosoftBuildingsLabeler` — 1.3B buildings via QuadKey index
- `GoogleBuildingsLabeler` — 1.8B buildings via S2 tile index
- `ESAWorldCoverLabeler` — 10m global land cover (11 classes, 2020/2021)
- `DynamicWorldLabeler` — near real-time land cover via Planetary Computer / GEE
- `SAMLabeler` — Segment Anything Model (vit_h, vit_l, vit_b, mobile_sam)
- `FoundationLabeler` — Prithvi / Clay / SatMAE / RemoteCLIP pseudo-labels
- `LabelStudioLabeler` — human-in-the-loop annotation via Label Studio

### Models (14 built-in architectures)
- Segmentation: UNet (ResNet-50, EfficientNet-B4), DeepLabV3+ (ResNet-101), SegFormer (B2, B5), FPN, PAN
- Detection: FCOS, RetinaNet
- Classification: ResNet-50, EfficientNet-B3, ViT-B/16
- Change detection: Siamese-UNet, ChangeFormer
- Super resolution: SRCNN, ESRGAN-Geo

### Training
- `GeoTrainer` — full training loop with AMP, gradient accumulation, DDP support
- Loss functions: Dice, Focal, DiceFocal, Tversky, WeightedCE, ChangeDetection
- Metrics: ConfusionMatrix, BinaryMetrics, AverageMeter
- Callbacks: EarlyStopping, ModelCheckpoint, MLflowLogger, LRSchedulerCallback
- `ModelExporter` — ONNX and TorchScript export + benchmarking
- Distributed training utilities (DDP setup/cleanup/wrap)

### Inference
- `TiledInference` — streaming tile inference with Gaussian blend stitching
- `EnsembleInference` — multi-model ensembling with TTA support
- `PostProcessor` — morphological cleanup, confidence thresholding, polygon vectorization

### Pipelines (10 end-to-end)
- `change_detection`, `land_cover`, `building_footprints`, `crop_monitoring`
- `disaster_assessment`, `deforestation`, `urban_growth`, `water_bodies`
- `solar_detection`, `carbon_estimation`

### Monitoring & Experiments
- `DriftDetector` — KS-test and MMD distribution drift detection
- `PerformanceTracker` — metric history and regression alerting
- `ExperimentTracker` — lightweight experiment tracking with reproducibility

### CLI
- `pygeovision data search/download` — PyGeoFetch data commands
- `pygeovision pipeline <name>` — run any pipeline from the CLI
- `pygeovision models list/info/cache` — model management
- `pygeovision train` — training from YAML config
- `pygeovision infer` — tiled inference on GeoTIFFs
- `pygeovision status` — environment check

---

## Format

Each release section uses: **Added**, **Changed**, **Deprecated**, **Removed**, **Fixed**, **Security**.
