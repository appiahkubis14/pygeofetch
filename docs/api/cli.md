# CLI Reference

PyGeoVision ships a complete command-line interface with 15 command groups. Install once, use everywhere.

---

## Installation

```bash
pip install pygeovision
pygeovision --help
# or the short alias:
pgv --help
```

---

## Command Groups

| Group | Purpose |
|-------|---------|
| `pgv data` | Search, download, authenticate satellite data |
| `pgv models` | List, inspect, download model architectures |
| `pgv infer` | Run inference on GeoTIFFs |
| `pgv label` | Auto-labeling from OSM, ESA, SAM, and more |
| `pgv explain` | XAI — GradCAM, SHAP saliency maps |
| `pgv monitor` | Drift detection, performance tracking |
| `pgv pipeline` | Run and schedule YAML pipelines |
| `pgv edge` | Export ONNX, benchmark, Jetson deployment |
| `pgv cloud` | Deploy to AWS, Azure, GCP |
| `pgv vlm` | Caption, VQA, and search with VLMs |
| `pgv timeseries` | NDVI/NDWI time series and anomaly detection |
| `pgv benchmark` | Run standardised benchmarks |
| `pgv search` | Quick satellite scene search |
| `pgv download` | Quick scene download |
| `pgv auth` | Manage provider credentials |

---

## `pgv data`

```bash
# Search Sentinel-2 scenes
pgv data search \
  --bbox -74.1 40.6 -73.7 40.9 \
  --providers planetary_computer \
  --cloud-cover-max 10 \
  --date-range 2024-06-01 2024-08-31

# Download
pgv data download \
  --bbox -74.1 40.6 -73.7 40.9 \
  --output ./data/ \
  --parallel 4 \
  --post-process reproject:EPSG:32618 cog

# Add provider credentials
pgv auth add planetary_computer --api-key YOUR_KEY
pgv auth add copernicus --username user --password pass
pgv auth list
```

---

## `pgv models`

```bash
# List all available models
pgv models list

# Filter by task
pgv models list --task segmentation
pgv models list --task detection
pgv models list --task foundation

# Filter by parameter budget
pgv models list --max-params 50

# Show detailed model info
pgv models info segformer-b2
pgv models info dinov3_vitl16_sat

# Download weights
pgv models download sam-vit-large
pgv models download dinov3_vitl16_sat

# Registry summary
pgv models summary
```

---

## `pgv infer`

```bash
# Single image inference
pgv infer predict scene.tif \
  --model segformer-b2 \
  --classes 7 \
  --chip-size 512 \
  --overlap 64 \
  --blend gaussian \
  --output ./output/pred.tif

# Batch directory
pgv infer batch ./data/scenes/ ./output/preds/ \
  --model unet-r50 \
  --workers 4

# With test-time augmentation
pgv infer predict scene.tif --model deeplab-r101 --tta
```

---

## `pgv label`

```bash
# OSM auto-labeling
pgv label osm -74.1 40.6 -73.7 40.9 \
  --categories buildings roads water \
  --output ./labels/osm.tif

# ESA WorldCover
pgv label worldcover -74.1 40.6 -73.7 40.9 \
  --output ./labels/worldcover.tif

# SAM automatic segmentation
pgv label sam scene.tif \
  --output ./labels/sam.tif \
  --points-per-side 32

# Label quality assessment
pgv label quality ./labels/buildings.tif
pgv label quality ./labels/buildings.tif --html   # Save HTML report
```

---

## `pgv explain`

```bash
# GradCAM saliency map
pgv explain gradcam scene.tif \
  --model segformer-b2 \
  --class-idx 1 \
  --output ./xai/gradcam.tif

# Uncertainty estimation
pgv explain uncertainty scene.tif \
  --model segformer-b2 \
  --n-passes 50 \
  --output ./xai/uncertainty.tif
```

---

## `pgv monitor`

```bash
# Detect data drift
pgv monitor drift ./reference/ ./current/ \
  --output ./monitoring/drift_report.json

# Track performance
pgv monitor performance --log val_iou=0.843 val_f1=0.887 --epoch 50
```

---

## `pgv pipeline`

```bash
# Run a pipeline
pgv pipeline run agriculture.yaml
pgv pipeline run agriculture.yaml --dry-run

# List built-in templates
pgv pipeline list-templates

# Schedule (cron)
pgv pipeline schedule agriculture.yaml --cron "0 6 * * *"
```

---

## `pgv edge`

```bash
# Export to ONNX
pgv edge export-onnx segformer-b2 \
  --output model.onnx \
  --classes 7 \
  --in-channels 4 \
  --input-size 512

# Benchmark ONNX model
pgv edge benchmark-onnx model.onnx --device cuda --runs 200

# Deploy to Jetson
pgv edge deploy-jetson model.onnx \
  --output model.trt \
  --precision fp16
```

---

## `pgv cloud`

```bash
pgv cloud deploy-aws model.onnx my-endpoint \
  --region us-east-1 \
  --instance ml.g4dn.xlarge

pgv cloud deploy-gcp model.onnx my-endpoint \
  --project my-gcp-project \
  --region us-central1

pgv cloud deploy-azure model.onnx my-endpoint \
  --subscription-id xxxx \
  --resource-group ml-rg \
  --workspace my-ws
```

---

## `pgv vlm`

```bash
# Generate caption
pgv vlm caption scene.tif

# VQA
pgv vlm query scene.tif "How many buildings are visible?"

# Text-to-image search
pgv vlm search "flooded farmland" ./data/scenes/ --top-k 10
```

---

## `pgv timeseries`

```bash
# Compute NDVI time series
pgv timeseries analyze jan.tif apr.tif jul.tif oct.tif \
  --index ndvi \
  --output ndvi_series.json

# Detect anomalies
pgv timeseries anomaly jan.tif feb.tif mar.tif \
  --threshold 2.5
```

---

## Global Options

```bash
pgv --version          # Print PyGeoVision version
pgv --verbose          # Enable debug logging
pgv --config my.yaml   # Load settings from YAML config file
pgv <command> --help   # Help for any command
```
