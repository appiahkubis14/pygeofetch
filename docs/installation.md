# Installation

## Requirements

- Python ≥ 3.10
- pip ≥ 23.0

---

## Quick Install

```bash
pip install pygeovision
```

---

## Install Extras

PyGeoVision is modular. Install only what you need:

```bash
# Geospatial stack (rasterio, geopandas, GDAL bindings)
pip install "pygeovision[geo]"

# AI model layer (torch, timm, segmentation-models-pytorch)
pip install "pygeovision[train]"

# Serving API (fastapi, uvicorn, pydantic)
pip install "pygeovision[serve]"

# Auto-labeling (requests, scipy, Pillow)
pip install "pygeovision[labeling]"

# Foundation models (transformers, huggingface_hub)
pip install "pygeovision[foundation]"

# Vision-Language Models (openclip-torch, transformers)
pip install "pygeovision[vlm]"

# Explainability (shap, captum)
pip install "pygeovision[xai]"

# Edge deployment (onnxruntime, onnx, onnxsim)
pip install "pygeovision[edge]"

# Cloud deployment (boto3, sagemaker, azure-ai-ml, google-cloud-aiplatform)
pip install "pygeovision[cloud]"

# Time series (scipy, statsmodels)
pip install "pygeovision[timeseries]"

# 3D / LiDAR (laspy, open3d)
pip install "pygeovision[geo3d]"

# Monitoring (scipy, matplotlib)
pip install "pygeovision[monitoring]"

# Development tools (pytest, ruff, mypy, black)
pip install "pygeovision[dev]"

# Everything
pip install "pygeovision[all]"
```

---

## Extras Reference

| Extra | Key Packages | Use Case |
|-------|-------------|----------|
| `geo` | rasterio, geopandas, rioxarray, shapely | GeoTIFF reading, CRS handling |
| `viz` | matplotlib, folium, plotly | Map visualisation |
| `train` | torch, timm, segmentation-models-pytorch, optuna | Model training |
| `serve` | fastapi, uvicorn, pydantic, PyJWT | REST inference API |
| `labeling` | requests, scipy, Pillow | Auto-label generation |
| `foundation` | transformers, huggingface_hub | DINOv3, Prithvi, SAM |
| `vlm` | openclip-torch | CLIP, RemoteCLIP, Moondream |
| `xai` | shap, captum | GradCAM, SHAP saliency |
| `edge` | onnxruntime, onnx | ONNX export and inference |
| `cloud` | boto3, sagemaker, azure-ai-ml, google-cloud-aiplatform | Cloud deployment |
| `geo3d` | laspy, open3d | LiDAR, point clouds |
| `monitoring` | scipy, matplotlib | Distribution drift detection |
| `dev` | pytest, ruff, mypy, black, pre-commit | Development workflow |
| `all` | Everything above | Full platform |

---

## System Dependencies

Some packages require system-level GDAL bindings. On Ubuntu/Debian:

```bash
sudo apt-get update && sudo apt-get install -y \
    gdal-bin libgdal-dev libproj-dev libgeos-dev \
    python3-dev build-essential
```

On macOS with Homebrew:

```bash
brew install gdal proj geos
```

---

## GPU Support

PyGeoVision automatically detects and uses CUDA, MPS (Apple Silicon), or falls back to CPU.

```bash
# Install PyTorch with CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Then install PyGeoVision
pip install "pygeovision[train,foundation]"
```

---

## Verify Installation

```python
import pygeovision as pgv

client = pgv.PyGeoVision()
print(client)
# PyGeoVision(v2.0 | datasets=503 | models=119 | geoai=independent)
```
