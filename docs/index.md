# PyGeoVision v2.0

**The complete, independent geospatial AI platform.**  
Zero GeoAI dependency. Production-ready. Salzburg-ready.

```bash
pip install pygeovision
```

## What is PyGeoVision?

PyGeoVision v2.0 is a fully independent geospatial AI platform that provides:

- **50+ model architectures** — segmentation, detection, classification, change detection, foundation models, VLM, 3D
- **503 benchmark datasets** — from EuroSAT to OmniEarth 10M corpus
- **15 CLI command groups** — data, models, infer, label, explain, monitor, edge, cloud, vlm, timeseries, and more
- **7 auto-labeling sources** — OSM, Microsoft Buildings, Google Buildings, ESA WorldCover, SAM, DINOv2, Active Learning
- **10 geospatial losses** — Dice, Focal, Tversky, Boundary-Aware, Lovász, OHEM, Mixed
- **Full serving layer** — FastAPI REST API with auth, WebSocket streaming, batch inference
- **Cloud deployment** — AWS SageMaker, Azure ML, GCP Vertex AI
- **Edge deployment** — ONNX Runtime, NVIDIA Jetson/TensorRT
- **Foundation models** — DINOv3 (12 variants), Prithvi-EO-2.0

## 5-Minute Quickstart

```python
import pygeovision as pgv

# Initialize client
client = pgv.PyGeoVision()

# Search for satellite imagery
results = client.search(
    bbox=[-74.1, 40.6, -73.7, 40.9],          # New York City
    date_range=["2024-01-01", "2024-12-31"],
    providers=["planetary_computer"],
    cloud_cover_max=10,
)

# Download
downloads = client.download(results, output_dir="./data/")

# Run building extraction
prediction = client.pipeline("building_extraction").run(
    bbox=[-74.1, 40.6, -73.7, 40.9]
)

# Auto-label with OSM
labels = client.labeling.osm(
    bbox=[-74.1, 40.6, -73.7, 40.9],
    categories=["buildings", "roads", "water"]
)
```

## Key Capabilities

| Layer | What's included |
|-------|----------------|
| 🗄 **Data** | 22 satellite providers, STAC search, CQL2 filtering, COG download |
| 🧠 **Models** | 50+ architectures: U-Net, SegFormer, SAM, YOLO, DINOv3, Prithvi |
| 🏷 **Labeling** | 7 auto-labeling sources + active learning + quality assessment |
| ⚡ **Inference** | Gaussian-blend tiling, batch, streaming, ensemble |
| 💡 **XAI** | GradCAM, SHAP, MC Dropout uncertainty, attention maps |
| 📊 **Monitoring** | Distribution drift (PSI+KL), performance tracking, alerts |
| 🖥 **Serving** | FastAPI + JWT auth + WebSocket + batch endpoints |
| ☁ **Cloud** | One-command deploy to AWS/Azure/GCP |
| 📱 **Edge** | ONNX export + Jetson TensorRT conversion |
| 📈 **Time Series** | NDVI/NDWI/EVI trends, anomaly detection, seasonal analysis |
| ☁ **Foundation** | DINOv3 (sat/web), Prithvi-EO-2.0 (multi-temporal) |

## Independence from GeoAI

PyGeoVision v2.0 is **completely independent** from geoai-py. All model implementations are self-contained in pure PyTorch + HuggingFace Transformers + timm.

```python
# Works with or without geoai-py installed
import pygeovision as pgv
client = pgv.PyGeoVision()
print(client)
# PyGeoVision(v2.0 | pygeofetch=✗ | geoai=independent | datasets=503 | ...)
```

## Architecture

```
PyGeoVision v2.0
├── data/           PyGeoFetch satellite data layer (22 providers)
├── models/         50+ architectures (independent of GeoAI)
├── labeling/       7 auto-labeling sources
├── losses/         10 geospatial loss functions
├── inference/      Tiled/batch/streaming/ensemble inference
├── explainability/ GradCAM, SHAP, uncertainty, attention
├── monitoring/     Drift detection, performance tracking, alerts
├── training/       GeoTrainer, distributed, mixed precision
├── serving/        FastAPI REST API, auth, WebSocket
├── pipelines/      YAML orchestration, scheduling
├── datasets/       503-entry benchmark registry
├── edge/           ONNX Runtime, Jetson TensorRT
├── cloud/          AWS/Azure/GCP deployment
└── advanced/       FewShot, MultiTask, AutoML, VLM, TimeSeries, 3D
```
