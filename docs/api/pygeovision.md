# PyGeoVision Client

The main entry point for all PyGeoVision functionality.

---

## Class: `PyGeoVision`

```python
import pygeovision as pgv

client = pgv.PyGeoVision(
    cache_dir="~/.cache/pygeovision",   # Weight and data cache
    log_level="INFO",                   # Logging verbosity
)
```

---

## Authentication

### `add_credentials(provider, **kwargs)`

Register API credentials for a satellite data provider.

```python
# Planetary Computer
client.add_credentials("planetary_computer", api_key="YOUR_KEY")

# AWS Open Data
client.add_credentials("aws_earth",
    access_key="AKIAIOSFODNN7EXAMPLE",
    secret_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    region="us-west-2"
)

# Copernicus Open Access Hub
client.add_credentials("copernicus",
    username="your_username",
    password="your_password"
)

# Maxar SecureWatch
client.add_credentials("maxar", api_key="YOUR_KEY")

# Planet
client.add_credentials("planet", api_key="YOUR_PLANET_KEY")
```

---

## Data Search

### `search(bbox, date_range, providers, **kwargs)`

Search satellite imagery with STAC and CQL2 filters.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `bbox` | `list[float]` | `[min_lon, min_lat, max_lon, max_lat]` in WGS84 |
| `date_range` | `list[str]` | `["YYYY-MM-DD", "YYYY-MM-DD"]` |
| `providers` | `list[str]` | Provider names (e.g. `["planetary_computer"]`) |
| `cloud_cover_max` | `float` | Maximum cloud cover % (default `20`) |
| `collections` | `list[str]` | STAC collection IDs (e.g. `["sentinel-2-l2a"]`) |
| `cql2_filter` | `str` | Advanced CQL2 filter expression |
| `sort_by` | `str` | Sort field: `"cloud_cover"`, `"date"` |
| `limit` | `int` | Maximum results per provider (default `50`) |

**Returns:** `list[SceneResult]`

```python
results = client.search(
    bbox=[-74.1, 40.6, -73.7, 40.9],
    date_range=["2024-06-01", "2024-08-31"],
    providers=["planetary_computer"],
    cloud_cover_max=10,
    sort_by="cloud_cover",
)

for r in results:
    print(f"{r.date}  cloud={r.cloud_cover:.0f}%  platform={r.platform}")
```

---

## Data Download

### `download(results, output_dir, **kwargs)`

Download and post-process satellite scenes.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `results` | `list[SceneResult]` | Results from `search()` |
| `output_dir` | `str` | Output directory |
| `post_process` | `list[str]` | Processing steps (see below) |
| `parallel` | `int` | Concurrent downloads (default `4`) |
| `bands` | `list[str]` | Specific bands to download |

**Post-processing options:**

- `reproject:EPSG:XXXX` — Reproject to a target CRS
- `cog` — Convert to Cloud-Optimised GeoTIFF
- `cloud_mask` — Apply cloud mask using QA band
- `clip` — Clip to the search bbox

```python
downloads = client.download(
    results[:3],
    output_dir="./data/nyc/",
    post_process=["reproject:EPSG:32618", "cog", "cloud_mask"],
    parallel=4,
)

for d in downloads:
    print(f"Downloaded: {d.path}  size={d.size_mb:.1f}MB")
```

---

## Auto-Labeling

Access via `client.labeling`:

```python
# OpenStreetMap
labels = client.labeling.osm(bbox, categories=["buildings","roads","water"])

# Microsoft Building Footprints
labels = client.labeling.microsoft_buildings(bbox)

# Google Open Buildings
labels = client.labeling.google_buildings(bbox)

# ESA WorldCover 2021 (11 classes, 10m)
labels = client.labeling.esa_worldcover(bbox)

# Dynamic World real-time (9 classes)
labels = client.labeling.dynamic_world(bbox, date_range=["2024-01-01","2024-12-31"])

# SAM automatic segmentation
labels = client.labeling.sam_auto("scene.tif", points_per_side=32)

# Foundation model (DINOv2 + k-means)
labels = client.labeling.foundation("scene.tif", n_clusters=8)

# Multi-source fusion (majority vote)
fused = client.labeling.pipeline(bbox,
    sources=["osm","esa_worldcover","microsoft_buildings"])

# Label quality assessment
report = client.labeling.quality("labels.tif")
print(f"Grade: {report['quality_grade']}  Score: {report['quality_score']:.0%}")
```

---

## Inference

Access via `client.inference`:

```python
from pygeovision.models import get_model

model = get_model("segformer-b2", num_classes=7)

# Tiled inference (Gaussian blend)
inf    = client.inference.tiled(model, chip_size=512, overlap=64)
result = inf.infer("scene.tif", "prediction.tif")

# Batch directory processing
batch  = client.inference.batch(model, n_workers=4)
result = batch.run_directory("./data/", "./predictions/")

# Ensemble (multiple models)
ens    = client.inference.ensemble([model1, model2], weights=[0.6, 0.4])
result = ens.infer("scene.tif", "ensemble_pred.tif")
```

---

## Monitoring

```python
# Drift detection
drift   = client.monitoring.drift_detector()
drift.fit(reference_images)
report  = drift.check(new_images)
print(f"Drift level: {report['data_drift']['drift_level']}")

# Performance tracking
tracker = client.monitoring.performance_tracker()
tracker.log(epoch=5, metrics={"val_iou": 0.84, "val_f1": 0.88})
trend   = tracker.trend("val_iou")
print(f"Trend: {trend['direction']}")
```

---

## GeoAI Engine

Access foundation models and advanced AI via `client.geoai`:

```python
# DINOv3
client.geoai.dinov3.load("dinov3_vitl16_sat")
features = client.geoai.dinov3.extract_features("sentinel2.tif")
height   = client.geoai.dinov3.canopy_height("forest.tif")
mask     = client.geoai.dinov3.zero_shot("image.tif", "solar panels")

# Prithvi
client.geoai.prithvi.load("prithvi_eo_2_0")
lc     = client.geoai.prithvi.land_cover("hls.tif")
change = client.geoai.prithvi.change_detection("2021.tif", "2024.tif")

# All foundation models
models = client.geoai.foundation_models.list()
```

---

## Pipelines

```python
from pygeovision.pipelines import Pipeline

# Load a YAML pipeline
p      = Pipeline.from_yaml("agriculture.yaml")
result = p.run(context={"client": client})

# Or build programmatically
from pygeovision.pipelines.steps import SearchStep, DownloadStep, InferStep

p = Pipeline("my_workflow")
p.add(SearchStep("search", params={"bbox": bbox, "providers": ["planetary_computer"]}))
p.add(DownloadStep("download", depends_on=["search"], params={"output_dir": "./data/"}))
p.add(InferStep("infer", depends_on=["download"], params={"model": "segformer-b2"}))
result = p.run(context={"client": client})
```
