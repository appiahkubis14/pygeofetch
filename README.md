# PyGeoFetch 🛰️

**Universal satellite data pipeline + geospatial processing platform** — unified access to 22+ satellite providers, 11 preprocessing operations, 17 spectral indices, full vector/raster post-processing, SAR analysis, and chainable YAML pipelines. One CLI, one Python API.

[![PyPI version](https://badge.fury.io/py/pygeofetch.svg)](https://pypi.org/project/pygeofetch/)
[![Python Versions](https://img.shields.io/pypi/pyversions/pygeofetch.svg)](https://pypi.org/project/pygeofetch/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/appiahkubis14/PyGeoFetch/actions/workflows/tests.yml/badge.svg)](https://github.com/appiahkubis14/PyGeoFetch/actions)
[![Coverage](https://codecov.io/gh/appiahkubis14/PyGeoFetch/branch/main/graph/badge.svg)](https://codecov.io/gh/appiahkubis14/PyGeoFetch)

---

## What is PyGeoFetch?

PyGeoFetch is the complete **data preparation layer** between raw satellite imagery and AI/ML analysis. It handles every step in the pipeline:

```
Raw Satellite Data (Sentinel-2, Landsat, SAR, VHR, DEM...)
        │
        ▼
  PyGeoFetch Acquire
        ├── Search 22+ providers with one query
        ├── Parallel downloads with resume + retry
        ├── Band selection (reduce 600 MB → 150 MB)
        └── STAC 1.0 compliant output
        │
        ▼
  PyGeoFetch Process
        ├── Preprocessing  (atmospheric correction, cloud mask, clip, reproject)
        ├── Spectral Indices  (NDVI, EVI, NDWI, TCT, PCA, LST, 17 total)
        ├── Post-processing  (vectorize, zonal stats, COG, compress)
        └── SAR Analysis  (despeckle, calibrate, flood map, coherence)
        │
        ▼
  PyGeoVision AI / Your Analysis
```

---

## Why PyGeoFetch?

| Feature | PyGeoFetch | EODAG | pystac-client | satpy | sentinelsat |
|---|---|---|---|---|---|
| **Providers** | **22+** | 10+ | STAC only | Limited | Sentinel only |
| **Processing Engine** | ✅ Full | ❌ | ❌ | Partial | ❌ |
| **Spectral Indices** | ✅ 17+ | ❌ | ❌ | ❌ | ❌ |
| **SAR Processing** | ✅ | ❌ | ❌ | ✅ | ❌ |
| **CLI** | ✅ Full | ❌ | ❌ | ❌ | ✅ Basic |
| **YAML Pipelines** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Auth Management** | ✅ Keyring | Partial | ❌ | ❌ | ✅ |
| **Parallel Downloads** | ✅ Adaptive | ✅ | ❌ | ❌ | ❌ |
| **STAC 1.0 Output** | ✅ Native | ❌ | ✅ | ❌ | ❌ |
| **COG Export** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Batch Processing** | ✅ Parallel | ❌ | ❌ | ❌ | ❌ |
| **Scheduler (cron)** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Commercial Providers** | ✅ Planet/Maxar | ❌ | ❌ | ❌ | ❌ |
| **9 Notebooks** | ✅ | ❌ | ❌ | ❌ | ❌ |

---

## Installation

```bash
# Core — free providers work immediately, no extras needed
pip install PyGeoFetch

# With raster processing (reproject, COG, spectral indices)
pip install "PyGeoFetch[geo]"

# With cloud provider extras
pip install "PyGeoFetch[cloud]"

# With cron scheduling
pip install "PyGeoFetch[schedule]"

# Everything
pip install "PyGeoFetch[all]"
```

**Requirements:** Python 3.9+

**Optional extras:**
| Extra | Packages | Enables |
|---|---|---|
| `geo` | rasterio, geopandas, pyarrow, shapely | Processing, COG, GeoParquet |
| `cloud` | boto3, pystac | S3 direct access, NASA cloud |
| `schedule` | croniter | Cron pipeline scheduling |
| `dev` | pytest, ruff, mypy, black | Development tools |

---

## Quick Start — 3 Minutes

### 1. Verify installation
```bash
PyGeoFetch doctor
# ✓ Python 3.11  ✓ httpx  ✓ pydantic  ✓ rich
# ✓ AWS Earth Search: HTTP 200
# ✓ Planetary Computer: HTTP 200
# ✓ Element 84: HTTP 200
```

### 2. Search (no login needed)
```bash
PyGeoFetch search run \
  --bbox "-74.1,40.6,-73.7,40.9" \
  --start-date 2024-01-01 \
  --end-date 2024-06-01 \
  --cloud-cover 0-10 \
  --providers aws_earth,planetary_computer \
  --format table \
  --output results.geojson
```

![search demo](docs/assets/search_demo.png)

### 3. Download
```bash
PyGeoFetch download run \
  --from-search results.geojson \
  --output ./data/ \
  --parallel 2 \
  --max-items 3 \
  --bands "B02,B03,B04"
```

![download demo](docs/assets/download_demo.png)

### 4. Process
```bash
# Compute NDVI
PyGeoFetch index ndvi --red B04.tif --nir B08.tif --output ndvi.tif

# Clip to study area
PyGeoFetch preprocess clip scene.tif --bbox "-74.1,40.6,-73.7,40.9"

# Export as Cloud Optimized GeoTIFF
PyGeoFetch post cog ndvi.tif --compress deflate
```

### 5. Python API
```python
from pygeofetch import PyGeoFetch
from pygeofetch.models.search_query import SearchQuery, BoundingBox
from pygeofetch.models.download_task import DownloadOptions
from pathlib import Path

client = PyGeoFetch()

# Search
results = client.search(
    SearchQuery(
        bbox=BoundingBox.from_string("-74.1,40.6,-73.7,40.9"),
        start_date="2024-01-01",
        end_date="2024-06-01",
        cloud_cover_max=10,
        sort_by="cloud_cover",
        sort_ascending=True,
    ),
    providers=["aws_earth", "planetary_computer"],
)
print(f"Found {len(results)} scenes")

# Download
dl = client.download(
    results[:3],
    destination=Path("./data/"),
    options=DownloadOptions(parallel=2, bands=["B02","B03","B04"]),
)

# Process
ndvi = client.indices.ndvi(red="B04.tif", nir="B08.tif")
cog  = client.post.cog(str(ndvi.output_path))

# End-to-end pipeline
result = (
    client.pipeline("my-workflow")
    .clip(bbox=(-74.1, 40.6, -73.7, 40.9))
    .ndvi(red="B04.tif", nir="B08.tif")
    .vectorize(threshold=0.3)
    .cog()
    .run(input="scene.tif", output_dir="./processed/")
)
```

---

## Supported Providers (22)

### Open Access — No Login Required (10)

| Provider ID | Satellites | Capabilities |
|---|---|---|
| `aws_earth` | Sentinel-2 COG, Landsat C2, NAIP | STAC |
| `planetary_computer` | Sentinel-1/2, Landsat 8/9, MODIS, NAIP, ALOS DEM | STAC, SAR |
| `element84` | Sentinel-2 L2A, Landsat C2, Sentinel-1 RTC, COP-DEM | STAC, SAR |
| `noaa_big_data` | GOES-16/17/18, NEXRAD radar | Weather |
| `esa_scihub` | Sentinel-1/2/3/5P (public mirrors) | SAR |
| `jaxa_earth` | ALOS 30m DSM, PALSAR-2 | SAR |
| `isro_bhuvan` | ResourceSat-2/2A (5.8m), Cartosat-1 (2.5m) | — |
| `inpe_cbers` | CBERS-4, CBERS-4A | — |
| `digitalglobe` | WorldView open disaster response | <1m VHR |
| `geoserver_generic` | Any OGC WMS/WFS/WCS endpoint | Generic |

### Authenticated Providers (12)

| Provider ID | Auth Type | Satellites | Cost |
|---|---|---|---|
| `usgs` | Username/Password | Landsat 1–9, ASTER, MODIS, SRTM | Free |
| `copernicus` | OAuth2 | Sentinel-1/2/3/5P full archive | Free |
| `nasa_earthdata` | OAuth2 | MODIS, VIIRS, ICESat-2, GEDI | Free |
| `nasa_earthdata_cloud` | OAuth2+S3 | NASA cloud datasets (direct S3) | Free |
| `alaska_satellite_facility` | Earthdata | Sentinel-1 C-SAR, ALOS PALSAR | Free |
| `opentopography` | API Key | SRTM, COP-DEM 30/90m, LiDAR | Free tier |
| `planet` | API Key | PlanetScope 3m, SkySat 0.5m | Subscription |
| `sentinel_hub` | OAuth2 Client | All Sentinels + Landsat, on-the-fly processing | Freemium |
| `maxar_gbdx` | API Token | WorldView 1/2/3/4 (30cm) | Subscription |
| `airbus_oneatlas` | API Key | Pléiades 1A/1B (50cm), SPOT 6/7 (1.5m) | Subscription |
| `google_earth_engine` | Service Account | Multi-petabyte GEE catalog | Free tier |
| `terrabotics` | API Key | Commercial archive + tasking | Subscription |

---

## Authentication

```bash
# Username / Password
PyGeoFetch auth add usgs --username USER --password PASS
PyGeoFetch auth add copernicus --username email@example.com --password PASS
PyGeoFetch auth add nasa_earthdata --username USER --password PASS

# API Key
PyGeoFetch auth add planet --api-key YOUR_KEY
PyGeoFetch auth add opentopography --api-key YOUR_KEY

# OAuth2 Client Credentials
PyGeoFetch auth add sentinel_hub --client-id ID --client-secret SECRET

# Interactive
PyGeoFetch auth login copernicus

# Manage
PyGeoFetch auth list
PyGeoFetch auth test usgs
PyGeoFetch auth remove planet --yes
```

**Environment variables** (for CI/CD):
```bash
export PYGEOFETCH_USGS_USERNAME=user
export PYGEOFETCH_USGS_PASSWORD=pass
export PYGEOFETCH_PLANET_API_KEY=PL-abc123
export PYGEOFETCH_COPERNICUS_USERNAME=email@example.com
export PYGEOFETCH_COPERNICUS_PASSWORD=pass
export PYGEOFETCH_SENTINEL_HUB_CLIENT_ID=id
export PYGEOFETCH_SENTINEL_HUB_CLIENT_SECRET=secret
export PYGEOFETCH_NASA_EARTHDATA_USERNAME=user
export PYGEOFETCH_OPENTOPOGRAPHY_API_KEY=key
```

Credentials are stored in your **system keyring** (macOS Keychain, Windows Credential Manager, Linux Secret Service) — never in plain-text files.

---

## Search

```bash
# Full-featured search
PyGeoFetch search run \
  --bbox "-74.1,40.6,-73.7,40.9" \
  --start-date 2024-01-01 --end-date 2024-06-01 \
  --cloud-cover 0-10 \
  --providers aws_earth,planetary_computer,copernicus \
  --satellites Sentinel-2 \
  --sort-by cloud_cover --sort-order asc \
  --max-results 50 \
  --format table \
  --output results.geojson

# CQL2 filter (Planetary Computer, Element84, AWS)
PyGeoFetch search run \
  --bbox "-74.1,40.6,-73.7,40.9" \
  --providers planetary_computer \
  --cql2 "eo:cloud_cover < 5 AND platform = 'sentinel-2b'"

# Geometry file AOI
PyGeoFetch search run \
  --geometry-file my_area.geojson \
  --cloud-cover 0-15 \
  --providers aws_earth
```

**Output formats:** `table` · `json` · `stac` · `geojson` · `geoparquet` · `csv` · `ids`

### Search flags

| Flag | Type | Description | Default |
|---|---|---|---|
| `--bbox` | string | `"minlon,minlat,maxlon,maxlat"` | — |
| `--geometry-file` | path | GeoJSON polygon AOI | — |
| `--start-date` | date | `YYYY-MM-DD` | — |
| `--end-date` | date | `YYYY-MM-DD` | today |
| `--cloud-cover` | range | `min-max` percent | `0-100` |
| `--providers` | list | comma-separated provider IDs | — |
| `--satellites` | list | filter by satellite name | — |
| `--sort-by` | choice | `datetime` `cloud_cover` `score` | `datetime` |
| `--sort-order` | choice | `asc` `desc` | `desc` |
| `--max-results` | int | per provider | `100` |
| `--cql2` | string | CQL2 filter expression | — |
| `--format` | choice | output format | `table` |
| `--output` | path | save to file | — |
| `--no-cache` | flag | bypass cache | — |
| `--timeout` | int | per-provider seconds | `60` |
| `--on-provider-failure` | choice | `skip` `abort` `retry` | `skip` |

---

## Download

```bash
# Basic
PyGeoFetch download run \
  --from-search results.geojson \
  --output ./data/ \
  --parallel 2 \
  --max-items 3

# RGB bands only (~150 MB vs 600 MB full Sentinel-2 scene)
PyGeoFetch download run \
  --from-search results.geojson \
  --output ./data/ \
  --bands "B02,B03,B04" \
  --max-items 5

# Full options
PyGeoFetch download run \
  --from-search results.geojson \
  --output ./data/ \
  --parallel 4 --retry 5 \
  --verify-checksum --resume \
  --bandwidth-limit 10MB \
  --on-failure skip \
  --notify webhook:https://hooks.slack.com/YOUR/WEBHOOK \
  --post-process "reproject:EPSG:4326,compress:lzw,cog"
```

### Sentinel-2 Band Reference

| Bands | Purpose | Resolution | ~Size/Scene |
|---|---|---|---|
| `B02,B03,B04` | RGB (Blue, Green, Red) | 10m | ~150 MB |
| `visual` | True colour TCI | 10m | ~200 MB |
| `B04,B08` | NDVI (Red + NIR) | 10m | ~100 MB |
| `B02,B03,B04,B08` | 4-band multispectral | 10m | ~200 MB |
| `B11,B12` | SWIR (fire, soil) | 20m | ~50 MB |
| `SCL` | Scene Classification (cloud mask) | 20m | ~20 MB |
| *(omit --bands)* | All data assets | 10/20/60m | ~600 MB |

### Post-processing chain

| Action | Syntax | Requires |
|---|---|---|
| `unzip` | `unzip` | — |
| `reproject` | `reproject:EPSG:4326` | rasterio |
| `compress` | `compress:lzw` · `compress:deflate` · `compress:zstd` | rasterio |
| `cog` | `cog` | rasterio |
| `clip` | `clip:file.geojson` | rasterio |
| `resample` | `resample:30` | rasterio |
| `ndvi` | `ndvi` | rasterio |
| `pan-sharpen` | `pan-sharpen` | rasterio |
| `merge` | `merge` | rasterio |

---

## Preprocessing (`client.preprocess`)

Complete preprocessing engine — all operations return a `ProcessingResult` with output path and metadata.

```bash
# Atmospheric Correction
PyGeoFetch preprocess atmos scene.tif --method dos1        # Dark Object Subtraction
PyGeoFetch preprocess atmos scene.tif --method sen2cor     # Sentinel-2 specific
PyGeoFetch preprocess atmos scene.tif --method flaash      # FLAASH (requires tool)

# Topographic Correction
PyGeoFetch preprocess topo-correct scene.tif dem.tif --method cosine
PyGeoFetch preprocess topo-correct scene.tif dem.tif --method c_correction

# Cloud Masking
PyGeoFetch preprocess cloud-mask scene.tif --method scl --scl-band SCL.tif
PyGeoFetch preprocess cloud-mask scene.tif --method fmask
PyGeoFetch preprocess cloud-fill cloudy.tif jan.tif mar.tif

# Geometric
PyGeoFetch preprocess clip    scene.tif --bbox "-74.1,40.6,-73.7,40.9"
PyGeoFetch preprocess clip    scene.tif --geometry study_area.geojson
PyGeoFetch preprocess reproject scene.tif --crs EPSG:32618 --resampling bilinear

# Resampling & Fusion
PyGeoFetch preprocess resample   scene.tif --resolution 30 --method bilinear
PyGeoFetch preprocess pansharpen pan_15m.tif ms_60m.tif --method brovey
PyGeoFetch preprocess tile       scene.tif --tile-size 512 --overlap 64

# Compositing
PyGeoFetch preprocess mosaic    s1.tif s2.tif s3.tif --method first
PyGeoFetch preprocess composite jan.tif feb.tif mar.tif --method median
```

**Python API:**
```python
result = client.preprocess.atmos("scene.tif", method="dos1")
result = client.preprocess.cloud_mask("scene.tif", method="scl", scl_band="SCL.tif")
result = client.preprocess.clip("scene.tif", bbox=(-74.1, 40.6, -73.7, 40.9))
result = client.preprocess.reproject("scene.tif", crs="EPSG:4326")
result = client.preprocess.resample("scene.tif", resolution=30)
result = client.preprocess.pansharpen(pan="pan.tif", ms="ms.tif", method="brovey")
result = client.preprocess.tile("scene.tif", tile_size=512, overlap=64)
result = client.preprocess.composite(["jan.tif","feb.tif","mar.tif"], method="median")
result = client.preprocess.mosaic(["s1.tif","s2.tif","s3.tif"])
result = client.preprocess.cloud_fill("cloudy.tif", time_series=["jan.tif","mar.tif"])
result = client.preprocess.topo_correct("scene.tif", dem="srtm.tif", method="c_correction")
```

---

## Spectral Indices (`client.indices`)

17+ spectral indices and transformations, all returning float32 GeoTIFF.

```bash
# Vegetation
PyGeoFetch index ndvi   --red B04.tif --nir B08.tif
PyGeoFetch index evi    --blue B02.tif --red B04.tif --nir B08.tif
PyGeoFetch index savi   --red B04.tif --nir B08.tif --soil-l 0.5

# Water
PyGeoFetch index ndwi   --green B03.tif --nir B08.tif
PyGeoFetch index mndwi  --green B03.tif --swir1 B11.tif

# Urban / Built-up
PyGeoFetch index ndbi   --nir B08.tif --swir1 B11.tif

# Snow / Ice
PyGeoFetch index ndsi   --green B03.tif --swir1 B11.tif

# Moisture
PyGeoFetch index ndmi   --nir B08.tif --swir1 B11.tif

# Fire / Burn Severity
PyGeoFetch index nbr    --nir B08.tif --swir2 B12.tif
PyGeoFetch index dnbr   --pre-nir B08.tif --pre-swir2 B12.tif \
                         --post-nir B08_post.tif --post-swir2 B12_post.tif

# Transformations
PyGeoFetch index tct    --blue B02.tif --green B03.tif --red B04.tif \
                         --nir B08.tif --swir1 B11.tif --swir2 B12.tif
PyGeoFetch index pca    B02.tif B03.tif B04.tif B08.tif --components 3
PyGeoFetch index texture B08.tif --window 7 --features contrast,homogeneity,energy

# Thermal
PyGeoFetch index lst    B10.tif --emissivity 0.97 --sensor landsat8

# Reflectance
PyGeoFetch index albedo B02.tif B03.tif B04.tif B08.tif B11.tif B12.tif

# Utilities
PyGeoFetch index band-math B04.tif B08.tif --expr "(B[1]-B[0])/(B[1]+B[0]+1e-6)"
PyGeoFetch index stack  B02.tif B03.tif B04.tif
```

### Index Reference

| Index | Formula | Range | Use Case |
|---|---|---|---|
| NDVI | (NIR-Red)/(NIR+Red) | -1 to +1 | Vegetation health (>0.3 = veg) |
| EVI | G·(NIR-Red)/(NIR+C1·Red-C2·Blue+L) | -1 to +1 | Dense canopy |
| SAVI | (NIR-Red)/(NIR+Red+L)·(1+L) | -1 to +1 | Sparse vegetation / soil |
| NDWI | (Green-NIR)/(Green+NIR) | -1 to +1 | Water bodies (>0 = water) |
| MNDWI | (Green-SWIR1)/(Green+SWIR1) | -1 to +1 | Urban water separation |
| NDBI | (SWIR1-NIR)/(SWIR1+NIR) | -1 to +1 | Built-up areas (>0 = urban) |
| NDSI | (Green-SWIR1)/(Green+SWIR1) | -1 to +1 | Snow/ice (>0.4 = snow) |
| NDMI | (NIR-SWIR1)/(NIR+SWIR1) | -1 to +1 | Canopy moisture |
| NBR | (NIR-SWIR2)/(NIR+SWIR2) | -1 to +1 | Pre-fire baseline |
| dNBR | NBR_pre - NBR_post | varies | Burn severity (>0.66 high) |
| TCT | Matrix coefficients | varies | Brightness, Greenness, Wetness |
| PCA | Eigen decomposition | varies | Dimensionality reduction |
| Texture | GLCM | varies | Contrast, homogeneity, energy |
| LST | Thermal → Kelvin/Celsius | K / °C | Surface temperature |
| Albedo | Narrowband to broadband | 0 to 1 | Surface reflectance |

---

## Post-Processing (`client.post`)

```bash
# Vectorize raster to polygons
PyGeoFetch post vectorize     ndvi.tif --threshold 0.3 --format geojson
PyGeoFetch post vectorize     classification.tif --min-area 100 --format gpkg

# Clean up vectors
PyGeoFetch post smooth        polygons.geojson --tolerance 0.5
PyGeoFetch post regularize    buildings.geojson          # orthogonalize footprints

# Analysis
PyGeoFetch post zonal-stats   ndvi.tif parcels.geojson --output stats.csv
PyGeoFetch post buffer        roads.geojson --distance 15
PyGeoFetch post centroids     polygons.geojson
PyGeoFetch post geometry-metrics polygons.geojson        # area, perimeter, compactness

# Export
PyGeoFetch post compress      scene.tif --method lzw
PyGeoFetch post cog           scene.tif --compress deflate --blocksize 512
```

**Python API:**
```python
vecs  = client.post.vectorize("ndvi.tif", threshold=0.3)
clean = client.post.smooth(str(vecs.output_path), tolerance=0.5)
reg   = client.post.regularize("buildings.geojson")
stats = client.post.zonal_stats("ndvi.tif", "parcels.geojson")
buf   = client.post.buffer("roads.geojson", distance=15)
cog   = client.post.cog("scene.tif", compress="deflate")
```

---

## SAR Processing (`client.sar`)

```bash
# Speckle filtering
PyGeoFetch sar despeckle sentinel1.tif --filter lee --window 5
PyGeoFetch sar despeckle sentinel1.tif --filter enhanced_lee --window 7
PyGeoFetch sar despeckle sentinel1.tif --filter frost
PyGeoFetch sar despeckle sentinel1.tif --filter gamma

# Radiometric calibration
PyGeoFetch sar calibrate sentinel1_dn.tif --output-type sigma0 --db
PyGeoFetch sar calibrate sentinel1_dn.tif --output-type gamma0 --db

# Flood mapping
PyGeoFetch sar flood-map post_flood.tif --threshold -15.0
PyGeoFetch sar flood-map post_flood.tif --reference pre_flood.tif

# Interferometric coherence
PyGeoFetch sar coherence slc_20240101.tif slc_20240113.tif --window 7
```

**Python API:**
```python
despeckled = client.sar.despeckle("sentinel1.tif", filter="enhanced_lee")
calibrated = client.sar.calibrate("sentinel1.tif", output_type="sigma0", in_db=True)
flood      = client.sar.flood_map("post.tif", threshold=-15.0, reference="pre.tif")
coherence  = client.sar.coherence("slc1.tif", "slc2.tif", window=7)
```

---

## Pipelines

### Python API pipeline builder

```python
result = (
    client.pipeline("sentinel2-ndvi")
    .atmos(method="dos1")
    .cloud_mask(method="scl", scl_band="SCL.tif")
    .clip(bbox=(-74.1, 40.6, -73.7, 40.9))
    .reproject(crs="EPSG:4326")
    .ndvi(red="B04.tif", nir="B08.tif")
    .vectorize(threshold=0.3)
    .smooth(tolerance=0.5)
    .cog(compress="deflate")
    .run(input="scene.tif", output_dir="./processed/")
)

print(result.success, result.duration_seconds)
for step in result.steps:
    print(f"  {step['step']}: {step['status']} ({step['duration']:.2f}s)")
```

### YAML pipeline definition

```yaml
name: weekly-sentinel2-monitoring
schedule: "0 6 * * 1"   # Every Monday 06:00 UTC
description: Weekly NDVI monitoring — search, download, process, export

steps:
  - search:
      providers: [aws_earth, planetary_computer]
      bbox: "-74.1,40.6,-73.7,40.9"
      date_range: last_7_days
      cloud_cover: "0-10"
      max_results: 20

  - filter:
      expression: "data.cloud_cover < 5"

  - download:
      parallel: 4
      output: ./raw/
      bands: [B04, B08]    # NDVI bands only

  - ndvi:
      red: B04.tif
      nir: B08.tif

  - vectorize:
      threshold: 0.3
      format: geojson

  - cog:
      compress: deflate
```

```bash
# Validate without running
PyGeoFetch proc-pipeline validate weekly.yaml

# Run once
PyGeoFetch proc-pipeline run weekly.yaml --input scene.tif

# Schedule (saves to ~/.pygeofetch/)
PyGeoFetch pipeline schedule weekly.yaml --name "ndvi-weekly"

# Pipeline templates
PyGeoFetch proc-pipeline template ndvi
PyGeoFetch proc-pipeline template change_detection
PyGeoFetch proc-pipeline template flood_map
PyGeoFetch proc-pipeline template urban_mapping
PyGeoFetch proc-pipeline template sar_analysis
PyGeoFetch proc-pipeline template land_cover
```

### Batch processing

```python
# Process multiple scenes in parallel with the same chain
results = client.batch_process(
    inputs=["scene1.tif", "scene2.tif", "scene3.tif"],
    chain=[
        ("clip",     {"bbox": (-74.1, 40.6, -73.7, 40.9)}),
        ("reproject",{"crs": "EPSG:4326"}),
        ("ndvi",     {}),
        ("cog",      {"compress": "deflate"}),
    ],
    output_dir="./processed/",
    parallel=4,
)
succeeded = [r for r in results if r.success]
print(f"{len(succeeded)}/{len(results)} succeeded")
```

---

## Data Acquisition Pipeline (`search` + `download`)

### Scheduling pipeline runs

```bash
# Schedule with cron (requires: pip install croniter)
PyGeoFetch pipeline schedule weekly.yaml --name "ndvi-weekly"
PyGeoFetch pipeline list-scheduled
PyGeoFetch pipeline history --limit 10
PyGeoFetch pipeline unschedule ndvi-weekly
```

**Common cron expressions:**
| Expression | Meaning |
|---|---|
| `0 6 * * 1` | Every Monday 06:00 UTC |
| `0 6 * * *` | Every day 06:00 UTC |
| `0 */6 * * *` | Every 6 hours |
| `0 6 1 * *` | First of every month |
| `0 6 * * 1,4` | Monday and Thursday |

---

## Cache Management

```bash
PyGeoFetch cache stats                    # show usage
PyGeoFetch cache clear                    # clear expired
PyGeoFetch cache clear --dry-run          # preview
PyGeoFetch cache ttl set 7200             # 2 hour TTL
PyGeoFetch cache location                 # show path
PyGeoFetch cache prune --max-size 1GB     # enforce size limit
```

---

## Configuration

Config is layered: defaults → `~/.pygeofetch/config.yaml` → env vars → CLI flags.

```yaml
# ~/.pygeofetch/config.yaml
download:
  parallel: 4
  retry_attempts: 5
  verify_checksum: false
  resume: true
  bandwidth_limit_mbps: null    # unlimited
  on_failure: skip              # skip | abort | retry

cache:
  ttl_seconds: 3600
  max_size_gb: 10

search:
  max_results: 100
  timeout_seconds: 60
  on_provider_failure: skip

auth:
  storage_backend: file         # file | keyring

logging:
  level: INFO
  format: console               # console | json
```

```bash
PyGeoFetch config show
PyGeoFetch config get download.parallel
PyGeoFetch config set download.parallel 8
PyGeoFetch config path
```

---

## Security

- **Credentials** stored in system keyring or encrypted file (`~/.pygeofetch/credentials.enc`), never plain text
- **TLS 1.2+** enforced on all connections, no `verify=False` anywhere
- **No telemetry** — PyGeoFetch never phones home
- **Atomic downloads** — `.tmp` then rename, no partial files on disk
- **SHA256 checksum** verification available via `--verify-checksum`
- Log filters redact passwords, tokens, and API keys automatically

---

## CLI Command Reference

```
SYSTEM
  PyGeoFetch doctor                   # diagnose installation and connectivity
  PyGeoFetch status [--json]          # provider and cache overview
  PyGeoFetch version

AUTH
  PyGeoFetch auth add PROVIDER [--username U] [--password P] [--api-key K]
  PyGeoFetch auth login PROVIDER      # interactive prompts
  PyGeoFetch auth list [--json]
  PyGeoFetch auth test PROVIDER
  PyGeoFetch auth remove PROVIDER [--yes]
  PyGeoFetch auth export [--output FILE]

PROVIDERS
  PyGeoFetch providers list [--auth|--no-auth] [--capabilities sar] [--json]
  PyGeoFetch providers info PROVIDER
  PyGeoFetch providers search "TERM"

SEARCH
  PyGeoFetch search run [16 flags] --format table|json|stac|geojson|geoparquet|csv|ids

DOWNLOAD
  PyGeoFetch download run [14 flags] --bands "B02,B03,B04" --post-process "cog"

PREPROCESSING
  PyGeoFetch preprocess atmos         scene.tif --method dos1|sen2cor|flaash
  PyGeoFetch preprocess cloud-mask    scene.tif --method scl|fmask|threshold
  PyGeoFetch preprocess cloud-fill    cloudy.tif t1.tif t2.tif
  PyGeoFetch preprocess clip          scene.tif --bbox "..." | --geometry file.geojson
  PyGeoFetch preprocess reproject     scene.tif --crs EPSG:4326
  PyGeoFetch preprocess resample      scene.tif --resolution 30
  PyGeoFetch preprocess pansharpen    pan.tif ms.tif --method brovey
  PyGeoFetch preprocess mosaic        s1.tif s2.tif --method first|last|min|max
  PyGeoFetch preprocess composite     *.tif --method median|mean|max|best_pixel
  PyGeoFetch preprocess tile          scene.tif --tile-size 512 --overlap 64
  PyGeoFetch preprocess topo-correct  scene.tif dem.tif --method cosine

SPECTRAL INDICES
  PyGeoFetch index ndvi|evi|savi|ndwi|mndwi|ndbi|ndsi|ndmi|nbr|dnbr
  PyGeoFetch index tct|pca|texture|lst|albedo|band-math|stack

POST-PROCESSING
  PyGeoFetch post vectorize|smooth|regularize|zonal-stats|buffer|centroids
  PyGeoFetch post geometry-metrics|compress|cog

SAR PROCESSING
  PyGeoFetch sar despeckle|calibrate|flood-map|coherence

PROCESSING PIPELINES
  PyGeoFetch proc-pipeline template ndvi|change_detection|flood_map|urban_mapping|...
  PyGeoFetch proc-pipeline validate FILE
  PyGeoFetch proc-pipeline run FILE [--input scene.tif]

DATA PIPELINES (search+download)
  PyGeoFetch pipeline run|validate|schedule|list-scheduled|unschedule|history

CACHE
  PyGeoFetch cache stats|clear|ttl|location|prune

CONFIG
  PyGeoFetch config show|get|set|path|reset

SHELL COMPLETION
  PyGeoFetch --install-completion bash|zsh|fish
```

---

## Notebooks

| Notebook | Topics |
|---|---|
| `01_getting_started.ipynb` | Install, doctor, first search, first download, Python API |
| `02_authentication_and_providers.ipynb` | All 22 providers, credentials, capability filters |
| `03_advanced_search.ipynb` | Federated search, CQL2 filters, geometry files, 7 output formats |
| `04_download_and_postprocessing.ipynb` | Band selection, parallel downloads, resume, post-processing |
| `05_pipelines_and_scheduling.ipynb` | YAML pipelines, scheduling, Python builder API |
| `06_real_world_workflows.ipynb` | NDVI time series, change detection, multi-sensor fusion, mosaics |
| `07_copernicus_and_authenticated_providers.ipynb` | Copernicus, USGS, NASA, Planet, ASF, OpenTopography |
| `08_cli_complete_reference.ipynb` | Every CLI command with runnable examples |
| `09_processing_complete.ipynb` | Full processing engine: all preprocessing, indices, SAR, pipelines |

```bash
cd notebooks/
jupyter lab
```

---

## Project Structure

```
pygeofetch/
├── pygeofetch/              # Python package
│   ├── __init__.py
│   ├── core/                # Engine, authenticator, searcher, downloader, scheduler, cache
│   ├── cli/                 # 11 CLI command files
│   │   ├── main.py          # CLI entry point
│   │   ├── auth_commands.py
│   │   ├── search_commands.py
│   │   ├── download_commands.py
│   │   ├── preprocess_commands.py  ← NEW
│   │   ├── index_commands.py       ← NEW
│   │   ├── postprocess_commands.py ← NEW
│   │   ├── sar_commands.py         ← NEW
│   │   ├── pipeline_process_commands.py ← NEW
│   │   ├── config_commands.py
│   │   └── ...
│   ├── processing/          # Processing engine ← NEW
│   │   ├── base.py          # ProcessingResult, helpers
│   │   ├── preprocessor.py  # A-D,H: atmos, cloud, geometric, resample, mosaic
│   │   ├── indices.py       # E: 17 spectral indices + transformations
│   │   ├── postprocessor.py # G: vectorize, smooth, zonal stats, COG
│   │   ├── sar.py           # F: despeckle, calibrate, flood, coherence
│   │   ├── pipeline.py      # Chainable builder + YAML loader
│   │   └── batch.py         # Parallel batch processing
│   ├── models/              # Pydantic models
│   ├── providers/           # 22 provider implementations
│   ├── utils/               # Logging, retry, geo, file, validators
│   └── config/              # Settings, defaults.yaml
├── notebooks/               # 9 Jupyter notebooks
├── tests/                   # 60 tests (unit + integration)
├── docs/assets/             # Screenshots (add after running)
├── pyproject.toml
├── README.md
├── QUICKSTART.md
├── CONTRIBUTING.md
├── Dockerfile
├── Makefile
├── .gitignore
└── .github/workflows/tests.yml
```

---

## Development

```bash
# Clone and install dev dependencies
git clone https://github.com/appiahkubis14/PyGeoFetch
cd PyGeoFetch
pip install -e ".[all]"

# Run tests
pytest tests/unit/ -v           # 60 tests, ~0.5s, no network needed
pytest tests/unit/ --cov=pygeofetch --cov-report=html

# Lint and format
ruff check pygeofetch/
black pygeofetch/
mypy pygeofetch/

# Docker
docker build -t pygeofetch .
docker run pygeofetch PyGeoFetch doctor
```

---

## Roadmap

- [ ] **v0.2** — BlackSky, KOMPSAT providers · streaming COG partial reads · TUI interactive mode
- [ ] **v0.3** — Web dashboard (`PyGeoFetch dashboard`) · REST API mode (`PyGeoFetch serve`)
- [ ] **v1.0** — PyGeoFetch Cloud (hosted API) · Enterprise SSO · Team workspaces

[Vote on features →](https://github.com/appiahkubis14/PyGeoFetch/discussions)

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). PRs welcome for new providers, processing steps, and bug fixes.

```bash
# Add a new provider
cp pygeofetch/providers/base.py pygeofetch/providers/my_provider.py
# Implement AbstractBaseProvider
# Register in pygeofetch/providers/__init__.py
# Add tests in tests/unit/test_providers.py
```

---

## License

MIT License — see [LICENSE](LICENSE).

© 2025 PyGeoFetch Contributors. Built for the geospatial community.

---

*Part of the **PyGeoVision** platform — [PyGeoFetch](https://github.com/appiahkubis14/PyGeoFetch) (data) + [GeoAI](https://opengeoai.org) (AI) = complete Earth observation pipeline.*
