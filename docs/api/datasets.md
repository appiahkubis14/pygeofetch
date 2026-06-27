# Dataset Registry

503-entry benchmark database spanning 14 geospatial AI research domains — the most comprehensive open registry of remote sensing datasets.

---

## Overview

```python
from pygeovision.datasets.registry import dataset_registry

# Total count
print(len(dataset_registry))   # 503

# Summary by domain
summary = dataset_registry.summary()
print(summary)
```

---

## `DatasetInfo` Schema

Every dataset entry contains 9 standardised attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | str | Dataset identifier |
| `domain` | str | Research domain |
| `year` | int | Publication year |
| `n_samples` | int | Total sample count |
| `sample_size` | str | Image dimensions (e.g. "512×512") |
| `n_classes` | int | Number of semantic classes |
| `modality` | str | Sensor type |
| `resolution_m` | float | Ground sample distance in metres |
| `volume_gb` | float | Approximate download size |
| `tasks` | list[str] | Applicable ML tasks |
| `description` | str | Human-readable summary |
| `download_url` | str | Direct download link |
| `paper_url` | str | Paper or project page |

---

## Search

```python
# Keyword search across name, description, and domain
results = dataset_registry.search("flood")
for d in results:
    print(f"{d.name:<30} {d.domain:<15} {d.year}")
```

---

## Filter

```python
# Filter by modality
sar_datasets = dataset_registry.filter(modality="sar")
s2_datasets  = dataset_registry.filter(modality="multispectral")

# Filter by resolution range
high_res = dataset_registry.filter(max_resolution_m=1.0)
medium   = dataset_registry.filter(min_resolution_m=5.0, max_resolution_m=30.0)

# Filter by task
seg_datasets = dataset_registry.filter(task="segmentation")
ts_datasets  = dataset_registry.filter(task="time_series")

# Filter by minimum sample count
large = dataset_registry.filter(min_samples=50000)

# Filter by domain
agri = dataset_registry.filter(domain="agriculture")

# Combine filters
high_res_seg = dataset_registry.filter(
    task="segmentation",
    max_resolution_m=0.5,
    min_samples=1000,
)
print(f"Found: {len(high_res_seg)} datasets")
```

---

## Domain Coverage

| Domain | Count | Example Datasets |
|--------|-------|-----------------|
| `urban` | 90+ | LEVIR-CD, iSAID, DOTA-v2, SpaceNet series |
| `agriculture` | 35+ | CropHarvest, TimeSen2Crop, BreizhCrops |
| `forestry` | 20+ | BioMassters, DeforestNet, CanopyHeight |
| `water` | 20+ | SEN1Floods11, WorldFloods, CoastalSeg |
| `disaster` | 20+ | xBD, FloodNet, RescueNet |
| `climate` | 20+ | DroughtWatch, GlacierNet, SolarPV |
| `sar` | 20+ | SARship, HRSID, OpenSARUrban |
| `foundation` | 20+ | Major-TOM, Clay-Pretrain, OmniEarth |
| `land_cover` | 20+ | GlobLand30, OpenEarthMap, FLAIR1/2 |
| `change_detection`| 15+ | WHU-CD, SYSU-CD, S2Looking |
| `3d` | 10+ | SensatUrban, DublinCity, DALES |
| `vlm` | 10+ | RSVQA, RS-Instructions, GeoChat |
| `time_series` | 10+ | EarthNet2021, Planet-UDM2 |
| `multimodal` | 10+ | DFC2021, FusionNet, MMSegRS |

---

## Top Datasets per Task

```python
# Best datasets for a given task (ranked by sample count and quality)
top5 = dataset_registry.top_for_task("segmentation", n=5)
for d in top5:
    print(f"{d.name:<30} {d.n_samples:>10,} samples  {d.resolution_m}m")
```

| Task | Top Dataset | Samples |
|------|-------------|---------|
| Segmentation | OpenEarthMap | 5,000 (1024px tiles) |
| Detection | DOTA-v2 | 195,000 instances |
| Change detection | SYSU-CD | 20,000 pairs |
| Classification | Sen2LCZ | 352,366 |
| Time series | SITS-Brazil | 1,000,000 |
| Foundation (pretrain) | OmniEarth | 10,000,000 |
| VLM | RS-Instructions | 100,000 |
| 3D / LiDAR | SensatUrban | 3,000,000,000 points |

---

## Similar Datasets

```python
# Find datasets similar to a given one
similar = dataset_registry.similar_to("EuroSAT", n=10)
for d in similar:
    print(f"{d.name}")
```

---

## Dataset Analysis

```python
from pygeovision.datasets.analysis import DatasetAnalyzer

analyzer = DatasetAnalyzer(dataset_registry)

# Domain distribution chart
analyzer.plot_domain_distribution(output="./docs/domain_dist.png")

# Resolution histogram
analyzer.plot_resolution_histogram(output="./docs/resolution_hist.png")

# Year timeline (growth of the field)
analyzer.plot_year_timeline(output="./docs/year_timeline.png")

# Modality breakdown
breakdown = analyzer.modality_breakdown()
print(breakdown)
# {'multispectral': 245, 'rgb': 130, 'sar': 55, ...}
```

---

## Benchmark Selection

```python
from pygeovision.datasets.benchmark import BenchmarkSelector

selector = BenchmarkSelector(dataset_registry)

# Select the recommended benchmark suite for a research task
suite = selector.recommend(
    task="segmentation",
    modality="multispectral",
    n_datasets=5,
)

for d in suite:
    print(f"{d.name:<30} {d.resolution_m}m  {d.n_samples:,} samples")
```

---

## Modalities

| Modality | Description |
|----------|-------------|
| `rgb` | 3-band visible (Red, Green, Blue) |
| `multispectral` | 4–13 bands including NIR, SWIR |
| `hyperspectral` | 100+ bands |
| `sar` | Synthetic Aperture Radar (Sentinel-1, ALOS-2) |
| `sar_optical` | SAR + optical fusion |
| `lidar` | Airborne or terrestrial LiDAR |
| `thermal` | Thermal infrared |
| `multimodal` | Multiple sensor types |
| `passive_microwave` | Passive microwave (SMOS, AMSR2) |
