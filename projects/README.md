# PyGeoVision v2.0 — Production Notebook Collection
## 25 Complete Real-World Geospatial AI Workflows

---

## Quick Start

```bash
pip install "pygeovision[geo,train,foundation]"
jupyter notebook
```

## Notebook Index

| # | Notebook | Domain | Real-World Problem Solved |
|---|----------|--------|--------------------------|
| 01 | Satellite Data Acquisition | Data | Auto-download Sentinel-2 for any study area from 22 providers |
| 02 | Building Footprint Extraction | Urban | Extract city-scale building footprints without manual labeling |
| 03 | Land Cover with Prithvi-EO-2.0 | Land Cover | 9-class mapping with 600M foundation model |
| 04 | Change Detection | Disaster | Identify damaged areas after hurricanes/earthquakes |
| 05 | Agricultural Crop Monitoring | Agriculture | Assess crop health for insurance companies |
| 06 | Forest Monitoring | Forestry | Detect illegal deforestation in real time |
| 07 | Water & Flood Mapping | Disaster | Rapid flood extent within hours of event |
| 08 | Solar Panel Detection | Energy | Inventory solar installations + energy potential |
| 09 | Disaster Damage Assessment | Emergency | 4-class building damage for rescue prioritization |
| 10 | Urban Growth Analysis | Urban | Decade-long urban expansion for city planning |
| 11 | Road Network Extraction | Infrastructure | Auto-extract road network for mapping agencies |
| 12 | Crop Type Mapping | Agriculture | 10-class crop mapping with time series |
| 13 | Glacier Monitoring | Climate | Track glacier retreat + RCP projections |
| 14 | Oil Spill Detection (SAR) | Environment | Night/cloud-proof SAR oil spill detection |
| 15 | Air Quality (NO2, PM2.5) | Environment | Urban air quality from Sentinel-5P |
| 16 | Wildfire Detection | Disaster | Burned area mapping with BAI + dNBR |
| 17 | Biodiversity Mapping | Ecology | Unsupervised habitat classification with DINOv3 |
| 18 | Infrastructure Monitoring | Civil Eng | Construction progress from satellite time series |
| 19 | Coastal & Wetland | Environment | Wetland loss detection and trend analysis |
| 20 | Climate Change (LST, NDVI) | Climate | Urban heat island quantification |
| 21 | Custom Model Training | ML/AI | Train custom models without manual annotation |
| 22 | Pipeline Deployment | MLOps | Production YAML pipelines + cloud deployment |
| 23 | Foundation Model Fine-Tuning | Deep Learning | DINOv3 + Prithvi fine-tuning guide |
| 24 | DINOv3 Embeddings | AI/Retrieval | Semantic image search from 10,000-scene archives |
| 25 | Vision-Language (CLIP+Moondream) | AI/NLP | Natural language satellite image querying |

---

## PyGeoVision Platform

| Metric | Value |
|--------|-------|
| Model architectures | 119 |
| Datasets registered | 503 |
| Tests passing | 580 |
| Satellite providers | 22 |
| Foundation models | DINOv3 (12 variants) + Prithvi-EO-2.0 (600M) |
| Auto-labeling sources | 7 (OSM, MS, Google, ESA, DynamicWorld, SAM, DINOv2) |

## Installation by Group

```bash
# Notebooks 01-07 (data + segmentation)
pip install "pygeovision[geo,train]"

# Notebooks 08-16 (change detection + environment)
pip install "pygeovision[geo,train,monitoring]"

# Notebooks 17-20 (time series + ecology)
pip install "pygeovision[geo,train,timeseries]"

# Notebooks 21-25 (AI + foundation models)
pip install "pygeovision[geo,train,foundation,vlm]"

# All notebooks
pip install "pygeovision[all]"
```

---

*PyGeoVision v2.0 — Built for the Salzburg International Geospatial AI Symposium*
