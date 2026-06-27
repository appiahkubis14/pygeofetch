# Climate & Carbon

Carbon stock estimation, sea ice monitoring, wildfire mapping, and climate-related land-cover dynamics.

---

## Forest Carbon Estimation

```python
import pygeovision as pgv
from pygeovision.models.foundation.dinov3 import CHMv2Model
from pygeovision.advanced.timeseries import GeoTimeSeries

client = pgv.PyGeoVision()

# Step 1: Canopy height from DINOv3 CHMv2
chm = CHMv2Model(device="cuda")
bio = chm.estimate_biomass("sentinel2_forest.tif")
print(f"Above-ground biomass: {bio['statistics']['mean_t_ha']:.0f} t DM/ha")
print(f"Carbon stock (×0.5):  {bio['statistics']['mean_t_ha']*0.5:.0f} tC/ha")

# Step 2: Long-term NDVI trend (vegetation health)
ts = GeoTimeSeries(sensor="sentinel2")
series = ts.compute_index_series(["2019.tif","2020.tif","2021.tif","2022.tif","2023.tif","2024.tif"],
                                  index="ndvi")
trend = ts.compute_trend(series)
print(f"NDVI trend: {trend['direction']} (slope={trend['slope']:.5f}/year)")
```

---

## Wildfire Burned Area Mapping

```python
from pygeovision.models import get_model
from pygeovision.inference.tiled import TiledInference

# BAI (Burn Area Index) time series
ts = GeoTimeSeries(sensor="sentinel2")
bai_series = ts.compute_index_series(["pre_fire.tif","post_fire.tif"], index="bai")

# Semantic segmentation of burned area
burn_model = get_model("segformer-b2", num_classes=3, in_channels=6)  # unburned, burned, recovering
inf = TiledInference(burn_model, chip_size=512)
result = inf.infer("post_fire.tif", "./output/burned_area.tif")
```

---

## Sea Ice Extent Monitoring

```python
from pygeovision.models import get_model
from pygeovision.inference.tiled import TiledInference

# SAR-based sea ice classification
ice_model = get_model("segformer-b2", num_classes=5, in_channels=2)  # SAR dual-pol
inf = TiledInference(ice_model, chip_size=512)
result = inf.infer("sentinel1_arctic.tif", "./output/sea_ice.tif")
```

---

## Solar Farm Detection

```python
from pygeovision.models.detection.yolo import GeoYOLO

detector = GeoYOLO("yolov8-m", num_classes=1, class_names=["solar_farm"])
result   = detector.detect("vhr_scene.tif", conf=0.4)
print(f"Solar farms: {result['n_detections']}")
```

---

## Permafrost Thaw Lake Detection

```python
# Thaw lakes appear as dark features in Sentinel-2 SWIR band
thaw_model = get_model("unet-r50", num_classes=2, in_channels=4)
inf        = TiledInference(thaw_model, chip_size=512)
result     = inf.infer("arctic_sentinel2.tif", "./output/thaw_lakes.tif")
```

---

## Key Datasets

| Dataset | Task | Notes |
|---------|------|-------|
| BioMassters | Biomass regression | Sentinel-1+2 |
| GFED | Fire emissions | Global 500m |
| GlacierNet | Glacier extent | Landsat time series |
| DroughtWatch | Drought classification | 4 classes |
| SolarPV | Solar panel detection | 15K samples |
| IceNet | Sea ice prediction | Arctic + Antarctic |
