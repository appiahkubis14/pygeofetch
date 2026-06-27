# Forestry

Deforestation monitoring, canopy height estimation, species mapping, and biomass assessment from satellite and LiDAR data.

---

## Deforestation Detection

```python
import pygeovision as pgv
from pygeovision.models.foundation.dinov3 import CHMv2Model
from pygeovision.models.change_detection.changeformer import ChangeFormer

client = pgv.PyGeoVision()
bbox   = [-62.0, -10.0, -61.0, -9.0]   # Amazon, Brazil

def get_scene(year):
    r = client.search(bbox=bbox, date_range=[f"{year}-07-01",f"{year}-09-30"],
                      providers=["planetary_computer"], cloud_cover_max=10)
    return client.download(r[:1], f"./data/amazon/{year}/")[0].path

before = get_scene(2020)
after  = get_scene(2024)

# ChangeFormer bi-temporal detection
cd = ChangeFormer(num_classes=2, in_channels=4, device="cuda")
change_result = cd.detect(before, after, "./output/amazon_change.tif")
print(f"Changed area: {change_result['change_pct']:.2f}%")

# DINOv3 CHMv2 deforestation (height-based)
chm = CHMv2Model(device="cuda")
defor = chm.detect_deforestation(before, after, min_height_loss_m=3.0,
                                   output_path="./output/deforestation.tif")
print(f"Deforested: {defor['area_ha']:.0f} ha ({defor['deforested_pct']:.2f}%)")
```

---

## Global Canopy Height Mapping

```python
from pygeovision.models.foundation.dinov3 import CHMv2Model

chm = CHMv2Model(device="cuda")

result = chm.predict_canopy_height(
    "sentinel2_boreal.tif",
    output_path="./output/boreal_canopy_height.tif",
)

print(f"Mean canopy height: {result['statistics']['mean_m']:.1f}m")
print(f"Max canopy height:  {result['statistics']['max_m']:.1f}m")
print(f"95th percentile:    {result['statistics']['p95_m']:.1f}m")
print(f"Tree cover (>2m):   {result['statistics']['coverage_pct']:.1f}%")

# Biomass estimation from canopy height
bio = chm.estimate_biomass("sentinel2_boreal.tif")
print(f"Mean biomass: {bio['statistics']['mean_t_ha']:.0f} t/ha (dry matter)")
```

---

## Forest Type Classification

```python
from pygeovision.models import get_model
from pygeovision.inference.tiled import TiledInference

# 8-class forest type model (coniferous, broadleaf, mixed, regenerating, ...)
model = get_model("segformer-b2", num_classes=8, in_channels=6)

inf = TiledInference(model=model, chip_size=512, overlap=64)
result = inf.infer("sentinel2_forest.tif", "./output/forest_types.tif")
```

---

## Time Series Phenology

```python
from pygeovision.advanced.timeseries import GeoTimeSeries

ts = GeoTimeSeries(sensor="sentinel2")
monthly = ["jan.tif", "mar.tif", "may.tif", "jul.tif", "sep.tif", "nov.tif"]

# EVI for forest canopy
evi_series = ts.compute_index_series(monthly, index="evi")
seasonal   = ts.decompose(evi_series, period=6)

print(f"Green-up peak:   {seasonal['peak_month']}")
print(f"Dormancy trough: {seasonal['trough_month']}")
print(f"Amplitude:       {seasonal['seasonal_amplitude']:.3f}")
```

---

## LiDAR Individual Tree Segmentation

```python
from pygeovision.advanced.pointcloud import LiDARProcessor

proc  = LiDARProcessor("forest_scan.las", crs="EPSG:25832")
chm   = proc.canopy_height_model(resolution_m=1.0, output="chm.tif")
trees = proc.segment_trees(chm_path="chm.tif", min_height_m=5.0,
                             output_path="trees.geojson")

print(f"Trees detected:   {trees['n_trees']}")
print(f"Mean height:      {trees['mean_height_m']:.1f}m")
print(f"Stem density:     {trees['stems_per_ha']:.0f} stems/ha")
print(f"Basal area:       {trees['basal_area_m2_ha']:.1f} m²/ha")
```

---

## Key Datasets

| Dataset | Task | Resolution |
|---------|------|-----------|
| BioMassters | Biomass estimation | 10m |
| CanopyHeight (GEDI+S2) | Height regression | 1m |
| DeforestNet | Deforestation detection | 10m |
| TropicalForest | Forest/non-forest | 0.3m |
| DALES | 3D LiDAR segmentation | 0.5m |
