# Point Cloud (3D)

LiDAR processing, Canopy Height Model generation, and 3D segmentation from airborne and terrestrial point clouds.

---

## `LiDARProcessor`

```python
from pygeovision.advanced.pointcloud import LiDARProcessor

proc = LiDARProcessor(
    input_path="scan.las",    # LAS, LAZ, PLY, PCD, NPZ
    crs="EPSG:25833",
    device="cuda",
)
```

---

## Point Cloud I/O

```python
# Load (auto-detects format)
pc = proc.load()
print(f"Points:  {pc['n_points']:,}")
print(f"Bounds:  {pc['bounds']}")
print(f"CRS:     {pc['crs']}")

# Filter by classification
ground    = proc.filter(classification=2)   # LAS classification 2 = ground
buildings = proc.filter(classification=6)   # 6 = building
trees     = proc.filter(classification=5)   # 5 = high vegetation

# Decimate (reduce density)
sparse = proc.decimate(target_density=10)  # points per m²

# Export
proc.export("output.las")
proc.export("output.ply")
proc.export("output.csv")
```

---

## Canopy Height Model (CHM)

```python
# Rasterise to Digital Surface Model and Terrain Model
dsm  = proc.rasterise(mode="max",   resolution_m=1.0, output="dsm.tif")
dtm  = proc.rasterise(mode="min",   resolution_m=1.0, output="dtm.tif")
chm  = proc.canopy_height_model(    resolution_m=1.0, output="chm.tif")
# CHM = DSM - DTM  (vegetation height above ground)

print(f"Mean canopy height: {chm['mean_m']:.1f}m")
print(f"Max canopy height:  {chm['max_m']:.1f}m")

# Individual tree segmentation
trees = proc.segment_trees(
    chm_path="chm.tif",
    min_height_m=2.0,
    output_path="trees.geojson",
)
print(f"Trees detected: {trees['n_trees']}")
print(f"Mean height:    {trees['mean_height_m']:.1f}m")
print(f"Basal area:     {trees['basal_area_m2_ha']:.1f} m²/ha")
```

---

## 3D Segmentation

Classify each point using deep learning (PointNet++, RandLA-Net, KPConv).

```python
from pygeovision.models import get_model

model = get_model("pointnet2-msg", num_classes=13)  # SemanticKITTI classes

result = proc.segment(
    model=model,
    classes=["ground","vegetation","building","vehicle","other"],
    batch_size=65536,   # Points per batch
    output_path="segmented.las",
)

for cls, pct in result['class_percentages'].items():
    print(f"  {cls:<15} {pct:.1f}%")
```

---

## Density Analysis

```python
density_map = proc.density_map(
    resolution_m=5.0,
    output="density.tif",
)

stats = proc.statistics()
print(f"Point density: {stats['mean_density_per_m2']:.1f} pts/m²")
print(f"Coverage:      {stats['coverage_pct']:.1f}%")
```
