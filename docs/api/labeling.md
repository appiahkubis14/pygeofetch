# Auto-Labeling

7 auto-labeling sources covering the full spectrum from vector databases to foundation model inference — no manual annotation required.

---

## OSM Labeler

Generate raster labels from OpenStreetMap vector data.

```python
from pygeovision.labeling.osm import OSMLabeler

labeler = OSMLabeler()

result = labeler.label(
    bbox=[-74.1, 40.6, -73.7, 40.9],       # WGS84 bounding box
    categories=["buildings", "roads", "water", "parks", "railways"],
    output_path="./labels/osm.tif",
    resolution_m=10.0,                       # Output pixel size
    crs="EPSG:4326",
)

print(f"Features: {result['n_features']}")
print(f"Output:   {result['output_path']}")
```

**Available categories:**

| Category | OSM Tag |
|----------|---------|
| `buildings` | `building=*` |
| `roads` | `highway=*` |
| `water` | `natural=water`, `waterway=*` |
| `parks` | `leisure=park`, `landuse=grass` |
| `railways` | `railway=*` |
| `farmland` | `landuse=farmland` |
| `forest` | `natural=wood`, `landuse=forest` |
| `industrial` | `landuse=industrial` |

---

## Microsoft Building Footprints

Global building footprints (~1.4 billion buildings worldwide).

```python
from pygeovision.labeling.microsoft import MicrosoftBuildingLabeler

labeler = MicrosoftBuildingLabeler()
result  = labeler.label(
    bbox=[-74.1, 40.6, -73.7, 40.9],
    output_path="./labels/ms_buildings.tif",
    min_area_m2=10.0,       # Filter tiny detections
    confidence_min=0.8,
)

print(f"Buildings: {result['n_features']}")
```

---

## Google Open Buildings

Open Buildings dataset — ~1.8 billion building footprints from Google.

```python
from pygeovision.labeling.google import GoogleBuildingLabeler

labeler = GoogleBuildingLabeler()
result  = labeler.label(
    bbox=[-74.1, 40.6, -73.7, 40.9],
    output_path="./labels/google_buildings.tif",
    confidence_threshold=0.75,
)
```

---

## ESA WorldCover

ESA WorldCover 2021 — global land cover at 10m resolution, 11 classes.

```python
from pygeovision.labeling.esa import ESAWorldCoverLabeler

labeler = ESAWorldCoverLabeler()
result  = labeler.label(
    bbox=[-74.1, 40.6, -73.7, 40.9],
    output_path="./labels/worldcover.tif",
    year=2021,
)
```

**Classes:**

| Class ID | Name | Color |
|----------|------|-------|
| 10 | Tree cover | Dark green |
| 20 | Shrubland | Light green |
| 30 | Grassland | Yellow-green |
| 40 | Cropland | Yellow |
| 50 | Built-up | Red |
| 60 | Bare / sparse veg. | Orange |
| 70 | Snow and ice | White |
| 80 | Permanent water | Blue |
| 90 | Herbaceous wetland | Teal |
| 95 | Mangroves | Dark teal |
| 100 | Moss and lichen | Beige |

---

## Dynamic World

Google Dynamic World — near-real-time land use/land cover, 9 classes, 10m.

```python
from pygeovision.labeling.dynamic_world import DynamicWorldLabeler

labeler = DynamicWorldLabeler()
result  = labeler.label(
    bbox=[-74.1, 40.6, -73.7, 40.9],
    date_range=["2024-06-01", "2024-08-31"],
    output_path="./labels/dynamic_world.tif",
    aggregation="mode",    # "mode" | "mean_probability"
)
```

**Classes:** water, trees, grass, flooded_veg, crops, shrub, built, bare, snow_ice

---

## SAM Auto-Labeler

Segment Anything Model automatic mask generation — zero-shot segmentation.

```python
from pygeovision.labeling.sam_auto import SAMAutoLabeler

labeler = SAMAutoLabeler(
    model_type="vit-large",   # "vit-huge" | "vit-large" | "vit-base"
    device="cuda",
)

result = labeler.auto_label(
    image_path="scene.tif",
    output_path="./labels/sam_masks.tif",
    points_per_side=32,        # Grid density (higher = more masks)
    min_area_m2=25.0,          # Filter tiny segments
    pred_iou_thresh=0.88,
    stability_score_thresh=0.95,
)

print(f"Masks generated: {result['n_masks']}")
```

---

## Foundation Model Labeler

DINOv2 feature extraction + k-means clustering — unsupervised labels.

```python
from pygeovision.labeling.foundation import FoundationModelLabeler

labeler = FoundationModelLabeler(
    model_name="dinov2-base",
    device="cuda",
)

result = labeler.cluster(
    image_path="scene.tif",
    output_path="./labels/foundation_clusters.tif",
    n_clusters=8,
    method="kmeans",   # "kmeans" | "gmm" | "spectral"
)

print(f"Clusters: {result['n_clusters']}")
print(f"Silhouette score: {result['silhouette_score']:.3f}")
```

---

## Active Learning

Select the most informative unlabelled samples to label next.

```python
from pygeovision.labeling.active import ActiveLearner

learner = ActiveLearner(
    model=trained_model,
    strategy="entropy",      # "entropy" | "margin" | "diversity" | "coreset"
    device="cuda",
)

# Score unlabelled images and return top-k most uncertain
candidates = learner.select(
    unlabelled_paths=["a.tif", "b.tif", "c.tif", ...],
    budget=20,               # Number of images to select
    chip_size=512,
)

for c in candidates:
    print(f"  {c['path']}  uncertainty={c['uncertainty']:.4f}")
```

---

## Label Quality Assessment

Measure the quality of any binary or multi-class label raster.

```python
from pygeovision.labeling.quality import LabelQualityAssessor

qa     = LabelQualityAssessor()
report = qa.assess(
    label_path="./labels/buildings.tif",
    reference_path=None,    # Optional ground truth for comparison
)

print(f"Grade:          {report['quality_grade']}")   # A / B / C / D / F
print(f"Score:          {report['quality_score']:.0%}")
print(f"Class balance:  {report['checks']['class_balance']['score']:.2f}")
print(f"Spatial noise:  {report['checks']['spatial_noise']['score']:.2f}")
print(f"Edge sharpness: {report['checks']['edge_quality']['score']:.2f}")

for rec in report['recommendations']:
    print(f"  → {rec}")
```

---

## Multi-Source Pipeline

Fuse labels from multiple sources using majority vote or weighted consensus.

```python
from pygeovision.labeling.pipeline import LabelingPipeline

pipeline = LabelingPipeline(
    sources=[
        {"type": "osm",                "weight": 1.0},
        {"type": "esa_worldcover",     "weight": 1.5},
        {"type": "microsoft_buildings","weight": 2.0},
    ],
    fusion="weighted_vote",    # "majority_vote" | "weighted_vote" | "union" | "intersection"
)

result = pipeline.run(
    bbox=[-74.1, 40.6, -73.7, 40.9],
    output_path="./labels/fused.tif",
)

print(f"Agreement map saved: {result['agreement_path']}")
```
