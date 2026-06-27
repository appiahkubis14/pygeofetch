# Explainability (XAI)

Four techniques for understanding what geospatial AI models are "looking at": GradCAM saliency maps, SHAP values, Monte Carlo Dropout uncertainty, and transformer attention maps.

---

## GradCAM

Class Activation Mapping using gradient signals from the final convolutional layer.

```python
from pygeovision.explainability.gradcam import GradCAM

cam = GradCAM(
    model=model,
    target_layer=None,     # Auto-detected from model architecture
    device="cuda",
)

# Single image
result = cam.explain(
    image_path="scene.tif",
    class_idx=1,           # Class to visualise (0=background, 1=building, ...)
    output_path="./xai/gradcam.tif",
)

# Batch (entire directory)
result = cam.batch_explain(
    input_path="scene.tif",
    output_path="./xai/gradcam_overlay.tif",
    class_idx=1,
    colormap="jet",        # "jet" | "plasma" | "viridis" | "hot"
    alpha=0.6,             # Overlay opacity on original image
)

print(f"Mean saliency: {result['mean_saliency']:.4f}")
print(f"Peak region:   {result['peak_bbox']}")
```

### Variants

```python
from pygeovision.explainability.gradcam import GradCAMPlusPlus, EigenCAM

# GradCAM++ — better for multiple instances of same class
cam = GradCAMPlusPlus(model=model)

# EigenCAM — class-agnostic, uses PCA on feature maps
cam = EigenCAM(model=model)
```

---

## SHAP for Geospatial Models

SHAP (SHapley Additive exPlanations) kernel values per spectral band and spatial superpixel.

```python
from pygeovision.explainability.shap_geo import GeoSHAP

shap_explainer = GeoSHAP(
    model=model,
    background_images=background_paths,   # Reference set for baseline
    n_superpixels=50,                     # SLIC superpixel count
    device="cuda",
)

result = shap_explainer.explain(
    image_path="scene.tif",
    output_path="./xai/shap.tif",
    class_idx=1,
    n_samples=200,        # More = more accurate but slower
)

# Band importance (which spectral bands matter most?)
for band, importance in result['band_importance'].items():
    print(f"  Band {band}: {importance:.4f}")

# Spatial SHAP map
print(f"Spatial map shape: {result['shap_map'].shape}")
```

---

## MC Dropout Uncertainty

Monte Carlo Dropout estimates prediction uncertainty by running inference N times with dropout active.

```python
from pygeovision.explainability.uncertainty import MCDropoutUncertainty

estimator = MCDropoutUncertainty(
    model=model,
    n_passes=50,           # Forward passes (more = smoother estimate)
    dropout_rate=0.1,      # Applied to all layers during inference
    device="cuda",
)

result = estimator.estimate(
    image_path="scene.tif",
    output_path="./xai/uncertainty.tif",
    chip_size=512,
)

# Per-pixel uncertainty maps
mean_pred = result['mean_prediction']         # (H, W) — average class prediction
entropy   = result['predictive_entropy']      # (H, W) — total uncertainty
aleatoric = result['aleatoric_uncertainty']   # (H, W) — data noise
epistemic = result['epistemic_uncertainty']   # (H, W) — model uncertainty

print(f"High-uncertainty pixels: {result['high_uncertainty_pct']:.1f}%")
print(f"Mean entropy: {result['mean_entropy']:.4f}")
```

### Interpreting Uncertainty

| Region | Meaning |
|--------|---------|
| Low entropy | Model is confident — predictions are reliable |
| High aleatoric | Inherently ambiguous input (e.g. mixed pixels at boundaries) |
| High epistemic | Model has not seen similar data — consider adding training examples |

---

## Transformer Attention Maps

Extract multi-head self-attention from ViT-based models (SegFormer, DINOv2/v3, Swin).

```python
from pygeovision.explainability.attention import AttentionMapExtractor

extractor = AttentionMapExtractor(
    model=model,
    layer_idx=-1,          # Which transformer block (-1 = last)
    head_idx=None,         # None = mean over all heads
)

result = extractor.extract(
    image_path="scene.tif",
    output_path="./xai/attention.tif",
    upscale=True,          # Bilinear upsample to full image resolution
)

attention_map = result['attention_map']    # (H, W)
head_maps     = result['per_head_maps']   # (n_heads, H, W)

print(f"Attention shape: {attention_map.shape}")
print(f"Max attention at pixel: {result['peak_pixel']}")
```

---

## Visualisation Utilities

```python
from pygeovision.explainability.gradcam import GradCAM

cam = GradCAM(model)

# Save side-by-side comparison (image | saliency | overlay)
cam.save_comparison(
    image_path="scene.tif",
    output_path="./xai/comparison.png",
    class_idx=1,
    dpi=150,
)

# Export as GeoTIFF (preserves CRS for GIS use)
cam.save_geotiff(
    image_path="scene.tif",
    output_path="./xai/gradcam_geo.tif",
    class_idx=1,
)
```
