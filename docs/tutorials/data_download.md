# Data Download

Download, post-process, and manage satellite imagery from 22 providers.

---

## Basic Download

```python
import pygeovision as pgv

client = pgv.PyGeoVision()

# Search first
results = client.search(
    bbox=[-74.1, 40.6, -73.7, 40.9],
    date_range=["2024-06-01", "2024-08-31"],
    providers=["planetary_computer"],
    cloud_cover_max=10,
)

# Download the best scene
downloads = client.download(
    results[:1],
    output_dir="./data/",
)

scene_path = downloads[0].path
print(f"Downloaded: {scene_path}")
```

---

## Post-Processing

Apply processing steps automatically during download:

```python
downloads = client.download(
    results,
    output_dir="./data/",
    post_process=[
        "reproject:EPSG:32618",   # UTM Zone 18N for New York
        "cog",                     # Cloud-Optimized GeoTIFF
        "cloud_mask",              # Apply scene classification layer
    ],
    parallel=4,                    # Download 4 scenes concurrently
)
```

### Available Post-Processing Steps

| Step | Effect |
|------|--------|
| `reproject:EPSG:XXXX` | Reproject to target CRS |
| `cog` | Convert to Cloud-Optimized GeoTIFF |
| `cloud_mask` | Zero-out cloud-contaminated pixels |
| `clip` | Clip to the search bbox |
| `normalise` | Scale to [0, 1] float32 |
| `stack` | Stack all bands into one file |

---

## Select Bands

Download only specific bands to save bandwidth:

```python
# Sentinel-2 — only RGB + NIR for 4-band multispectral
downloads = client.download(
    results[:3],
    output_dir="./data/",
    bands=["B02", "B03", "B04", "B08"],   # Blue, Green, Red, NIR
)
```

---

## Inspect Downloads

```python
for d in downloads:
    import rasterio
    with rasterio.open(d.path) as src:
        print(f"Path:       {d.path}")
        print(f"Size:       {d.size_mb:.1f} MB")
        print(f"Bands:      {src.count}")
        print(f"Shape:      {src.height} × {src.width}")
        print(f"CRS:        {src.crs}")
        print(f"Resolution: {src.res[0]:.1f}m")
```

---

## Large-Area Download

For areas larger than one scene, PyGeoVision automatically mosaics:

```python
# Search a large area (multiple tiles)
results = client.search(
    bbox=[-75.0, 40.0, -73.0, 41.5],   # 2° × 1.5° — likely 3–4 tiles
    date_range=["2024-07-01", "2024-07-31"],
    cloud_cover_max=5,
)

# Download and mosaic into one seamless GeoTIFF
mosaic = client.download_mosaic(
    results,
    output_path="./data/nyregion_mosaic.tif",
    method="median",        # "first" | "last" | "median" | "mean"
    nodata=0,
)
print(f"Mosaic: {mosaic.path}  ({mosaic.size_mb:.0f} MB)")
```

---

## Resumable Downloads

Downloads are resumable — restart after interruption without re-downloading completed files:

```python
# If interrupted, re-run with the same output_dir:
downloads = client.download(results, output_dir="./data/")
# Only missing files are downloaded
```
