# Data Search

Learn how to search satellite imagery with STAC and CQL2 filters.

---

## Basic Search

```python
import pygeovision as pgv

client = pgv.PyGeoVision()

results = client.search(
    bbox=[-74.1, 40.6, -73.7, 40.9],
    date_range=["2024-06-01", "2024-08-31"],
    providers=["planetary_computer"],
    cloud_cover_max=10,
)

print(f"Found: {len(results)} scenes")

for r in results[:5]:
    print(f"  {r.date}  cloud={r.cloud_cover:.1f}%  id={r.id}")
```

---

## Filter by Collection

```python
results = client.search(
    bbox=[-74.1, 40.6, -73.7, 40.9],
    date_range=["2024-01-01", "2024-12-31"],
    providers=["planetary_computer"],
    collections=["sentinel-2-l2a"],   # Surface reflectance (L2A)
    cloud_cover_max=20,
    sort_by="cloud_cover",            # Sort lowest cloud first
    limit=50,
)
```

---

## CQL2 Advanced Filters

PyGeoVision supports full CQL2-JSON and CQL2-TEXT filtering:

```python
# Text filter: Sentinel-2B only, very low cloud cover
results = client.search(
    bbox=[-74.1, 40.6, -73.7, 40.9],
    date_range=["2024-06-01", "2024-08-31"],
    cql2_filter="eo:cloud_cover < 5 AND platform = 'sentinel-2b'",
)

# Filter by orbit direction
results = client.search(
    bbox=[10.0, 47.0, 15.0, 50.0],
    cql2_filter="sat:orbit_state = 'descending' AND eo:cloud_cover < 15",
)

# Filter by processing level and solar zenith angle
results = client.search(
    bbox=[-74.1, 40.6, -73.7, 40.9],
    cql2_filter="processing:level = 'L2A' AND view:sun_elevation > 30",
)
```

---

## Multi-Provider Search

Search multiple providers simultaneously and merge results:

```python
results = client.search(
    bbox=[-74.1, 40.6, -73.7, 40.9],
    date_range=["2024-06-01", "2024-08-31"],
    providers=["planetary_computer", "aws_earth", "copernicus"],
    cloud_cover_max=15,
    deduplicate=True,     # Remove scenes covering the same area/date
)

# Group by provider
from collections import Counter
by_provider = Counter(r.provider for r in results)
print(dict(by_provider))
```

---

## Inspect Results

```python
for r in results[:3]:
    print(f"ID:         {r.id}")
    print(f"Date:       {r.date}")
    print(f"Platform:   {r.platform}")
    print(f"Cloud:      {r.cloud_cover:.1f}%")
    print(f"Bands:      {r.bands}")
    print(f"Resolution: {r.resolution_m}m")
    print(f"CRS:        {r.crs}")
    print(f"Preview:    {r.preview_url}")
    print()
```

---

## Scene Preview

```python
# Download a quick-look JPEG preview
r = results[0]
r.download_preview("./previews/scene_preview.jpg")
```

---

## Time Series Search

Find the same area across multiple months:

```python
months = [
    ("2024-01-01", "2024-01-31"),
    ("2024-04-01", "2024-04-30"),
    ("2024-07-01", "2024-07-31"),
    ("2024-10-01", "2024-10-31"),
]

time_series = []
for start, end in months:
    r = client.search(
        bbox=[-74.1, 40.6, -73.7, 40.9],
        date_range=[start, end],
        providers=["planetary_computer"],
        cloud_cover_max=20,
        limit=1,          # Best scene per month
    )
    if r:
        time_series.append(r[0])

print(f"Monthly composites found: {len(time_series)}")
```
