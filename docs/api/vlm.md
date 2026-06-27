# Vision-Language Models

CLIP-based geo retrieval, Moondream VQA, and zero-shot scene understanding for satellite imagery.

---

## CLIPGeo

OpenCLIP / RemoteCLIP for satellite image retrieval and zero-shot classification.

```python
from pygeovision.advanced.vlm.clip_geo import CLIPGeo

clip = CLIPGeo(
    model_name="remoteclip-l14",   # "openclip-b32" | "remoteclip-b32" | "remoteclip-l14"
    device="cuda",
)

# Text-to-image search across a directory
results = clip.search(
    query="flooded agricultural fields",
    image_dir="./data/scenes/",
    top_k=10,
)

for r in results:
    print(f"{r['score']:.4f}  {r['path']}")

# Zero-shot classification
probs = clip.classify(
    image="scene.tif",
    labels=["urban area", "dense forest", "cropland", "water body", "bare soil"],
)
print(probs)
# {'urban area': 0.68, 'cropland': 0.18, ...}

# Image-to-image retrieval
similar = clip.similar_images("query.tif", image_dir="./data/", top_k=5)

# Build a searchable index over a large image collection
clip.build_index("./data/scenes/", index_path="./index.faiss")
results = clip.search_index("flooding", index_path="./index.faiss", top_k=20)
```

---

## MoondreamGeo

Moondream2 — a compact (1.8B) vision-language model adapted for satellite imagery VQA and captioning.

```python
from pygeovision.advanced.vlm.moondream_geo import MoondreamGeo

moon = MoondreamGeo(device="cuda")

# Image captioning
caption = moon.caption("scene.tif")
print(caption)
# "Aerial view of a dense residential area with regular street grid and green parks."

# Visual question answering
answer = moon.vqa("scene.tif", "How many buildings are visible?")
print(answer)   # "Approximately 120 buildings"

answer = moon.vqa("scene.tif", "Is there any flooding visible?")
print(answer)   # "No visible flooding. The area appears dry and clear."

# Batch captioning
captions = moon.batch_caption(["a.tif", "b.tif", "c.tif"])

# Change description (two images)
description = moon.describe_change("before.tif", "after.tif")
print(description)
# "Significant deforestation visible. Approximately 40% of tree cover removed."
```

---

## GeoRetrieval

Semantic retrieval across large satellite image archives using CLIP embeddings and FAISS indexing.

```python
from pygeovision.advanced.vlm.retrieval import GeoRetrieval

retrieval = GeoRetrieval(
    model="remoteclip-l14",
    device="cuda",
    index_type="IVF",    # "Flat" | "IVF" | "HNSW"
    nprobe=32,
)

# Build index (one-time, ~1min per 10k images)
retrieval.build_index(
    image_dir="./archive/",
    index_path="./geo_index.faiss",
    metadata_path="./geo_index_meta.json",
    batch_size=64,
)

# Query by text
results = retrieval.search(
    query="airport runway with parked aircraft",
    index_path="./geo_index.faiss",
    top_k=20,
)

# Query by image
results = retrieval.search_by_image("query_scene.tif", top_k=20)

# Cluster the archive (unsupervised scene taxonomy)
clusters = retrieval.cluster(
    index_path="./geo_index.faiss",
    n_clusters=20,
    output_dir="./clusters/",
)
```
