After months of building, testing, and breaking things until they stopped breaking — pygeofetch is live. An open-source Python package that unifies satellite data access and processing into one coherent API.

🌍 The problem:
Copernicus speaks OData. USGS speaks M2M. NASA speaks Earthdata Login. Planet has its own schema. Every provider has different auth, queries, and formats — and every project means re-learning half of it from scratch. For institutions in Africa and the Global South, that friction is often the barrier between "we need to monitor this" and "we're monitoring it."

🛰️ What pygeofetch does:
One CLI. One Python API. A single interface to search, download, and process satellite data across major providers — Copernicus, USGS Landsat, NASA Earthdata,planetary_computer,element84 and more.

🔧 Under the hood:
• Federated search with real deduplication and scoring
• Resilient downloads — live progress, auto-resume, retry logic
• Full optical preprocessing (DOS1/DOS2/Sen2Cor, cloud masking, pan-sharpening, mosaicking)
• 17+ spectral indices (NDVI, NDWI, EVI, NBR, dNBR, LST...)
• Per-pixel time-series trend fitting, zonal extraction, anomaly detection
• Postprocessing — vectorize, zonal stats, Cloud Optimized GeoTIFF
• SAR processing — despeckling, calibration, flood mapping, coherence
• Complete InSAR pipeline in pure Python — coregistration, interferogram, SNAPHU unwrapping, SBAS inversion
• Sentinel-1 SLC support for real InSAR workflows
• YAML pipeline orchestration for repeatable, scheduled workflows
pygeofetch_linkedin_announcement.md
💡 Why this matters:
I've tested this against real problems — a 6-year vegetation trend analysis over the Obuasi mining belt and flood extent mapping in Accra. Tools that only work in a demo don't help anyone.This is early. Some providers are more battle-tested than others. But the core has been hardened against real APIs, real edge cases, and real data — until it held up.

📚 Free tutorials will soon be avaliable at: https://www.eocoreint.com
🖥️ pip install pygeofetch
📦 GitHub: https://github.com/EOCoreINT/pygeofetch
📚 Docs: https://appiahkubis14.github.io/pygeofetch-docs/

This is the first step toward open-source Earth observation tooling built for Africa and the Global South. PyGeoVision — the AI layer — is coming next.
Earth observation should be powerful *and* open. Let's build that together. 🚀

#pygeofetch #EarthObservation #RemoteSensing #SatelliteData #OpenSource #pygeovision #Africa #Python