After months of building, testing, and breaking things until they stopped breaking ,now I'm excited to share pygeofetch, an open-source Python package built to unify satellite data access and processing into one coherent API.

🌍 The problem:

If you've worked with Earth observation data, you know the pain. Copernicus speaks OData. USGS speaks M2M tokens. NASA speaks Earthdata Login. Planet has its own schema entirely. Every provider has its own auth flow, its own query language, its own data format and every new project means re-learning half of it from scratch.

For researchers and institutions across Africa and the Global South, that friction isn't just annoying. It's often the actual barrier standing between "we need to monitor this" and "we're monitoring it."

🛰️ What pygeofetch does:

One CLI. One Python API. A single interface to search, download, and process satellite data across major providers — Copernicus, USGS Landsat, NASA Earthdata, and more, with active work ongoing to harden and expand coverage further.

🔧 Under the hood:

• Federated search with real deduplication and scoring, not just a thin wrapper
• Resilient downloads — live progress, automatic resume, retry logic tuned against what providers actually do when you hit them too hard
• A full optical preprocessing engine — atmospheric correction (DOS1/DOS2/Sen2Cor), cloud masking (SCL/FMask), topographic correction, pan-sharpening, mosaicking and multi-date compositing
• 17+ spectral indices (NDVI, NDWI, EVI, NBR, dNBR, LST...)
• Real time-series analysis — not just before/after snapshots, but per-pixel trend fitting across multiple dates, zonal time series extraction, and anomaly detection against a baseline period
• Postprocessing that turns raster results into GIS-ready output — vectorize, zonal statistics, Cloud Optimized GeoTIFF
• A full SAR processing chain — despeckling, radiometric calibration, coherence
• A complete InSAR pipeline in pure Python — coregistration, interferogram generation, SNAPHU phase unwrapping, SBAS time series inversion, atmospheric correction
• Sentinel-1 SLC support built for real InSAR workflows, not just browsing
• YAML pipeline orchestration for repeatable, scheduled workflows

💡 Why this matters to me:

Institutions across Africa need satellite intelligence to monitor deforestation, catch illegal mining before it spreads, track crop health, watch cities grow, and respond when disasters hit. I've been building and testing this against real problems — a 6-year vegetation trend analysis over the Obuasi mining belt in Ashanti Region, tracking land degradation against real Landsat data pulled live from USGS, boundary-clipped and time-series-analyzed end to end — plus flood extent mapping in Accra. Tools that only work in a demo don't help anyone.

This is early. Some providers are more battle-tested than others, and I'd rather say that plainly than oversell it. But the core has been genuinely hardened  not just written once and left alone, but pushed against real APIs, real edge cases, and real data until it actually held up. It's growing, and it's yours to use, inspect, and improve.

📚 Free tutorials and resources will be available on: https://www.eocoreint.com
🖥️ pip install pygeofetch
📦 GitHub: https://github.com/EOCoreINT/pygeofetch
📚 Docs: https://appiahkubis14.github.io/pygeofetch-docs/

This is the first step toward open-source Earth observation tooling built for evryone. PyGeoVision — the AI layer on top of this — is coming next.

Earth observation should be powerful *and* open. Let's build that together. 🚀

#pygeofetch #EarthObservation #RemoteSensing #SatelliteData #OpenSource #pygeovision #GeoAI #Africa #GlobalSouth #Python