After months of building, testing, and breaking things until they stopped breaking — I'm excited to share pygeofetch, an open-source Python package built to unify satellite data access and processing into one coherent API.

🌍 The problem:

If you've worked with Earth observation data, you know the pain. Copernicus speaks OData. USGS speaks M2M tokens. NASA speaks Earthdata Login. Planet has its own schema entirely. Every provider has its own auth flow, its own query language, its own data format and every new project means re-learning half of it from scratch.

For researchers and institutions across Africa and the Global South, that friction isn't just annoying. It's often the actual barrier standing between "we need to monitor this" and "we're monitoring it."

🛰️ What pygeofetch does:

One CLI. One Python API. A single interface to search, download, and process satellite data across major providers — Copernicus, USGS Landsat, NASA Earthdata,Microsoft Planetary Computer,element84,AWS Earth and more, with active work ongoing to harden and expand coverage further.

🔧 Under the hood:

• Federated search with real deduplication and scoring, not just a thin wrapper
• Resilient downloads — live progress, automatic resume, retry logic tuned against what providers actually do when you hit them too hard
• A full optical preprocessing engine — atmospheric correction (DOS1/DOS2/Sen2Cor), cloud masking (SCL/FMask), topographic correction, pan-sharpening, mosaicking and multi-date compositing
• 17+ spectral indices (NDVI, NDWI, EVI, NBR, dNBR, LST...)
• Postprocessing that turns raster results into GIS-ready output — vectorize, zonal statistics, Cloud Optimized GeoTIFF
• A full SAR processing chain — despeckling, radiometric calibration, flood mapping, coherence
• A complete InSAR pipeline in pure Python — coregistration, interferogram generation, SNAPHU phase unwrapping, SBAS time series inversion, atmospheric correction
• Sentinel-1 SLC support built for real InSAR workflows, not just browsing
• YAML pipeline orchestration for repeatable, scheduled workflows

💡 Why this matters to me:

Institutions across Africa need satellite intelligence to monitor deforestation, catch illegal mining before it spreads, track crop health, watch cities grow, and respond when disasters hit. I've been building and testing this against real problems — mining-related land degradation in Ashanti Region, flood extent mapping in Accra — because tools that only work in a demo don't help anyone.

This is early. Some providers are more battle-tested than others, and I'd rather say that plainly than oversell it. But the core is solid, it's growing, and it's yours to use, inspect, and improve and report any issue for improvement.

📚 Free tutorials and resources will soon be available at: https://www.eocoreint.com
🖥️ pip install pygeofetch
📦 GitHub: https://github.com/EOCoreINT/pygeofetch
📚 Docs: https://pygeofetch.readthedocs.io

This is the first step toward open-source Earth observation tooling built for Africa and the Global South. PyGeoVision — the AI layer on top of this — is coming next.

Earth observation should be powerful *and* open. Let's build that together. 🚀

#pygeofetch #EarthObservation #RemoteSensing #SatelliteData #OpenSource #Africa #GlobalSouth #Python