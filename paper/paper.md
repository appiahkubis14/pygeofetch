---
title: 'PyGeoFetch: A Unified Python Framework for Multi-Provider Satellite Data Acquisition and Geospatial Processing'

tags:
  - Python
  - satellite remote sensing
  - earth observation
  - geospatial
  - SAR
  - Sentinel
  - Landsat
  - data pipeline

authors:
  - name: Samuel Appiah Kubi
    orcid: 0000-0000-0000-0000
    affiliation: "1"

affiliations:
  - index: 1
    name: EOCoreINT, Accra, Ghana

date: 12 July 2026

bibliography: paper.bib
---

# Summary

PyGeoFetch is an open-source Python package that provides a unified interface for
discovering, acquiring, and processing satellite Earth observation data from 22+
providers through a single, consistent API. Satellite imagery is increasingly central
to environmental science, climate monitoring, agriculture, disaster response, and urban
planning. However, accessing this data requires navigating a fragmented landscape of
provider-specific APIs, authentication schemes, data formats, and naming conventions—a
substantial barrier that consumes researcher time and introduces reproducibility errors.

PyGeoFetch eliminates this friction by presenting all providers through one common
search, download, and processing interface while preserving full access to
provider-specific capabilities including Synthetic Aperture Radar (SAR) processing,
Sentinel-1 Single Look Complex (SLC) products, and precise orbit file retrieval for
Interferometric SAR (InSAR) workflows. The package covers the complete satellite data
workflow: federated search, credential management, parallel download with integrity
validation, and a 41-step chainable processing pipeline covering atmospheric correction,
cloud masking, 17 spectral indices, SAR calibration, and Cloud Optimized GeoTIFF (COG)
export. PyGeoFetch targets earth scientists, remote sensing practitioners, and
geospatial engineers who need reliable, reproducible data pipelines without
reimplementing provider-specific access logic for every project.

PyGeoFetch serves as the data acquisition backbone for PyGeoVision
[@appiahkubi2026pygeovision], a companion library for geospatial AI inference,
and is released under the MIT License to maximise reuse across academic and
operational contexts.

# Statement of Need

Satellite data access today requires researchers to master a fragmented ecosystem of
incompatible interfaces. The Copernicus Data Space Ecosystem [@copernicus2023] uses
OData REST queries with OAuth2 authentication; the USGS Earth Explorer Machine-to-Machine
API [@usgs_m2m] requires EROS account tokens and dataset-specific payload structures;
Planetary Computer [@microsoft_pc] uses STAC 1.0 with anonymous access; AWS Earth
[@aws_earth] exposes a separate STAC endpoint; NASA Earthdata [@earthdata] uses URS
tokens; Planet Labs [@planet] uses a proprietary search schema. Each provider exposes different metadata
field names for the same concepts and delivers data in different archive formats.

The consequences are concrete: reproducing an analysis that used multiple providers
often requires weeks of re-engineering authentication and query logic. Students
frequently restrict their work to a single provider not because it has the best data,
but because they cannot afford the time to learn additional APIs [@baumann2021]. PyGeoFetch
addresses this by providing: (1) a single `SearchQuery` object and `client.search()` call
that queries any combination of supported providers simultaneously and returns a uniform
`SatelliteData` list; (2) the first open-source library to expose Sentinel-1C and
Sentinel-1D (the active constellation since May and April 2026 respectively) with SLC
product type routing; (3) programmatic retrieval of ESA POEORB and RESORB orbit files with
local caching—a prerequisite for millimetre-precision InSAR deformation mapping; (4) a 41-step
processing pipeline returning consistent `ProcessingResult` objects; and (5) file-type-aware
download validation preventing silent partial-download failures.

# State of the Field

Several packages address portions of this problem. `sentinelsat` [@sentinelsat] provides
Copernicus access but is limited to Sentinel missions without processing integration.
`landsatxplore` [@landsatxplore] wraps the USGS M2M API for Landsat only.
`pystac-client` [@pystac_client] is an excellent STAC browser but covers only
STAC-compliant endpoints, excluding USGS, Planet, and Maxar. `stackstac` [@stackstac]
and `odc-stac` [@odcstac] address lazy STAC loading but not acquisition. `eo-learn`
[@eolearn] provides processing pipelines but requires a Sentinel Hub subscription.
`leafmap` [@leafmap] and `geemap` [@geemap] offer Earth Engine integration, but
Earth Engine's server-side model differs fundamentally from the local download-and-process
workflow needed for offline analysis and HPC batch processing.

The closest analogue in scope is `eodag` [@eodag], which also abstracts multiple
providers. PyGeoFetch differs in three key areas: an integrated post-download processing
pipeline (eodag stops at download), the full Sentinel-1 SLC workflow including orbit
file retrieval, and built-in file validation preventing the silent corruption bug class
that costs significant researcher time when discovered late in a processing chain.
PyGeoFetch's choice to process data locally rather than in-cloud enables reproducibility
across funding cycles, HPC batch workflows, custom processing chains, and offline use in
field environments with limited connectivity. PyGeoFetch serves as the data acquisition
backbone for PyGeoVision [@appiahkubi2026pygeovision], providing the foundation for
end-to-end geospatial AI workflows.

# Software Design

PyGeoFetch is structured around five layers reflecting the natural stages of a satellite
data workflow.

**Provider layer** (`pygeofetch/providers/`): each of the 22 providers inherits from
`AbstractBaseProvider`, which enforces `authenticate()`, `search()`, and `download()`
methods. Provider-specific logic (OData filters for Copernicus, M2M JSON payloads for
USGS, STAC queries for Planetary Computer) is fully encapsulated. Authentication
supports five credential types (username/password, API key, OAuth2 client credentials,
bearer token, access/secret key pair), all stored via the system keyring through a
single `add_credentials(provider, dict)` call.

**Search layer** (`pygeofetch/core/searcher.py`): `FederatedSearcher` dispatches queries
to multiple providers concurrently via `ThreadPoolExecutor`, deduplicates results by
scene ID and spatial overlap, and returns a ranked list of `SatelliteData` objects. A
`SearchQuery` model captures bounding box, date range, cloud cover limit, satellite
platform, product type (GRD or SLC), polarisation, and pass direction, along with an
optional CQL2 filter string for STAC-compliant providers.

**Download layer** (`pygeofetch/core/downloader.py`): `AdaptiveDownloader` manages
parallel downloads with exponential back-off retries and resume support. After each file
completes, `_validate_downloaded_file()` applies format-aware integrity checks: ZIP and
TAR archives are verified with `zipfile.testzip()` and `tarfile.getmembers()`; GeoTIFF
and JPEG 2000 files are opened with rasterio [@rasterio] and a sample tile is read to
distinguish a truncated response from a valid file. A `DownloadProgress` display adapts
to the execution environment: a braille spinner with in-place updates in terminals, and
an HTML gradient progress widget in Jupyter notebooks via IPython's `display_id`
mechanism.

**Processing layer** (`pygeofetch/processing/`): a fluent builder API chains 41
operations—11 preprocessing steps (atmospheric correction via DOS1/Sen2Cor, cloud
masking, reprojection, pan-sharpening, tiling), 17 spectral indices (NDVI, EVI, SAVI,
NDWI, MNDWI, NDBI, NDSI, NDMI, NBR, dNBR, Tasselled Cap, PCA, texture, LST, albedo,
band math, band stack), 9 post-processing steps (vectorisation, zonal statistics, COG
export), and 4 SAR operations (Lee/Enhanced-Lee despeckle, radiometric calibration to
sigma-naught, flood mapping, InSAR coherence). Pipelines may also be defined in YAML
for version-controlled, reproducible workflows schedulable with cron expressions.

**Orbit file layer** (`pygeofetch/core/orbits.py`): `fetch_orbit_file()` parses
acquisition timestamp and satellite identifier from any Sentinel-1 product name, checks
a local cache, and if not cached fetches the ESA orbit directory listing, matches the
file whose validity window covers the acquisition time, and downloads it. If the
acquisition is fewer than 21 days old a warning recommends restituted orbits, reflecting
ESA's publication schedule for precise orbital solutions.

A key design trade-off was synchronous HTTP (via `httpx` [@httpx] and `requests`) over
an async-first approach. Satellite scenes are typically 500 MB to 8 GB: at these sizes,
network bandwidth dominates latency, and `ThreadPoolExecutor` achieves equivalent
throughput with a simpler programming model suitable for research contexts.

PyGeoFetch is released under the MIT License, enabling maximum reuse across academic,
commercial, and operational contexts.

# Research Impact Statement

PyGeoFetch directly enables research workflows that were previously impractical without
custom engineering. InSAR time-series analysis—combining Sentinel-1 SLC search,
precise orbit retrieval, and coherence estimation—can now be scripted in a reproducible
pipeline feeding tools such as SNAP [@snap] or MintPy [@mintpy], without manual
per-scene orbit file downloads. Multi-sensor change detection studies combining
contemporaneous SAR and optical imagery [@chini2018; @valero2021] benefit from
federated search returning results from both sensor families in a single call. Burnt-area
severity mapping using pre- and post-fire dNBR is expressible in ten lines of Python
or a version-controlled YAML pipeline, replacing one-off cloud console workflows.

The activation of Sentinel-1C (May 2025) and Sentinel-1D (April 2026), following
Sentinel-1B's 2022 failure and Sentinel-1A's planned decommissioning, creates a data
continuity challenge for long-running monitoring systems. PyGeoFetch's platform
normalisation and updated provider platform lists ensure existing search queries
automatically include new constellation members without code changes.

The package ships with 70 integration contract tests that verify the complete data
contract: search result ordering and length invariants, download integrity validation,
orbit file caching, band alias resolution across Sentinel-2 and Landsat naming
conventions, and all 41 processing pipeline builder methods. PyGeoFetch is distributed
under the MIT License to maximise reuse across academic and operational contexts.

# AI Usage Disclosure

Portions of the PyGeoFetch codebase, test suite, and this paper were developed with
assistance from Claude (Anthropic, Claude Sonnet 4.6). AI assistance was used for code
generation of provider adapters and processing stubs, test scaffolding, logging module
refactoring, and copy-editing of this manuscript. All AI-generated code was reviewed
and validated against the 70-test contract suite and live provider APIs by the human
author. Core design decisions—the five-layer provider abstraction architecture, SLC
routing logic, file-type-aware validation strategy, and orbit file caching design—were
made by the human author. The human author reviewed and edited all AI-assisted text and
takes full responsibility for its accuracy.

# Acknowledgements

The author thanks the European Space Agency for free and open access to Sentinel mission
data through the Copernicus Data Space Ecosystem, the United States Geological Survey
for the Landsat archive through the Earth Explorer M2M API, and NASA for the Earthdata
platform. The author acknowledges the open-source communities behind rasterio [@rasterio],
pydantic [@pydantic], GDAL [@gdal], and pystac [@pystac], whose libraries underpin the
PyGeoFetch implementation.

# References

