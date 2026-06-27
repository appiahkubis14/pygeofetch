# Authentication

PyGeoVision supports 22 satellite data providers. This tutorial shows how to set up credentials for the most commonly used ones.

---

## Planetary Computer (Microsoft)

The easiest provider to start with — many datasets are free without authentication.

```bash
# Install the Planetary Computer client
pip install planetary-computer
```

```python
import pygeovision as pgv

client = pgv.PyGeoVision()

# Free tier — no key required for public collections
results = client.search(
    bbox=[-74.1, 40.6, -73.7, 40.9],
    date_range=["2024-06-01", "2024-08-31"],
    providers=["planetary_computer"],
)

# Premium tier — request a free API key at planetarycomputer.microsoft.com
client.add_credentials("planetary_computer", api_key="YOUR_KEY")
```

---

## AWS Open Data

Sentinel-2 and Landsat are available free on AWS S3 via the AWS Open Data program.

```bash
pip install boto3 awscli
aws configure   # Enter your AWS credentials
```

```python
client.add_credentials("aws_earth",
    access_key="AKIAIOSFODNN7EXAMPLE",
    secret_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    region="us-west-2",
)
```

---

## Copernicus Data Space (ESA)

Free access to Sentinel-1, Sentinel-2, Sentinel-3, and Sentinel-5P.

Register at: [dataspace.copernicus.eu](https://dataspace.copernicus.eu)

```python
client.add_credentials("copernicus",
    username="your_username",
    password="your_password",
)
```

---

## USGS EarthExplorer

Free access to Landsat, ASTER, MODIS, and more.

Register at: [ers.cr.usgs.gov](https://ers.cr.usgs.gov/register)

```python
client.add_credentials("usgs",
    username="your_username",
    password="your_password",
)
```

---

## Planet Labs

Commercial VHR imagery (Planet Scope, SkySat).

```python
client.add_credentials("planet", api_key="YOUR_PLANET_API_KEY")
```

---

## Maxar

WorldView and GeoEye imagery.

```python
client.add_credentials("maxar",
    api_key="YOUR_MAXAR_API_KEY",
    customer_id="YOUR_CUSTOMER_ID",
)
```

---

## Saving Credentials

PyGeoVision saves credentials to `~/.pygeovision/credentials.json` (encrypted at rest):

```bash
pgv auth add planetary_computer --api-key YOUR_KEY
pgv auth add copernicus --username user --password pass
pgv auth list   # Show configured providers
pgv auth remove planetary_computer
```

---

## Verifying Credentials

```python
# Test that credentials work
result = client.test_credentials("planetary_computer")
print(f"Connected: {result['success']}")
print(f"Quota remaining: {result.get('quota_remaining', 'unlimited')}")
```

---

## Provider Comparison

| Provider | Coverage | Resolution | Free Tier | Latency |
|----------|----------|-----------|-----------|---------|
| Planetary Computer | Global | 10m (S2) | Yes | Low |
| AWS Open Data | Global | 10m (S2) | Yes | Low |
| Copernicus | Global | 10m (S2) | Yes | Medium |
| USGS | Global | 30m (Landsat) | Yes | Medium |
| Planet | Global | 3–5m | No | Low |
| Maxar | Global | 0.3m | No | Low |
| Airbus SPOT | Global | 1.5m | No | Medium |
