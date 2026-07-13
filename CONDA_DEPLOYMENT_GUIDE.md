# PyGeoFetch — Conda & Conda-Forge Deployment Guide

Complete step-by-step guide to publish PyGeoFetch on PyPI, conda, and conda-forge.

---

## Overview

```
Step 1 → Prepare & test the package locally
Step 2 → Publish to PyPI (required before conda-forge)
Step 3 → Submit to conda-forge (automated builds for all platforms)
Step 4 → Optional: publish your own conda channel
```

---

## Prerequisites

Install the tooling once:

```bash
# Build and publish tools
pip install build twine conda-build anaconda-client grayskull

# Or via conda
conda install -c conda-forge conda-build anaconda-client grayskull
```

You need accounts at:
- **PyPI**: https://pypi.org/account/register/
- **TestPyPI**: https://test.pypi.org/account/register/
- **GitHub**: https://github.com (for conda-forge feedstock PR)
- **Anaconda Cloud** (optional): https://anaconda.org

---

## STEP 1 — Prepare the package

### 1a. Clean and verify

```bash
cd PyGeoFetch-1.1.0-complete/pygeofetch

# Remove old build artifacts
rm -rf dist/ build/ *.egg-info/

# Confirm the package installs cleanly
pip install -e ".[all]"

# Run the contract tests
pytest tests/test_pgv_integration_contract.py -v
# Expected: 70 passed
```

### 1b. Verify pyproject.toml is correct

Key fields that conda-forge checks:
```toml
[project]
name = "pygeofetch"          # must be lowercase for PyPI
version = "1.1.0"            # must match git tag
license = { file = "LICENSE" }
requires-python = ">=3.9"
```

### 1c. Build the distribution

```bash
# Build both sdist (.tar.gz) and wheel (.whl)
python -m build

# You should see:
# dist/
#   pygeofetch-1.1.0.tar.gz      ← source distribution (conda uses this)
#   pygeofetch-1.1.0-py3-none-any.whl  ← wheel
```

### 1d. Check the built package

```bash
# Verify both packages are valid
twine check dist/*

# Expected output:
# Checking dist/pygeofetch-1.1.0.tar.gz: PASSED
# Checking dist/pygeofetch-1.1.0-py3-none-any.whl: PASSED
```

---

## STEP 2 — Publish to PyPI

### 2a. Test on TestPyPI first (always do this)

```bash
# Upload to TestPyPI
twine upload --repository testpypi dist/*

# When prompted:
#   username: __token__
#   password: pypi-AgEIcHlwaS5vcm...  (your TestPyPI API token)

# Install from TestPyPI to verify
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    pygeofetch==1.1.0

# Quick smoke test
python -c "from pygeofetch import PyGeoFetch; print('OK')"
PyGeoFetch --version
```

### 2b. Publish to real PyPI

```bash
twine upload dist/*

# username: __token__
# password: pypi-...  (your real PyPI API token)
```

Get your PyPI API token at: https://pypi.org/manage/account/token/

### 2c. Note the SHA256 hash

After uploading, get the SHA256 of the source tarball — conda-forge needs it:

```bash
# From the dist/ folder:
sha256sum dist/pygeofetch-1.1.0.tar.gz

# Or download from PyPI and hash it:
pip download pygeofetch==1.1.0 --no-deps -d /tmp/pgf/
sha256sum /tmp/pgf/pygeofetch-1.1.0.tar.gz
```

Keep this hash — you'll paste it into meta.yaml in Step 3.

---

## STEP 3 — Submit to conda-forge

conda-forge maintains "feedstocks" — one GitHub repo per package.
The process is: generate recipe → fork staged-recipes → open PR.

### 3a. Generate the recipe automatically with grayskull

```bash
# grayskull reads from PyPI and generates a conda recipe
grayskull pypi pygeofetch --output conda-recipe/

# This auto-fills: dependencies, version, sha256, license
# Review the generated meta.yaml and clean it up
```

### 3b. Update meta.yaml with the real SHA256

Open `conda-recipe/meta.yaml` and replace the placeholder:

```yaml
source:
  url: https://pypi.io/packages/source/p/pygeofetch/pygeofetch-1.1.0.tar.gz
  sha256: PASTE_YOUR_REAL_SHA256_HERE   # ← replace this line
```

### 3c. Test the recipe locally

```bash
# Build the conda package locally first
conda build conda-recipe/ -c conda-forge

# If successful, you'll see:
# anaconda upload /path/to/pygeofetch-1.1.0-py39_0.tar.bz2

# Test install the local build
conda install --use-local pygeofetch
PyGeoFetch --version
PyGeoFetch doctor
```

### 3d. Fork conda-forge/staged-recipes

```bash
# 1. Fork https://github.com/conda-forge/staged-recipes on GitHub

# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/staged-recipes.git
cd staged-recipes

# 3. Create a branch
git checkout -b add-pygeofetch

# 4. Create the recipe folder
mkdir -p recipes/pygeofetch
cp /path/to/conda-recipe/meta.yaml recipes/pygeofetch/meta.yaml

# 5. Commit
git add recipes/pygeofetch/meta.yaml
git commit -m "Add pygeofetch recipe"

# 6. Push
git push origin add-pygeofetch
```

### 3e. Open the Pull Request

Go to: https://github.com/conda-forge/staged-recipes
Click **"New pull request"** from your fork.

The PR description should include:
```
## Package info
- Package name: pygeofetch
- PyPI URL: https://pypi.org/project/pygeofetch/
- License: MIT
- Description: Universal satellite data pipeline — 22+ providers

## Testing
- All 70 contract tests passing
- `conda build` succeeds locally
- `PyGeoFetch --version` and `PyGeoFetch doctor` both work
```

The conda-forge bots will:
1. Run CI on Linux, macOS, and Windows
2. Request review from maintainers
3. Merge → automatically create `pygeofetch-feedstock` repo

After merge (usually 1-3 days):
```bash
conda install -c conda-forge pygeofetch
```

---

## STEP 4 — Optional: Your own conda channel (Anaconda Cloud)

For faster iteration or private distribution before conda-forge approval:

```bash
# Install anaconda-client
conda install anaconda-client

# Log in
anaconda login
# Enter your anaconda.org username and password

# Build the package
conda build conda-recipe/ -c conda-forge

# Upload to your personal channel
anaconda upload /path/to/pygeofetch-1.1.0-py39_0.tar.bz2

# Users can then install from your channel:
conda install -c YOUR_ANACONDA_USERNAME pygeofetch
```

---

## meta.yaml — Full Reference

```yaml
{% set name = "pygeofetch" %}
{% set version = "1.1.0" %}

package:
  name: {{ name }}
  version: {{ version }}

source:
  url: https://pypi.io/packages/source/{{ name[0] }}/{{ name }}/{{ name }}-{{ version }}.tar.gz
  sha256: <PASTE_SHA256_HERE>

build:
  number: 0
  noarch: python
  script: {{ PYTHON }} -m pip install . -vv --no-deps --no-build-isolation
  entry_points:
    - PyGeoFetch = pygeofetch.cli.main:cli

requirements:
  host:
    - python >=3.9
    - pip
    - setuptools >=68
    - wheel
  run:
    - python >=3.9
    - httpx >=0.27
    - click >=8.1
    - pydantic >=2.5
    - pydantic-settings >=2.1
    - pyyaml >=6.0
    - cryptography >=42.0
    - keyring >=24.3
    - tenacity >=8.2
    - python-dateutil >=2.9
    - requests >=2.31
    - anyio >=4.2

test:
  imports:
    - pygeofetch
    - pygeofetch.core.engine
  commands:
    - PyGeoFetch --version
    - PyGeoFetch --help
  requires:
    - pytest

about:
  home: https://github.com/pygeofetch/PyGeoFetch
  license: MIT
  license_family: MIT
  license_file: LICENSE
  summary: Universal satellite data pipeline — unified access to 22+ providers
  doc_url: https://pygeofetch.readthedocs.io

extra:
  recipe-maintainers:
    - your-github-username
```

---

## Versioning for future releases

```bash
# Bump version in pyproject.toml
# Then tag the commit
git tag -a v1.0.1 -m "Release 1.0.1"
git push origin v1.0.1

# Rebuild and republish
rm -rf dist/
python -m build
twine upload dist/*

# Then update conda-forge feedstock:
# Go to https://github.com/conda-forge/pygeofetch-feedstock
# The conda-forge bot (regro-cf-autotick-bot) usually opens a PR
# automatically when it detects a new PyPI version.
# Review and merge it — no manual recipe update needed.
```

---

## Troubleshooting

| Error | Fix |
|---|---|
| `twine check` fails | Check README.md is valid reStructuredText or set `readme = {file="README.md", content-type="text/markdown"}` |
| `conda build` fails on imports | Add missing dep to `run:` section of meta.yaml |
| sha256 mismatch | Re-download the tarball from PyPI and rehash |
| PR bot says "dependencies not on conda-forge" | Check each dep exists: `conda search -c conda-forge PACKAGE` |
| `noarch: python` fails | Package has C extensions — remove `noarch` and add platform selectors |
| CI fails on Windows | Usually path separators — check any `os.path` calls |

---

## Install commands after publishing

```bash
# From PyPI
pip install pygeofetch
pip install "pygeofetch[geo]"        # +rasterio, geopandas, shapely
pip install "pygeofetch[all]"        # everything

# From conda-forge (after approval)
conda install -c conda-forge pygeofetch
mamba install -c conda-forge pygeofetch   # faster

# From your personal channel (immediate)
conda install -c YOUR_USERNAME pygeofetch
```
