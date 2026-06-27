# Contributing

We welcome contributions of all kinds — bug fixes, new models, dataset entries, documentation improvements, and tutorials.

---

## Development Setup

```bash
# 1. Fork and clone the repository
git clone https://github.com/yourusername/pygeovision
cd pygeovision

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install in editable mode with dev tools
pip install -e ".[dev,geo,train,foundation]"

# 4. Install pre-commit hooks
pre-commit install
```

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/test_models.py -v

# With coverage
pytest tests/ --cov=pygeovision --cov-report=html

# Fast (skip slow integration tests)
pytest tests/ -m "not slow" -v
```

---

## Code Style

PyGeoVision uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
ruff check pygeovision/       # lint
ruff format pygeovision/      # format
mypy pygeovision/             # type checking
```

---

## Adding a New Model

1. Add a `ModelSpec` entry to `pygeovision/models/registry.py`:

```python
ModelSpec(
    name="my-net-b0",
    task="segmentation",
    family="mynet",
    params_m=12.5,
    hf_id="myorg/my-net-base",
    description="MyNet-B0 — efficient satellite segmentation",
    supports_multispectral=True,
    pretrained_on="sentinel2",
)
```

2. Optionally create `pygeovision/models/segmentation/my_net.py` with a `build_mynet()` factory function.

3. Add tests in `tests/test_model_zoo.py`.

4. Open a pull request.

---

## Adding a Dataset Entry

Add a `DatasetInfo` entry to the appropriate domain list in `pygeovision/datasets/registry.py`:

```python
DatasetInfo(
    name="MyDataset",
    domain="urban",
    year=2024,
    n_samples=50000,
    sample_size="512×512",
    n_classes=12,
    modality="multispectral",
    resolution_m=10.0,
    volume_gb=25.0,
    tasks=["segmentation"],
    description="Urban land cover from Sentinel-2",
    download_url="https://example.com/mydataset",
    paper_url="https://arxiv.org/abs/...",
)
```

---

## Pull Request Guidelines

- Keep PRs focused — one feature or fix per PR.
- Add or update tests for any changed functionality.
- Update the relevant `.md` doc page.
- All CI checks must pass before merging.
- Write clear commit messages: `feat: add YOLOv10 to detection registry`.

---

## Commit Message Format

```
<type>: <short summary>

<body (optional)>
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`

---

## Reporting Bugs

Open a GitHub Issue with:
- Python version and OS
- PyGeoVision version (`python -c "import pygeovision; print(pygeovision.__version__)"`)
- Minimal reproducible example
- Full traceback
