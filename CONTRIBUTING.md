# Contributing to PyGeoVision

Thank you for your interest in contributing to PyGeoVision! This guide covers all pathways for contributing.

## Development Setup

```bash
git clone https://github.com/pygeovision/pygeovision
cd pygeovision
pip install -e ".[dev,geoai,geo]"
pre-commit install
```

## Ways to Contribute

### 🐛 Bug Reports
Open an issue with:
- PyGeoVision version (`pygeovision status`)
- Python version and OS
- Minimal reproducible example
- Full traceback

### 🔧 Bug Fixes
1. Fork the repo
2. Create a branch: `git checkout -b fix/describe-the-bug`
3. Write a regression test in `tests/`
4. Fix the bug
5. Open a PR

### ✨ New Features
- Open a GitHub Discussion first to discuss the design
- Follow the existing code patterns
- Add tests for new functionality
- Update docs in `docs/`

### 📊 New Datasets
Add to `pygeovision/datasets/registry.py`:
```python
DatasetInfo(
    name="MyDataset",
    domain="urban",           # must be in DOMAINS list
    year=2024,
    n_samples=10000,
    sample_size="512×512",
    n_classes=5,
    modality="rgb",           # must be in MODALITIES list
    resolution_m=0.3,
    volume_gb=15.0,
    tasks=["segmentation"],
    description="Short description",
    download_url="https://...",
    paper_url="https://arxiv.org/...",
)
```

### 🤖 New Models
Add to `pygeovision/ai/models/zoo.py`:
```python
ModelSpec(
    name="my_model_b",
    task="segmentation",
    architecture="MyArch-B",
    backbone="my_backbone_base",
    pretrained_available=True,
    hf_model_id="username/my-model",
    params_m=85.0,
    description="My architecture description",
    tags=["transformer", "remote_sensing"],
)
```

### 🔄 New Pipelines
Add to `pygeovision/ai/pipelines/domains.py`:
```python
class MyDomainPipeline(BasePipeline):
    name = "my_domain_pipeline"
    description = "Description of what it does"
    domain = "urban"
    satellite = "sentinel-2"
    tags = ["urban", "segmentation"]

    def run(self, bbox, output_dir="./output", date="2024-06", **kwargs):
        # 1. Search via PyGeoFetch
        # 2. Download + post-process
        # 3. Run GeoAI model
        # 4. Return PipelineResult
        ...

# Register it
_PIPELINE_REGISTRY["my_domain_pipeline"] = MyDomainPipeline
```

## Code Standards

- **Formatting**: `black .` (line length 100)
- **Linting**: `ruff check .`
- **Type checking**: `mypy pygeovision/`
- **Tests**: `pytest tests/ -v --cov=pygeovision`
- **Docstrings**: Google-style docstrings

## PR Checklist

- [ ] Tests pass: `pytest tests/`
- [ ] Linting: `ruff check . && black --check .`
- [ ] Type annotations added
- [ ] Docstring on new public functions/classes
- [ ] CHANGELOG.md updated
- [ ] No breaking changes (or discussed in issue)

## Release Process

1. Update `_version.py`
2. Update `CHANGELOG.md`
3. Tag the release: `git tag v2.x.x`
4. CI publishes to PyPI automatically

## Community

- **GitHub Discussions**: Feature requests, questions, showcase
- **Issues**: Bug reports, specific problems
- **PRs**: Code contributions
