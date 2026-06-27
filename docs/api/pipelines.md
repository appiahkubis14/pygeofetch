# Pipeline Orchestration

YAML-based workflow orchestration for end-to-end geospatial AI pipelines — from satellite search to exported results. Supports dependency management, cron scheduling, retry logic, and parallel execution.

---

## YAML Pipeline Format

```yaml
name: building_extraction
description: "Extract building footprints from Sentinel-2 imagery"
version: "2.0"

defaults:
  cloud_cover_max: 10
  resolution_m: 10
  output_dir: ./output/buildings/

steps:
  - name: search
    action: search
    params:
      bbox: [-74.1, 40.6, -73.7, 40.9]
      providers: [planetary_computer]
      date_range: ["2024-06-01", "2024-08-31"]
      cloud_cover_max: 10

  - name: download
    action: download
    depends_on: [search]
    retry_on_fail: 3
    params:
      output_dir: ./data/
      parallel: 4
      post_process: [reproject:EPSG:32618, cog]

  - name: label
    action: label
    depends_on: [download]
    params:
      source: microsoft_buildings
      output_dir: ./labels/

  - name: infer
    action: infer
    depends_on: [download]
    params:
      model: segformer-b2
      num_classes: 2
      chip_size: 512
      overlap: 64
      blend_mode: gaussian

  - name: export
    action: export
    depends_on: [infer]
    params:
      format: geojson
      output_dir: ./output/buildings/
      simplify_tolerance: 0.5
```

---

## `Pipeline`

Load and run pipelines programmatically.

```python
from pygeovision.pipelines import Pipeline

# Load from YAML file
p = Pipeline.from_yaml("building_extraction.yaml")
print(p)
# Pipeline(name='building_extraction', steps=5)

# Dry run — validate steps without executing
result = p.run(dry_run=True)
print(f"Steps would execute: {result.steps_completed}")

# Full run
result = p.run(context={"client": pgv_client})
print(f"Success: {result.success}")
print(f"Steps completed: {result.steps_completed}")
print(f"Steps failed: {result.steps_failed}")
print(f"Duration: {result.duration_s:.1f}s")
```

### Build Programmatically

```python
from pygeovision.pipelines import Pipeline
from pygeovision.pipelines.steps import SearchStep, DownloadStep, InferStep, ExportStep

p = Pipeline("my_pipeline", description="Custom workflow")

p.add(SearchStep("search", params={
    "bbox":       [-74.1, 40.6, -73.7, 40.9],
    "providers":  ["planetary_computer"],
    "cloud_cover_max": 10,
}))

p.add(DownloadStep("download",
    depends_on=["search"],
    params={"output_dir": "./data/"},
    retry_on_fail=3,
))

p.add(InferStep("infer",
    depends_on=["download"],
    params={"model": "segformer-b2", "num_classes": 7},
))

p.add(ExportStep("export",
    depends_on=["infer"],
    params={"format": "geojson", "output_dir": "./results/"},
))

# Add event hooks
p.on("before_step", lambda step, ctx: print(f"Starting: {step.name}"))
p.on("on_error",    lambda step, ctx, exc: print(f"Failed: {step.name} — {exc}"))

result = p.run()
```

---

## `PipelineOrchestrator`

Manage and run multiple named pipelines in one place.

```python
from pygeovision.pipelines import PipelineOrchestrator, Pipeline

orch = PipelineOrchestrator(client=pgv_client)

# Register from YAML files
orch.register_yaml("./pipelines/agriculture.yaml")
orch.register_yaml("./pipelines/urban.yaml")
orch.register_yaml("./pipelines/flood.yaml")

# Or register Pipeline objects
orch.register("custom", Pipeline.from_yaml("custom.yaml"))

print("Available:", orch.list())
# ['agriculture', 'urban', 'flood', 'custom']

# Run by name
result = orch.run("agriculture")
result = orch.run("urban", context={"bbox": [-74.1, 40.6, -73.7, 40.9]})
```

---

## Pipeline Scheduler

Schedule pipelines on cron expressions for automated, recurring workflows.

```python
from pygeovision.pipelines.scheduler import PipelineScheduler

scheduler = PipelineScheduler()

# Add jobs with cron expressions
scheduler.add_job(
    name="daily_flood_check",
    fn=lambda: orch.run("flood"),
    cron="0 6 * * *",       # Every day at 06:00 UTC
)

scheduler.add_job(
    name="weekly_land_cover",
    fn=lambda: orch.run("land_cover"),
    cron="0 2 * * 1",       # Every Monday at 02:00 UTC
)

# Start the background scheduler thread
scheduler.start()

# Check job status
for job in scheduler.status():
    print(f"{job['name']}: {job['n_runs']} runs, {job['errors']} errors")

# Trigger a job immediately
scheduler.run_now("daily_flood_check")

# Stop
scheduler.stop()
```

---

## YAML Parser

Validate and inspect pipeline configs before execution.

```python
from pygeovision.pipelines.yaml_parser import PipelineYAMLParser

parser = PipelineYAMLParser()

# Load and validate
config = parser.load("agriculture.yaml")

# Validate a dict (useful when building configs dynamically)
parser.validate(config)

# Parse a YAML string
config = parser.loads("""
name: quick_test
steps:
  - name: search
    action: search
    params:
      bbox: [-74.1, 40.6, -73.7, 40.9]
""")

# Save to file
parser.dump(config, "output_config.yaml")
```

---

## Built-In Templates

Six ready-to-run pipeline templates are included:

| Template | Domain | Key Steps |
|----------|--------|-----------|
| `agriculture.yaml` | Agriculture | Search → Download → Infer (crop type) → Export |
| `forestry.yaml` | Forestry | Search → Download → Canopy height → Deforestation |
| `urban.yaml` | Urban | Search → Download → Building extraction → Export |
| `water.yaml` | Water | Search → Download → Flood mapping → Alert |
| `disaster.yaml` | Disaster | Pre/post search → Change detection → Damage map |
| `climate.yaml` | Climate | Time series → Carbon estimation → Report |

```python
# Use a template
p = Pipeline.from_yaml("pygeovision/pipelines/templates/agriculture.yaml")
p.run(context={"client": client, "bbox": [8.5, 47.3, 8.7, 47.4]})
```
