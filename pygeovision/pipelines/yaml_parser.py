"""YAML pipeline configuration parser and validator."""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
logger = logging.getLogger(__name__)


REQUIRED_FIELDS = {"name", "steps"}
VALID_ACTIONS = {
    "search", "download", "infer", "export", "label",
    "train", "evaluate", "alert", "notify", "custom",
}


class PipelineYAMLParser:
    """Parse and validate PyGeoVision YAML pipeline configurations.

    YAML format::

        name: building_extraction
        description: Extract building footprints from Sentinel-2
        schedule: "0 6 * * *"   # Run daily at 6am UTC
        steps:
          - name: search
            action: search
            params:
              bbox: [-74.1, 40.6, -73.7, 40.9]
              providers: [planetary_computer]
              date_range: ["2024-01-01", "2024-12-31"]
              cloud_cover_max: 15
          - name: download
            action: download
            depends_on: [search]
            params:
              output_dir: ./data/
              parallel: 4
          - name: infer
            action: infer
            depends_on: [download]
            params:
              model: unet-r50
              num_classes: 2
          - name: export
            action: export
            depends_on: [infer]
            params:
              format: geojson
              output_dir: ./results/

    Example::

        parser = PipelineYAMLParser()
        pipeline = parser.load("agriculture.yaml")
        parser.validate(pipeline)
    """

    def load(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Load and parse a YAML pipeline file."""
        try:
            import yaml
        except ImportError:
            raise ImportError("pip install pyyaml")
        with open(path) as f:
            config = yaml.safe_load(f)
        self.validate(config)
        return config

    def loads(self, yaml_str: str) -> Dict[str, Any]:
        """Parse a YAML pipeline string."""
        try:
            import yaml
            config = yaml.safe_load(yaml_str)
        except ImportError:
            raise ImportError("pip install pyyaml")
        self.validate(config)
        return config

    def validate(self, config: Dict) -> None:
        """Validate a pipeline configuration dict."""
        missing = REQUIRED_FIELDS - set(config.keys())
        if missing:
            raise ValueError(f"Pipeline config missing required fields: {missing}")

        steps = config.get("steps", [])
        if not steps:
            raise ValueError("Pipeline must have at least one step")

        step_names = set()
        for step in steps:
            if "name" not in step:
                raise ValueError("Each step must have a 'name' field")
            if "action" not in step:
                raise ValueError(f"Step '{step['name']}' missing 'action' field")
            if step["action"] not in VALID_ACTIONS:
                logger.warning("Step '%s' has unknown action '%s'",
                               step["name"], step["action"])
            step_names.add(step["name"])

        # Check dependency references
        for step in steps:
            for dep in step.get("depends_on", []):
                if dep not in step_names:
                    raise ValueError(f"Step '{step['name']}' depends on unknown step '{dep}'")

    def from_dict(self, config: Dict) -> "ParsedPipeline":
        from pygeovision.pipelines.orchestrator import Pipeline
        return Pipeline.from_config(config)

    def dump(self, config: Dict, path: Union[str, Path]) -> None:
        """Save a pipeline config to YAML."""
        try:
            import yaml
        except ImportError:
            raise ImportError("pip install pyyaml")
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
