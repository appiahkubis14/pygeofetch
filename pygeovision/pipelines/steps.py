"""Pipeline step definitions."""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
logger = logging.getLogger(__name__)


@dataclass
class Step:
    name: str
    action: str
    params: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    retry_on_fail: int = 3
    timeout_s: int = 3600
    _fn: Optional[Callable] = field(default=None, repr=False)

    def run(self, context: Dict) -> Dict:
        """Execute this step."""
        if self._fn:
            return self._fn(context, **self.params)
        return {"step": self.name, "status": "no-op"}


@dataclass
class SearchStep(Step):
    action: str = "search"

    def run(self, context: Dict) -> Dict:
        client = context.get("client")
        if not client:
            return {"error": "No client in context"}
        return client.search(**self.params)


@dataclass
class DownloadStep(Step):
    action: str = "download"

    def run(self, context: Dict) -> Dict:
        client = context.get("client")
        results = context.get("search_results", [])
        if not client:
            return {"error": "No client in context"}
        return client.download(results, **self.params)


@dataclass
class InferStep(Step):
    action: str = "infer"

    def run(self, context: Dict) -> Dict:
        downloads = context.get("downloads", [])
        model = context.get("model")
        if not model:
            return {"note": "No model in context — using geoai subsystem"}
        from pygeovision.inference.tiled import TiledInference
        results = []
        for dl in downloads:
            path = dl.path if hasattr(dl, "path") else str(dl)
            out  = path.replace(".tif", "_pred.tif")
            inf  = TiledInference(model=model, **self.params)
            results.append(inf.infer(path, out))
        return {"inferences": results}


@dataclass
class ExportStep(Step):
    action: str = "export"

    def run(self, context: Dict) -> Dict:
        predictions = context.get("predictions", [])
        fmt = self.params.get("format", "geojson")
        out_dir = self.params.get("output_dir", "./output/")
        return {"exported": len(predictions), "format": fmt, "output_dir": out_dir}
