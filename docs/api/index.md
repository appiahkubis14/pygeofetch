# API Reference

Complete reference for all 18 PyGeoVision modules.

---

## Modules

| Module | Description | Key Classes |
|--------|-------------|-------------|
| [PyGeoVision Client](pygeovision.md) | Main client — search, download, pipeline | `PyGeoVision` |
| [Model Layer](models.md) | 119 architectures from the registry | `ModelRegistry`, `get_model()` |
| [Auto-Labeling](labeling.md) | 7 labeling sources + active learning | `OSMLabeler`, `SAMAutoLabeler` |
| [Loss Functions](losses.md) | 10 geospatial losses for training | `DiceLoss`, `FocalLoss`, `GeospatialMixedLoss` |
| [Inference Engine](inference.md) | Tiled, batch, streaming, ensemble | `TiledInference`, `BatchInferenceEngine` |
| [Explainability](explainability.md) | GradCAM, SHAP, uncertainty | `GradCAM`, `MCDropoutUncertainty` |
| [Monitoring](monitoring.md) | Drift detection + performance tracking | `DriftDetector`, `ModelPerformanceTracker` |
| [Training Framework](training.md) | GeoTrainer + distributed + mixed precision | `GeoTrainer`, `CheckpointManager` |
| [Serving API](serving.md) | FastAPI inference server | `InferenceServer`, `APIKeyAuth` |
| [Pipeline Orchestration](pipelines.md) | YAML workflows + scheduling | `Pipeline`, `PipelineOrchestrator` |
| [Dataset Registry](datasets.md) | 503-entry benchmark database | `DatasetRegistry`, `DatasetInfo` |
| [Foundation Models](foundation.md) | DINOv3 + Prithvi-EO-2.0 | `DINOv3Backbone`, `Prithvi`, `CHMv2Model` |
| [Edge Deployment](edge.md) | ONNX Runtime + Jetson TensorRT | `ONNXRuntimeInference`, `JetsonDeployer` |
| [Cloud Deployment](cloud.md) | AWS / Azure / GCP | `AWSDeployer`, `GCPDeployer` |
| [Advanced AI](advanced.md) | FewShot, MultiTask, AutoML | `FewShotLearner`, `AutoML` |
| [Vision-Language Models](vlm.md) | CLIP, Moondream, geo retrieval | `CLIPGeo`, `MoondreamGeo` |
| [Time Series](timeseries.md) | NDVI/NDWI trends + anomaly detection | `GeoTimeSeries` |
| [Point Cloud (3D)](pointcloud.md) | LiDAR, CHM, 3D segmentation | `LiDARProcessor` |
| [CLI Reference](cli.md) | 15 command groups | `pgv data`, `pgv models`, `pgv infer` |

---

## Import Paths

```python
# Main client
import pygeovision as pgv
client = pgv.PyGeoVision()

# Model layer
from pygeovision.models import get_model, model_registry

# Foundation models
from pygeovision.models.foundation.dinov3 import DINOv3Backbone, CHMv2Model, DINOv3Text
from pygeovision.models.foundation.prithvi import Prithvi, PrithviMultiTemporal, PrithviTasks

# Inference
from pygeovision.inference.tiled import TiledInference
from pygeovision.inference.batch import BatchInferenceEngine

# Training
from pygeovision.training.trainer    import GeoTrainer
from pygeovision.training.callbacks  import EarlyStopping, ModelCheckpoint
from pygeovision.training.checkpoint import CheckpointManager

# Losses
from pygeovision.losses.segmentation import DiceLoss, FocalLoss, TverskyLoss

# Serving
from pygeovision.serving import InferenceServer, create_app

# Pipelines
from pygeovision.pipelines import Pipeline, PipelineOrchestrator

# Dataset registry
from pygeovision.datasets.registry import dataset_registry

# Edge / Cloud
from pygeovision.edge.onnx_rt import ONNXRuntimeInference
from pygeovision.cloud.deploy  import AWSDeployer, GCPDeployer
```
