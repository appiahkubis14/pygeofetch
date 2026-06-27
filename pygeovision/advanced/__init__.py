"""
PyGeoVision Advanced AI Layer — few-shot, multi-task, AutoML, VLM, 3D, time series.
All independent of GeoAI.
"""
from pygeovision.advanced.few_shot   import FewShotLearner
from pygeovision.advanced.multitask  import MultiTaskLearner
from pygeovision.advanced.automl     import GeoAutoML

__all__ = ["FewShotLearner", "MultiTaskLearner", "GeoAutoML"]
