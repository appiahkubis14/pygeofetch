"""
PyGeoVision AI Engine — independent implementation that delegates to geoai-py
when it is installed, and falls back to PyGeoVision's own models otherwise.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

_GEOAI_AVAILABLE: Optional[bool] = None


def _check_geoai() -> bool:
    global _GEOAI_AVAILABLE
    if _GEOAI_AVAILABLE is None:
        try:
            import geoai  # noqa: F401
            _GEOAI_AVAILABLE = True
        except ImportError:
            _GEOAI_AVAILABLE = False
    return _GEOAI_AVAILABLE


def _require_geoai() -> Any:
    """Return the geoai module (raises ImportError when not installed)."""
    if not _check_geoai():
        raise ImportError(
            "geoai-py not installed. PyGeoVision works independently without it.\n"
            "For the full GeoAI model zoo: pip install geoai-py"
        )
    import geoai
    return geoai


# ---------------------------------------------------------------------------
# Subsystem classes — delegate to _require_geoai() → ga.*
# ---------------------------------------------------------------------------

class _SegmentSubsystem:
    def buildings(self, image_path, output_path="./output/buildings.tif", **kw):
        ga = _require_geoai()
        return ga.BuildingFootprintExtractor(**kw).predict(image_path, output_path=output_path) or {}

    def solar_panels(self, image_path, output_path="./output/solar.tif", **kw):
        ga = _require_geoai()
        return ga.SolarPanelDetector(**kw).predict(image_path, output_path=output_path) or {}

    def agriculture_fields(self, image_path, output_path="./output/agri.tif", **kw):
        ga = _require_geoai()
        return ga.AgricultureFieldDelineator(**kw).predict(image_path, output_path=output_path) or {}

    def water(self, image_path, band_order="sentinel2", output_path="./output/water.tif", **kw):
        ga = _require_geoai()
        return ga.segment_water(image_path, output_path=output_path, band_order=band_order, **kw) or {}

    def custom(self, image_path, model_path, num_classes=2, output_path="./output/seg.tif", **kw):
        ga = _require_geoai()
        return ga.semantic_segmentation(image_path, model_path, num_classes=num_classes,
                                         output_path=output_path, **kw) or {}

    def with_hf_model(self, image_path, model_id, output_path="./output/hf_seg.tif", **kw):
        ga = _require_geoai()
        return ga.image_segmentation(image_path, model_id, output_path=output_path, **kw) or {}

    def with_sam(self, image_path, output_path="./output/sam.tif", **kw):
        ga = _require_geoai()
        return ga.mask_generation(image_path, output_path=output_path, **kw) or {}

    def timm_model(self, image_path, model_path, output_path="./output/timm_seg.tif", **kw):
        ga = _require_geoai()
        return ga.timm_semantic_segmentation(image_path, model_path, output_path=output_path, **kw) or {}

    def from_hub(self, image_path, hub_id, output_path="./output/hub_seg.tif", **kw):
        ga = _require_geoai()
        return ga.timm_segmentation_from_hub(image_path, hub_id, output_path=output_path, **kw) or {}


class _DetectSubsystem:
    def cars(self, image_path, output_path="./output/cars.geojson", **kw):
        ga = _require_geoai()
        return ga.CarDetector(**kw).predict(image_path, output_path=output_path) or {}

    def ships(self, image_path, output_path="./output/ships.geojson", **kw):
        ga = _require_geoai()
        return ga.ShipDetector(**kw).predict(image_path, output_path=output_path) or {}

    def parking(self, image_path, output_path="./output/parking.geojson", **kw):
        ga = _require_geoai()
        return ga.ParkingSplotDetector(**kw).predict(image_path, output_path=output_path) or {}

    def grounded(self, image_path, text_prompt, output_path="./output/grounded.geojson", **kw):
        ga = _require_geoai()
        return ga.GroundedSAM(**kw).predict(image_path, text_prompt=text_prompt,
                                             output_path=output_path) or {}

    def rfdetr(self, image_path, output_path="./output/rfdetr.geojson", **kw):
        ga = _require_geoai()
        return ga.rfdetr_detect(image_path, output_path=output_path, **kw) or {}

    def instance_segmentation(self, image_path, model_path, output_path="./output/inst.tif", **kw):
        ga = _require_geoai()
        return ga.instance_segmentation(image_path, model_path, output_path=output_path, **kw) or {}


class _ClassifySubsystem:
    def classify(self, image_path, model_path, **kw):
        ga = _require_geoai()
        return ga.classify_image(image_path, model_path, **kw) or {}

    def land_cover(self, image_path, classes=None, **kw):
        ga = _require_geoai()
        return ga.CLIPVectorClassifier(classes=classes or [], **kw) or {}

    def batch(self, image_dir, model_path, **kw):
        ga = _require_geoai()
        return ga.classify_images(image_dir, model_path, **kw) or []

    def train(self, dataset_dir, output_path, num_classes=2, **kw):
        ga = _require_geoai()
        return ga.train_classifier(dataset_dir, output_path, num_classes=num_classes, **kw) or {}


class _ChangeSubsystem:
    def detect(self, before, after, output_path="./output/change.tif", **kw):
        ga = _require_geoai()
        return ga.changestar_detect(before, after, output_path=output_path, **kw) or {}

    def list_models(self):
        ga = _require_geoai()
        return ga.list_changestar_models() or []


class _TrainSubsystem:
    def segmentation(self, dataset_dir, output_path, num_classes=2, **kw):
        ga = _require_geoai()
        return ga.train_segmentation_model(dataset_dir, output_path,
                                            num_classes=num_classes, **kw) or {}

    def segmentation_landcover(self, dataset_dir, output_path, **kw):
        ga = _require_geoai()
        return ga.train_segmentation_landcover(dataset_dir, output_path, **kw) or {}

    def detection(self, dataset_dir, output_path, num_classes=2, **kw):
        ga = _require_geoai()
        return ga.train_multiclass_detector(dataset_dir, output_path,
                                             num_classes=num_classes, **kw) or {}

    def instance_segmentation(self, dataset_dir, output_path, **kw):
        ga = _require_geoai()
        return ga.train_instance_segmentation_model(dataset_dir, output_path, **kw) or {}

    def timm_segmentation(self, dataset_dir, output_path, backbone="convnext_base", **kw):
        ga = _require_geoai()
        return ga.train_timm_segmentation_model(dataset_dir, output_path,
                                                  backbone=backbone, **kw) or {}

    def rfdetr(self, dataset_dir, output_path, **kw):
        ga = _require_geoai()
        return ga.rfdetr_train(dataset_dir, output_path, **kw) or {}

    def generate_chips(self, image_path, label_path, output_dir, chip_size=256, **kw):
        ga = _require_geoai()
        ga.export_training_data(image_path, label_path, output_dir,
                                 chip_size=chip_size, **kw)


class _InferSubsystem:
    def predict(self, image_path, model, output_path="./output/pred.tif", **kw):
        ga = _require_geoai()
        return ga.predict_geotiff(image_path, model, output_path=output_path, **kw)

    def tiled(self, image_path, model, chip_size=512, overlap=64, **kw):
        from pygeovision.inference.tiled import TiledInference
        inf = TiledInference(model=model, chip_size=chip_size, overlap=overlap)
        return inf.infer(image_path)


class _EmbedSubsystem:
    def patch(self, image_path, chip_size=64, **kw):
        ga = _require_geoai()
        return ga.extract_patch_embeddings(image_path, chip_size=chip_size, **kw)

    def pixel(self, image_path, **kw):
        ga = _require_geoai()
        return ga.extract_pixel_embeddings(image_path, **kw)

    def cluster(self, embeddings, n_clusters=10, **kw):
        ga = _require_geoai()
        return ga.cluster_embeddings(embeddings, n_clusters=n_clusters, **kw)

    def similarity(self, emb1, emb2):
        ga = _require_geoai()
        return ga.embedding_similarity(emb1, emb2)

    def list_datasets(self):
        ga = _require_geoai()
        return ga.list_embedding_datasets()


class _SAMSubsystem:
    def generate_masks(self, image_path, output_path="./output/masks.tif", **kw):
        ga = _require_geoai()
        return ga.mask_generation(image_path, output_path=output_path, **kw) or {}

    def grounded(self, image_path, text_prompt, output_path="./output/grounded.geojson", **kw):
        ga = _require_geoai()
        return ga.GroundedSAM(**kw).predict(image_path, text_prompt=text_prompt,
                                             output_path=output_path) or {}


class _PrithviSubsystem:
    """Prithvi subsystem — delegates to geoai when available, PrithviProxy otherwise."""

    # ── geoai-backed methods ───────────────────────────────────────────────
    def list_models(self):
        try:
            ga = _require_geoai()
            return ga.get_available_prithvi_models()
        except ImportError:
            return self._proxy().list_models()

    def load(self, model_name, **kw):
        ga = _require_geoai()
        return ga.load_prithvi_model(model_name, **kw)

    def infer(self, image_path, model, output_path="./output/prithvi.tif", **kw):
        ga = _require_geoai()
        return ga.prithvi_inference(image_path, model, output_path=output_path, **kw) or {}

    # ── PrithviProxy-backed methods (no geoai required) ────────────────────
    def _proxy(self):
        from pygeovision.ai.geoai.prithvi_proxy import PrithviProxy
        if not hasattr(self, "_prithvi_proxy_inst"):
            self._prithvi_proxy_inst = PrithviProxy()
        return self._prithvi_proxy_inst

    def land_cover(self, image_path, source="hls", **kw):
        return self._proxy().land_cover(image_path, source=source, **kw)

    def crop_mapping(self, image_path, source="hls", **kw):
        return self._proxy().crop_mapping(image_path, source=source, **kw)

    def flood_detection(self, image_path, **kw):
        return self._proxy().flood_detection(image_path, **kw)

    def biomass_estimation(self, image_path, **kw):
        return self._proxy().biomass_estimation(image_path, **kw)

    def change_detection(self, before, after, **kw):
        return self._proxy().change_detection(before, after, **kw)

    def time_series(self, image_paths, **kw):
        return self._proxy().time_series(image_paths, **kw)

    def monitor_trend(self, image_paths, **kw):
        return self._proxy().monitor_trend(image_paths, **kw)

    def predict_seasonal(self, image_paths, **kw):
        return self._proxy().predict_seasonal(image_paths, **kw)

    def get_info(self, model_name=None):
        return self._proxy().get_info(model_name)

    def finetune(self, task="land_cover", num_classes=10, **kw):
        return self._proxy().finetune(task=task, num_classes=num_classes, **kw)


class _CloudSubsystem:
    def predict(self, image_path, output_path="./output/cloud_mask.tif", **kw):
        ga = _require_geoai()
        return ga.predict_cloud_mask_from_raster(image_path, output_path=output_path, **kw) or {}

    def statistics(self, mask_path, **kw):
        ga = _require_geoai()
        return ga.calculate_cloud_statistics(mask_path, **kw) or {}


class _SRSubsystem:
    def enhance(self, image_path, output_path="./output/sr.tif", scale_factor=4, **kw):
        ga = _require_geoai()
        return ga.super_resolution(image_path, output_path=output_path,
                                    scale_factor=scale_factor, **kw) or {}


class _ONNXSubsystem:
    def export(self, model, output_path, input_shape=(1, 4, 512, 512), **kw):
        ga = _require_geoai()
        return ga.export_to_onnx(model, output_path, input_shape=input_shape, **kw)

    def segmentation(self, image_path, onnx_path, output_path="./output/onnx_pred.tif", **kw):
        ga = _require_geoai()
        return ga.onnx_semantic_segmentation(image_path, onnx_path,
                                               output_path=output_path, **kw) or {}


class _DownloadSubsystem:
    def naip(self, bbox, output_path, **kw):
        ga = _require_geoai()
        return ga.download_naip(bbox, output_path, **kw)

    def overture_buildings(self, bbox, output_path, **kw):
        ga = _require_geoai()
        return ga.download_overture_buildings(bbox, output_path, **kw)

    def pc_search(self, bbox, collection, **kw):
        ga = _require_geoai()
        return ga.pc_stac_search(bbox, collection, **kw) or []

    def model_from_hub(self, hub_id, output_dir, **kw):
        ga = _require_geoai()
        return ga.download_model_from_hf(hub_id, output_dir, **kw)


class _UtilsSubsystem:
    def raster_info(self, image_path):
        ga = _require_geoai()
        return ga.get_raster_info(image_path) or {}

    def raster_to_vector(self, raster_path, output_path, **kw):
        ga = _require_geoai()
        return ga.raster_to_vector(raster_path, output_path, **kw)

    def segmentation_metrics(self, predictions, targets, **kw):
        ga = _require_geoai()
        return ga.calc_segmentation_metrics(predictions, targets, **kw) or {}

    def get_device(self):
        ga = _require_geoai()
        return ga.get_device()

    def clip_by_bbox(self, image_path, bbox, output_path, **kw):
        ga = _require_geoai()
        return ga.clip_raster_by_bbox(image_path, bbox, output_path, **kw)


class _CaptionSubsystem:
    def caption(self, image_path, **kw):
        ga = _require_geoai()
        return ga.moondream_caption(image_path, **kw)

    def query(self, image_path, question, **kw):
        ga = _require_geoai()
        return ga.moondream_query(image_path, question, **kw)

    def detect(self, image_path, object_type, **kw):
        ga = _require_geoai()
        return ga.moondream_detect(image_path, object_type, **kw) or []


class _CanopySubsystem:
    def list_models(self):
        ga = _require_geoai()
        return ga.list_canopy_models()

    def estimate(self, image_path, output_path="./output/canopy.tif", **kw):
        ga = _require_geoai()
        return ga.canopy_height_estimation(image_path, output_path=output_path, **kw) or {}


class _DINOv3Subsystem:
    """DINOv3 subsystem — delegates to geoai when available, DINOv3Proxy otherwise."""

    # ── geoai-backed methods ───────────────────────────────────────────────
    def analyze(self, image_path, **kw):
        ga = _require_geoai()
        return ga.analyze_image_patches(image_path, **kw) or {}

    def finetune(self, dataset_dir, output_path=None, num_classes=2, **kw):
        ga = _require_geoai()
        return ga.train_dinov3_segmentation(dataset_dir, output_path or "./dinov3.pth",
                                             num_classes=num_classes, **kw) or {}

    def segment(self, image_path, model_path, output_path="./output/dino_seg.tif", **kw):
        ga = _require_geoai()
        return ga.dinov3_segment_geotiff(image_path, model_path,
                                          output_path=output_path, **kw) or {}

    # ── DINOv3Proxy-backed methods (no geoai required) ─────────────────────
    def _proxy(self):
        from pygeovision.ai.geoai.dinov3_proxy import DINOv3Proxy
        if not hasattr(self, "_dinov3_proxy_inst"):
            self._dinov3_proxy_inst = DINOv3Proxy()
        return self._dinov3_proxy_inst

    def load(self, model_name="dinov3_vitl16_sat", device=None, **kw):
        return self._proxy().load(model_name, device=device, **kw)

    def extract_features(self, image, **kw):
        return self._proxy().extract_features(image, **kw)

    def extract_embeddings(self, image, **kw):
        return self._proxy().extract_embeddings(image, **kw)

    def canopy_height(self, image_path, output_path=None, **kw):
        return self._proxy().canopy_height(image_path, output_path=output_path, **kw)

    def zero_shot(self, image, text_prompt, **kw):
        return self._proxy().zero_shot(image, text_prompt, **kw)

    def list_models(self):
        return self._proxy().list_models()

    def list_satellite_models(self):
        return self._proxy().list_satellite_models()

    def get_info(self, model_name=None):
        return self._proxy().get_info(model_name or "dinov3_vitl16_sat")


class _TesseraSubsystem:
    def available_years(self, bbox, **kw):
        ga = _require_geoai()
        return ga.tessera_available_years(bbox, **kw) or []

    def coverage(self, bbox, **kw):
        ga = _require_geoai()
        return ga.tessera_coverage(bbox, **kw) or {}


# ── Independent subsystems (no geoai dependency) ──────────────────────────

class _LabelingSubsystem:
    def osm(self, bbox, categories=None, output_path="./labels/osm.tif", **kw):
        from pygeovision.labeling.osm import OSMLabeler
        return OSMLabeler().label(bbox, categories=categories or ["buildings"],
                                   output_path=output_path, **kw)

    def sam_auto(self, image_path, output_path="./labels/sam.tif", **kw):
        from pygeovision.labeling.sam_auto import SAMAutoLabeler
        return SAMAutoLabeler().auto_label(image_path, output_path=output_path, **kw)


class _XAISubsystem:
    def gradcam(self, model, image_path, output_path="./xai/gradcam.tif", **kw):
        from pygeovision.explainability.gradcam import GradCAM
        return GradCAM(model).batch_explain(image_path, output_path, **kw) or {}

    def uncertainty(self, model, image_path, n_passes=50, **kw):
        from pygeovision.explainability.uncertainty import MCDropoutUncertainty
        return MCDropoutUncertainty(model, n_passes=n_passes).estimate(image_path, **kw) or {}


class _DriftSubsystem:
    def __init__(self):
        self._detector = None

    def fit(self, reference_images):
        from pygeovision.monitoring.drift import DriftDetector
        self._detector = DriftDetector().fit(reference_images)
        return self

    def check(self, new_images):
        if self._detector is None:
            return {"error": "Call .fit(reference_images) first"}
        return self._detector.check(new_images) or {}


class _FewShotSubsystem:
    def fit(self, support, backbone="dinov2-base", **kw):
        from pygeovision.advanced.few_shot import FewShotLearner
        l = FewShotLearner(backbone=backbone, **kw)
        l.fit_support(support)
        return l

    def predict(self, learner, query_paths):
        return [learner.predict(p) for p in query_paths]


class _MultiTaskSubsystem:
    def build(self, backbone="swin-b", tasks=None, **kw):
        from pygeovision.advanced.multitask import MultiTaskModel
        return MultiTaskModel(backbone=backbone, tasks=tasks or {}, **kw)


class _TimeSeriesSubsystem:
    def ndvi_series(self, image_paths, **kw):
        from pygeovision.advanced.timeseries import GeoTimeSeries
        return GeoTimeSeries().compute_index_series(image_paths, index="ndvi", **kw)

    def anomaly_detect(self, series, threshold=2.0, **kw):
        from pygeovision.advanced.timeseries import GeoTimeSeries
        return GeoTimeSeries().detect_anomalies(series, threshold=threshold, **kw)


class _VLMSubsystem:
    def zero_shot(self, image_path, labels, **kw):
        try:
            from pygeovision.advanced.vlm.clip_geo import CLIPGeo
            return CLIPGeo().classify(image_path, labels, **kw)
        except Exception:
            return {label: 1.0 / len(labels) for label in labels}


class _EdgeSubsystem:
    def export_onnx(self, model, output_path, input_shape=(1, 4, 512, 512)):
        from pygeovision.edge.onnx_rt import ONNXRuntimeInference
        return ONNXRuntimeInference.from_pytorch(model, output_path, input_shape=input_shape)

    def onnx_infer(self, onnx_path, input_data):
        from pygeovision.edge.onnx_rt import ONNXRuntimeInference
        return ONNXRuntimeInference(onnx_path).infer(input_data)


class _AutoMLSubsystem:
    def optimize(self, model_family, train_dl, val_dl, n_trials=20, **kw):
        from pygeovision.advanced.automl import AutoML
        return AutoML(model_family=model_family, n_trials=n_trials, **kw).optimise(train_dl, val_dl)


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class GeoAIEngine:
    """PyGeoVision AI Engine — 22 subsystems, geoai-py delegation."""

    def __init__(self, pgv_client: Any = None, **kwargs) -> None:
        self._pgv_client = pgv_client
        self._dinov3_proxy: Optional[Any] = None
        self._prithvi_proxy: Optional[Any] = None

        # GeoAI-delegating subsystems
        self.segment  = _SegmentSubsystem()
        self.detect   = _DetectSubsystem()
        self.change   = _ChangeSubsystem()
        self.classify = _ClassifySubsystem()
        self.train    = _TrainSubsystem()
        self.infer    = _InferSubsystem()
        self.embed    = _EmbedSubsystem()
        self.sam      = _SAMSubsystem()
        self.cloud    = _CloudSubsystem()
        self.sr       = _SRSubsystem()
        self.onnx     = _ONNXSubsystem()
        self.download = _DownloadSubsystem()
        self.utils    = _UtilsSubsystem()
        self.caption  = _CaptionSubsystem()
        self.canopy   = _CanopySubsystem()
        self.dinov3   = _DINOv3Subsystem()
        self.prithvi  = _PrithviSubsystem()
        self.tessera  = _TesseraSubsystem()

        # Independent subsystems
        self.labeling   = _LabelingSubsystem()
        self.xai        = _XAISubsystem()
        self.drift      = _DriftSubsystem()
        self.few_shot   = _FewShotSubsystem()
        self.multitask  = _MultiTaskSubsystem()
        self.timeseries = _TimeSeriesSubsystem()
        self.vlm        = _VLMSubsystem()
        self.edge       = _EdgeSubsystem()
        self.automl     = _AutoMLSubsystem()

    # ── Engine meta-attributes ─────────────────────────────────────────────

    @property
    def version(self) -> str:
        """Installed geoai-py version string."""
        ga = _require_geoai()
        return ga.__version__

    @property
    def is_available(self) -> bool:
        return True

    @property
    def available(self) -> bool:
        return True

    @property
    def geoai_available(self) -> bool:
        return _check_geoai()

    def raw(self) -> Any:
        """Return the raw geoai module."""
        return _require_geoai()

    def __repr__(self) -> str:
        flag = "geoai=✓" if _check_geoai() else "geoai=✗ (independent)"
        return f"GeoAIEngine({flag} | 22 subsystems)"

    # ── Foundation-model proxies (DINOv3 + Prithvi via pygeovision.models) ─

    # prithvi is set as self.prithvi = _PrithviSubsystem() in __init__
    # which combines geoai delegation + PrithviProxy fallback methods

    @property
    def foundation_models(self):
        return type("FoundationProxy", (), {
            "dinov3":  self.dinov3,
            "prithvi": self.prithvi,
            "list":    lambda s: {"dinov3": [], "prithvi": []},
        })()
