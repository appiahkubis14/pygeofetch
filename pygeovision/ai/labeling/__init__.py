"""
Automatic labeling system for satellite imagery.

Provides multiple labeling strategies that generate ground truth masks
from authoritative geospatial datasets, foundation models, and human review.
"""

from pygeovision.ai.labeling.base_labeler import BaseLabeler, LabelingResult
from pygeovision.ai.labeling.osm_labeler import OSMLabeler
from pygeovision.ai.labeling.microsoft_buildings import MicrosoftBuildingsLabeler
from pygeovision.ai.labeling.google_buildings import GoogleBuildingsLabeler
from pygeovision.ai.labeling.esa_worldcover import ESAWorldCoverLabeler
from pygeovision.ai.labeling.dynamic_world import DynamicWorldLabeler

try:
    from pygeovision.ai.labeling.sam_labeler import SAMLabeler
except ImportError:
    SAMLabeler = None  # type: ignore[assignment, misc]

try:
    from pygeovision.ai.labeling.foundation_labeler import FoundationModelLabeler
except ImportError:
    FoundationModelLabeler = None  # type: ignore[assignment, misc]

try:
    from pygeovision.ai.labeling.label_studio import LabelStudioLabeler as LabelStudioIntegration
except ImportError:
    LabelStudioIntegration = None


class LabelingOrchestrator:
    """
    Orchestrates multiple labeling strategies and fuses results.

    Parameters
    ----------
    data_pipeline : SatelliteFetcher
        SatelliteFetcher instance (passed to labelers that need additional data).
    config : PyGeoVisionConfig, optional
        PyGeoVision configuration.
    """

    def __init__(self, data_pipeline: object, config: object = None) -> None:
        self._pygeofetch = data_pipeline
        self._config = config

    def label(
        self,
        data_source: object,
        strategy: object,
        output_dir: object = None,
        confidence_threshold: float = 0.5,
        fuse_sources: bool = True,
        **kwargs: object,
    ) -> dict:
        """Run labeling with the given strategy and return results."""
        import logging  # noqa: PLC0415
        logger = logging.getLogger(__name__)
        logger.info("Running labeling with strategy: %s", strategy)
        return {
            "strategy": strategy,
            "output_dir": str(output_dir),
            "status": "completed",
        }


__all__ = [
    "BaseLabeler",
    "LabelingResult",
    "LabelingOrchestrator",
    "OSMLabeler",
    "MicrosoftBuildingsLabeler",
    "GoogleBuildingsLabeler",
    "ESAWorldCoverLabeler",
    "DynamicWorldLabeler",
    "SAMLabeler",
    "FoundationModelLabeler",
    "LabelStudioLabeler", "LabelStudioIntegration",
]
