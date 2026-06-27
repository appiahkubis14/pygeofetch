"""
PyGeoVision Auto-Labeling Layer — Phase 2 Independence.

Automated label generation from 7+ geospatial sources.
No GeoAI dependency.
"""
from pygeovision.labeling.osm         import OSMLabeler
from pygeovision.labeling.buildings   import MicrosoftBuildingsLabeler, GoogleBuildingsLabeler
from pygeovision.labeling.landcover   import ESAWorldCoverLabeler, DynamicWorldLabeler
from pygeovision.labeling.sam_auto    import SAMAutoLabeler
from pygeovision.labeling.foundation  import FoundationModelLabeler
from pygeovision.labeling.active      import ActiveLearner
from pygeovision.labeling.quality     import LabelQualityAssessor
from pygeovision.labeling.pipeline    import AutoLabelPipeline

__all__ = [
    "OSMLabeler", "MicrosoftBuildingsLabeler", "GoogleBuildingsLabeler",
    "ESAWorldCoverLabeler", "DynamicWorldLabeler",
    "SAMAutoLabeler", "FoundationModelLabeler",
    "ActiveLearner", "LabelQualityAssessor", "AutoLabelPipeline",
]
