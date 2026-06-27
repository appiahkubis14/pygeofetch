"""AutoLabelPipeline — orchestrate multiple label sources with voting fusion."""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
logger = logging.getLogger(__name__)


class AutoLabelPipeline:
    """Orchestrate multiple auto-labeling sources and fuse with majority voting.

    Example::

        pipeline = AutoLabelPipeline(
            sources=["osm", "microsoft_buildings", "esa_worldcover"],
            fusion="majority_vote",
        )
        result = pipeline.run(
            bbox=(-74.05, 40.70, -73.95, 40.80),
            output_dir="./labels/nyc/",
        )
    """

    SOURCES = ["osm", "microsoft_buildings", "google_buildings",
               "esa_worldcover", "dynamic_world", "sam_auto", "foundation"]

    def __init__(
        self,
        sources: Optional[List[str]] = None,
        fusion: str = "majority_vote",  # majority_vote | union | intersection | priority
        quality_threshold: float = 0.6,
        reference_raster: Optional[str] = None,
    ) -> None:
        self.sources = sources or ["osm", "esa_worldcover"]
        self.fusion = fusion
        self.quality_threshold = quality_threshold
        self.reference_raster = reference_raster

    def run(
        self,
        bbox: Tuple[float, ...],
        output_dir: str = "./labels/",
        categories: Optional[List[str]] = None,
        assess_quality: bool = True,
    ) -> Dict[str, Any]:
        """Run all configured labeling sources and fuse results."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        partial_results: Dict[str, Dict] = {}

        for source in self.sources:
            try:
                if source == "osm":
                    from pygeovision.labeling.osm import OSMLabeler
                    r = OSMLabeler().label(
                        bbox, categories=categories or ["buildings", "roads", "water"],
                        output_path=f"{output_dir}/osm.tif",
                        reference_raster=self.reference_raster,
                    )
                elif source == "microsoft_buildings":
                    from pygeovision.labeling.buildings import MicrosoftBuildingsLabeler
                    r = MicrosoftBuildingsLabeler().label(
                        bbox, output_path=f"{output_dir}/ms_buildings.tif",
                        reference_raster=self.reference_raster,
                    )
                elif source == "google_buildings":
                    from pygeovision.labeling.buildings import GoogleBuildingsLabeler
                    r = GoogleBuildingsLabeler().label(
                        bbox, output_path=f"{output_dir}/google_buildings.tif",
                        reference_raster=self.reference_raster,
                    )
                elif source == "esa_worldcover":
                    from pygeovision.labeling.landcover import ESAWorldCoverLabeler
                    r = ESAWorldCoverLabeler().label(
                        bbox, output_path=f"{output_dir}/esa_worldcover.tif",
                    )
                elif source == "dynamic_world":
                    from pygeovision.labeling.landcover import DynamicWorldLabeler
                    r = DynamicWorldLabeler().label(
                        bbox, output_path=f"{output_dir}/dynamic_world.tif",
                    )
                else:
                    r = {"success": False, "error": f"Source '{source}' not implemented"}

                partial_results[source] = r
                logger.info("Source '%s': %s", source, "OK" if r.get("success") else r.get("error"))
            except Exception as exc:
                partial_results[source] = {"success": False, "error": str(exc)}
                logger.warning("Source '%s' failed: %s", source, exc)

        # Fuse successful results
        successful = [k for k, v in partial_results.items() if v.get("success")]
        label_paths = [partial_results[s]["output_path"] for s in successful if "output_path" in partial_results[s]]

        fused_path = None
        if len(label_paths) > 1:
            fused_path = self._fuse(label_paths, bbox, f"{output_dir}/fused_labels.tif")
        elif len(label_paths) == 1:
            fused_path = label_paths[0]

        result = {
            "success": len(successful) > 0,
            "sources_succeeded": successful,
            "sources_failed": [k for k in partial_results if k not in successful],
            "label_paths": label_paths,
            "fused_path": str(fused_path) if fused_path else None,
            "partial_results": partial_results,
        }

        if assess_quality and fused_path:
            from pygeovision.labeling.quality import LabelQualityAssessor
            qa = LabelQualityAssessor()
            result["quality"] = qa.assess(fused_path)

        return result

    def _fuse(self, paths: List[str], bbox: Tuple, output: str) -> Optional[str]:
        """Fuse multiple label rasters by majority voting."""
        try:
            import numpy as np, rasterio
            arrays = []
            ref_profile = None
            for p in paths:
                try:
                    with rasterio.open(p) as src:
                        arrays.append(src.read(1))
                        if ref_profile is None:
                            ref_profile = src.profile.copy()
                except Exception:
                    continue

            if not arrays or ref_profile is None:
                return None

            # Resize all to same shape
            ref_shape = arrays[0].shape
            resized = []
            for a in arrays:
                if a.shape != ref_shape:
                    import cv2
                    a = cv2.resize(a, (ref_shape[1], ref_shape[0]), interpolation=cv2.INTER_NEAREST)
                resized.append(a)

            stack = np.stack(resized, axis=0)

            if self.fusion == "majority_vote":
                from scipy import stats
                fused = stats.mode(stack, axis=0).mode[0].astype(np.uint8)
            elif self.fusion == "union":
                fused = (stack.max(axis=0) > 0).astype(np.uint8)
            elif self.fusion == "intersection":
                fused = (stack.min(axis=0) > 0).astype(np.uint8)
            elif self.fusion == "priority":
                # Priority: first source wins
                fused = resized[0].copy()
                for a in resized[1:]:
                    fused[fused == 0] = a[fused == 0]
            else:
                fused = stack[0]

            Path(output).parent.mkdir(parents=True, exist_ok=True)
            ref_profile.update(dtype="uint8", count=1)
            with rasterio.open(output, "w", **ref_profile) as dst:
                dst.write(fused[np.newaxis])
            return output
        except Exception as exc:
            logger.error("Fusion failed: %s", exc)
            return None
