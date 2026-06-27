"""
ESA WorldCover + Google Dynamic World Auto-Labelers (E1, E2).
Free global land cover at 10m resolution.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
logger = logging.getLogger(__name__)

# ESA WorldCover 2021 class map (10m, 11 classes)
ESA_WORLDCOVER_CLASSES = {
    10: "Tree cover", 20: "Shrubland", 30: "Grassland",
    40: "Cropland", 50: "Built-up", 60: "Bare/sparse vegetation",
    70: "Snow and ice", 80: "Permanent water bodies",
    90: "Herbaceous wetland", 95: "Mangroves", 100: "Moss and lichen",
}

# Google Dynamic World 9-class map (10m, near real-time)
DW_CLASSES = {
    0: "water", 1: "trees", 2: "grass", 3: "flooded_vegetation",
    4: "crops", 5: "shrub_and_scrub", 6: "built", 7: "bare", 8: "snow_and_ice",
}


class ESAWorldCoverLabeler:
    """Generate land cover labels from ESA WorldCover 2020 or 2021.

    Free, open-access global land cover at 10m resolution.
    11 classes covering all land surface types globally.
    Data available via Microsoft Planetary Computer STAC.

    Example::

        labeler = ESAWorldCoverLabeler(year=2021)
        result = labeler.label(
            bbox=(-87.7, 41.8, -87.5, 41.9),   # Chicago
            output_path="./labels/chicago_lc.tif",
        )
    """

    COLLECTION = "esa-worldcover"
    PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"

    def __init__(self, year: int = 2021) -> None:
        if year not in (2020, 2021):
            raise ValueError("ESA WorldCover available for year=2020 or 2021")
        self.year = year

    def label(
        self,
        bbox: Tuple[float, ...],
        output_path: Union[str, Path] = "./labels/esa_worldcover.tif",
        remap_classes: Optional[Dict[int, int]] = None,
        clip_to_bbox: bool = True,
    ) -> Dict[str, Any]:
        """Download and clip ESA WorldCover tiles for the bbox.

        Args:
            bbox: (lon_min, lat_min, lon_max, lat_max)
            output_path: Output label GeoTIFF
            remap_classes: Optional dict to remap ESA class values (e.g. {50: 1} for binary urban)
            clip_to_bbox: Clip output to exact bbox extent

        Returns:
            Dict with output_path, classes_found, n_classes
        """
        output_path = Path(output_path)
        lon_min, lat_min, lon_max, lat_max = bbox

        try:
            import requests
        except ImportError:
            raise ImportError("pip install requests")

        # Search PC STAC for WorldCover tiles
        search_url = f"{self.PC_STAC}/search"
        params = {
            "collections": [self.COLLECTION],
            "bbox": [lon_min, lat_min, lon_max, lat_max],
            "limit": 10,
        }
        try:
            resp = requests.post(search_url, json=params, timeout=30)
            items = resp.json().get("features", []) if resp.status_code == 200 else []
        except Exception as exc:
            logger.error("ESA WorldCover STAC search failed: %s", exc)
            items = []

        if not items:
            return {"success": False, "error": "No ESA WorldCover tiles found — check bbox coverage"}

        # Download and mosaic tiles
        tile_paths = []
        for item in items[:4]:
            asset_url = item.get("assets", {}).get("map", {}).get("href", "")
            if not asset_url:
                continue
            try:
                import pystac_client, planetary_computer as pc
                signed_url = pc.sign(asset_url)
                tile_path = output_path.parent / f"_esa_tile_{item['id']}.tif"
                _download_cog_window(signed_url, tile_path, bbox)
                tile_paths.append(str(tile_path))
            except ImportError:
                # Fallback: direct HTTPS download
                try:
                    r = requests.get(asset_url, stream=True, timeout=60)
                    tile_path = output_path.parent / f"_esa_tile_{item['id']}.tif"
                    tile_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(tile_path, "wb") as f:
                        for chunk in r.iter_content(8192):
                            f.write(chunk)
                    tile_paths.append(str(tile_path))
                except Exception as exc2:
                    logger.warning("ESA tile download failed: %s", exc2)

        # Merge tiles and optionally remap
        result_path = _merge_and_clip(tile_paths, bbox, output_path, remap_classes, clip_to_bbox)
        # Cleanup temp tiles
        for tp in tile_paths:
            try:
                Path(tp).unlink()
            except Exception:
                pass

        classes_found = self._count_classes(str(result_path))
        return {
            "success": bool(result_path),
            "output_path": str(output_path),
            "year": self.year,
            "classes_found": classes_found,
            "class_map": ESA_WORLDCOVER_CLASSES,
        }

    def _count_classes(self, path: str) -> Dict[int, int]:
        try:
            import rasterio, numpy as np
            with rasterio.open(path) as src:
                data = src.read(1)
            unique, counts = np.unique(data, return_counts=True)
            return {int(k): int(v) for k, v in zip(unique, counts)}
        except Exception:
            return {}

    @staticmethod
    def class_map() -> Dict[int, str]:
        return ESA_WORLDCOVER_CLASSES


class DynamicWorldLabeler:
    """Generate near real-time land cover labels from Google Dynamic World.

    10m resolution, 9 classes, near real-time (within 3 days of Sentinel-2).
    Available via Google Earth Engine or planetary_computer.

    Example::

        labeler = DynamicWorldLabeler()
        result = labeler.label(
            bbox=(-0.15, 51.47, -0.10, 51.52),
            date_range=("2024-06-01", "2024-08-31"),
            output_path="./labels/london_dw.tif",
        )
    """

    EE_COLLECTION = "GOOGLE/DYNAMICWORLD/V1"
    PC_COLLECTION = "io-lulc-9-class"

    def __init__(self, backend: str = "planetary_computer") -> None:
        self.backend = backend

    def label(
        self,
        bbox: Tuple[float, ...],
        date_range: Tuple[str, str] = ("2024-01-01", "2024-12-31"),
        output_path: Union[str, Path] = "./labels/dynamic_world.tif",
        label_mode: str = "most_likely",  # most_likely | all_probabilities
    ) -> Dict[str, Any]:
        """Generate Dynamic World land cover label for a time period.

        Args:
            bbox: (lon_min, lat_min, lon_max, lat_max)
            date_range: (start, end) date strings
            output_path: Output label GeoTIFF
            label_mode: "most_likely" for single-band argmax, or "all_probabilities" for 9-band softmax

        Returns:
            Dict with output_path, n_classes, class_map
        """
        output_path = Path(output_path)

        if self.backend == "planetary_computer":
            return self._label_via_pc(bbox, date_range, output_path, label_mode)
        elif self.backend == "earth_engine":
            return self._label_via_ee(bbox, date_range, output_path)
        else:
            return {"success": False, "error": f"Unknown backend: {self.backend}"}

    def _label_via_pc(self, bbox, date_range, output_path, label_mode):
        try:
            import requests
            lon_min, lat_min, lon_max, lat_max = bbox
            params = {
                "collections": ["io-lulc-9-class"],
                "bbox": [lon_min, lat_min, lon_max, lat_max],
                "datetime": f"{date_range[0]}/{date_range[1]}",
                "limit": 5,
            }
            resp = requests.post(
                "https://planetarycomputer.microsoft.com/api/stac/v1/search",
                json=params, timeout=30,
            )
            items = resp.json().get("features", []) if resp.status_code == 200 else []
            if not items:
                return {"success": False, "error": "No Dynamic World data found for bbox+date"}

            # Use most recent item
            item = items[0]
            asset_url = item["assets"].get("data", {}).get("href", "")
            if not asset_url:
                return {"success": False, "error": "No data asset in Dynamic World STAC item"}

            output_path.parent.mkdir(parents=True, exist_ok=True)
            r = requests.get(asset_url, stream=True, timeout=60)
            with open(output_path, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            return {
                "success": True, "output_path": str(output_path),
                "class_map": DW_CLASSES, "n_classes": len(DW_CLASSES),
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def _label_via_ee(self, bbox, date_range, output_path):
        try:
            import ee
            lon_min, lat_min, lon_max, lat_max = bbox
            region = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])
            col = (ee.ImageCollection(self.EE_COLLECTION)
                   .filterBounds(region)
                   .filterDate(*date_range))
            label = col.select("label").mode().clip(region)
            task = ee.batch.Export.image.toDrive(
                image=label, description="dynamic_world_label",
                scale=10, region=region,
            )
            task.start()
            return {"success": True, "output_path": "See Google Drive", "ee_task": str(task.id)}
        except ImportError:
            return {"success": False, "error": "pip install earthengine-api"}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    @staticmethod
    def class_map() -> Dict[int, str]:
        return DW_CLASSES


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _download_cog_window(url: str, output: Path, bbox: Tuple) -> None:
    """Download a window of a COG GeoTIFF from a URL."""
    try:
        import rasterio
        from rasterio.windows import from_bounds
        output.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(url) as src:
            window = from_bounds(*bbox, transform=src.transform)
            data = src.read(window=window)
            transform = rasterio.transform.from_bounds(*bbox, data.shape[-1], data.shape[-2])
            with rasterio.open(str(output), "w", driver="GTiff",
                                height=data.shape[-2], width=data.shape[-1],
                                count=src.count, dtype=data.dtype,
                                crs=src.crs, transform=transform) as dst:
                dst.write(data)
    except Exception as exc:
        logger.warning("COG window download failed: %s", exc)


def _merge_and_clip(
    tile_paths: List[str],
    bbox: Tuple,
    output: Path,
    remap: Optional[Dict[int, int]],
    clip: bool,
) -> Optional[Path]:
    if not tile_paths:
        return None
    try:
        import numpy as np
        import rasterio
        from rasterio.merge import merge
        from rasterio.mask import mask as rio_mask
        from shapely.geometry import box
        import json

        sources = [rasterio.open(p) for p in tile_paths]
        mosaic, transform = merge(sources)
        for s in sources:
            s.close()

        if remap:
            for src_val, dst_val in remap.items():
                mosaic[mosaic == src_val] = dst_val

        profile = sources[0].profile
        profile.update(height=mosaic.shape[1], width=mosaic.shape[2],
                        transform=transform, compress="lzw", count=1)
        output.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(str(output), "w", **profile) as dst:
            dst.write(mosaic)

        if clip:
            lon_min, lat_min, lon_max, lat_max = bbox
            shapes = [json.loads(box(lon_min, lat_min, lon_max, lat_max).to_json())]
            with rasterio.open(str(output)) as src:
                clipped, clip_transform = rio_mask(src, shapes, crop=True)
                clip_profile = src.profile.copy()
            clip_profile.update(height=clipped.shape[1], width=clipped.shape[2], transform=clip_transform)
            with rasterio.open(str(output), "w", **clip_profile) as dst:
                dst.write(clipped)

        return output
    except Exception as exc:
        logger.error("Merge failed: %s", exc)
        return None
