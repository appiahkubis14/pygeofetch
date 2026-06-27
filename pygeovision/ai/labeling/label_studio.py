"""
Label Studio Integration.

Exports satellite imagery tiles to Label Studio for human annotation and
imports completed labels back into PyGeoVision. Supports image segmentation,
bounding box detection, and polygon annotation tasks.

Label Studio: https://labelstud.io/

Example:
    >>> from pygeovision.ai.labeling.label_studio import LabelStudioLabeler
    >>> labeler = LabelStudioLabeler(
    ...     url="http://localhost:8080",
    ...     api_key="your-api-key",
    ...     project_id=1,
    ... )
    >>> # Export tiles for annotation
    >>> labeler.export_tiles(tiles, task_type="segmentation")
    >>> # Import completed annotations as labels
    >>> results = labeler.label_tiles(tiles, output_dir="./labels/")
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pygeovision.ai.labeling.base_labeler import BaseLabeler, LabelingResult
from pygeovision.core.exceptions import LabelingError

logger = logging.getLogger(__name__)

# Label Studio annotation types
_TASK_TYPES = ("segmentation", "bbox_detection", "polygon", "classification")

# Default Label Studio config templates per task type
_LS_CONFIGS: Dict[str, str] = {
    "segmentation": """
<View>
  <Image name="image" value="$image"/>
  <BrushLabels name="tag" toName="image">
    {label_elements}
  </BrushLabels>
</View>
""",
    "bbox_detection": """
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    {label_elements}
  </RectangleLabels>
</View>
""",
    "polygon": """
<View>
  <Image name="image" value="$image"/>
  <PolygonLabels name="label" toName="image">
    {label_elements}
  </PolygonLabels>
</View>
""",
    "classification": """
<View>
  <Image name="image" value="$image"/>
  <Choices name="choice" toName="image" choice="single">
    {label_elements}
  </Choices>
</View>
""",
}


@dataclass
class LabelStudioConfig:
    """Configuration for the Label Studio integration.

    Attributes:
        url: Label Studio server URL (e.g. 'http://localhost:8080').
        api_key: Label Studio API key.
        project_id: Existing project ID, or None to create a new project.
        task_type: Annotation task type.
        class_names: List of class names for annotation labels.
        export_format: Format for exporting completed annotations.
        tile_export_dir: Directory to write tiles for LS consumption.
        image_format: Export image format ('PNG' or 'JPEG').
        poll_interval: Seconds between annotation completion polls.
        completion_threshold: Min fraction of tasks annotated before import.
        request_timeout: HTTP timeout in seconds.
    """

    url: str = "http://localhost:8080"
    api_key: str = ""
    project_id: Optional[int] = None
    task_type: str = "segmentation"
    class_names: List[str] = field(default_factory=lambda: ["background", "foreground"])
    export_format: str = "JSON"
    tile_export_dir: Optional[Path] = None
    image_format: str = "PNG"
    poll_interval: float = 30.0
    completion_threshold: float = 1.0
    request_timeout: int = 30


class LabelStudioLabeler(BaseLabeler):
    """Human-in-the-loop labeler via Label Studio integration.

    Exports satellite tiles to Label Studio for human annotation and
    imports completed labels back as GeoTIFF label masks.

    Workflow:
        1. ``export_tiles()`` — Upload tiles to Label Studio as annotation tasks.
        2. Annotators complete tasks in the Label Studio UI.
        3. ``label_tiles()`` — Download completed annotations and convert
           to pixel-aligned GeoTIFF label masks.

    Args:
        url: Label Studio server URL.
        api_key: Label Studio API key.
        project_id: ID of an existing Label Studio project.
            If None, a new project is created automatically.
        task_type: Annotation type: 'segmentation', 'bbox_detection',
            'polygon', or 'classification'.
        class_names: List of class names shown to annotators.
        tile_export_dir: Local directory for tile images exported to LS.
        num_workers: Number of parallel workers.
        skip_existing: Skip tiles that already have labels.

    Example:
        >>> labeler = LabelStudioLabeler(
        ...     url="http://localhost:8080",
        ...     api_key="abc123",
        ...     task_type="polygon",
        ...     class_names=["building", "road", "water"],
        ... )
        >>> labeler.export_tiles(tiles)
        >>> # After annotation in Label Studio UI...
        >>> results = labeler.label_tiles(tiles, "./labels/")
    """

    def __init__(
        self,
        url: str = "http://localhost:8080",
        api_key: str = "",
        project_id: Optional[int] = None,
        task_type: str = "segmentation",
        class_names: Optional[List[str]] = None,
        tile_export_dir: Optional[Path] = None,
        num_workers: int = 4,
        skip_existing: bool = True,
    ) -> None:
        super().__init__(
            labeler_name="label_studio",
            num_workers=num_workers,
            skip_existing=skip_existing,
        )
        if task_type not in _TASK_TYPES:
            raise ValueError(
                f"task_type must be one of {_TASK_TYPES}, got {task_type!r}"
            )
        if not api_key:
            raise LabelingError(
                "Label Studio API key is required. "
                "Find it at: Label Studio → Account & Settings → Access Token"
            )

        self.config = LabelStudioConfig(
            url=url.rstrip("/"),
            api_key=api_key,
            project_id=project_id,
            task_type=task_type,
            class_names=class_names or ["background", "foreground"],
            tile_export_dir=tile_export_dir
            or Path.home() / ".pygeovision" / "cache" / "label_studio_tiles",
        )
        self.config.tile_export_dir.mkdir(parents=True, exist_ok=True)
        self._session: Optional[Any] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def export_tiles(
        self,
        tiles: List[Any],
        project_title: str = "PyGeoVision Annotation",
        wait_for_completion: bool = False,
    ) -> int:
        """Export imagery tiles to Label Studio as annotation tasks.

        Creates a Label Studio project (if needed) and uploads tile images
        as tasks for human annotation.

        Args:
            tiles: List of TileMetadata objects or paths to tile GeoTIFFs.
            project_title: Title for a newly created project.
            wait_for_completion: If True, block until all tasks are annotated.

        Returns:
            Label Studio project ID.
        """
        import requests

        # Ensure project exists
        if self.config.project_id is None:
            self.config.project_id = self._create_project(project_title)
            logger.info("Created Label Studio project ID=%d", self.config.project_id)
        else:
            logger.info("Using existing Label Studio project ID=%d", self.config.project_id)

        # Export tiles and upload tasks
        uploaded = 0
        for tile in tiles:
            tile_path = tile if isinstance(tile, Path) else Path(getattr(tile, "path", str(tile)))
            img_path = self._export_tile_image(tile_path)
            task_id = self._upload_task(img_path, tile_path)
            if task_id:
                uploaded += 1

        logger.info(
            "Uploaded %d tiles to Label Studio project %d",
            uploaded, self.config.project_id
        )

        if wait_for_completion:
            self._wait_for_annotations()

        return self.config.project_id

    # ------------------------------------------------------------------
    # BaseLabeler abstract properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return 'label_studio'

    @property
    def supported_tasks(self) -> list:
        return ['segmentation', 'detection', 'classification']

    # ------------------------------------------------------------------
    # BaseLabeler interface
    # ------------------------------------------------------------------

    def label_tile(
        self,
        tile_path: Path,
        tile_metadata: Any,
        output_path: Path,
    ) -> LabelingResult:
        """Download completed Label Studio annotations and convert to GeoTIFF.

        Fetches the annotation for the tile from Label Studio and converts
        polygon/brush/bbox annotations to a pixel-aligned label mask.

        Args:
            tile_path: Path to the GeoTIFF tile.
            tile_metadata: TileMetadata with bounds, CRS, shape.
            output_path: Destination path for the label GeoTIFF.

        Returns:
            LabelingResult with annotation statistics.
        """
        try:
            import rasterio
        except ImportError as exc:
            raise LabelingError(
                "label_studio labeler requires rasterio. Install: pip install rasterio"
            ) from exc

        try:
            if self.config.project_id is None:
                raise LabelingError(
                    "No Label Studio project configured. "
                    "Call export_tiles() first or set project_id."
                )

            with rasterio.open(tile_path) as src:
                height, width = src.height, src.width
                crs = src.crs
                transform = src.transform

            # Find the task for this tile
            task = self._find_task_for_tile(tile_path)
            if task is None:
                logger.warning(
                    "No Label Studio task found for %s. "
                    "Run export_tiles() first.",
                    tile_path.name,
                )
                mask = np.zeros((height, width), dtype=np.uint8)
                self._write_label_geotiff(
                    mask, output_path,
                    {"driver": "GTiff", "dtype": "uint8", "width": width,
                     "height": height, "count": 1, "crs": crs,
                     "transform": transform, "compress": "lzw"}
                )
                return LabelingResult(
                    tile_path=tile_path, label_path=output_path,
                    success=False, labeler="label_studio",
                    error="no_task_found",
                )

            # Check annotation completeness
            annotations = task.get("annotations", [])
            if not annotations:
                logger.warning(
                    "Task for %s has no annotations yet.", tile_path.name
                )
                return LabelingResult(
                    tile_path=tile_path, label_path=output_path,
                    success=False, labeler="label_studio",
                    error="no_annotations",
                )

            # Convert the most recent annotation to a mask
            latest = annotations[-1]
            mask = self._annotation_to_mask(
                annotation=latest,
                height=height,
                width=width,
                task_type=self.config.task_type,
            )

            meta = {
                "driver": "GTiff",
                "dtype": "uint8",
                "width": width,
                "height": height,
                "count": 1,
                "crs": crs,
                "transform": transform,
                "compress": "lzw",
            }
            self._write_label_geotiff(mask, output_path, meta)

            unique = np.unique(mask)
            stats = {
                self.config.class_names[c] if c < len(self.config.class_names) else f"class_{c}":
                float(np.sum(mask == c)) / mask.size
                for c in unique
            }

            return LabelingResult(
                tile_path=tile_path,
                label_path=output_path,
                success=True,
                labeler="label_studio",
                class_distribution=stats,
                metadata={
                    "project_id": self.config.project_id,
                    "task_id": task.get("id"),
                    "annotation_id": latest.get("id"),
                    "annotator": latest.get("completed_by", {}).get("email", "unknown"),
                    "task_type": self.config.task_type,
                },
            )

        except Exception as exc:
            logger.error(
                "LabelStudioLabeler failed for %s: %s", tile_path, exc
            )
            return LabelingResult(
                tile_path=tile_path,
                label_path=output_path,
                success=False,
                labeler="label_studio",
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_session(self) -> Any:
        """Get or create an authenticated requests.Session."""
        import requests
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update(
                {"Authorization": f"Token {self.config.api_key}"}
            )
        return self._session

    def _api_get(self, endpoint: str) -> Any:
        """GET from the Label Studio API."""
        session = self._get_session()
        url = f"{self.config.url}/api/{endpoint}"
        resp = session.get(url, timeout=self.config.request_timeout)
        resp.raise_for_status()
        return resp.json()

    def _api_post(self, endpoint: str, data: Any = None, files: Any = None) -> Any:
        """POST to the Label Studio API."""
        session = self._get_session()
        url = f"{self.config.url}/api/{endpoint}"
        if files:
            resp = session.post(url, data=data, files=files,
                                timeout=self.config.request_timeout)
        else:
            resp = session.post(url, json=data, timeout=self.config.request_timeout)
        resp.raise_for_status()
        return resp.json()

    def _create_project(self, title: str) -> int:
        """Create a new Label Studio project.

        Args:
            title: Project title.

        Returns:
            New project ID.
        """
        label_elements = "\n    ".join(
            f'<Label value="{name}"/>' for name in self.config.class_names
        )
        label_config = _LS_CONFIGS[self.config.task_type].format(
            label_elements=label_elements
        )
        result = self._api_post("projects", {
            "title": title,
            "label_config": label_config,
        })
        return result["id"]

    def _export_tile_image(self, tile_path: Path) -> Path:
        """Convert a GeoTIFF tile to a PNG image for Label Studio.

        Args:
            tile_path: Path to the source GeoTIFF.

        Returns:
            Path to the exported PNG image.
        """
        import rasterio
        from PIL import Image

        out_path = self.config.tile_export_dir / (tile_path.stem + ".png")
        if out_path.exists():
            return out_path

        with rasterio.open(tile_path) as src:
            # Read up to 3 bands for RGB visualization
            n = min(src.count, 3)
            bands = src.read(list(range(1, n + 1))).astype(np.float32)

        # Normalize to uint8
        rgb_bands = []
        for band in bands:
            p2, p98 = np.percentile(band[band > 0], (2, 98)) if np.any(band > 0) else (0, 1)
            normalized = np.clip((band - p2) / max(p98 - p2, 1e-6) * 255, 0, 255).astype(np.uint8)
            rgb_bands.append(normalized)

        if len(rgb_bands) == 1:
            rgb_bands = rgb_bands * 3
        elif len(rgb_bands) == 2:
            rgb_bands.append(rgb_bands[0])

        rgb = np.stack(rgb_bands, axis=-1)
        Image.fromarray(rgb).save(out_path, format="PNG")
        return out_path

    def _upload_task(self, img_path: Path, tile_path: Path) -> Optional[int]:
        """Upload an image as a Label Studio task.

        Args:
            img_path: Path to the PNG image.
            tile_path: Original tile path (used as task metadata).

        Returns:
            Task ID, or None on failure.
        """
        try:
            with open(img_path, "rb") as fh:
                result = self._api_post(
                    f"projects/{self.config.project_id}/import",
                    files={"file": (img_path.name, fh, "image/png")},
                    data={"meta": json.dumps({"tile_path": str(tile_path)})},
                )
            # result is a list of created tasks
            if isinstance(result, list) and result:
                return result[0].get("id")
        except Exception as exc:
            logger.warning("Failed to upload task for %s: %s", img_path.name, exc)
        return None

    def _find_task_for_tile(self, tile_path: Path) -> Optional[Dict[str, Any]]:
        """Find the Label Studio task associated with a tile path.

        Args:
            tile_path: Path to the GeoTIFF tile.

        Returns:
            Task dict, or None if not found.
        """
        try:
            tasks = self._api_get(
                f"projects/{self.config.project_id}/tasks?page_size=1000"
            )
            task_list = tasks.get("tasks", tasks) if isinstance(tasks, dict) else tasks
            for task in task_list:
                meta = task.get("meta", {})
                if isinstance(meta, str):
                    try:
                        meta = json.loads(meta)
                    except json.JSONDecodeError:
                        meta = {}
                if str(tile_path) in str(meta.get("tile_path", "")):
                    return task
                # Also match by filename
                task_file = task.get("data", {}).get("image", "")
                if tile_path.stem in task_file:
                    return task
        except Exception as exc:
            logger.warning("Failed to find task for %s: %s", tile_path, exc)
        return None

    def _annotation_to_mask(
        self,
        annotation: Dict[str, Any],
        height: int,
        width: int,
        task_type: str,
    ) -> np.ndarray:
        """Convert a Label Studio annotation to a pixel mask.

        Args:
            annotation: Label Studio annotation dict with 'result' key.
            height: Mask height in pixels.
            width: Mask width in pixels.
            task_type: Annotation type for parsing strategy.

        Returns:
            uint8 label mask.
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        results = annotation.get("result", [])

        class_map = {
            name: idx for idx, name in enumerate(self.config.class_names)
        }

        for result in results:
            r_type = result.get("type", "")
            value = result.get("value", {})

            try:
                if r_type == "brushlabels":
                    mask = self._parse_brush_label(mask, value, class_map)
                elif r_type in ("rectanglelabels", "rectangle"):
                    mask = self._parse_bbox_label(mask, value, class_map, height, width)
                elif r_type in ("polygonlabels", "polygon"):
                    mask = self._parse_polygon_label(mask, value, class_map, height, width)
                elif r_type == "choices":
                    # Classification: fill entire tile with class ID
                    choices = value.get("choices", [])
                    if choices:
                        class_id = class_map.get(choices[0], 1)
                        mask[:] = class_id
            except Exception as exc:
                logger.warning("Failed to parse annotation result: %s", exc)

        return mask

    def _parse_brush_label(
        self,
        mask: np.ndarray,
        value: Dict[str, Any],
        class_map: Dict[str, int],
    ) -> np.ndarray:
        """Parse a brush (segmentation) annotation."""
        import base64
        import zlib
        from PIL import Image
        import io

        labels = value.get("brushlabels", [])
        class_id = class_map.get(labels[0], 1) if labels else 1

        rle = value.get("rle")
        if rle:
            # RLE-encoded mask
            decoded = base64.b64decode(rle)
            decompressed = zlib.decompress(decoded)
            brush_mask = np.frombuffer(decompressed, dtype=np.uint8)
            h = value.get("height", mask.shape[0])
            w = value.get("width", mask.shape[1])
            brush_mask = brush_mask.reshape(h, w)
            # Resize to tile dimensions if needed
            if brush_mask.shape != mask.shape:
                img = Image.fromarray(brush_mask)
                img = img.resize((mask.shape[1], mask.shape[0]), Image.NEAREST)
                brush_mask = np.array(img)
            mask[brush_mask > 0] = class_id
        return mask

    def _parse_bbox_label(
        self,
        mask: np.ndarray,
        value: Dict[str, Any],
        class_map: Dict[str, int],
        height: int,
        width: int,
    ) -> np.ndarray:
        """Parse a bounding box annotation."""
        labels = value.get("rectanglelabels", [])
        class_id = class_map.get(labels[0], 1) if labels else 1

        # LS uses percentage coordinates
        x = int(value.get("x", 0) / 100 * width)
        y = int(value.get("y", 0) / 100 * height)
        w = int(value.get("width", 0) / 100 * width)
        h = int(value.get("height", 0) / 100 * height)

        x2 = min(x + w, width)
        y2 = min(y + h, height)
        mask[y:y2, x:x2] = class_id
        return mask

    def _parse_polygon_label(
        self,
        mask: np.ndarray,
        value: Dict[str, Any],
        class_map: Dict[str, int],
        height: int,
        width: int,
    ) -> np.ndarray:
        """Parse a polygon annotation."""
        try:
            from PIL import ImageDraw, Image as PILImage
        except ImportError:
            return mask

        labels = value.get("polygonlabels", [])
        class_id = class_map.get(labels[0], 1) if labels else 1

        points = value.get("points", [])
        if len(points) < 3:
            return mask

        # Convert percentage coords to pixels
        pixel_points = [
            (int(p[0] / 100 * width), int(p[1] / 100 * height))
            for p in points
        ]

        img = PILImage.fromarray(mask)
        draw = ImageDraw.Draw(img)
        draw.polygon(pixel_points, fill=class_id)
        return np.array(img, dtype=np.uint8)

    def _wait_for_annotations(self) -> None:
        """Block until the completion threshold is reached."""
        if self.config.project_id is None:
            return
        logger.info("Waiting for Label Studio annotations (threshold=%.0f%%)…",
                    self.config.completion_threshold * 100)
        while True:
            try:
                stats = self._api_get(f"projects/{self.config.project_id}/")
                total = stats.get("task_number", 0)
                done = stats.get("num_tasks_with_annotations", 0)
                if total > 0:
                    frac = done / total
                    logger.info(
                        "Annotation progress: %d/%d (%.1f%%)", done, total, frac * 100
                    )
                    if frac >= self.config.completion_threshold:
                        logger.info("Annotation threshold reached.")
                        break
            except Exception as exc:
                logger.warning("Poll failed: %s", exc)
            time.sleep(self.config.poll_interval)
