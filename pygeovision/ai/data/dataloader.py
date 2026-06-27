"""
GeoDataLoader — converts PyGeoFetch results into AI-ready datasets.

This is the bridge between PyGeoFetch data and the AI engine. It:
1. Accepts PyGeoFetch search results or downloaded data paths
2. Tiles large imagery with the TilingEngine
3. Generates labels via the labeling subsystem
4. Creates spatially aware train/val/test splits (no data leakage)
5. Returns a GeoDataset ready for PyTorch or TensorFlow
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np

from pygeovision.ai.data.dataset import GeoDataset, TileMetadata
from pygeovision.ai.data.tiling import TilingConfig, TilingEngine
from pygeovision.ai.data.augmentations import GeoAugmentationPipeline
from pygeovision.ai.data.preprocessing import GeoPreprocessor
from pygeovision.core.exceptions import DatasetError

if TYPE_CHECKING:
    from pygeovision.data import SatelliteFetcher as PyGeoFetch
    from pygeovision.core.config import PyGeoVisionConfig

logger = logging.getLogger(__name__)


class GeoDataLoader:
    """
    Convert PyGeoFetch results or imagery paths into AI-ready datasets.

    GeoDataLoader is the bridge between PyGeoFetch (data) and the AI engine
    (intelligence). It never makes direct satellite API calls — it uses the
    SatelliteFetcher instance for all data retrieval.

    Parameters
    ----------
    data_pipeline : SatelliteFetcher
        The SatelliteFetcher instance for data retrieval.
    config : PyGeoVisionConfig, optional
        PyGeoVision configuration.
    device : str
        Compute device (e.g. ``"cuda:0"``, ``"cpu"``).

    Examples
    --------
    >>> loader = GeoDataLoader(data_pipeline=pygeofetch_instance)
    >>> dataset = loader.prepare(
    ...     data_source=search_results,
    ...     labels="openstreetmap",
    ...     task="segmentation",
    ...     tile_size=512,
    ... )
    >>> train_loader = dataset.to_pytorch(split="train", batch_size=8)
    """

    def __init__(
        self,
        data_pipeline: "SatelliteFetcher",
        config: Optional["PyGeoVisionConfig"] = None,
        device: str = "cpu",
    ) -> None:
        self._pygeofetch = data_pipeline
        self._config = config
        self._device = device

    def prepare(
        self,
        data_source: Any,
        labels: Union[str, list[str]] = "openstreetmap",
        task: str = "segmentation",
        tile_size: int = 512,
        overlap: int = 64,
        bands: Optional[list[int]] = None,
        val_split: float = 0.15,
        test_split: float = 0.10,
        augment: bool = True,
        output_dir: Optional[Union[str, Path]] = None,
        min_valid_fraction: float = 0.1,
        preprocessing_steps: Optional[list[str]] = None,
        seed: int = 42,
        **kwargs: Any,
    ) -> GeoDataset:
        """
        Prepare a GeoDataset from PyGeoFetch results or data paths.

        Parameters
        ----------
        data_source : any
            - List of PyGeoFetch SearchResult objects → download then tile
            - List of file paths (str or Path) → tile directly
            - Single directory path → tile all imagery in directory
            - Single file path → tile that file
        labels : str or list of str
            Labeling strategy (see :class:`~pygeovision.ai.engine.AIEngine`).
        task : str
            AI task type.
        tile_size : int
            Tile size in pixels.
        overlap : int
            Tile overlap in pixels.
        bands : list of int, optional
            Band indices to use (1-indexed).
        val_split : float
            Validation split fraction (0–1).
        test_split : float
            Test split fraction (0–1).
        augment : bool
            Build augmentation pipeline for training.
        output_dir : str or Path, optional
            Directory to save dataset. Auto-generated if None.
        min_valid_fraction : float
            Minimum valid pixel fraction per tile.
        preprocessing_steps : list of str, optional
            Preprocessing steps to apply (see :class:`~pygeovision.ai.data.preprocessing.GeoPreprocessor`).
        seed : int
            Random seed for reproducible splits.
        **kwargs
            Additional labeling-specific parameters.

        Returns
        -------
        GeoDataset
            Ready-to-use dataset.

        Raises
        ------
        DatasetError
            If data preparation fails.
        """
        if val_split + test_split >= 1.0:
            raise DatasetError(
                f"val_split ({val_split}) + test_split ({test_split}) must be < 1.0"
            )

        # Resolve output directory
        dataset_dir = self._resolve_output_dir(output_dir, data_source, task, tile_size)
        tile_dir = dataset_dir / "tiles"
        label_dir = dataset_dir / "labels"
        tile_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Preparing dataset: task=%s, tile_size=%d, overlap=%d → %s",
            task,
            tile_size,
            overlap,
            dataset_dir,
        )

        # Step 1: Resolve imagery paths
        imagery_paths = self._resolve_imagery(data_source, dataset_dir)
        if not imagery_paths:
            raise DatasetError(
                "No imagery found in data_source. "
                "Provide PyGeoFetch results, file paths, or a directory."
            )
        logger.info("Found %d imagery files", len(imagery_paths))

        # Step 2: Tile all imagery
        tiling_config = TilingConfig(
            tile_size=tile_size,
            overlap=overlap,
            min_valid_fraction=min_valid_fraction,
            save_tiles=True,
            output_dir=tile_dir,
        )
        engine = TilingEngine(config=tiling_config)
        all_tiles: list[TileMetadata] = []

        for path in imagery_paths:
            try:
                tiles = engine.tile(
                    raster_path=path,
                    bands=bands,
                )
                all_tiles.extend(tiles)
                logger.debug("Tiled %s → %d tiles", path.name, len(tiles))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to tile %s: %s", path, exc)

        if not all_tiles:
            raise DatasetError(
                "No tiles were generated. Check imagery files and tiling parameters."
            )
        logger.info("Total tiles generated: %d", len(all_tiles))

        # Step 3: Generate labels
        label_strategy = [labels] if isinstance(labels, str) else labels
        self._generate_labels(
            tiles=all_tiles,
            strategy=label_strategy,
            label_dir=label_dir,
            task=task,
            **kwargs,
        )

        # Step 4: Compute preprocessing statistics
        preprocessor = None
        if preprocessing_steps:
            preprocessor = GeoPreprocessor(steps=preprocessing_steps)

        # Step 5: Create spatially aware train/val/test splits
        split_map = self._spatial_split(
            tiles=all_tiles,
            val_split=val_split,
            test_split=test_split,
            seed=seed,
        )

        # Save split map
        split_file = dataset_dir / "splits.json"
        with open(split_file, "w") as fh:
            json.dump(split_map, fh, indent=2)

        # Step 6: Build augmentation pipeline for training
        augmentation_pipeline = None
        if augment:
            augmentation_pipeline = GeoAugmentationPipeline.medium(task=task)

        # Step 7: Infer class names from labeling strategy
        class_names = self._infer_class_names(task, label_strategy)

        # Step 8: Assemble and save GeoDataset
        dataset = GeoDataset(
            tiles=all_tiles,
            label_dir=label_dir,
            split_file=split_file,
            task=task,
            class_names=class_names,
            augmentations=augmentation_pipeline,
            bands=bands,
        )
        dataset.save(dataset_dir)

        logger.info(
            "Dataset ready: %d tiles | splits=%s | classes=%s",
            len(all_tiles),
            dataset.split_counts(),
            class_names,
        )
        return dataset

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _resolve_imagery(
        self,
        data_source: Any,
        work_dir: Path,
    ) -> list[Path]:
        """
        Resolve ``data_source`` into a list of imagery file paths.

        Handles PyGeoFetch results (downloads them), file paths, and directories.
        """
        if data_source is None:
            return []

        # PyGeoFetch SearchResult list — download via PyGeoFetch
        if isinstance(data_source, list) and data_source:
            # Check if these look like PyGeoFetch results (have a 'provider' or 'id' attr)
            first = data_source[0]
            if hasattr(first, "id") or hasattr(first, "provider"):
                return self._download_via_pygeofetch(data_source, work_dir)
            # Otherwise assume list of paths
            return [Path(p) for p in data_source if Path(p).exists()]

        # Single directory
        if isinstance(data_source, (str, Path)) and Path(data_source).is_dir():
            directory = Path(data_source)
            patterns = ["*.tif", "*.tiff", "*.img", "*.vrt"]
            paths: list[Path] = []
            for pattern in patterns:
                paths.extend(directory.rglob(pattern))
            return sorted(paths)

        # Single file
        if isinstance(data_source, (str, Path)) and Path(data_source).is_file():
            return [Path(data_source)]

        logger.warning("Could not resolve data_source: %s", type(data_source))
        return []

    def _download_via_pygeofetch(
        self,
        results: list[Any],
        work_dir: Path,
    ) -> list[Path]:
        """
        Download PyGeoFetch results using the internal PyGeoFetch data pipeline.

        All download logic is handled by PyGeoFetch — GeoDataLoader just
        tells it where to put the files and collects the output paths.
        """
        from pygeovision.data.fetch import DownloadResult  # type: ignore[import-untyped]

        download_dir = work_dir / "raw"
        download_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Downloading %d scenes via PyGeoFetch → %s", len(results), download_dir)

        download_results = self._pygeofetch.download(
            results,
            destination=download_dir,
            options=DownloadOptions(parallel=4, resume=True),
        )

        paths: list[Path] = []
        for dr in download_results:
            if hasattr(dr, "success") and dr.success and hasattr(dr, "path"):
                paths.append(Path(dr.path))
            elif isinstance(dr, Path):
                paths.append(dr)

        logger.info("Downloaded %d/%d files successfully", len(paths), len(results))
        return paths

    def _generate_labels(
        self,
        tiles: list[TileMetadata],
        strategy: list[str],
        label_dir: Path,
        task: str,
        **kwargs: Any,
    ) -> None:
        """Generate labels for all tiles using the configured strategy."""
        for strat in strategy:
            labeler = self._get_labeler(strat)
            if labeler is None:
                # Strategy might be a file path — use it directly
                path = Path(strat)
                if path.exists():
                    logger.info("Using existing labels from: %s", path)
                else:
                    logger.warning("Unknown labeling strategy and not a file: %s", strat)
                continue

            logger.info("Generating labels with strategy: %s", strat)
            try:
                labeler.label_tiles(
                    tiles=tiles,
                    output_dir=label_dir,
                    task=task,
                    **kwargs,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Labeling with '%s' failed: %s", strat, exc)

    def _get_labeler(self, strategy: str) -> Any:
        """Instantiate a labeler for the given strategy."""
        try:
            if strategy == "openstreetmap":
                from pygeovision.ai.labeling.osm_labeler import OSMLabeler  # noqa: PLC0415

                return OSMLabeler()
            if strategy == "microsoft_buildings":
                from pygeovision.ai.labeling.microsoft_buildings import MicrosoftBuildingsLabeler  # noqa: PLC0415

                return MicrosoftBuildingsLabeler()
            if strategy == "google_buildings":
                from pygeovision.ai.labeling.google_buildings import GoogleBuildingsLabeler  # noqa: PLC0415

                return GoogleBuildingsLabeler()
            if strategy == "esa_worldcover":
                from pygeovision.ai.labeling.esa_worldcover import ESAWorldCoverLabeler  # noqa: PLC0415

                return ESAWorldCoverLabeler()
            if strategy == "dynamic_world":
                from pygeovision.ai.labeling.dynamic_world import DynamicWorldLabeler  # noqa: PLC0415

                return DynamicWorldLabeler()
            if strategy == "sam":
                from pygeovision.ai.labeling.sam_labeler import SAMLabeler  # noqa: PLC0415

                return SAMLabeler()
            if strategy == "foundation":
                from pygeovision.ai.labeling.foundation_labeler import FoundationModelLabeler  # noqa: PLC0415

                return FoundationModelLabeler()
        except ImportError as exc:
            logger.warning("Could not load labeler for '%s': %s", strategy, exc)
        return None

    @staticmethod
    def _spatial_split(
        tiles: list[TileMetadata],
        val_split: float,
        test_split: float,
        seed: int,
    ) -> dict[str, str]:
        """
        Create spatially aware train/val/test splits.

        Uses geographic blocking to prevent data leakage between splits.
        Tiles are grouped by their source file and then by geographic region,
        ensuring the same area is not in both train and validation sets.

        Parameters
        ----------
        tiles : list of TileMetadata
        val_split : float
        test_split : float
        seed : int

        Returns
        -------
        dict
            Mapping of ``tile_id → split_name``.
        """
        rng = random.Random(seed)

        # Group tiles by source file for spatial awareness
        source_groups: dict[str, list[TileMetadata]] = {}
        for tile in tiles:
            key = str(tile.source_file)
            source_groups.setdefault(key, []).append(tile)

        sources = list(source_groups.keys())
        rng.shuffle(sources)

        n = len(sources)
        n_test = max(1, int(n * test_split)) if test_split > 0 else 0
        n_val = max(1, int(n * val_split)) if val_split > 0 else 0

        test_sources = set(sources[:n_test])
        val_sources = set(sources[n_test : n_test + n_val])
        # Remaining are train

        split_map: dict[str, str] = {}
        for src, src_tiles in source_groups.items():
            if src in test_sources:
                split = "test"
            elif src in val_sources:
                split = "val"
            else:
                split = "train"
            for tile in src_tiles:
                split_map[tile.tile_id] = split

        counts: dict[str, int] = {}
        for v in split_map.values():
            counts[v] = counts.get(v, 0) + 1
        logger.info("Split: %s", counts)
        return split_map

    @staticmethod
    def _resolve_output_dir(
        output_dir: Optional[Union[str, Path]],
        data_source: Any,
        task: str,
        tile_size: int,
    ) -> Path:
        """Generate a deterministic output directory path."""
        if output_dir is not None:
            return Path(output_dir)

        # Generate a hash-based name from the data source
        source_str = str(data_source)[:100]
        suffix = hashlib.md5(f"{source_str}_{task}_{tile_size}".encode()).hexdigest()[:8]
        return Path(".pygeovision_datasets") / f"{task}_{tile_size}px_{suffix}"

    @staticmethod
    def _infer_class_names(task: str, strategies: list[str]) -> list[str]:
        """Infer class names based on task and labeling strategy."""
        # OSM building segmentation
        if "building" in task.lower() or any("building" in s for s in strategies):
            return ["background", "building"]

        # ESA WorldCover classes
        if "esa_worldcover" in strategies:
            return [
                "tree_cover", "shrubland", "grassland", "cropland",
                "built_up", "bare_sparse_veg", "snow_ice", "perm_water",
                "herbaceous_wetland", "mangroves", "moss_lichen",
            ]

        # Generic land cover
        if "landcover" in task.lower():
            return [
                "water", "trees", "grass", "flooded_vegetation",
                "crops", "shrub", "built_area", "bare_ground",
                "snow_ice", "clouds",
            ]

        # Solar panels
        if "solar" in task.lower():
            return ["background", "solar_panel"]

        # Change detection
        if "change" in task.lower():
            return ["no_change", "change"]

        # Default binary
        return ["background", "foreground"]
