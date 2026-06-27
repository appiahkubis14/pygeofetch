"""
Geospatial dataset classes for PyTorch and TensorFlow.

GeoDataset and TileDataset wrap tiled satellite imagery with labels and
expose standard ML dataset interfaces, while preserving all geospatial
metadata (CRS, transform, bounds, tile coordinates).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# Optional: PyTorch
try:
    import torch
    from torch.utils.data import Dataset as TorchDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    TorchDataset = object  # type: ignore[assignment, misc]

# Optional: TensorFlow
try:
    import tensorflow as tf

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


@dataclass
class TileMetadata:
    """
    Geospatial metadata for a single imagery tile.

    Attributes
    ----------
    tile_id : str
        Unique tile identifier.
    source_file : Path
        Path to the source raster file.
    bounds : tuple of float
        Tile bounds as (min_x, min_y, max_x, max_y) in the native CRS.
    crs : str
        Coordinate reference system as an EPSG string (e.g. ``"EPSG:4326"``).
    transform : list of float
        Affine transform coefficients (6 values).
    row_off : int
        Pixel row offset within source raster.
    col_off : int
        Pixel column offset within source raster.
    height : int
        Tile height in pixels.
    width : int
        Tile width in pixels.
    bands : list of int
        Band indices used (1-indexed).
    resolution_m : float
        Ground sampling distance in metres.
    overlap : int
        Overlap with adjacent tiles in pixels.
    provider : str
        Satellite data provider ID.
    satellite : str
        Satellite name.
    datetime : str
        Acquisition datetime (ISO 8601).
    cloud_cover : float
        Cloud cover percentage (0-100).
    extra : dict
        Additional provider-specific metadata.
    """

    tile_id: str
    source_file: Path
    bounds: tuple[float, float, float, float]
    crs: str
    transform: list[float]
    row_off: int
    col_off: int
    height: int
    width: int
    bands: list[int]
    resolution_m: float = 10.0
    overlap: int = 0
    provider: str = ""
    satellite: str = ""
    datetime: str = ""
    cloud_cover: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        return {
            "tile_id": self.tile_id,
            "source_file": str(self.source_file),
            "bounds": list(self.bounds),
            "crs": self.crs,
            "transform": self.transform,
            "row_off": self.row_off,
            "col_off": self.col_off,
            "height": self.height,
            "width": self.width,
            "bands": self.bands,
            "resolution_m": self.resolution_m,
            "overlap": self.overlap,
            "provider": self.provider,
            "satellite": self.satellite,
            "datetime": self.datetime,
            "cloud_cover": self.cloud_cover,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TileMetadata":
        """Deserialise from a dict."""
        data = data.copy()
        data["source_file"] = Path(data["source_file"])
        data["bounds"] = tuple(data["bounds"])  # type: ignore[assignment]
        return cls(**data)


class GeoDataset:
    """
    Geospatial dataset that bridges tiled satellite imagery with ML frameworks.

    A GeoDataset holds the complete prepared dataset including train/val/test
    splits. It can be converted to PyTorch DataLoaders or TensorFlow Datasets.

    Parameters
    ----------
    tiles : list of TileMetadata
        All tiles in the dataset.
    label_dir : Path
        Directory containing label rasters/masks aligned to tiles.
    split_file : Path, optional
        JSON file mapping tile IDs to splits (``"train"``, ``"val"``, ``"test"``).
    task : str
        AI task: ``"segmentation"``, ``"detection"``, ``"classification"``,
        ``"change_detection"``.
    class_names : list of str, optional
        Ordered list of class names.
    augmentations : callable, optional
        Augmentation pipeline applied to training samples.
    normalize : bool
        Normalise image pixels to [0, 1]. Defaults to True.
    bands : list of int, optional
        Band indices to load (1-indexed). None = all bands.

    Examples
    --------
    >>> dataset = GeoDataset(tiles=tile_list, label_dir=Path("./labels/"))
    >>> train_loader = dataset.to_pytorch(split="train", batch_size=8)
    >>> for images, masks, meta in train_loader:
    ...     loss = criterion(model(images), masks)
    """

    def __init__(
        self,
        tiles: list[TileMetadata],
        label_dir: Path,
        split_file: Optional[Path] = None,
        task: str = "segmentation",
        class_names: Optional[list[str]] = None,
        augmentations: Optional[Callable] = None,
        normalize: bool = True,
        bands: Optional[list[int]] = None,
    ) -> None:
        self.tiles = tiles
        self.label_dir = Path(label_dir)
        self.task = task
        self.class_names = class_names or []
        self.augmentations = augmentations
        self.normalize = normalize
        self.bands = bands

        # Build split mapping
        self._split_map: dict[str, str] = {}
        if split_file and split_file.exists():
            with open(split_file) as fh:
                self._split_map = json.load(fh)
        else:
            # Default: no split info yet
            self._split_map = {t.tile_id: "train" for t in tiles}

        # Statistics (computed lazily)
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None

        logger.info(
            "GeoDataset: %d tiles | task=%s | classes=%d",
            len(tiles),
            task,
            len(self.class_names),
        )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def num_classes(self) -> int:
        """Number of target classes."""
        return len(self.class_names) if self.class_names else 0

    @property
    def num_bands(self) -> int:
        """Number of input bands."""
        if self.bands:
            return len(self.bands)
        if self.tiles:
            return len(self.tiles[0].bands)
        return 0

    def splits(self) -> dict[str, list[TileMetadata]]:
        """
        Return a mapping of split name → list of tiles.

        Returns
        -------
        dict
            Keys: ``"train"``, ``"val"``, ``"test"`` (or whatever splits are defined).
        """
        result: dict[str, list[TileMetadata]] = {}
        for tile in self.tiles:
            split = self._split_map.get(tile.tile_id, "train")
            result.setdefault(split, []).append(tile)
        return result

    def split_counts(self) -> dict[str, int]:
        """Return tile counts per split."""
        return {k: len(v) for k, v in self.splits().items()}

    # ------------------------------------------------------------------
    # PyTorch interface
    # ------------------------------------------------------------------

    def to_pytorch(
        self,
        split: str = "train",
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
        shuffle: Optional[bool] = None,
        sampler: Optional[Any] = None,
    ) -> Any:
        """
        Convert to a PyTorch DataLoader.

        Parameters
        ----------
        split : str
            Dataset split: ``"train"``, ``"val"``, or ``"test"``.
        batch_size : int
            Batch size. Defaults to 8.
        num_workers : int
            DataLoader worker count. Defaults to 4.
        pin_memory : bool
            Pin memory for faster GPU transfer. Defaults to True.
        shuffle : bool, optional
            Shuffle data. Defaults to True for train, False otherwise.
        sampler : optional
            Custom sampler (overrides shuffle).

        Returns
        -------
        torch.utils.data.DataLoader

        Raises
        ------
        ImportError
            If PyTorch is not installed.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required. Install with: pip install torch")

        from torch.utils.data import DataLoader  # noqa: PLC0415

        tile_dataset = TileDataset(
            tiles=self.splits().get(split, []),
            label_dir=self.label_dir,
            task=self.task,
            augmentations=self.augmentations if split == "train" else None,
            normalize=self.normalize,
            bands=self.bands,
        )

        if shuffle is None:
            shuffle = split == "train" and sampler is None

        return DataLoader(
            tile_dataset,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            sampler=sampler,
            drop_last=split == "train",
        )

    # ------------------------------------------------------------------
    # TensorFlow interface
    # ------------------------------------------------------------------

    def to_tensorflow(
        self,
        split: str = "train",
        batch_size: int = 8,
        shuffle: bool = True,
        prefetch: int = 2,
    ) -> Any:
        """
        Convert to a TensorFlow tf.data.Dataset.

        Parameters
        ----------
        split : str
            Dataset split.
        batch_size : int
            Batch size.
        shuffle : bool
            Shuffle data. Defaults to True.
        prefetch : int
            Number of batches to prefetch. Defaults to 2.

        Returns
        -------
        tf.data.Dataset

        Raises
        ------
        ImportError
            If TensorFlow is not installed.
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required. Install with: pip install tensorflow")

        split_tiles = self.splits().get(split, [])

        def generator():
            for tile in split_tiles:
                image, mask, meta = _load_tile_numpy(
                    tile=tile,
                    label_dir=self.label_dir,
                    normalize=self.normalize,
                    bands=self.bands,
                )
                yield image.astype(np.float32), mask.astype(np.int64)

        # Infer shapes from first tile
        if not split_tiles:
            raise ValueError(f"No tiles found for split '{split}'")

        sample_image, sample_mask, _ = _load_tile_numpy(
            tile=split_tiles[0],
            label_dir=self.label_dir,
            normalize=self.normalize,
            bands=self.bands,
        )
        image_shape = sample_image.shape
        mask_shape = sample_mask.shape

        ds = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=image_shape, dtype=tf.float32),
                tf.TensorSpec(shape=mask_shape, dtype=tf.int64),
            ),
        )
        if shuffle and split == "train":
            ds = ds.shuffle(buffer_size=min(len(split_tiles), 1000))
        ds = ds.batch(batch_size).prefetch(prefetch)
        return ds

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, output_dir: Union[str, Path]) -> Path:
        """
        Save dataset metadata to disk (tiles, splits, class names).

        Parameters
        ----------
        output_dir : str or Path
            Directory to save metadata files.

        Returns
        -------
        Path
            Path to the dataset directory.
        """
        dest = Path(output_dir)
        dest.mkdir(parents=True, exist_ok=True)

        meta = {
            "task": self.task,
            "class_names": self.class_names,
            "num_tiles": len(self.tiles),
            "split_counts": self.split_counts(),
        }
        with open(dest / "dataset_meta.json", "w") as fh:
            json.dump(meta, fh, indent=2)

        tiles_data = [t.to_dict() for t in self.tiles]
        with open(dest / "tiles.json", "w") as fh:
            json.dump(tiles_data, fh, indent=2)

        with open(dest / "splits.json", "w") as fh:
            json.dump(self._split_map, fh, indent=2)

        logger.info("Dataset saved to %s (%d tiles)", dest, len(self.tiles))
        return dest

    @classmethod
    def load(cls, dataset_dir: Union[str, Path]) -> "GeoDataset":
        """
        Load a dataset from disk.

        Parameters
        ----------
        dataset_dir : str or Path
            Directory containing dataset metadata files.

        Returns
        -------
        GeoDataset
        """
        src = Path(dataset_dir)

        with open(src / "dataset_meta.json") as fh:
            meta = json.load(fh)

        with open(src / "tiles.json") as fh:
            tiles = [TileMetadata.from_dict(d) for d in json.load(fh)]

        split_file = src / "splits.json"

        return cls(
            tiles=tiles,
            label_dir=src / "labels",
            split_file=split_file if split_file.exists() else None,
            task=meta.get("task", "segmentation"),
            class_names=meta.get("class_names", []),
        )

    def __len__(self) -> int:
        return len(self.tiles)

    def __repr__(self) -> str:
        counts = self.split_counts()
        return (
            f"GeoDataset(tiles={len(self.tiles)}, task={self.task!r}, "
            f"splits={counts})"
        )


class TileDataset(TorchDataset if TORCH_AVAILABLE else object):  # type: ignore[misc]
    """
    PyTorch Dataset for individual tiles.

    Lazily loads imagery tiles and their labels from disk, applying
    optional augmentations.

    Parameters
    ----------
    tiles : list of TileMetadata
        Tiles to include.
    label_dir : Path
        Directory containing label files.
    task : str
        AI task type.
    augmentations : callable, optional
        Augmentation callable (e.g. from albumentations).
    normalize : bool
        Normalise to [0, 1]. Defaults to True.
    bands : list of int, optional
        Band indices to load.

    Examples
    --------
    >>> tile_dataset = TileDataset(tiles=tiles, label_dir=Path("./labels/"))
    >>> image, mask, meta = tile_dataset[0]
    >>> print(image.shape, mask.shape)
    torch.Size([3, 512, 512]) torch.Size([512, 512])
    """

    def __init__(
        self,
        tiles: list[TileMetadata],
        label_dir: Path,
        task: str = "segmentation",
        augmentations: Optional[Callable] = None,
        normalize: bool = True,
        bands: Optional[list[int]] = None,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required. Install with: pip install torch")
        self.tiles = tiles
        self.label_dir = Path(label_dir)
        self.task = task
        self.augmentations = augmentations
        self.normalize = normalize
        self.bands = bands

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> tuple[Any, Any, dict[str, Any]]:
        """
        Load and return one tile.

        Parameters
        ----------
        idx : int
            Index.

        Returns
        -------
        tuple
            ``(image_tensor, label_tensor, metadata_dict)``
        """
        import torch  # noqa: PLC0415

        tile = self.tiles[idx]
        image_np, label_np, meta = _load_tile_numpy(
            tile=tile,
            label_dir=self.label_dir,
            normalize=self.normalize,
            bands=self.bands,
        )

        # Apply augmentations (albumentations format)
        if self.augmentations is not None:
            # Transpose to HWC for albumentations
            image_hwc = np.moveaxis(image_np, 0, -1)
            result = self.augmentations(image=image_hwc, mask=label_np)
            image_np = np.moveaxis(result["image"], -1, 0)
            label_np = result["mask"]

        image_t = torch.from_numpy(image_np.astype(np.float32))
        label_t = torch.from_numpy(label_np.astype(np.int64))

        return image_t, label_t, meta

    def __repr__(self) -> str:
        return f"TileDataset(n={len(self.tiles)}, task={self.task!r})"


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------


def _load_tile_numpy(
    tile: TileMetadata,
    label_dir: Path,
    normalize: bool = True,
    bands: Optional[list[int]] = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Load a tile's imagery and label as numpy arrays.

    Returns
    -------
    tuple
        ``(image_array [C,H,W], label_array [H,W], metadata_dict)``
    """
    try:
        import rasterio  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "rasterio is required for tile loading. "
            "Install with: pip install rasterio"
        ) from exc

    # Load image
    with rasterio.open(tile.source_file) as src:
        band_indices = bands or list(range(1, src.count + 1))
        window = rasterio.windows.Window(
            col_off=tile.col_off,
            row_off=tile.row_off,
            width=tile.width,
            height=tile.height,
        )
        image = src.read(band_indices, window=window).astype(np.float32)

    if normalize:
        # Normalise to [0, 1] using per-band min/max
        for c in range(image.shape[0]):
            band = image[c]
            b_min, b_max = band.min(), band.max()
            if b_max > b_min:
                image[c] = (band - b_min) / (b_max - b_min)

    # Load label
    label_file = label_dir / f"{tile.tile_id}.tif"
    if label_file.exists():
        with rasterio.open(label_file) as lsrc:
            label = lsrc.read(1).astype(np.int64)
    else:
        label = np.zeros((tile.height, tile.width), dtype=np.int64)

    meta = tile.to_dict()
    return image, label, meta
