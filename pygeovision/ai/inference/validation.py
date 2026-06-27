"""
Inference validation utilities for PyGeoVision.

Tools for validating model predictions against ground truth labels,
computing spatial error maps, and generating validation reports.

Example:
    >>> from pygeovision.ai.inference.validation import PredictionValidator
    >>> validator = PredictionValidator(num_classes=10)
    >>> report = validator.validate("prediction.tif", "ground_truth.tif")
    >>> print(f"mIoU: {report.metrics['mean_iou']:.4f}")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Validation results comparing predictions to ground truth.

    Attributes:
        metrics: Dict of metric names to values (mIoU, accuracy, etc.).
        per_class_iou: IoU score for each class.
        per_class_precision: Precision for each class.
        per_class_recall: Recall for each class.
        per_class_f1: F1 score for each class.
        confusion_matrix: Full (num_classes, num_classes) confusion matrix.
        error_rate: Fraction of mislabelled pixels.
        num_pixels: Total pixels evaluated.
        class_names: Optional class name mapping.
    """

    metrics: Dict[str, float] = field(default_factory=dict)
    per_class_iou: List[float] = field(default_factory=list)
    per_class_precision: List[float] = field(default_factory=list)
    per_class_recall: List[float] = field(default_factory=list)
    per_class_f1: List[float] = field(default_factory=list)
    confusion_matrix: Optional[np.ndarray] = None
    error_rate: float = 0.0
    num_pixels: int = 0
    class_names: Optional[List[str]] = None

    def summary(self) -> str:
        """Return a formatted validation summary string."""
        lines = ["=== Validation Report ==="]
        for k, v in self.metrics.items():
            lines.append(f"  {k}: {v:.4f}")
        if self.class_names and self.per_class_iou:
            lines.append("\nPer-class IoU:")
            for i, (name, iou) in enumerate(zip(self.class_names, self.per_class_iou)):
                lines.append(f"  [{i}] {name}: {iou:.4f}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the report to a JSON-compatible dict."""
        return {
            "metrics": self.metrics,
            "per_class_iou": self.per_class_iou,
            "per_class_precision": self.per_class_precision,
            "per_class_recall": self.per_class_recall,
            "per_class_f1": self.per_class_f1,
            "error_rate": self.error_rate,
            "num_pixels": self.num_pixels,
            "class_names": self.class_names,
        }

    def save(self, path: Union[str, Path]) -> Path:
        """Save the report as a JSON file.

        Args:
            path: Output file path.

        Returns:
            Path to the saved file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path


class PredictionValidator:
    """Validate raster model predictions against ground truth labels.

    Computes standard segmentation metrics (mIoU, accuracy, F1) by
    comparing a prediction GeoTIFF to a ground truth label GeoTIFF.
    Handles mismatched resolutions by resampling to match.

    Args:
        num_classes: Number of semantic classes.
        ignore_index: Class index to exclude from metrics.
        class_names: Optional list of class names for reporting.

    Example:
        >>> validator = PredictionValidator(num_classes=10, ignore_index=255)
        >>> report = validator.validate("pred.tif", "gt.tif")
        >>> report.save("validation_report.json")
        >>> print(report.summary())
    """

    def __init__(
        self,
        num_classes: int = 2,
        ignore_index: int = 255,
        class_names: Optional[List[str]] = None,
    ) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_names = class_names

    def validate(
        self,
        prediction_path: Union[str, Path],
        ground_truth_path: Union[str, Path],
        output_error_map: Optional[Union[str, Path]] = None,
    ) -> ValidationReport:
        """Compare a prediction GeoTIFF to a ground truth GeoTIFF.

        Args:
            prediction_path: Path to the predicted label mask GeoTIFF.
            ground_truth_path: Path to the ground truth label GeoTIFF.
            output_error_map: If provided, save an error map GeoTIFF
                (0=correct, 1=wrong) to this path.

        Returns:
            ValidationReport with all computed metrics.
        """
        try:
            import rasterio
            from rasterio.enums import Resampling
        except ImportError as exc:
            raise ImportError("validate() requires rasterio. pip install rasterio") from exc

        with rasterio.open(prediction_path) as src:
            pred = src.read(1).astype(np.int64)
            pred_transform = src.transform
            pred_profile = src.profile.copy()

        with rasterio.open(ground_truth_path) as src:
            gt = src.read(1).astype(np.int64)
            if gt.shape != pred.shape:
                # Resample GT to match prediction shape
                from rasterio.warp import reproject
                gt_resampled = np.zeros(pred.shape, dtype=np.int64)
                with rasterio.open(ground_truth_path) as gt_src:
                    reproject(
                        source=gt_src.read(1),
                        destination=gt_resampled,
                        src_transform=gt_src.transform,
                        src_crs=gt_src.crs,
                        dst_transform=pred_transform,
                        dst_crs=gt_src.crs,
                        resampling=Resampling.nearest,
                    )
                gt = gt_resampled
                logger.info("Resampled ground truth to match prediction shape %s", pred.shape)

        report = self.validate_arrays(pred, gt)

        if output_error_map is not None:
            error_map = (pred != gt).astype(np.uint8)
            output_error_map = Path(output_error_map)
            output_error_map.parent.mkdir(parents=True, exist_ok=True)
            pred_profile.update(dtype="uint8", count=1, compress="lzw")
            with rasterio.open(output_error_map, "w", **pred_profile) as dst:
                dst.write(error_map[np.newaxis, ...])
            logger.info("Error map saved to %s", output_error_map)

        return report

    def validate_arrays(
        self,
        pred: np.ndarray,
        gt: np.ndarray,
    ) -> ValidationReport:
        """Compute metrics from prediction and ground truth numpy arrays.

        Args:
            pred: Predicted label array (H, W) of int class IDs.
            gt: Ground truth label array (H, W) of int class IDs.

        Returns:
            ValidationReport with all metrics.
        """
        pred = pred.flatten().astype(np.int64)
        gt = gt.flatten().astype(np.int64)

        # Mask out ignore_index
        valid = gt != self.ignore_index
        pred = pred[valid]
        gt = gt[valid]

        num_pixels = int(valid.sum())
        if num_pixels == 0:
            logger.warning("No valid pixels to evaluate.")
            return ValidationReport(
                metrics={"mean_iou": 0.0, "accuracy": 0.0},
                num_pixels=0,
                class_names=self.class_names,
            )

        # Build confusion matrix
        n = self.num_classes
        conf = np.zeros((n, n), dtype=np.int64)
        mask = (gt >= 0) & (gt < n) & (pred >= 0) & (pred < n)
        np.add.at(conf, (gt[mask], pred[mask]), 1)

        tp = np.diag(conf).astype(float)
        fp = conf.sum(0).astype(float) - tp
        fn = conf.sum(1).astype(float) - tp
        eps = 1e-7

        iou = (tp / (tp + fp + fn + eps)).tolist()
        precision = (tp / (tp + fp + eps)).tolist()
        recall = (tp / (tp + fn + eps)).tolist()
        f1 = [2 * p * r / (p + r + eps) for p, r in zip(precision, recall)]

        mean_iou = float(np.nanmean([v for v in iou if not np.isnan(v)]))
        accuracy = float(tp.sum() / num_pixels)
        freq = conf.sum(1).astype(float) / conf.sum()
        fw_iou = float((freq * np.array(iou)).sum())

        error_rate = 1.0 - accuracy

        return ValidationReport(
            metrics={
                "mean_iou": mean_iou,
                "accuracy": accuracy,
                "frequency_weighted_iou": fw_iou,
                "error_rate": error_rate,
                "mean_f1": float(np.mean(f1)),
                "mean_precision": float(np.mean(precision)),
                "mean_recall": float(np.mean(recall)),
            },
            per_class_iou=iou,
            per_class_precision=precision,
            per_class_recall=recall,
            per_class_f1=f1,
            confusion_matrix=conf,
            error_rate=error_rate,
            num_pixels=num_pixels,
            class_names=self.class_names,
        )

    def cross_validate(
        self,
        pred_paths: List[Union[str, Path]],
        gt_paths: List[Union[str, Path]],
    ) -> ValidationReport:
        """Validate across multiple prediction/GT pairs and aggregate.

        Args:
            pred_paths: List of prediction GeoTIFF paths.
            gt_paths: List of ground truth GeoTIFF paths.

        Returns:
            Aggregated ValidationReport across all pairs.
        """
        if len(pred_paths) != len(gt_paths):
            raise ValueError("pred_paths and gt_paths must have the same length.")

        # Accumulate confusion matrices
        n = self.num_classes
        conf_total = np.zeros((n, n), dtype=np.int64)
        total_pixels = 0

        for pred_p, gt_p in zip(pred_paths, gt_paths):
            report = self.validate(pred_p, gt_p)
            if report.confusion_matrix is not None:
                conf_total += report.confusion_matrix
            total_pixels += report.num_pixels

        # Compute final metrics from aggregated confusion matrix
        eps = 1e-7
        tp = np.diag(conf_total).astype(float)
        fp = conf_total.sum(0).astype(float) - tp
        fn = conf_total.sum(1).astype(float) - tp

        iou = (tp / (tp + fp + fn + eps)).tolist()
        precision = (tp / (tp + fp + eps)).tolist()
        recall = (tp / (tp + fn + eps)).tolist()
        f1 = [2 * p * r / (p + r + eps) for p, r in zip(precision, recall)]

        mean_iou = float(np.nanmean([v for v in iou if not np.isnan(v)]))
        accuracy = float(tp.sum() / max(total_pixels, 1))

        return ValidationReport(
            metrics={
                "mean_iou": mean_iou,
                "accuracy": accuracy,
                "mean_f1": float(np.mean(f1)),
            },
            per_class_iou=iou,
            per_class_precision=precision,
            per_class_recall=recall,
            per_class_f1=f1,
            confusion_matrix=conf_total,
            error_rate=1.0 - accuracy,
            num_pixels=total_pixels,
            class_names=self.class_names,
        )
