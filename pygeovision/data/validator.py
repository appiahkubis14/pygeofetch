"""
pygeovision.data.validator
==========================

Mandatory data validation layer.  Every dataset is checked before any
model touches it — no exceptions.

Validates:
  - Null / Inf / NaN values
  - Data type correctness
  - CRS consistency
  - Value range bounds
  - Band shape consistency
  - Required band presence
  - Statistical outliers
  - No-data coverage
  - Model-specific requirements

Usage::

    from pygeovision.data.validator import DataValidator

    v = DataValidator()

    # Validate a GeoTIFF (returns ValidationReport)
    report = v.validate("sentinel2.tif")
    if not report.passed:
        print(report.summary())

    # Validate + auto-fix and return a clean numpy array
    arr = v.validate_for_inference("sentinel2.tif", model_type="segmentation")

    # Generate full HTML report
    v.generate_report("sentinel2.tif", "validation_report.html")
"""
from __future__ import annotations

import dataclasses
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ValidationIssue:
    """A single issue detected during validation."""
    severity: str           # "error" | "warning" | "info"
    check:    str           # which check detected it
    message:  str
    fix:      Optional[str] = None   # applied fix, if any

    def __str__(self) -> str:
        icon = {"error": "✗", "warning": "⚠", "info": "ℹ"}.get(self.severity, "?")
        fix_str = f"  → {self.fix}" if self.fix else ""
        return f"  [{icon} {self.severity.upper()}] {self.check}: {self.message}{fix_str}"


@dataclasses.dataclass
class ValidationReport:
    """Complete validation report for a dataset."""
    source:        str
    passed:        bool
    issues:        List[ValidationIssue]  = dataclasses.field(default_factory=list)
    stats:         Dict[str, Any]         = dataclasses.field(default_factory=dict)
    auto_fixed:    bool = False

    @property
    def errors(self)   -> List[ValidationIssue]: return [i for i in self.issues if i.severity == "error"]
    @property
    def warnings(self) -> List[ValidationIssue]: return [i for i in self.issues if i.severity == "warning"]

    def summary(self) -> str:
        lines = [
            f"ValidationReport: {self.source}",
            f"  Status : {'✓ PASSED' if self.passed else '✗ FAILED'}",
            f"  Errors : {len(self.errors)}",
            f"  Warnings: {len(self.warnings)}",
        ]
        if self.stats:
            lines.append(f"  Stats  : {self.stats}")
        for issue in self.issues:
            lines.append(str(issue))
        return "\n".join(lines)

    def raise_if_failed(self):
        if not self.passed:
            msgs = "; ".join(e.message for e in self.errors)
            raise ValidationError(f"Validation failed for '{self.source}': {msgs}")


class ValidationError(ValueError):
    """Raised when validation cannot auto-fix a critical issue."""


# ---------------------------------------------------------------------------
# Core validator
# ---------------------------------------------------------------------------

class DataValidator:
    """Mandatory data validation before any model runs.

    Args:
        mode: ``"strict"`` (raise on errors) | ``"fix"`` (auto-fix errors)
              | ``"warn"`` (log warnings, never raise).
        max_nodata_pct: Maximum acceptable no-data percentage (default 50 %).
        outlier_sigma: Z-score threshold for outlier detection (default 4.0).
    """

    def __init__(
        self,
        mode: str = "fix",
        max_nodata_pct: float = 50.0,
        outlier_sigma: float = 4.0,
    ):
        if mode not in ("strict", "fix", "warn"):
            raise ValueError("mode must be 'strict', 'fix', or 'warn'")
        self.mode           = mode
        self.max_nodata_pct = max_nodata_pct
        self.outlier_sigma  = outlier_sigma

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def validate(
        self,
        data: Union[str, "np.ndarray"],
        *,
        expected_dtype: Optional[str] = None,
        expected_shape: Optional[Tuple[int, ...]] = None,
        required_bands: Optional[int] = None,
        value_range: Optional[Tuple[float, float]] = None,
        target_crs: Optional[str] = None,
        auto_fix: bool = True,
    ) -> ValidationReport:
        """Run all validation checks on a GeoTIFF path or numpy array.

        Args:
            data: Path to a GeoTIFF or a ``(C, H, W)`` numpy array.
            expected_dtype: If set, check data type (e.g. ``"float32"``).
            expected_shape: If set, check ``(C, H, W)`` shape.
            required_bands: Minimum number of bands required.
            value_range: ``(min, max)`` acceptable value range.
            target_crs: If set, check that the raster CRS matches.
            auto_fix: Apply fixes when mode == "fix".

        Returns:
            :class:`ValidationReport`.

        Example::

            report = validator.validate(
                "sentinel2.tif",
                required_bands=6,
                value_range=(0, 10000),
            )
            report.raise_if_failed()
        """
        source = str(data) if isinstance(data, str) else f"array{getattr(data, 'shape', '')}"
        issues: List[ValidationIssue] = []
        stats:  Dict[str, Any] = {}
        arr:    Optional[np.ndarray] = None
        profile: Dict = {}

        # --- Load if path ---
        if isinstance(data, (str, Path)):
            arr, profile = self._load_raster(str(data))
        elif isinstance(data, np.ndarray):
            arr = data.copy()
        else:
            raise TypeError(f"data must be a path or numpy array, got {type(data)}")

        # Ensure (C, H, W)
        if arr.ndim == 2:
            arr = arr[np.newaxis]

        C, H, W = arr.shape
        stats.update({"bands": C, "height": H, "width": W, "dtype": str(arr.dtype)})

        # --- Checks ---
        issues += self.check_nulls(arr,     auto_fix=auto_fix)
        issues += self.check_dtype(arr,     expected_dtype, auto_fix=auto_fix)
        issues += self.check_shape(arr,     expected_shape)
        issues += self.check_bands(arr,     required_bands)
        issues += self.check_bounds(arr,    value_range,    auto_fix=auto_fix)
        issues += self.check_nodata(arr,    profile.get("nodata"))
        issues += self.check_outliers(arr,  auto_fix=auto_fix)

        if profile and target_crs:
            issues += self.check_crs(profile.get("crs"), target_crs)

        # Compute statistics
        valid_mask = np.isfinite(arr)
        if valid_mask.any():
            valid = arr[valid_mask]
            stats.update({
                "value_min":     float(valid.min()),
                "value_max":     float(valid.max()),
                "value_mean":    float(valid.mean()),
                "value_std":     float(valid.std()),
                "nan_pct":       float((~valid_mask).mean() * 100),
                "zero_pct":      float((arr == 0).mean() * 100),
            })

        errors   = [i for i in issues if i.severity == "error"]
        passed   = len(errors) == 0
        fixed    = any(i.fix for i in issues)

        report = ValidationReport(
            source=source, passed=passed,
            issues=issues, stats=stats, auto_fixed=fixed,
        )

        if not passed:
            if self.mode == "strict":
                report.raise_if_failed()
            elif self.mode == "fix":
                logger.warning("Auto-fixed %d validation error(s) for '%s'",
                               len(errors), source)
            else:
                for e in errors:
                    logger.warning("Validation %s: %s", e.severity, e.message)

        return report

    # ------------------------------------------------------------------
    # Individual checks (each returns a list of ValidationIssue)
    # ------------------------------------------------------------------

    def check_nulls(
        self,
        arr: np.ndarray,
        auto_fix: bool = True,
    ) -> List[ValidationIssue]:
        """Check for NaN and Inf values; replace with 0 when auto_fix=True."""
        issues = []
        nan_count = int(np.isnan(arr).sum())
        inf_count = int(np.isinf(arr).sum())

        if nan_count > 0:
            fix = None
            if auto_fix:
                arr[np.isnan(arr)] = 0.0
                fix = f"replaced {nan_count} NaN values with 0"
            issues.append(ValidationIssue(
                "error" if not auto_fix else "warning",
                "check_nulls",
                f"{nan_count} NaN value(s) found",
                fix,
            ))

        if inf_count > 0:
            fix = None
            if auto_fix:
                arr[np.isinf(arr)] = 0.0
                fix = f"replaced {inf_count} Inf value(s) with 0"
            issues.append(ValidationIssue(
                "error" if not auto_fix else "warning",
                "check_nulls",
                f"{inf_count} Inf value(s) found",
                fix,
            ))

        return issues

    def check_dtype(
        self,
        arr: np.ndarray,
        expected_dtype: Optional[str] = None,
        auto_fix: bool = True,
    ) -> List[ValidationIssue]:
        """Verify array dtype; cast to expected_dtype when auto_fix=True."""
        issues = []
        if expected_dtype is None:
            return issues

        actual = str(arr.dtype)
        if actual != expected_dtype:
            fix = None
            if auto_fix:
                arr = arr.astype(expected_dtype)
                fix = f"cast {actual} → {expected_dtype}"
            issues.append(ValidationIssue(
                "warning", "check_dtype",
                f"dtype is {actual}, expected {expected_dtype}", fix,
            ))
        return issues

    def check_crs(
        self,
        actual_crs: Any,
        target_crs: str,
    ) -> List[ValidationIssue]:
        """Check that the raster CRS matches target_crs."""
        issues = []
        if actual_crs is None:
            issues.append(ValidationIssue(
                "error", "check_crs",
                "Raster has no CRS (undefined projection).",
                "Assign the correct CRS before proceeding.",
            ))
            return issues

        # Normalise to EPSG string for comparison
        actual_str = str(actual_crs)
        target_str = str(target_crs)
        if target_str not in actual_str and actual_str not in target_str:
            issues.append(ValidationIssue(
                "warning", "check_crs",
                f"CRS mismatch: raster={actual_str}, expected={target_str}",
                "Use clip_to_bbox(bbox_crs=...) or reproject() before inference.",
            ))
        return issues

    def check_bounds(
        self,
        arr: np.ndarray,
        value_range: Optional[Tuple[float, float]] = None,
        auto_fix: bool = True,
    ) -> List[ValidationIssue]:
        """Check that all values are within [min, max]; clip if auto_fix."""
        issues = []
        if value_range is None:
            return issues

        lo, hi = value_range
        out_of_range = int(((arr < lo) | (arr > hi)).sum())

        if out_of_range > 0:
            fix = None
            if auto_fix:
                np.clip(arr, lo, hi, out=arr)
                fix = f"clipped {out_of_range} pixel(s) to [{lo}, {hi}]"
            issues.append(ValidationIssue(
                "warning" if auto_fix else "error",
                "check_bounds",
                f"{out_of_range} pixel(s) outside [{lo}, {hi}]", fix,
            ))
        return issues

    def check_shape(
        self,
        arr: np.ndarray,
        expected_shape: Optional[Tuple[int, ...]] = None,
    ) -> List[ValidationIssue]:
        """Check that arr.shape matches expected_shape."""
        issues = []
        if expected_shape is None:
            return issues

        if arr.shape != expected_shape:
            issues.append(ValidationIssue(
                "error", "check_shape",
                f"Shape mismatch: actual={arr.shape}, expected={expected_shape}",
                "Use resample() or stack_bands() to match shapes.",
            ))
        return issues

    def check_bands(
        self,
        arr: np.ndarray,
        required_bands: Optional[int] = None,
    ) -> List[ValidationIssue]:
        """Check that the array has at least required_bands channels."""
        issues = []
        if required_bands is None:
            return issues

        C = arr.shape[0] if arr.ndim >= 3 else 1
        if C < required_bands:
            issues.append(ValidationIssue(
                "error", "check_bands",
                f"Too few bands: have {C}, need ≥ {required_bands}",
                f"Stack additional bands or use a model expecting {C} channels.",
            ))
        return issues

    def check_nodata(
        self,
        arr: np.ndarray,
        nodata_val: Optional[float] = None,
    ) -> List[ValidationIssue]:
        """Check for excessive no-data coverage."""
        issues = []
        if nodata_val is None:
            nodata_val = 0.0

        nodata_pct = float((arr == nodata_val).mean() * 100)
        if nodata_pct > self.max_nodata_pct:
            issues.append(ValidationIssue(
                "warning", "check_nodata",
                f"{nodata_pct:.1f}% of pixels are no-data (threshold {self.max_nodata_pct}%)",
                "Consider acquiring a less-cloud-affected scene.",
            ))
        return issues

    def check_outliers(
        self,
        arr: np.ndarray,
        auto_fix: bool = True,
    ) -> List[ValidationIssue]:
        """Detect and optionally clip statistical outliers (per-band z-score)."""
        issues = []
        sigma = self.outlier_sigma

        for b in range(arr.shape[0]):
            band  = arr[b]
            valid = band[np.isfinite(band)]
            if valid.size < 2:
                continue
            mu, std = float(valid.mean()), float(valid.std())
            if std < 1e-10:
                continue
            n_outliers = int(((np.abs(band - mu) / std) > sigma).sum())
            if n_outliers > 0:
                fix = None
                if auto_fix:
                    lo = mu - sigma * std
                    hi = mu + sigma * std
                    arr[b] = np.clip(band, lo, hi)
                    fix = f"clipped to [{lo:.2f}, {hi:.2f}]"
                issues.append(ValidationIssue(
                    "warning", "check_outliers",
                    f"Band {b}: {n_outliers} outlier(s) beyond {sigma}σ", fix,
                ))
        return issues

    # ------------------------------------------------------------------
    # Model-type specific validation
    # ------------------------------------------------------------------

    _MODEL_REQUIREMENTS: Dict[str, Dict] = {
        "segmentation": {
            "min_bands":   1,
            "dtype":       "float32",
            "value_range": (0.0, 1.0),
        },
        "detection": {
            "min_bands":   3,
            "dtype":       "float32",
            "value_range": (0.0, 1.0),
        },
        "change_detection": {
            "min_bands":   4,
            "dtype":       "float32",
            "value_range": (0.0, 1.0),
        },
        "classification": {
            "min_bands":   1,
            "dtype":       "float32",
            "value_range": (0.0, 1.0),
        },
        "foundation": {
            "min_bands":   3,
            "dtype":       "float32",
            "value_range": (-3.0, 3.0),   # z-score normalised
        },
        "regression": {
            "min_bands":   1,
            "dtype":       "float32",
            "value_range": None,
        },
    }

    def validate_for_inference(
        self,
        data: Union[str, np.ndarray],
        model_type: str = "segmentation",
        auto_normalise: bool = True,
    ) -> np.ndarray:
        """Validate and prepare data for model inference.

        Runs all checks, auto-fixes where possible, and returns a clean
        numpy array of shape ``(C, H, W)`` in ``float32``.

        Args:
            data: GeoTIFF path or ``(C, H, W)`` numpy array.
            model_type: One of ``"segmentation"``, ``"detection"``,
                ``"change_detection"``, ``"classification"``,
                ``"foundation"``, ``"regression"``.
            auto_normalise: If the data is outside the expected range,
                auto-normalise to [0, 1] using min-max per band.

        Returns:
            Validated ``float32`` numpy array ``(C, H, W)``.

        Example::

            arr = validator.validate_for_inference(
                "sentinel2.tif",
                model_type="segmentation",
            )
            # arr is float32, [0,1], no NaN/Inf, no outliers
        """
        req = self._MODEL_REQUIREMENTS.get(model_type, {})

        # Load
        if isinstance(data, (str, Path)):
            arr, _ = self._load_raster(str(data))
        else:
            arr = np.asarray(data, dtype=np.float32)

        if arr.ndim == 2:
            arr = arr[np.newaxis]

        # Cast to float32
        arr = arr.astype(np.float32)

        # Fix nulls
        arr[~np.isfinite(arr)] = 0.0

        # Fix outliers
        self.check_outliers(arr, auto_fix=True)

        # Check minimum bands
        min_bands = req.get("min_bands", 1)
        if arr.shape[0] < min_bands:
            logger.warning(
                "validate_for_inference: %d bands present, model_type='%s' needs ≥ %d. "
                "Repeating bands cyclically.",
                arr.shape[0], model_type, min_bands,
            )
            repeats = math.ceil(min_bands / arr.shape[0])
            arr = np.concatenate([arr] * repeats, axis=0)[:min_bands]

        # Auto-normalise if needed
        vr = req.get("value_range")
        if auto_normalise and vr is not None:
            lo, hi = vr
            arr_max = float(arr.max())
            arr_min = float(arr.min())
            if arr_max > hi or arr_min < lo:
                # Per-band min-max to target range
                for b in range(arr.shape[0]):
                    band  = arr[b]
                    bmin, bmax = float(band.min()), float(band.max())
                    denom = bmax - bmin if bmax - bmin > 1e-10 else 1.0
                    arr[b] = (band - bmin) / denom * (hi - lo) + lo

        logger.debug(
            "validate_for_inference: %s | shape=%s | dtype=%s | range=[%.3f, %.3f]",
            model_type, arr.shape, arr.dtype, float(arr.min()), float(arr.max()),
        )
        return arr

    def validate_for_training(
        self,
        images: Union[str, np.ndarray, list],
        labels: Optional[Union[str, np.ndarray]] = None,
        num_classes: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Validate a training batch (images + optional labels).

        Returns:
            Dict with ``"images"``, ``"labels"`` (if provided),
            ``"report"``, and ``"valid_samples"``.

        Example::

            result = validator.validate_for_training(
                images="./chips/",
                labels="./labels/",
                num_classes=5,
            )
            print(result["report"].summary())
        """
        issues: List[ValidationIssue] = []

        # Load images
        if isinstance(images, (str, Path)):
            from pathlib import Path as _P
            paths = sorted(_P(str(images)).rglob("*.tif"))
            arrs  = [self._load_raster(str(p))[0] for p in paths[:5]]  # sample
            source = str(images)
        elif isinstance(images, np.ndarray):
            arrs   = [images]
            source = "array"
        else:
            arrs   = images
            source = "list"

        # Check all images have same shape
        shapes = [a.shape for a in arrs]
        if len(set(shapes)) > 1:
            issues.append(ValidationIssue(
                "error", "check_shape",
                f"Inconsistent image shapes: {set(shapes)}",
                "Ensure all chips are the same size.",
            ))

        # Check labels if provided
        result_labels = None
        if labels is not None:
            if isinstance(labels, (str, Path)):
                lbl_paths = sorted(Path(str(labels)).rglob("*.tif"))
                label_arrs = [self._load_raster(str(p))[0] for p in lbl_paths[:5]]
            elif isinstance(labels, np.ndarray):
                label_arrs = [labels]
            else:
                label_arrs = labels

            # Check label values
            for la in label_arrs:
                la = la.astype(np.int32)
                unique_cls = np.unique(la)
                if num_classes and unique_cls.max() >= num_classes:
                    issues.append(ValidationIssue(
                        "error", "check_bands",
                        f"Label has class {unique_cls.max()} but num_classes={num_classes}",
                    ))
            result_labels = label_arrs

        errors = [i for i in issues if i.severity == "error"]
        report = ValidationReport(
            source=source,
            passed=len(errors) == 0,
            issues=issues,
            stats={"n_samples": len(arrs), "shapes": list(set(shapes))},
        )
        if self.mode == "strict":
            report.raise_if_failed()

        return {"images": arrs, "labels": result_labels,
                "report": report, "valid_samples": len(arrs)}

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(
        self,
        data: Union[str, np.ndarray],
        output_path: str,
        fmt: str = "html",
    ) -> str:
        """Generate a full validation report as HTML or JSON.

        Args:
            data: GeoTIFF path or numpy array.
            output_path: Where to write the report file.
            fmt: ``"html"`` or ``"json"``.

        Returns:
            ``output_path``.
        """
        report = self.validate(data)
        from pathlib import Path as _P
        _P(output_path).parent.mkdir(parents=True, exist_ok=True)

        if fmt == "json":
            import json
            payload = {
                "source":   report.source,
                "passed":   report.passed,
                "stats":    report.stats,
                "issues":   [
                    {"severity": i.severity, "check": i.check,
                     "message": i.message, "fix": i.fix}
                    for i in report.issues
                ],
            }
            with open(output_path, "w") as f:
                json.dump(payload, f, indent=2)
        else:
            # Minimal but functional HTML report
            rows = ""
            for iss in report.issues:
                colour = {"error": "#fee2e2", "warning": "#fef9c3", "info": "#e0f2fe"}.get(iss.severity, "#fff")
                rows += (
                    f"<tr style='background:{colour}'>"
                    f"<td>{iss.severity.upper()}</td>"
                    f"<td>{iss.check}</td>"
                    f"<td>{iss.message}</td>"
                    f"<td>{iss.fix or ''}</td></tr>\n"
                )
            stat_rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in report.stats.items())
            status_color = "#16a34a" if report.passed else "#dc2626"
            html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>PyGeoVision Validation Report</title>
<style>body{{font-family:monospace;padding:2rem;}}
table{{border-collapse:collapse;width:100%;margin:1rem 0;}}
th,td{{border:1px solid #e2e8f0;padding:8px 12px;text-align:left;}}
th{{background:#f1f5f9;}}h1{{font-size:1.4rem;}}
.badge{{padding:4px 12px;border-radius:4px;color:#fff;font-weight:bold;}}
</style></head><body>
<h1>🛰️ PyGeoVision — Validation Report</h1>
<p><b>Source:</b> {report.source}</p>
<p><b>Status:</b> <span class="badge" style="background:{status_color}">
{'✓ PASSED' if report.passed else '✗ FAILED'}</span></p>
<h2>Statistics</h2>
<table><tr><th>Metric</th><th>Value</th></tr>{stat_rows}</table>
<h2>Issues ({len(report.issues)})</h2>
<table><tr><th>Severity</th><th>Check</th><th>Message</th><th>Auto-fix applied</th></tr>
{rows if rows else '<tr><td colspan="4">No issues found.</td></tr>'}
</table></body></html>"""
            with open(output_path, "w") as f:
                f.write(html)

        logger.info("Validation report written → %s", output_path)
        return output_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_raster(self, path: str) -> Tuple[np.ndarray, Dict]:
        """Load a GeoTIFF into a ``(C, H, W)`` float32 array + profile."""
        try:
            import rasterio
        except ImportError:
            raise ImportError("pip install rasterio") from None

        with rasterio.open(path) as src:
            data    = src.read().astype(np.float32)
            profile = dict(src.profile)
            profile["crs"] = src.crs
        return data, profile
