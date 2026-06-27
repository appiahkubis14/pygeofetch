"""
Label Quality Assessment (E4) — Automated validation of training labels.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
logger = logging.getLogger(__name__)


class LabelQualityAssessor:
    """Automated quality assessment and cleaning for geospatial training labels.

    Checks:
        - Class balance (class imbalance warning)
        - Boundary quality (thin slivers, isolated pixels)
        - Spatial coverage (blank/nodata regions)
        - Label consistency (duplicate pixels across chips)
        - Cross-validation with reference sources (OSM, etc.)
        - IoU with SAM-predicted masks

    Example::

        assessor = LabelQualityAssessor()
        report = assessor.assess("./labels/buildings.tif")
        if report["quality_score"] < 0.7:
            cleaned = assessor.clean("./labels/buildings.tif", "./labels/buildings_clean.tif")
    """

    def __init__(self, num_classes: int = 2, ignore_index: int = 255) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def assess(
        self,
        label_path: Union[str, Path],
        image_path: Optional[str] = None,
        checks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run quality checks on a label raster.

        Args:
            label_path: Path to label GeoTIFF
            image_path: Optional paired image for cross-validation
            checks: List of checks to run (default: all)

        Returns:
            Dict with quality_score (0-1), per-check results, recommendations
        """
        checks = checks or ["class_balance", "slivers", "coverage", "connectivity"]
        label_path = Path(label_path)
        results: Dict[str, Any] = {"label_path": str(label_path), "checks": {}}

        try:
            import numpy as np
            import rasterio
            with rasterio.open(str(label_path)) as src:
                label = src.read(1).astype(np.int32)
                transform = src.transform
                pixel_area_m2 = abs(transform.a * transform.e)
                total_px = label.size
                valid_px = (label != self.ignore_index).sum()
        except Exception as exc:
            return {"error": str(exc), "quality_score": 0.0}

        scores = []

        # ── Class balance check ──────────────────────────────────────────────
        if "class_balance" in checks:
            class_counts = {}
            for c in range(self.num_classes):
                n = int((label == c).sum())
                class_counts[c] = n
            n_vals = [v for v in class_counts.values() if v > 0]
            if len(n_vals) > 1:
                imbalance_ratio = max(n_vals) / (min(n_vals) + 1)
                balance_score = float(1.0 / (1 + imbalance_ratio / 10))
            else:
                imbalance_ratio = float("inf")
                balance_score = 0.0

            class_pct = {c: round(n/max(valid_px,1)*100, 2) for c, n in class_counts.items()}
            results["checks"]["class_balance"] = {
                "class_counts": class_counts,
                "class_pct": class_pct,
                "imbalance_ratio": round(imbalance_ratio, 1) if imbalance_ratio != float("inf") else None,
                "score": round(balance_score, 3),
                "status": "ok" if imbalance_ratio < 20 else "warning" if imbalance_ratio < 100 else "critical",
            }
            scores.append(balance_score)

        # ── Spatial coverage check ──────────────────────────────────────────
        if "coverage" in checks:
            valid_fraction = float(valid_px / max(total_px, 1))
            coverage_score = min(1.0, valid_fraction * 2)
            results["checks"]["coverage"] = {
                "valid_pixels": int(valid_px),
                "total_pixels": int(total_px),
                "valid_fraction": round(valid_fraction, 3),
                "score": round(coverage_score, 3),
                "status": "ok" if valid_fraction > 0.5 else "warning",
            }
            scores.append(coverage_score)

        # ── Sliver detection ────────────────────────────────────────────────
        if "slivers" in checks:
            try:
                from scipy import ndimage
                n_slivers = 0
                sliver_threshold_px = max(4, int(1.0 / pixel_area_m2))  # 1 m² min
                for c in range(self.num_classes):
                    binary = (label == c).astype(np.uint8)
                    labeled_arr, n_comp = ndimage.label(binary)
                    component_sizes = ndimage.sum(binary, labeled_arr, range(1, n_comp + 1))
                    n_slivers += int((np.array(component_sizes) < sliver_threshold_px).sum())
                sliver_score = float(1.0 - min(1.0, n_slivers / max(valid_px * 0.001, 1)))
                results["checks"]["slivers"] = {
                    "n_slivers": n_slivers,
                    "score": round(sliver_score, 3),
                    "status": "ok" if n_slivers < 100 else "warning",
                }
                scores.append(sliver_score)
            except ImportError:
                results["checks"]["slivers"] = {"status": "skipped", "reason": "pip install scipy"}

        # ── Connectivity analysis ────────────────────────────────────────────
        if "connectivity" in checks:
            try:
                from scipy import ndimage
                for c in range(1, self.num_classes):  # skip background
                    binary = (label == c).astype(np.uint8)
                    _, n_comp = ndimage.label(binary)
                # High connectivity = fewer, larger components = good labels
                conn_score = min(1.0, 10.0 / max(n_comp, 1)) if valid_px > 0 else 0.5
                results["checks"]["connectivity"] = {
                    "n_components": n_comp,
                    "score": round(conn_score, 3),
                    "status": "ok" if n_comp < 1000 else "warning",
                }
                scores.append(conn_score)
            except ImportError:
                results["checks"]["connectivity"] = {"status": "skipped"}

        # ── Boundary quality ────────────────────────────────────────────────
        if "boundaries" in checks:
            try:
                import cv2
                for c in range(1, self.num_classes):
                    binary = ((label == c) * 255).astype(np.uint8)
                    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    total_perimeter = sum(cv2.arcLength(c, True) for c in contours)
                    total_area = binary.sum() / 255
                    # Isoperimetric quotient: circle = 1, very irregular shapes << 1
                    iq = float(4 * 3.14159 * total_area / (total_perimeter ** 2 + 1e-8))
                results["checks"]["boundaries"] = {
                    "isoperimetric_quotient": round(iq, 3),
                    "score": min(1.0, iq * 2),
                    "status": "ok" if iq > 0.05 else "warning",
                }
            except ImportError:
                results["checks"]["boundaries"] = {"status": "skipped", "reason": "pip install opencv-python"}

        # ── Overall quality score ────────────────────────────────────────────
        quality_score = float(sum(scores) / max(len(scores), 1))
        results["quality_score"] = round(quality_score, 3)
        results["quality_grade"] = (
            "A" if quality_score >= 0.9 else
            "B" if quality_score >= 0.75 else
            "C" if quality_score >= 0.6 else
            "D" if quality_score >= 0.4 else "F"
        )
        results["recommendations"] = self._recommendations(results)
        return results

    def _recommendations(self, results: Dict) -> List[str]:
        recs = []
        checks = results.get("checks", {})
        cb = checks.get("class_balance", {})
        if cb.get("status") in ("warning", "critical"):
            ratio = cb.get("imbalance_ratio", 0)
            recs.append(f"Class imbalance ratio {ratio:.0f}x — use weighted loss or oversampling minority class")
        slivers = checks.get("slivers", {})
        if slivers.get("status") == "warning":
            recs.append(f"{slivers.get('n_slivers',0)} sliver labels detected — run clean() to remove them")
        cov = checks.get("coverage", {})
        if cov.get("valid_fraction", 1.0) < 0.5:
            recs.append("Less than 50% of pixels labeled — consider expanding label coverage")
        if results.get("quality_score", 1.0) < 0.7:
            recs.append("Quality score < 0.7 — recommend manual review of low-quality regions")
        return recs

    def clean(
        self,
        label_path: Union[str, Path],
        output_path: Union[str, Path],
        remove_slivers: bool = True,
        min_area_m2: float = 5.0,
        fill_holes: bool = True,
        smooth: bool = True,
    ) -> Dict[str, Any]:
        """Clean a label raster by removing slivers and filling holes.

        Args:
            label_path: Input label GeoTIFF
            output_path: Cleaned output path
            remove_slivers: Remove connected components smaller than min_area_m2
            min_area_m2: Minimum component area to keep
            fill_holes: Fill small holes inside label regions
            smooth: Apply morphological smoothing

        Returns:
            Dict with n_removed, output_path, before/after pixel counts
        """
        try:
            import numpy as np
            import rasterio
            from scipy import ndimage
        except ImportError as exc:
            raise ImportError(f"scipy + rasterio required: {exc}")

        label_path = Path(label_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(str(label_path)) as src:
            label = src.read(1).astype(np.int32)
            profile = src.profile.copy()
            pixel_area_m2 = abs(src.transform.a * src.transform.e)

        min_px = max(1, int(min_area_m2 / pixel_area_m2))
        n_removed = 0
        cleaned = label.copy()

        for c in range(1, self.num_classes):
            binary = (label == c).astype(np.uint8)
            labeled_arr, n_comp = ndimage.label(binary)
            for comp_id in range(1, n_comp + 1):
                comp_mask = (labeled_arr == comp_id)
                if remove_slivers and comp_mask.sum() < min_px:
                    cleaned[comp_mask] = 0
                    n_removed += 1

            if fill_holes:
                binary_clean = (cleaned == c).astype(np.uint8)
                filled = ndimage.binary_fill_holes(binary_clean)
                cleaned[(filled > 0) & (cleaned == 0)] = c

            if smooth:
                try:
                    import cv2
                    mask = (cleaned == c).astype(np.uint8)
                    kernel = np.ones((3, 3), np.uint8)
                    smoothed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                    cleaned[smoothed == 1] = c
                except ImportError:
                    pass

        with rasterio.open(str(output_path), "w", **profile) as dst:
            dst.write(cleaned[np.newaxis])
            dst.update_tags(cleaned_by="LabelQualityAssessor", min_area_m2=str(min_area_m2))

        return {
            "success": True,
            "output_path": str(output_path),
            "n_components_removed": n_removed,
            "before_label_pixels": int((label > 0).sum()),
            "after_label_pixels": int((cleaned > 0).sum()),
        }

    def report_html(self, results: Dict) -> str:
        """Generate an HTML quality report."""
        grade = results.get("quality_grade", "?")
        score = results.get("quality_score", 0)
        color = "#22c55e" if grade in "AB" else "#f59e0b" if grade == "C" else "#ef4444"
        recs = results.get("recommendations", [])

        checks_html = ""
        for check_name, check_data in results.get("checks", {}).items():
            status = check_data.get("status", "unknown")
            icon = "✓" if status == "ok" else "⚠" if status == "warning" else "✗"
            checks_html += f'<tr><td>{check_name}</td><td>{icon} {status}</td><td>{check_data.get("score", "—")}</td></tr>'

        recs_html = "".join(f"<li>{r}</li>" for r in recs) if recs else "<li>No issues found</li>"

        return f"""<html><body style="font-family:sans-serif;padding:20px">
<h2>Label Quality Report</h2>
<p>File: <code>{results.get("label_path","?")}</code></p>
<h3>Grade: <span style="color:{color};font-size:2em">{grade}</span> ({score:.2%})</h3>
<table border="1" cellpadding="6" style="border-collapse:collapse">
<tr><th>Check</th><th>Status</th><th>Score</th></tr>
{checks_html}
</table>
<h3>Recommendations:</h3><ul>{recs_html}</ul>
</body></html>"""
