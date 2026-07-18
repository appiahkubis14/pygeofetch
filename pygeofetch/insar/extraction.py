"""
SLCExtractor — extract usable VV/VH measurement GeoTIFFs from downloaded
Sentinel-1 SLC .SAFE archives.

Sentinel-1 SLC products are delivered as a .SAFE folder (inside a .zip)
containing 6 separate measurement TIFFs — one per sub-swath (IW1/IW2/IW3)
per polarisation (VV/VH). InterferogramGenerator needs a single flat
complex GeoTIFF per scene, so this module:

  1. Lists the measurement TIFFs inside the downloaded zip for a given
     polarisation.
  2. Reads each sub-swath's embedded Ground Control Points (GCPs) via
     rasterio — Sentinel-1 SLC TIFFs carry these directly, so no
     annotation XML parsing is needed for a coverage check.
  3. Picks the sub-swath whose GCP-derived footprint actually overlaps
     the requested AOI.
  4. Extracts just that one TIFF to disk as a flat, directly-usable file.

Takes DownloadResult objects directly (from client.download()) rather than
re-deriving file paths from scene metadata — DownloadResult.output_path
already holds the exact path the provider wrote to, which avoids an entire
class of filename/subfolder-mismatch bugs.
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from pygeofetch.models.download_task import DownloadResult
    from pygeofetch.models.search_query import BoundingBox

logger = logging.getLogger("pygeofetch.insar.extraction")


class SLCExtractor:
    """
    Extract usable measurement GeoTIFFs from downloaded Sentinel-1 SLC archives.

    Example::

        from pygeofetch import PyGeoFetch
        from pygeofetch.insar import SLCExtractor

        client = PyGeoFetch()
        results = client.download([ref_scene, sec_scene], destination=out_dir)

        extractor = SLCExtractor(polarisation="VV")
        ref_tif, sec_tif = extractor.extract_pair(
            results[0], results[1], aoi=aoi_bbox, output_dir=out_dir,
        )

        # ref_tif / sec_tif are now flat GeoTIFFs ready for
        # InterferogramGenerator.process_pair(ref_tif, sec_tif, ...)
    """

    def __init__(self, polarisation: str = "VV") -> None:
        self._pol = polarisation.lower()

    # ── public API ────────────────────────────────────────────────────────────

    def extract_pair(
        self,
        reference: Union["DownloadResult", str, Path],
        secondary: Union["DownloadResult", str, Path],
        aoi: "BoundingBox",
        output_dir: Union[str, Path],
    ) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Extract the AOI-matching sub-swath from both a reference and
        secondary SLC download in one call.

        Args:
            reference: DownloadResult from client.download() (preferred —
                       uses its .output_path directly, no guessing), or a
                       direct path to the downloaded .SAFE.zip.
            secondary: Same, for the secondary scene.
            aoi:       BoundingBox to match against sub-swath footprints.
            output_dir: Where to write the extracted flat GeoTIFFs.

        Returns:
            (reference_tif, secondary_tif) — either may be None if
            extraction failed for that scene (see logged warnings for why).
        """
        ref_zip = self._resolve_path(reference)
        sec_zip = self._resolve_path(secondary)

        if ref_zip is None or sec_zip is None:
            missing = "reference" if ref_zip is None else "secondary"
            logger.error(
                "Could not resolve a usable file path for the %s scene — "
                "check that its download completed successfully.",
                missing,
            )
            return None, None

        logger.info("Reference archive: %s", ref_zip.name)
        ref_tif = self.extract_scene(ref_zip, aoi, output_dir, label="reference")

        logger.info("Secondary archive: %s", sec_zip.name)
        sec_tif = self.extract_scene(sec_zip, aoi, output_dir, label="secondary")

        return ref_tif, sec_tif

    def extract_scene(
        self,
        zip_path: Union[str, Path],
        aoi: "BoundingBox",
        output_dir: Union[str, Path],
        label: str = "",
    ) -> Optional[Path]:
        """
        Find the sub-swath covering the AOI in one SLC archive and extract it.

        Args:
            zip_path:   Path to the downloaded .SAFE.zip.
            aoi:        BoundingBox to match against sub-swath footprints.
            output_dir: Where to write the extracted flat GeoTIFF.
            label:      Used to build the output filename
                       (f"{label}_{polarisation}.tif").

        Returns:
            Path to the extracted GeoTIFF, or None if no sub-swath in this
            archive overlaps the AOI (check logs for per-sub-swath
            footprints when this happens).
        """
        zip_path = Path(zip_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not zip_path.exists():
            logger.error("Archive not found: %s", zip_path)
            return None

        aoi_tuple = (aoi.min_lon, aoi.min_lat, aoi.max_lon, aoi.max_lat)
        members = self._find_subswaths(zip_path)

        if not members:
            logger.warning(
                "No %s measurement TIFFs found inside %s. This archive may "
                "not be a standard Sentinel-1 SLC .SAFE.zip, or the "
                "polarisation requested (%s) is not present.",
                self._pol.upper(),
                zip_path.name,
                self._pol.upper(),
            )
            return None

        logger.info(
            "Found %d %s sub-swath(s) in %s",
            len(members),
            self._pol.upper(),
            zip_path.name,
        )

        matched_member = None
        for member in members:
            footprint = self._gcp_footprint(zip_path, member)
            if footprint is None:
                continue
            swath = self._swath_label(member)
            overlaps = self._bbox_overlaps(footprint, aoi_tuple)
            logger.debug(
                "  %s: footprint=%s  overlaps AOI: %s",
                swath,
                tuple(round(v, 2) for v in footprint),
                overlaps,
            )
            if overlaps and matched_member is None:
                matched_member = member

        if matched_member is None:
            logger.warning(
                "No sub-swath in %s overlaps the requested AOI %s. "
                "Check your bbox or scene selection — the scene's overall "
                "footprint may cover the AOI while no single sub-swath does "
                "if the AOI straddles a sub-swath boundary.",
                zip_path.name,
                aoi_tuple,
            )
            return None

        out_path = (
            output_dir / f"{label}_{self._pol}.tif"
            if label
            else output_dir / f"{zip_path.stem}_{self._pol}.tif"
        )

        with zipfile.ZipFile(zip_path) as zf:
            with zf.open(matched_member) as src_f, open(out_path, "wb") as dst_f:
                dst_f.write(src_f.read())

        logger.info(
            "Extracted %s -> %s", self._swath_label(matched_member), out_path.name
        )
        return out_path

    def list_subswaths(self, zip_path: Union[str, Path]) -> List[str]:
        """
        List the measurement TIFF entries for the configured polarisation
        inside an SLC .SAFE zip, without extracting anything.
        """
        return self._find_subswaths(Path(zip_path))

    # ── internal helpers ──────────────────────────────────────────────────────

    def _resolve_path(
        self, source: Union["DownloadResult", str, Path]
    ) -> Optional[Path]:
        """
        Resolve a usable file path from a DownloadResult, string, or Path.

        Prefers DownloadResult.output_path (the exact path the provider
        actually wrote to) over any path re-derivation, which avoids
        filename/subfolder-mismatch bugs entirely.
        """
        # DownloadResult (duck-typed check to avoid a hard import dependency)
        if hasattr(source, "output_path") or hasattr(source, "output_paths"):
            output_path = getattr(source, "output_path", None)
            if output_path is not None:
                p = Path(output_path)
                if p.exists():
                    return p
                logger.warning(
                    "DownloadResult.output_path does not exist on disk: %s", p
                )
            output_paths = getattr(source, "output_paths", None) or []
            for p in output_paths:
                p = Path(p)
                if p.exists():
                    return p
            success = getattr(source, "success", None)
            error = getattr(source, "error", None)
            if success is False:
                logger.error("Download did not succeed: %s", error)
            return None

        # Plain path
        p = Path(source)
        if p.exists():
            return p
        logger.error("Path does not exist: %s", p)
        return None

    def _find_subswaths(self, zip_path: Path) -> List[str]:
        try:
            with zipfile.ZipFile(zip_path) as zf:
                names = zf.namelist()
        except zipfile.BadZipFile as exc:
            logger.error("Cannot open %s as a zip archive: %s", zip_path.name, exc)
            return []

        marker = f"-{self._pol}-"
        matches = [
            n
            for n in names
            if "/measurement/" in n and marker in n and n.endswith(".tiff")
        ]
        return sorted(matches)

    def _gcp_footprint(
        self, zip_path: Path, member_name: str
    ) -> Optional[Tuple[float, float, float, float]]:
        """Read embedded GCPs from a zipped measurement TIFF and return its
        approximate (min_lon, min_lat, max_lon, max_lat) footprint."""
        try:
            import rasterio
        except ImportError:
            raise ImportError('rasterio required: pip install "pygeofetch[geo]"')

        vsi_path = f"/vsizip/{zip_path}/{member_name}"
        try:
            with rasterio.open(vsi_path) as src:
                gcps, _gcp_crs = src.gcps
                if not gcps:
                    return None
                lons = [g.x for g in gcps]
                lats = [g.y for g in gcps]
                return (min(lons), min(lats), max(lons), max(lats))
        except Exception as exc:
            logger.warning("Could not read GCPs from %s: %s", member_name, exc)
            return None

    def _bbox_overlaps(
        self, a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]
    ) -> bool:
        """a, b = (min_lon, min_lat, max_lon, max_lat)."""
        return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])

    def _swath_label(self, member_name: str) -> str:
        lower = member_name.lower()
        for swath in ("iw1", "iw2", "iw3", "ew1", "ew2", "ew3", "ew4", "ew5"):
            if swath in lower:
                return swath.upper()
        return "?"
