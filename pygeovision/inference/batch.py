"""Batch inference engine (B2, B6) — parallel multi-file inference."""
from __future__ import annotations
import logging, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
logger = logging.getLogger(__name__)


class BatchInferenceEngine:
    """High-throughput batch inference for directories of GeoTIFFs (B2, B6).

    Processes entire directories with configurable parallelism.
    Supports distributed inference across multiple GPUs/nodes.

    Example::

        engine = BatchInferenceEngine(model, device="cuda", n_workers=4)
        results = engine.run_directory(
            "./data/sentinel2/", "./results/predictions/",
            pattern="*.tif",
        )
    """

    def __init__(
        self,
        model: Any,
        device: Optional[str] = None,
        n_workers: int = 1,
        chip_size: int = 512,
        overlap: int = 64,
        num_classes: int = 2,
        blend_mode: str = "gaussian",
        half_precision: bool = True,
        on_error: str = "skip",       # skip | abort | warn
    ) -> None:
        self.model = model
        self.device = device
        self.n_workers = n_workers
        self.chip_size = chip_size
        self.overlap = overlap
        self.num_classes = num_classes
        self.blend_mode = blend_mode
        self.half_precision = half_precision
        self.on_error = on_error

    def run_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        pattern: str = "*.tif",
        band_selection: Optional[List[int]] = None,
        skip_existing: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict[str, Any]:
        """Process all matching GeoTIFFs in a directory.

        Args:
            input_dir: Input directory
            output_dir: Output directory (preserves subdirectory structure)
            pattern: Glob pattern for input files
            skip_existing: Skip files already processed
            progress_callback: Called with (current, total, filename) for progress

        Returns:
            Dict with n_success, n_failed, failed_files, total_time_s
        """
        from pygeovision.inference.tiled import TiledInference

        input_dir  = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        all_files = sorted(input_dir.rglob(pattern))
        if not all_files:
            return {"success": False, "error": f"No files matching '{pattern}' in {input_dir}"}

        logger.info("BatchInference: %d files | %d workers | device=%s",
                    len(all_files), self.n_workers, self.device)

        # Build tasks
        tasks = []
        for f in all_files:
            rel = f.relative_to(input_dir)
            out_path = output_dir / rel
            if skip_existing and out_path.exists():
                continue
            tasks.append((f, out_path))

        logger.info("BatchInference: %d files to process (%d skipped)",
                    len(tasks), len(all_files) - len(tasks))

        results = {"succeeded": [], "failed": [], "skipped": len(all_files) - len(tasks)}
        t_start = time.time()

        def _process_one(args):
            in_path, out_path = args
            try:
                inf = TiledInference(
                    model=self.model,
                    chip_size=self.chip_size,
                    overlap=self.overlap,
                    blend_mode=self.blend_mode,
                    num_classes=self.num_classes,
                    device=self.device,
                    half_precision=self.half_precision,
                )
                r = inf.infer(in_path, out_path, band_selection=band_selection)
                return in_path, True, r
            except Exception as exc:
                return in_path, False, {"error": str(exc)}

        completed = 0
        with ThreadPoolExecutor(max_workers=self.n_workers) as pool:
            futures = {pool.submit(_process_one, t): t for t in tasks}
            for fut in as_completed(futures):
                in_path, ok, info = fut.result()
                completed += 1
                if ok:
                    results["succeeded"].append(str(in_path))
                else:
                    results["failed"].append({"path": str(in_path), "error": info.get("error")})
                    if self.on_error == "abort":
                        pool.shutdown(wait=False, cancel_futures=True)
                        break
                if progress_callback:
                    progress_callback(completed, len(tasks), str(in_path))

        total_time = time.time() - t_start
        results.update({
            "n_success": len(results["succeeded"]),
            "n_failed": len(results["failed"]),
            "total_time_s": round(total_time, 1),
            "throughput_fps": round(len(results["succeeded"]) / max(total_time, 0.001), 2),
        })
        logger.info("BatchInference complete: %d/%d | %.1fs | %.2f fps",
                    results["n_success"], len(tasks), total_time, results["throughput_fps"])
        return results

    def run_multi_gpu(
        self,
        input_files: List[str],
        output_dir: str,
        gpu_ids: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Distribute inference across multiple GPUs (B6)."""
        try:
            import torch
            n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        except ImportError:
            n_gpus = 0

        if n_gpus <= 1:
            return self.run_directory(
                Path(input_files[0]).parent, output_dir
            )

        gpu_ids = gpu_ids or list(range(n_gpus))
        # Split files across GPUs
        splits = [input_files[i::len(gpu_ids)] for i in range(len(gpu_ids))]

        def _gpu_worker(files, gpu_id):
            engine = BatchInferenceEngine(
                self.model, device=f"cuda:{gpu_id}",
                n_workers=2, chip_size=self.chip_size,
                overlap=self.overlap, num_classes=self.num_classes,
            )
            results = {"succeeded": [], "failed": []}
            for f in files:
                out = Path(output_dir) / Path(f).name
                from pygeovision.inference.tiled import TiledInference
                inf = TiledInference(self.model, device=f"cuda:{gpu_id}",
                                      chip_size=self.chip_size, num_classes=self.num_classes)
                r = inf.infer(f, str(out))
                (results["succeeded"] if r.get("success") else results["failed"]).append(f)
            return results

        all_results = {"succeeded": [], "failed": []}
        with ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
            futures = {pool.submit(_gpu_worker, splits[i], gpu_ids[i]): i for i in range(len(gpu_ids))}
            for fut in as_completed(futures):
                r = fut.result()
                all_results["succeeded"].extend(r["succeeded"])
                all_results["failed"].extend(r["failed"])

        return all_results
