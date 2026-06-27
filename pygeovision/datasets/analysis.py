"""
EarthNets Dataset Analysis Tools (Phase 5.1).
Volume trends, resolution distributions, correlation matrix,
t-SNE/UMAP clustering, radar charts — all from the 500+ catalog.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DatasetAnalyzer:
    """Analyse the 500+ dataset catalog using EarthNets methodology.

    Example::

        from pygeovision.datasets.analysis import DatasetAnalyzer
        from pygeovision.datasets.registry import dataset_registry

        analyzer = DatasetAnalyzer(dataset_registry)
        analyzer.volume_trend()                  # volume added per year
        analyzer.resolution_distribution()       # histogram of GSD values
        analyzer.domain_distribution()           # pie/bar of domain counts
        analyzer.correlation_matrix()            # dataset similarity heatmap
        analyzer.cluster(method="tsne")          # 2D embedding of all datasets
        analyzer.radar_chart(["EuroSAT", "BigEarthNet", "LoveDA"])
    """

    def __init__(self, registry: Any = None) -> None:
        if registry is None:
            from pygeovision.datasets.registry import dataset_registry
            registry = dataset_registry
        self.registry = registry
        self._datasets = registry.all()

    # ── 5.1 Volume trend ──────────────────────────────────────────────
    def volume_trend(self, show: bool = True) -> Dict[int, float]:
        """Cumulative storage volume added per year (GB → TB)."""
        year_vol: Dict[int, float] = {}
        for d in self._datasets:
            year_vol[d.year] = year_vol.get(d.year, 0.0) + d.volume_gb
        # Cumulative
        sorted_years = sorted(year_vol.keys())
        cumulative: Dict[int, float] = {}
        total = 0.0
        for y in sorted_years:
            total += year_vol[y]
            cumulative[y] = round(total / 1024, 2)   # GB → TB

        if show:
            print(f"\nVolume trend (cumulative TB by year):")
            for y, tb in sorted(cumulative.items()):
                bar = "█" * min(int(tb * 2), 60)
                print(f"  {y}: {bar} {tb:.1f} TB")
        return cumulative

    # ── Resolution distribution ───────────────────────────────────────
    def resolution_distribution(self, bins: Optional[List[float]] = None, show: bool = True) -> Dict[str, int]:
        """Histogram of spatial resolution (GSD) values."""
        bins = bins or [0.0, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 100.0, float("inf")]
        labels = ["<0.1m","0.1-0.5m","0.5-1m","1-5m","5-10m","10-30m","30-100m",">100m"]
        counts = {l: 0 for l in labels}
        for d in self._datasets:
            r = d.resolution_m
            for i in range(len(bins) - 1):
                if bins[i] <= r < bins[i + 1]:
                    counts[labels[i]] += 1
                    break
        if show:
            print(f"\nSpatial resolution distribution:")
            for label, count in counts.items():
                bar = "█" * count
                print(f"  {label:>10}: {bar} ({count})")
        return counts

    # ── Domain distribution ───────────────────────────────────────────
    def domain_distribution(self, show: bool = True) -> Dict[str, int]:
        """Count datasets per research domain."""
        counts: Dict[str, int] = {}
        for d in self._datasets:
            counts[d.domain] = counts.get(d.domain, 0) + 1
        if show:
            print(f"\nDataset distribution by domain ({len(counts)} domains):")
            for domain, count in sorted(counts.items(), key=lambda x: -x[1]):
                bar = "█" * count
                print(f"  {domain:<22}: {bar} ({count})")
        return counts

    # ── Modality distribution ─────────────────────────────────────────
    def modality_distribution(self, show: bool = True) -> Dict[str, int]:
        """Count datasets per data modality."""
        counts: Dict[str, int] = {}
        for d in self._datasets:
            counts[d.modality] = counts.get(d.modality, 0) + 1
        if show:
            print(f"\nDataset distribution by modality:")
            for mod, count in sorted(counts.items(), key=lambda x: -x[1]):
                bar = "█" * count
                print(f"  {mod:<18}: {bar} ({count})")
        return counts

    # ── Correlation matrix ────────────────────────────────────────────
    def correlation_matrix(
        self,
        names: Optional[List[str]] = None,
        show: bool = True,
    ) -> List[List[float]]:
        """Compute pairwise EarthNets similarity matrix.

        Returns an N×N matrix where entry [i][j] is the similarity
        between datasets i and j (0=dissimilar, 1=identical).
        """
        datasets = (
            [self.registry[n] for n in names]
            if names
            else self._datasets[:30]  # limit for display
        )
        n = len(datasets)
        matrix = [[0.0] * n for _ in range(n)]
        import math
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i][j] = 1.0
                    continue
                a, b = datasets[i], datasets[j]
                domain_match = 1.0 if a.domain == b.domain else 0.0
                modal_match  = 1.0 if a.modality == b.modality else 0.3
                res_sim = 1.0 / (1.0 + abs(math.log1p(a.resolution_m) - math.log1p(b.resolution_m)))
                task_sim = len(set(a.tasks) & set(b.tasks)) / max(len(set(a.tasks) | set(b.tasks)), 1)
                scale_sim = min(a.n_samples, b.n_samples) / max(a.n_samples, b.n_samples, 1)
                matrix[i][j] = 0.35*domain_match + 0.2*modal_match + 0.2*res_sim + 0.15*task_sim + 0.1*scale_sim

        if show:
            names_out = [d.name[:12] for d in datasets]
            print(f"\nEarthNets Similarity Matrix (top-{n} datasets):")
            header = "".join(f"{nm:>13}" for nm in names_out)
            print(f"  {'':>14}{header}")
            for i, row in enumerate(matrix):
                vals = "".join(f"{v:>13.2f}" for v in row)
                print(f"  {names_out[i]:>14}{vals}")

        return matrix

    # ── t-SNE / UMAP clustering ────────────────────────────────────────
    def cluster(
        self,
        method: str = "tsne",
        n_components: int = 2,
        save_path: Optional[str] = None,
    ) -> Optional[Any]:
        """Cluster all datasets in 2D using t-SNE or UMAP.

        Requires: scikit-learn (t-SNE) or umap-learn (UMAP).
        """
        try:
            import numpy as np
        except ImportError:
            logger.warning("numpy required for clustering: pip install numpy")
            return None

        # Build feature matrix
        domain_map = {d: i for i, d in enumerate(sorted(set(x.domain for x in self._datasets)))}
        modal_map = {m: i for i, m in enumerate(sorted(set(x.modality for x in self._datasets)))}
        import math
        features = []
        for d in self._datasets:
            feat = [
                domain_map.get(d.domain, 0) / max(len(domain_map), 1),
                modal_map.get(d.modality, 0) / max(len(modal_map), 1),
                math.log1p(d.resolution_m) / 10.0,
                math.log1p(d.n_samples) / 20.0,
                math.log1p(d.volume_gb) / 10.0,
                (d.year - 2000) / 25.0,
                min(d.n_classes, 100) / 100.0,
            ]
            features.append(feat)

        X = np.array(features)
        labels = [d.name for d in self._datasets]
        domains = [d.domain for d in self._datasets]

        try:
            if method == "umap":
                from umap import UMAP
                reducer = UMAP(n_components=n_components, random_state=42, n_neighbors=15, min_dist=0.1)
            else:
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=n_components, random_state=42, perplexity=min(30, len(X)-1))

            embedding = reducer.fit_transform(X)
            logger.info("%s clustering complete: shape=%s", method.upper(), embedding.shape)

            if save_path:
                try:
                    import matplotlib.pyplot as plt
                    unique_domains = list(set(domains))
                    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_domains)))
                    color_map = {d: colors[i] for i, d in enumerate(unique_domains)}
                    fig, ax = plt.subplots(figsize=(14, 10))
                    for domain in unique_domains:
                        idx = [i for i, dom in enumerate(domains) if dom == domain]
                        ax.scatter(embedding[idx, 0], embedding[idx, 1],
                                   c=[color_map[domain]], label=domain, alpha=0.7, s=40)
                    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
                    ax.set_title(f"Dataset Embedding ({method.upper()}) — {len(self._datasets)} datasets")
                    ax.set_xlabel(f"{method.upper()}-1"); ax.set_ylabel(f"{method.upper()}-2")
                    plt.tight_layout()
                    plt.savefig(save_path, dpi=150, bbox_inches="tight")
                    logger.info("Cluster plot saved → %s", save_path)
                except ImportError:
                    logger.warning("matplotlib required for plot")
            return embedding
        except ImportError as exc:
            logger.warning("%s not available: %s. pip install scikit-learn umap-learn", method, exc)
            return None

    # ── Radar chart ───────────────────────────────────────────────────
    def radar_chart(
        self,
        dataset_names: List[str],
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> Optional[Any]:
        """Radar chart comparing datasets across 6 EarthNets dimensions."""
        try:
            import numpy as np
            import math
        except ImportError:
            logger.warning("numpy required: pip install numpy")
            return None

        datasets = []
        for name in dataset_names:
            try:
                datasets.append(self.registry[name])
            except KeyError:
                logger.warning("Dataset '%s' not found; skipping.", name)

        if not datasets:
            return None

        # 6 normalised dimensions
        max_res_log = 8.0
        all_samples = [d.n_samples for d in self._datasets]
        all_vols = [d.volume_gb for d in self._datasets]
        max_log_samples = math.log1p(max(all_samples))
        max_log_vol = math.log1p(max(all_vols))

        def _features(d: Any) -> List[float]:
            return [
                1.0 - min(math.log1p(d.resolution_m) / max_res_log, 1.0),  # Resolution (finer=higher)
                math.log1p(d.n_samples) / max_log_samples,                 # Scale
                min(d.n_classes / 50.0, 1.0),                              # Diversity
                min(d.year - 2000, 24) / 24.0,                             # Recency
                min(math.log1p(d.volume_gb) / max_log_vol, 1.0),           # Volume
                1.0 if d.download_url else 0.0,                            # Accessibility
            ]

        dims = ["Resolution", "Scale", "Diversity", "Recency", "Volume", "Access"]
        N = len(dims)
        angles = [2 * math.pi * i / N for i in range(N)]
        angles_closed = angles + [angles[0]]

        if show:
            print(f"\nRadar comparison: {[d.name for d in datasets]}")
            print(f"  {'Dimension':<15} " + "  ".join(f"{d.name:>12}" for d in datasets))
            for i, dim in enumerate(dims):
                vals = [_features(d)[i] for d in datasets]
                print(f"  {dim:<15} " + "  ".join(f"{v:>12.2f}" for v in vals))

        try:
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection="polar")
            colors = plt.cm.tab10(np.linspace(0, 0.8, len(datasets)))
            for d, color in zip(datasets, colors):
                vals = _features(d)
                vals_closed = vals + [vals[0]]
                ax.plot(angles_closed, vals_closed, "o-", linewidth=2, color=color, label=d.name)
                ax.fill(angles_closed, vals_closed, alpha=0.1, color=color)
            ax.set_xticks(angles)
            ax.set_xticklabels(dims, fontsize=11)
            ax.set_ylim(0, 1)
            ax.set_yticks([0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=8)
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
            ax.set_title("EarthNets Dataset Radar Chart", pad=20)
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                logger.info("Radar chart saved → %s", save_path)
            return fig
        except ImportError:
            logger.warning("matplotlib required for radar chart")
            return None

    # ── Summary report ────────────────────────────────────────────────
    def full_report(self) -> Dict[str, Any]:
        """Generate complete EarthNets-style catalog analysis report."""
        s = self.registry.summary()
        return {
            "catalog_summary": s,
            "volume_trend": self.volume_trend(show=False),
            "resolution_distribution": self.resolution_distribution(show=False),
            "domain_distribution": self.domain_distribution(show=False),
            "modality_distribution": self.modality_distribution(show=False),
            "top_segmentation": [d.name for d in self.registry.top_for_task("segmentation", n=5)],
            "top_detection":    [d.name for d in self.registry.top_for_task("detection", n=5)],
            "top_classification": [d.name for d in self.registry.top_for_task("classification", n=5)],
            "top_change_detection": [d.name for d in self.registry.top_for_task("change_detection", n=5)],
        }
