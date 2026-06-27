"""
AutoML for geospatial models (D6) — automated architecture search and HPO.
"""
from __future__ import annotations
import logging
from typing import Any, Callable, Dict, List, Optional, Union
logger = logging.getLogger(__name__)


class GeoAutoML:
    """Automated hyperparameter optimisation for geospatial models (D6).

    Uses Optuna (or Ray Tune) to automatically search:
        - Learning rate, weight decay
        - Architecture choices (backbone, head)
        - Loss function and weights
        - Augmentation strengths
        - Batch size and scheduler

    Example::

        automl = GeoAutoML(metric="val_iou", n_trials=50)
        best_config = automl.search(
            train_fn=my_train_function,
            search_space={
                "lr": ("float", 1e-5, 1e-2, "log"),
                "backbone": ("categorical", ["resnet50", "resnet101", "efficientnet_b4"]),
                "loss": ("categorical", ["combo", "tversky", "focal"]),
            }
        )
    """

    SEARCH_BACKENDS = ["optuna", "ray_tune"]

    def __init__(
        self,
        metric: str = "val_iou",
        direction: str = "maximize",
        n_trials: int = 50,
        timeout_s: Optional[int] = None,
        backend: str = "optuna",
        n_jobs: int = 1,
        storage: Optional[str] = None,
        study_name: Optional[str] = None,
    ) -> None:
        self.metric = metric
        self.direction = direction
        self.n_trials = n_trials
        self.timeout_s = timeout_s
        self.backend = backend
        self.n_jobs = n_jobs
        self.storage = storage
        self.study_name = study_name or "pgv_automl"
        self._study = None

    def search(
        self,
        train_fn: Callable[[Dict], float],
        search_space: Dict[str, Any],
        pruner: Optional[str] = "median",
        sampler: Optional[str] = "tpe",
    ) -> Dict[str, Any]:
        """Run hyperparameter search.

        Args:
            train_fn: Function that takes a config dict and returns the metric value
            search_space: Dict of {param_name: (type, *args)}
                Types: "float" (low, high, [log]), "int" (low, high), "categorical" (choices)
            pruner: "median" | "hyperband" | None
            sampler: "tpe" | "cmaes" | "random"

        Returns:
            Dict with best_params, best_value, n_trials, study
        """
        if self.backend == "optuna":
            return self._search_optuna(train_fn, search_space, pruner, sampler)
        elif self.backend == "ray_tune":
            return self._search_ray(train_fn, search_space)
        else:
            return {"success": False, "error": f"Unknown backend: {self.backend}"}

    def _search_optuna(self, train_fn, search_space, pruner, sampler) -> Dict[str, Any]:
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            return {"success": False, "error": "pip install optuna"}

        pruners_map = {
            "median":    optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
            "hyperband": optuna.pruners.HyperbandPruner(),
            "none":      optuna.pruners.NopPruner(),
        }
        samplers_map = {
            "tpe":    optuna.samplers.TPESampler(seed=42),
            "cmaes":  optuna.samplers.CmaEsSampler(seed=42),
            "random": optuna.samplers.RandomSampler(seed=42),
        }

        def _objective(trial: optuna.Trial) -> float:
            config = {}
            for name, spec in search_space.items():
                kind = spec[0]
                if kind == "float":
                    low, high = spec[1], spec[2]
                    log = len(spec) > 3 and spec[3] == "log"
                    config[name] = trial.suggest_float(name, low, high, log=log)
                elif kind == "int":
                    config[name] = trial.suggest_int(name, spec[1], spec[2])
                elif kind == "categorical":
                    config[name] = trial.suggest_categorical(name, spec[1])
                else:
                    config[name] = spec[1]
            try:
                value = train_fn(config)
                return float(value) if value is not None else float("-inf")
            except Exception as exc:
                logger.warning("Trial failed: %s", exc)
                raise optuna.exceptions.TrialPruned()

        self._study = optuna.create_study(
            direction=self.direction,
            pruner=pruners_map.get(pruner or "median"),
            sampler=samplers_map.get(sampler or "tpe"),
            study_name=self.study_name,
            storage=self.storage,
            load_if_exists=True,
        )

        self._study.optimize(
            _objective,
            n_trials=self.n_trials,
            timeout=self.timeout_s,
            n_jobs=self.n_jobs,
            show_progress_bar=True,
        )

        best = self._study.best_trial
        return {
            "success": True,
            "best_params": best.params,
            "best_value": round(best.value, 6),
            "n_trials_completed": len(self._study.trials),
            "metric": self.metric,
            "direction": self.direction,
            "study_name": self.study_name,
        }

    def _search_ray(self, train_fn, search_space) -> Dict[str, Any]:
        try:
            from ray import tune, air
        except ImportError:
            return {"success": False, "error": "pip install ray[tune]"}

        ray_space = {}
        for name, spec in search_space.items():
            kind = spec[0]
            if kind == "float":
                ray_space[name] = tune.loguniform(spec[1], spec[2]) if (len(spec) > 3 and spec[3] == "log") else tune.uniform(spec[1], spec[2])
            elif kind == "int":
                ray_space[name] = tune.randint(spec[1], spec[2] + 1)
            elif kind == "categorical":
                ray_space[name] = tune.choice(spec[1])

        def _ray_objective(config):
            val = train_fn(config)
            tune.report(**{self.metric: val})

        analysis = tune.run(
            _ray_objective,
            config=ray_space,
            num_samples=self.n_trials,
            metric=self.metric,
            mode=self.direction[:3],  # "max" or "min"
        )

        best = analysis.best_config
        best_val = analysis.best_result.get(self.metric)
        return {
            "success": True,
            "best_params": best,
            "best_value": best_val,
            "n_trials_completed": self.n_trials,
        }

    def importance(self) -> Dict[str, float]:
        """Return hyperparameter importance from the completed study."""
        if self._study is None:
            return {}
        try:
            import optuna
            imp = optuna.importance.get_param_importances(self._study)
            return {k: round(v, 4) for k, v in imp.items()}
        except Exception:
            return {}

    def plot_optimization_history(self, save_path: Optional[str] = None) -> None:
        if self._study is None: return
        try:
            import optuna
            fig = optuna.visualization.plot_optimization_history(self._study)
            if save_path:
                fig.write_image(save_path)
            else:
                fig.show()
        except Exception as exc:
            logger.warning("Plot failed: %s", exc)
