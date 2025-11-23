# -*- coding: utf-8 -*-
"""
Lightweight Bayesian-style hyperparameter tuner to answer the
"too many knobs" criticism with an automated search over a reduced set.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from types import SimpleNamespace

import numpy as np

from dataset_use.NYC.src.algorithms import algorithms_bridge as bridge


@dataclass
class ReducedHyperparams:
    """
    Five core knobs that summarize the adaptive postprocess behavior.
    """

    variance_quantile: float = 0.75  # high-variance cutoff (quantile)
    sparsity_threshold: int = 5  # reports < threshold => sparse
    privacy_tension: float = 0.75  # budget consumption ratio to tighten smoothing
    change_sensitivity: float = 2.0  # change > sensitivity * sigma => change detected
    smoothing_intensity: float = 0.5  # [0,1], maps to alpha/proc_var

    def to_full_params(self, data_variance: float = 100.0) -> Dict[str, Any]:
        """
        Expand the reduced knobs into full parameter names used by sa_htd_paper.
        """
        dv = float(max(data_variance, 1e-6))
        var_high = max(1.0, dv * (1.0 + (self.variance_quantile - 0.5) * 2.0))
        var_low = 0.25 * var_high

        intensity = float(np.clip(self.smoothing_intensity, 0.0, 1.0))
        alpha_lap = intensity * 0.4  # cap at 0.4
        proc_var = intensity * 30.0  # cap at 30

        return {
            # aligns with LightweightVarianceAwarePostprocessor params
            "post_var_threshold_low": var_low,
            "post_var_threshold_high": var_high,
            "post_sparse_threshold": int(self.sparsity_threshold),
            "post_privacy_tension_ratio": float(self.privacy_tension),
            "post_change_sensitivity": float(self.change_sensitivity),
            "post_change_window": 3,
            "post_convergence_window": 5,
            "post_convergence_threshold": 0.05,
            "post_warmup_rounds": 3,
            "enable_privacy_adaptive": True,
            "use_privacy_aware_postprocess": True,
            # smoothing strength
            "post_lap_alpha": alpha_lap,
            "post_process_var": proc_var,
        }


class BayesianHyperparameterTuner:
    """
    Simplified TPE-style optimizer: random explore first, then sample near good trials.
    """

    def __init__(
        self,
        objective_func: Callable[[ReducedHyperparams], float],
        n_trials: int = 50,
        seed: int = 2025,
    ):
        self.objective = objective_func
        self.n_trials = int(n_trials)
        self.rng = np.random.default_rng(seed)

        self.search_space = {
            "variance_quantile": (0.6, 0.9),
            "sparsity_threshold": (3, 10),  # integer
            "privacy_tension": (0.6, 0.9),
            "change_sensitivity": (1.0, 3.0),
            "smoothing_intensity": (0.2, 0.8),
        }

        self.history: List[Tuple[ReducedHyperparams, float]] = []
        self.best_params: Optional[ReducedHyperparams] = None
        self.best_score: float = float("inf")

    def sample_params(self, use_tpe: bool = True) -> ReducedHyperparams:
        if not use_tpe or len(self.history) < 10:
            return ReducedHyperparams(
                variance_quantile=self.rng.uniform(*self.search_space["variance_quantile"]),
                sparsity_threshold=int(
                    self.rng.integers(
                        self.search_space["sparsity_threshold"][0],
                        self.search_space["sparsity_threshold"][1] + 1,
                    )
                ),
                privacy_tension=self.rng.uniform(*self.search_space["privacy_tension"]),
                change_sensitivity=self.rng.uniform(*self.search_space["change_sensitivity"]),
                smoothing_intensity=self.rng.uniform(*self.search_space["smoothing_intensity"]),
            )
        return self._tpe_sample()

    def _tpe_sample(self) -> ReducedHyperparams:
        sorted_history = sorted(self.history, key=lambda x: x[1])
        n_good = max(5, len(sorted_history) // 4)
        good_samples = [h[0] for h in sorted_history[:n_good]]
        if not good_samples:
            return self.sample_params(use_tpe=False)

        good_dict = {
            "variance_quantile": [s.variance_quantile for s in good_samples],
            "sparsity_threshold": [s.sparsity_threshold for s in good_samples],
            "privacy_tension": [s.privacy_tension for s in good_samples],
            "change_sensitivity": [s.change_sensitivity for s in good_samples],
            "smoothing_intensity": [s.smoothing_intensity for s in good_samples],
        }

        def sample_around(values, bounds, is_int: bool = False):
            mean = float(np.mean(values))
            std = float(np.std(values) + 1e-6)
            sampled = mean + self.rng.normal(0.0, 0.5 * std)
            sampled = float(np.clip(sampled, *bounds))
            return int(round(sampled)) if is_int else sampled

        return ReducedHyperparams(
            variance_quantile=sample_around(
                good_dict["variance_quantile"],
                self.search_space["variance_quantile"],
            ),
            sparsity_threshold=sample_around(
                good_dict["sparsity_threshold"],
                self.search_space["sparsity_threshold"],
                is_int=True,
            ),
            privacy_tension=sample_around(
                good_dict["privacy_tension"],
                self.search_space["privacy_tension"],
            ),
            change_sensitivity=sample_around(
                good_dict["change_sensitivity"],
                self.search_space["change_sensitivity"],
            ),
            smoothing_intensity=sample_around(
                good_dict["smoothing_intensity"],
                self.search_space["smoothing_intensity"],
            ),
        )

    def optimize(self) -> Tuple[ReducedHyperparams, float]:
        print(f"[Bayesian Tuning] Starting {self.n_trials} trials...")

        for trial in range(self.n_trials):
            params = self.sample_params(use_tpe=(trial >= 10))
            score = float(self.objective(params))
            if not np.isfinite(score):
                score = float("inf")

            self.history.append((params, score))

            if score < self.best_score:
                self.best_score = score
                self.best_params = params
                print(f"  Trial {trial + 1}: New best RMSE={score:.3f} | {params}")
            elif trial % 10 == 0:
                print(f"  Trial {trial + 1}: RMSE={score:.3f}")

        print(f"\n[Bayesian Tuning] Done. Best RMSE={self.best_score:.3f}")
        if self.best_params is not None:
            print(f"Best params: {self.best_params}")

        return self.best_params, self.best_score

    def plot_optimization_trace(self) -> Dict[str, Any]:
        trials = list(range(len(self.history)))
        scores = [h[1] for h in self.history]
        best_so_far = []
        current_best = float("inf")
        for score in scores:
            current_best = min(current_best, score)
            best_so_far.append(current_best)

        return {
            "trials": trials,
            "scores": scores,
            "best_so_far": best_so_far,
        }


def make_sahtd_objective(
    rounds_iter_factory: Callable[[], Iterable],
    base_params: Any,
    n_workers: int,
    data_variance: float = 100.0,
    max_rounds: Optional[int] = None,
) -> Callable[[ReducedHyperparams], float]:
    """
    Wrap sa_htd_paper into a numeric objective for tuning.

    rounds_iter_factory must yield a fresh iterator per call (to avoid exhaustion).
    """

    def _merge_params(overrides: Dict[str, Any]) -> SimpleNamespace:
        if base_params is None:
            base_dict: Dict[str, Any] = {}
        elif isinstance(base_params, dict):
            base_dict = dict(base_params)
        else:
            base_dict = dict(getattr(base_params, "__dict__", vars(base_params)))
        base_dict.update(overrides)
        return SimpleNamespace(**base_dict)

    def objective(hp: ReducedHyperparams) -> float:
        params_ns = _merge_params(hp.to_full_params(data_variance=data_variance))
        logs = bridge.sahtd_paper_bridge(rounds_iter_factory(), n_workers, params=params_ns)
        if max_rounds is not None and isinstance(logs, list):
            logs = logs[:max_rounds]

        rmse_vals = np.array([float(log.get("rmse", np.nan)) for log in logs], float)
        if rmse_vals.size == 0 or not np.isfinite(rmse_vals).any():
            return float("inf")
        return float(np.nanmean(rmse_vals[np.isfinite(rmse_vals)]))

    return objective
