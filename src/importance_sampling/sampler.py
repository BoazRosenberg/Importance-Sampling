"""Iterative Importance Sampling (IIS) algorithm for hierarchical model estimation.

A lightweight, general-purpose module for fitting multi-subject computational, cognitive,
or statistical models using iterative population importance sampling.
"""

from __future__ import annotations

import copy
import inspect
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.special import logsumexp

from importance_sampling.utils import (
    save_model as _save_model_func,
    load_model as _load_model_func,
    time_to_text,
)


class Sampler:
    """Iterative Importance Sampler for hierarchical model fitting.

    Parameters
    ----------
    data : Sequence[Any]
        A sequence where data[s] contains the dataset for subject s.
    model : Callable
        A model function with signature:
        model(subj_data, parameters, mode="log_likelihood")
        - In "log_likelihood" mode: returns a 1D array of log-likelihoods for candidate parameter samples.
        - In "simulate" mode: returns choice probabilities (p_choices) across trials.
    hyper_params : Dict[str, Dict[str, Any]]
        Initial population hyper-priors in latent normal space. Format:
        {'param_name': {'mean': float, 'sd': float, 'transform': optional_callable}}
    transformations : Optional[Dict[str, Callable]], default=None
        Functions that map latent normal parameters into valid model ranges
        (e.g., sigmoid for [0, 1], softplus or exp for positive values).
        If provided in hyper_params under the 'transform' key, this can be omitted.
    multinormal : Union[bool, str, List[Tuple[str, str]]], default=False
        - False: independent parameters (diagonal correlation matrix).
        - "full": estimate full correlation matrix between all parameters.
        - list of tuples: estimate correlations only for specific pairs, e.g. [('lr', 'inv_temp')].
    n_choices : int, default=2
        Number of choice options available per decision (e.g. 2 for a two-armed bandit).
        This is the amount of choices the likelihood is based on and is important for the BIC calculations.
    model_name : str, default="unnamed_model"
        A descriptive name for the model.
    description : str, default=""
        Optional details or notes.
    type : str, default="B"
        Optional model category tag.
    random_state : Optional[Union[int, np.random.Generator]], default=None
        Random seed or generator for reproducible sampling.
    """

    def __init__(
        self,
        data: Sequence[Any],
        model: Callable,
        hyper_params: Dict[str, Dict[str, Any]],
        transformations: Optional[Dict[str, Callable]] = None,
        multinormal: Union[bool, str, List[Tuple[str, str]]] = False,
        n_choices: int = 2,
        model_name: str = "unnamed_model",
        description: str = "",
        type: str = "B",
        random_state: Optional[Union[int, np.random.Generator]] = None,
    ):
        self.data = list(data)
        self.model = model

        # Number of choice options available per decision (e.g. 2 for a two-armed bandit).
        # This is the amount of choices the likelihood is based on and is important for the BIC calculations.
        self.n_choices = n_choices

        self.model_name = model_name
        self.description = description
        self.type = type

        self.n_subjects = len(self.data)
        self.subjects = list(range(self.n_subjects))
        self.params = list(hyper_params.keys())
        self.n_params = len(self.params)

        self.rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )

        # Extract latent normal mean and sd, plus optional transforms
        self.hyper_params: Dict[str, Dict[str, float]] = {}
        auto_transforms: Dict[str, Callable] = {}
        for k, v in hyper_params.items():
            self.hyper_params[k] = {
                "mean": float(v["mean"]),
                "sd": float(v["sd"]),
            }
            if "transform" in v and callable(v["transform"]):
                auto_transforms[k] = v["transform"]

        # Merge explicitly provided transformations or default to identity
        self.transformations: Dict[str, Callable] = {}
        for p in self.params:
            if transformations and p in transformations:
                self.transformations[p] = transformations[p]
            elif p in auto_transforms:
                self.transformations[p] = auto_transforms[p]
            else:
                self.transformations[p] = lambda x: x

        # Correlation matrix
        self.multinormal = multinormal
        self.cor_matrix = np.identity(self.n_params, dtype=np.float64)
        self.cor_matrices = [self.cor_matrix.copy()]

        self.hyper_params_list = [copy.deepcopy(self.hyper_params)]
        self.samples: Optional[Dict[str, Any]] = None
        self.mean_params: Optional[Dict[str, List[float]]] = None

        self.creation_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        self.total_fit_time = 0.0

        self.iterations = 0

        # Primary Evidence Attributes
        self.evidence: List[float] = []                    # General evidence (total log marginal LL across subjects)
        self.subj_evidence: List[List[float]] = []         # Subject-level evidence (shape: [n_iterations, n_subjects])
        self.evidence_change: List[float] = [0.0]

        self.mean_accuracy: List[float] = []
        self.BIC: List[float] = []
        self.BIC_change: List[float] = [0.0]
        self.mean_accuracy_change: List[float] = [0.0]

    # -------------------------------------------------------------------------
    # Backwards Compatibility Aliases
    # -------------------------------------------------------------------------
    @property
    def log_likelihood(self) -> List[float]:
        """Alias for self.evidence."""
        return self.evidence

    @property
    def log_likelihoods(self) -> List[List[float]]:
        """Alias for self.subj_evidence."""
        return self.subj_evidence

    @property
    def log_likelihood_change(self) -> List[float]:
        """Alias for self.evidence_change."""
        return self.evidence_change

    # -------------------------------------------------------------------------
    # Core Sampling & Resampling
    # -------------------------------------------------------------------------

    def _invoke_model(
        self,
        subj_data: Any,
        transformed_params: Dict[str, np.ndarray],
        mode: str = "log_likelihood",
        raw_samples: Optional[Dict[str, np.ndarray]] = None,
    ) -> Any:
        """Calls the model with pre-transformed parameters.

        Supports both modern 3-argument signature:
            model(subj_data, parameters, mode="log_likelihood")
        and legacy 4-argument signature:
            model(subj_data, parameters, transformations, mode="log_likelihood")
        Also supports either mode="log_likelihood" or mode="loglikelihood".
        """
        def _call_fn(m_arg: str):
            try:
                sig = inspect.signature(self.model)
                if len(sig.parameters) <= 3 or "transformations" not in sig.parameters:
                    return self.model(subj_data, transformed_params, mode=m_arg)
                else:
                    p = raw_samples if raw_samples is not None else transformed_params
                    return self.model(subj_data, p, self.transformations, mode=m_arg)
            except (ValueError, TypeError):
                try:
                    return self.model(subj_data, transformed_params, mode=m_arg)
                except TypeError:
                    p = raw_samples if raw_samples is not None else transformed_params
                    return self.model(subj_data, p, self.transformations, mode=m_arg)

        res = _call_fn(mode)
        if res is None:
            if mode == "log_likelihood":
                res = _call_fn("loglikelihood")
            elif mode == "loglikelihood":
                res = _call_fn("log_likelihood")
        return res

    def sample_resample(
        self, subj: int, n_samples: int = 1000
    ) -> Tuple[Dict[str, np.ndarray], float, Dict[str, float]]:
        """Sample candidate parameters from current prior, compute likelihoods, and resample."""
        subj_data = self.data[subj]

        # Draw standardized multinormal samples
        try:
            std_normals = self.rng.multivariate_normal(
                np.zeros(self.n_params), self.cor_matrix, size=n_samples
            ).T
        except np.linalg.LinAlgError:
            jitter = self.cor_matrix + np.eye(self.n_params) * 1e-6
            std_normals = self.rng.multivariate_normal(
                np.zeros(self.n_params), jitter, size=n_samples
            ).T

        # Scale by subject hyper_params in latent space
        raw_samples = {
            param: (
                std_normals[i] * self.hyper_params[param]["sd"]
                + self.hyper_params[param]["mean"]
            )
            for i, param in enumerate(self.params)
        }

        # Pre-transform candidate parameter particles into their valid bounds
        transformed_samples = {
            param: self.transformations[param](raw_samples[param])
            for param in self.params
        }

        # Calculate likelihoods from model (model receives pre-transformed parameters)
        log_likelihoods = np.asarray(
            self._invoke_model(subj_data, transformed_samples, mode="log_likelihood", raw_samples=raw_samples),
            dtype=np.float64,
        )

        if np.isnan(log_likelihoods).all():
            raise ValueError(f"All log-likelihoods are NaN for subject {subj}")

        # Compute importance weights stably: exp(LL - logsumexp(LL))
        log_sum_exp = logsumexp(log_likelihoods)
        weights = np.exp(log_likelihoods - log_sum_exp)
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)

        weights_sum = np.sum(weights)
        if weights_sum <= 0 or not np.isfinite(weights_sum):
            weights = np.ones(n_samples) / n_samples
        else:
            weights = weights / weights_sum

        # Weighted posterior mean parameters (in latent space)
        mean_params = {param: float(np.sum(weights * raw_samples[param])) for param in self.params}

        # Multinomial particle resampling
        resample_idx = self.rng.choice(n_samples, size=n_samples, p=weights, replace=True)
        resample = {param: raw_samples[param][resample_idx] for param in self.params}
        resample["log_likelihood"] = log_likelihoods[resample_idx]

        # Log marginal likelihood for subject: log( (1/N) * sum(exp(LL)) )
        log_mean_likelihood = float(log_sum_exp - np.log(n_samples))

        return resample, log_mean_likelihood, mean_params

    def adjust_hyper_priors(self, n_samples: int = 1000, disable_progress: bool = True) -> None:
        """Run one iteration of importance sampling across all subjects and update priors."""
        subject_samples: Dict[str, List[np.ndarray]] = {param: [] for param in self.params}
        mean_params: Dict[str, List[float]] = {param: [] for param in self.params}
        log_likelihoods: List[float] = []

        for s in range(self.n_subjects):
            resample, log_mean_ll, s_mean = self.sample_resample(s, n_samples=n_samples)
            for param in self.params:
                subject_samples[param].append(resample[param])
                mean_params[param].append(s_mean[param])
            log_likelihoods.append(log_mean_ll)

        # Pool particles across all subjects to update population distribution
        pooled_samples = {
            param: np.concatenate(subject_samples[param])
            for param in self.params
        }

        # Recalculate population mean and standard deviation
        updated_priors = {
            param: {
                "mean": float(np.mean(pooled_samples[param])),
                "sd": float(np.std(pooled_samples[param], ddof=1)),
            }
            for param in self.params
        }

        self.hyper_params_list.append(copy.deepcopy(updated_priors))
        self.hyper_params = updated_priors
        self.mean_params = mean_params
        self.samples = subject_samples

        # Update correlation matrix
        if not self.multinormal:
            self.cor_matrix = np.identity(self.n_params, dtype=np.float64)
        elif self.multinormal == "full":
            stacked = np.array([pooled_samples[k] for k in self.params], dtype=np.float64)
            cov_matrix = np.cov(stacked)
            std_devs = np.sqrt(np.diag(cov_matrix))
            std_outer = np.outer(std_devs, std_devs)
            std_outer[std_outer == 0] = 1e-8
            corr = cov_matrix / std_outer
            np.fill_diagonal(corr, 1.0)
            self.cor_matrix = np.nan_to_num(corr, nan=0.0)
            self.cor_matrices.append(self.cor_matrix.copy())
        elif isinstance(self.multinormal, list):
            param_indices = {p: i for i, p in enumerate(self.params)}
            for k1, k2 in self.multinormal:
                if k1 in param_indices and k2 in param_indices:
                    x = np.asarray(pooled_samples[k1], dtype=np.float64)
                    y = np.asarray(pooled_samples[k2], dtype=np.float64)
                    corr_val = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else 0.0
                    i, j = param_indices[k1], param_indices[k2]
                    self.cor_matrix[i, j] = np.nan_to_num(corr_val, nan=0.0)
                    self.cor_matrix[j, i] = self.cor_matrix[i, j]
            self.cor_matrices.append(self.cor_matrix.copy())

        # Track evidence & BIC
        iter_evidence = float(np.nansum(log_likelihoods))
        self.evidence.append(iter_evidence)
        self.subj_evidence.append(log_likelihoods)

        sample_size = max(1, self.n_subjects * self.n_choices)
        self.mean_accuracy.append(iter_evidence / sample_size)

        if self.multinormal == "full":
            n_hyper = 2 * self.n_params + self.n_params * (self.n_params - 1) / 2
        else:
            n_hyper = 2 * self.n_params
        bic_val = float(-2.0 * iter_evidence + n_hyper * np.log(sample_size))
        self.BIC.append(bic_val)

        if self.iterations > 0:
            self.evidence_change.append(self.evidence[-1] - self.evidence[-2])
            self.mean_accuracy_change.append(self.mean_accuracy[-1] - self.mean_accuracy[-2])
            self.BIC_change.append(self.BIC[-1] - self.BIC[-2])

    def iterative_model_fit(
        self,
        n_iterations: int = 10,
        n_samples: int = 1000,
        epsilon: float = 0.01,
        n_mean: int = 10,
        stop_at_convergence: bool = True,
        verbose: bool = True,
    ) -> "Sampler":
        """Run the iterative importance sampling estimation loop until convergence."""
        start = time.time()

        for i in range(n_iterations):
            self.adjust_hyper_priors(n_samples=n_samples, disable_progress=True)
            self.iterations += 1

            # Convergence check over window of n_mean iterations
            if i >= n_mean:
                change = (self.evidence[-1] - self.evidence[-n_mean]) / n_mean
                if change < epsilon and stop_at_convergence:
                    if verbose:
                        print(
                            f"\n✨ Converged after {i + 1} iterations.\n"
                            f"Total evidence: {round(self.evidence[-1], 4)}"
                        )
                    self.total_fit_time += time.time() - start
                    break

            elapsed = time.time() - start
            predicted_duration = (elapsed / (i + 1)) * (n_iterations - i - 1)

            if verbose:
                prefix = f"\n {self.model_name}\n Iteration: {i + 1} of {n_iterations}"
                change_n_mean = f", over last {n_mean} iterations: {round(change, 4)}" if i >= n_mean else ""
                desc = (
                    f"\nCurrent time:    {time.strftime('%H:%M:%S', time.localtime())}\n"
                    f"Elapsed:         {time_to_text(elapsed)}\n"
                    f"Time remaining:  {time_to_text(predicted_duration)}\n"
                    f"Evidence:        {round(self.evidence[-1], 4):>10} "
                    f"(change: {round(self.evidence_change[-1], 4):>7}"
                    f"{change_n_mean})\n"
                )
                print(prefix + desc)
        else:
            if verbose:
                print(
                    f"\nCompleted all {n_iterations} iterations without convergence.\n"
                    f"Final evidence: {round(self.evidence[-1], 4)}"
                )
            self.total_fit_time += time.time() - start

        return self

    # -------------------------------------------------------------------------
    # Simulation Utility
    # -------------------------------------------------------------------------

    def simulate(
        self,
        mode: str = "resample",
        subjects: Union[str, Sequence[int]] = "all",
        override_params: Optional[Dict[str, Any]] = None,
        n_samples: int = 1000,
    ) -> List[Any]:
        """Simulate choices using fitted parameters, returning subject data with mean choice probability.

        Returns
        -------
        List[Dict[str, Any]]
            The list of subject data as it was, with an added 'mean_choice_probability' array for each subject.
        """
        sim_data = []
        target_subjects = self.subjects if subjects == "all" else list(subjects)

        for s in target_subjects:
            subj_data = self.data[s]

            if mode == "resample" and self.samples is not None:
                raw_p = {k: self.samples[k][s] for k in self.params}
                transformed_p = {k: self.transformations[k](raw_p[k]) for k in self.params}
            elif mode == "hyper_params":
                raw_p = {
                    k: self.rng.normal(
                        self.hyper_params[k]["mean"], self.hyper_params[k]["sd"], size=n_samples
                    )
                    for k in self.params
                }
                transformed_p = {k: self.transformations[k](raw_p[k]) for k in self.params}
            elif mode == "override_params" and override_params is not None:
                raw_p = override_params
                transformed_p = override_params
            else:
                raw_p = {k: np.array([self.mean_params[k][s]]) for k in self.params}
                transformed_p = {k: self.transformations[k](raw_p[k]) for k in self.params}

            # Invoke model in simulate mode to get choice probabilities across trials
            p_choices = self._invoke_model(subj_data, transformed_p, mode="simulate", raw_samples=raw_p)

            p_arr = np.asarray(p_choices, dtype=np.float64)
            if p_arr.ndim > 1:
                # Shape (n_trials, n_samples) -> average across candidate parameter draws
                mean_p = np.mean(p_arr, axis=-1)
            else:
                mean_p = p_arr

            # Return original subject data dictionary/object with mean_choice_probability
            if hasattr(subj_data, "copy"):
                subj_sim = subj_data.copy()
            elif isinstance(subj_data, dict):
                subj_sim = dict(subj_data)
            else:
                subj_sim = {"data": subj_data}

            subj_sim["mean_choice_probability"] = mean_p
            sim_data.append(subj_sim)

        return sim_data

    # -------------------------------------------------------------------------
    # Persistence & Summary
    # -------------------------------------------------------------------------

    def save_model(self, filename: str = "", directory: str = "saved_models") -> str:
        """Save this sampler object to disk."""
        return _save_model_func(self, filename=filename, directory=directory)

    def summary(self) -> None:
        """Print fit summary statistics."""
        final_ev = self.evidence[-1] if self.evidence else np.nan
        final_bic = self.BIC[-1] if self.BIC else np.nan
        final_acc = self.mean_accuracy[-1] if self.mean_accuracy else np.nan

        print("Model name:           ", self.model_name)
        print("Description:          ", self.description)
        print("Parameters:           ", self.params)
        print("Number of subjects:   ", self.n_subjects)
        print("Choices per subject:  ", self.n_choices)
        print("Iterations:           ", self.iterations)
        print("Evidence (total):     ", round(final_ev, 4))
        print("Mean accuracy:        ", round(final_acc, 4))
        print("BIC:                  ", round(final_bic, 4))
        print("\nFitted Hyper-Priors:")
        for p in self.params:
            mu = round(self.hyper_params[p]["mean"], 4)
            sd = round(self.hyper_params[p]["sd"], 4)
            print(f"  • {p:>12s}: mean = {mu:8.4f}, sd = {sd:8.4f}")

    def create_report(
        self,
        filename: Optional[str] = None,
        show: bool = True,
        transformed: bool = True,
        renderer: Optional[str] = None,
    ) -> Any:
        """Generate an interactive report widget for model diagnostics and results.

        Visualizes:
        1. Hyperparameter evolution across iterations (solid line for mean, shaded area for +/- 1 SD).
        2. Model fit and evidence convergence (total evidence and BIC).
        3. Individual subject posterior means and parameter distribution.

        Parameters
        ----------
        filename : Optional[str], default=None
            If provided (e.g. 'model_report.html'), exports a self-contained, interactive
            HTML report that can be opened in any web browser without a Python runtime.
        show : bool, default=True
            Whether to display the interactive figure in the current environment
            (e.g., Jupyter notebook, Google Colab, or browser).
        transformed : bool, default=True
            If True, displays hyperparameters transformed into their valid domain bounds
            (e.g., [0, 1] for learning rate, strictly positive for inverse temperature).
            If False, displays parameters in latent normal space.
        renderer : Optional[str], default=None
            Plotly renderer to use when displaying the figure (e.g., 'browser', 'notebook', 'colab').

        Returns
        -------
        plotly.graph_objects.Figure
            The interactive Plotly Figure object containing the multi-panel report.
        """
        from importance_sampling.report import create_report as _create_report_func

        return _create_report_func(
            self,
            filename=filename,
            show=show,
            transformed=transformed,
            renderer=renderer,
        )


# -----------------------------------------------------------------------------
# Module Function: Load Model
# -----------------------------------------------------------------------------

def load_model(filename: str, directory: str = "saved_models") -> Sampler:
    """Load a previously saved Sampler instance from disk."""
    return _load_model_func(filename, directory=directory)
