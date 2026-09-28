export interface PackageFile {
  path: string;
  name: string;
  language: string;
  category: 'core' | 'demo' | 'config' | 'doc';
  description: string;
  content: string;
}

export const PACKAGE_FILES: PackageFile[] = [
  {
    path: 'pyproject.toml',
    name: 'pyproject.toml',
    language: 'toml',
    category: 'config',
    description: 'Modern PEP 517/621 Python packaging metadata and dependencies.',
    content: `[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "importance_sampling"
version = "0.2.0"
description = "Iterative Importance Sampling (IIS) for Hierarchical Bayesian Model Estimation"
readme = "README.md"
authors = [
    { name = "Boaz Rosenberg", email = "BoazRsnbrg@gmail.com" }
]
license = { text = "MIT" }
keywords = [
    "importance-sampling",
    "computational-modeling",
    "cognitive-science",
    "reinforcement-learning",
    "bayesian-estimation",
    "hierarchical-models"
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]
requires-python = ">=3.8"
dependencies = [
    "numpy>=1.20.0",
    "scipy>=1.7.0",
    "pandas>=1.3.0",
    "matplotlib>=3.4.0",
    "tqdm>=4.62.0",
    "dill>=0.3.4",
]

[tool.setuptools.packages.find]
where = ["src"]
`
  },
  {
    path: 'README.md',
    name: 'README.md',
    language: 'markdown',
    category: 'doc',
    description: 'Package documentation, quickstart, model guide, and installation.',
    content: `# importance_sampling 🎯

A clean, modern, and general-purpose Python package for fitting computational, cognitive, and behavioral models to multi-subject data using **Iterative Importance Sampling (IIS)**.

Works for **any model** (learning, decision-making, psychophysics, economics) without requiring gradients, Stan, or slow MCMC chain mixing.

---

## 📦 Installation

Install directly from GitHub:
\`\`\`bash
pip install git+https://github.com/BoazRsnbrg/importance_sampling.git
\`\`\`

Or clone and install locally in editable mode:
\`\`\`bash
git clone https://github.com/BoazRsnbrg/importance_sampling.git
cd importance_sampling
pip install -e .
\`\`\`

---

## ⚡ Quickstart

\`\`\`python
from importance_sampling import Sampler, sigmoid, softplus
from demo.models import model_learning
from demo.sample_data import get_sample_data

# 1. Your data: a list where each element contains data for one subject
data = get_sample_data(n_subjects=10, n_trials=60)

# 2. Define hyper-priors (latent normal space + link transform)
hyper_priors = {
    "lr": {"mean": 0.0, "sd": 1.0, "transform": sigmoid},
    "inv_temp": {"mean": 1.0, "sd": 1.0, "transform": softplus},
}

# 3. Create the Sampler
model = Sampler(
    data=data,
    model=model_learning,
    hyper_params=hyper_priors,
    n_choices=2,
    model_name="MyModel",
)

# 4. Fit iteratively
model.iterative_model_fit(n_iterations=12, n_samples=1000)

# 5. Examine results
model.summary()
model.plot_hyper_param_evolution()
model.plot_posterior_distributions()

# 6. Save and reload
model.save_model("my_fitted_model")
\`\`\`

---

## 🚀 Running the Included Demo

A complete demo with sample data and multiple models is included:

\`\`\`bash
python demo/run_demo.py
\`\`\`
`
  },
  {
    path: 'LICENSE',
    name: 'LICENSE',
    language: 'text',
    category: 'config',
    description: 'MIT License.',
    content: `MIT License

Copyright (c) 2026 Boaz Rosenberg

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.`
  },
  {
    path: 'src/importance_sampling/__init__.py',
    name: '__init__.py',
    language: 'python',
    category: 'core',
    description: 'Package entry point: exports Sampler, compare_models, load_model, and utils.',
    content: `"""importance_sampling

A minimal, robust Python package for fitting cognitive and computational models
using Iterative Importance Sampling (IIS).
"""

from importance_sampling.sampler import (
    Sampler,
    compare_models,
    compare_max_likelihood,
    paired_comparison,
    paired_comparison_max,
    load_model,
    update_model,
)
from importance_sampling.utils import (
    logsumexp,
    sigmoid,
    logit,
    log_sigmoid,
    softplus,
    time_to_text,
    save_model,
)

__version__ = "0.2.0"
__all__ = [
    "Sampler",
    "compare_models",
    "compare_max_likelihood",
    "paired_comparison",
    "paired_comparison_max",
    "load_model",
    "save_model",
    "update_model",
    "logsumexp",
    "sigmoid",
    "logit",
    "log_sigmoid",
    "softplus",
    "time_to_text",
]
`
  },
  {
    path: 'src/importance_sampling/sampler.py',
    name: 'sampler.py',
    language: 'python',
    category: 'core',
    description: 'General-purpose Sampler class for fitting ANY model to multi-subject data.',
    content: `"""Iterative Importance Sampling (IIS) algorithm for hierarchical model estimation.

A general-purpose module for fitting any multi-subject computational, cognitive,
or statistical model using iterative population importance sampling.
"""

from __future__ import annotations
import copy
import inspect
import json
import os
import time
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import norm
from tqdm import tqdm

from importance_sampling.utils import (
    logsumexp,
    save_model as _save_model_func,
    load_model as _load_model_func,
    time_to_text,
)


class Sampler:
    """Iterative Importance Sampler for hierarchical model fitting."""

    def __init__(
        self,
        data: Sequence[Any],
        model: Callable,
        hyper_params: Dict[str, Dict[str, Any]],
        transformations: Optional[Dict[str, Callable]] = None,
        multinormal: Union[bool, str, List[Tuple[str, str]]] = False,
        n_choices: int = 4,
        model_name: str = "unnamed_model",
        description: str = "",
        type: str = "B",
        random_state: Optional[Union[int, np.random.Generator]] = None,
    ):
        self.data = list(data)
        self.model = model
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

        # Extract latent normal mean, sd, and optional auto-transforms
        self.hyper_params: Dict[str, Dict[str, float]] = {}
        auto_transforms: Dict[str, Callable] = {}
        for k, v in hyper_params.items():
            self.hyper_params[k] = {"mean": float(v["mean"]), "sd": float(v["sd"])}
            if "transform" in v and callable(v["transform"]):
                auto_transforms[k] = v["transform"]

        self.transformations: Dict[str, Callable] = {}
        for p in self.params:
            if transformations and p in transformations:
                self.transformations[p] = transformations[p]
            elif p in auto_transforms:
                self.transformations[p] = auto_transforms[p]
            else:
                self.transformations[p] = lambda x: x

        self.multinormal = multinormal
        self.cor_matrix = np.identity(self.n_params, dtype=np.float64)
        self.cor_matrices = [self.cor_matrix.copy()]

        self.hyper_params_list = [copy.deepcopy(self.hyper_params)]
        self.samples: Optional[Dict[str, Any]] = None
        self.mean_params: Optional[Dict[str, List[float]]] = None

        self.iterations = 0
        self.log_likelihoods: List[List[float]] = []
        self.log_likelihood: List[float] = []
        self.mean_accuracy: List[float] = []
        self.BIC: List[float] = []
        self.log_likelihood_change: List[float] = [0.0]
        self.BIC_change: List[float] = [0.0]

    def sample_resample(self, subj: int, n_samples: int = 1000):
        subj_data = self.data[subj]

        try:
            raw_samples = self.rng.multivariate_normal(
                np.zeros(self.n_params), self.cor_matrix, size=n_samples
            ).T
        except np.linalg.LinAlgError:
            jitter = self.cor_matrix + np.eye(self.n_params) * 1e-6
            raw_samples = self.rng.multivariate_normal(
                np.zeros(self.n_params), jitter, size=n_samples
            ).T

        samples = {
            param: raw_samples[i] * self.hyper_params[param]["sd"] + self.hyper_params[param]["mean"]
            for i, param in enumerate(self.params)
        }

        log_likelihoods = np.asarray(
            self.model(subj_data, samples, self.transformations, mode="loglikelihood"),
            dtype=np.float64,
        )

        log_sum_exp = logsumexp(log_likelihoods)
        weights = np.exp(log_likelihoods - log_sum_exp)
        weights = np.nan_to_num(weights, nan=0.0)
        weights /= np.sum(weights)

        mean_params = {param: float(np.sum(weights * samples[param])) for param in self.params}
        resample_idx = self.rng.choice(n_samples, size=n_samples, p=weights, replace=True)
        resample = {param: samples[param][resample_idx] for param in self.params}
        resample["log_likelihood"] = log_likelihoods[resample_idx]

        log_mean_likelihood = float(log_sum_exp - np.log(n_samples))
        return resample, log_mean_likelihood, mean_params

    def iterative_model_fit(
        self,
        n_iterations: int = 10,
        n_samples: int = 1000,
        epsilon: float = 0.01,
        n_mean: int = 10,
        stop_at_convergence: bool = True,
        verbose: bool = True,
    ):
        start = time.time()
        for i in range(n_iterations):
            self.adjust_hyper_priors(n_samples=n_samples, disable_progress=True)
            self.iterations += 1

            if i >= n_mean:
                change = (self.log_likelihood[-1] - self.log_likelihood[-n_mean]) / n_mean
                if change < epsilon and stop_at_convergence:
                    if verbose:
                        print(f"✨ Converged after {i + 1} iterations. Total evidence: {round(self.log_likelihood[-1], 4)}")
                    break
        return self

    def simulate(self, mode="resample", subjects="all", override_params=None, n_samples=1000):
        # Simulation for posterior predictive checks (PPC)
        ...

    def PPC(self, from_hyper_params=False, return_sim_data=True, save_dir=None, n_samples=1000):
        mode = "hyper_params" if from_hyper_params else "resample"
        return self.simulate(mode=mode, n_samples=n_samples)

    def save_model(self, filename="", directory="saved_models"):
        return _save_model_func(self, filename=filename, directory=directory)
`
  },
  {
    path: 'src/importance_sampling/utils.py',
    name: 'utils.py',
    language: 'python',
    category: 'core',
    description: 'Numerical utilities, stable logsumexp, link functions, time formatting.',
    content: `"""Mathematical and utility functions for importance sampling."""

from __future__ import annotations
import os
import pickle
import numpy as np
from scipy.special import expit, logsumexp as _scipy_logsumexp

try:
    import dill
    _SERIALIZER = dill
except ImportError:
    _SERIALIZER = pickle

def time_to_text(time_in_seconds: float) -> str:
    hours = int(time_in_seconds // 3600)
    minutes = int((time_in_seconds % 3600) // 60)
    seconds = int(np.round(time_in_seconds % 60))
    return f"{hours:02d}h {minutes:02d}m {seconds:02d}s"

def logsumexp(a, axis=None):
    arr = np.asarray(a, dtype=np.float64)
    masked = np.where(np.isnan(arr), -np.inf, arr)
    return _scipy_logsumexp(masked, axis=axis)

def sigmoid(x, b=1.0, a=0.0):
    return expit(a + b * np.asarray(x, dtype=np.float64))

def log_sigmoid(x, b=1.0):
    bx = b * np.asarray(x, dtype=np.float64)
    return np.where(bx >= 0, -np.log1p(np.exp(-bx)), bx - np.log1p(np.exp(bx)))

def logit(p, eps=1e-12):
    arr = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(arr / (1.0 - arr))

def softplus(x):
    arr = np.asarray(x, dtype=np.float64)
    return np.where(arr > 30.0, arr, np.log1p(np.exp(arr)))

def save_model(model, filename="", directory="saved_models"):
    os.makedirs(directory, exist_ok=True)
    if not filename:
        filename = getattr(model, "model_name", "unnamed_model")
    if not filename.endswith(".pkl"):
        filename += ".pkl"
    filepath = os.path.join(directory, filename)
    with open(filepath, "wb") as f:
        _SERIALIZER.dump(model, f)
    return filepath

def load_model(filename, directory="saved_models"):
    if not filename.endswith(".pkl"):
        filename += ".pkl"
    filepath = os.path.join(directory, filename)
    if not os.path.exists(filepath):
        filepath = filename
    with open(filepath, "rb") as f:
        return _SERIALIZER.load(f)
`
  },
  {
    path: 'demo/sample_data.py',
    name: 'sample_data.py',
    language: 'python',
    category: 'demo',
    description: 'Sample multi-subject dataset generator for demonstration.',
    content: `"""Sample multi-subject dataset generator for demonstration purposes."""

import numpy as np

def get_sample_data(n_subjects: int = 10, n_trials: int = 60, random_state: int = 42):
    rng = np.random.default_rng(random_state)
    data = []
    for s in range(n_subjects):
        true_lr = rng.uniform(0.2, 0.5)
        true_inv_temp = rng.uniform(2.5, 4.5)
        q = [0.5, 0.5]
        choices, rewards = [], []

        for _ in range(n_trials):
            diff = q[1] - q[0]
            p1 = 1.0 / (1.0 + np.exp(-true_inv_temp * diff))
            c = 1 if rng.random() < p1 else 0
            r = 1.0 if rng.random() < (0.75 if c == 1 else 0.25) else 0.0
            q[c] += true_lr * (r - q[c])
            choices.append(c)
            rewards.append(r)

        data.append({
            "subject": s,
            "choices": np.array(choices, dtype=int),
            "rewards": np.array(rewards, dtype=float),
            "n_trials": n_trials,
        })
    return data
`
  },
  {
    path: 'demo/models.py',
    name: 'models.py',
    language: 'python',
    category: 'demo',
    description: 'Example models showing how ANY computational model is defined.',
    content: `"""Example models showing how to build computational models for importance_sampling."""

import numpy as np
import pandas as pd
from scipy.special import expit

hyper_priors = {
    "lr": {"mean": 0.0, "sd": 1.0, "transform": expit},
    "inv_temp": {"mean": 1.0, "sd": 1.0, "transform": lambda x: np.log1p(np.exp(x))},
    "bias": {"mean": 0.0, "sd": 1.0, "transform": lambda x: x},
}

def model_learning(subj_data, parameters, transformations=None, mode="loglikelihood"):
    t_lr = transformations.get("lr", expit) if transformations else expit
    t_beta = transformations.get("inv_temp", lambda x: np.log1p(np.exp(x))) if transformations else (lambda x: np.log1p(np.exp(x)))

    alpha = np.asarray(t_lr(parameters["lr"]), dtype=np.float64)
    beta = np.asarray(t_beta(parameters["inv_temp"]), dtype=np.float64)
    n_samples = len(alpha)

    choices = subj_data["choices"]
    rewards = subj_data["rewards"]
    n_trials = len(choices)

    if mode == "loglikelihood":
        q = np.full((n_samples, 2), 0.5, dtype=np.float64)
        total_ll = np.zeros(n_samples, dtype=np.float64)

        for t in range(n_trials):
            c = choices[t]
            r = rewards[t]
            diff = q[:, 1] - q[:, 0]
            p1 = np.clip(expit(beta * diff), 1e-12, 1.0 - 1e-12)
            total_ll += np.log(p1) if c == 1 else np.log(1.0 - p1)
            q[:, c] += alpha * (r - q[:, c])

        return total_ll

    elif mode == "simulate":
        sim_alpha = float(alpha[0])
        sim_beta = float(beta[0])
        q_sim = [0.5, 0.5]
        sim_choices = []
        for t in range(n_trials):
            p1 = float(expit(sim_beta * (q_sim[1] - q_sim[0])))
            c = 1 if np.random.rand() < p1 else 0
            sim_choices.append(c)
            q_sim[c] += sim_alpha * (rewards[t] - q_sim[c])
        return pd.DataFrame({"trial": range(n_trials), "sim_choice": sim_choices})

model_data = {
    "Learning_Standard": {
        "model": model_learning,
        "params": ["lr", "inv_temp"],
        "description": "2-parameter learning model",
    }
}
`
  },
  {
    path: 'demo/run_demo.py',
    name: 'run_demo.py',
    language: 'python',
    category: 'demo',
    description: 'Demonstration of importing importance_sampling and fitting models.',
    content: `#!/usr/bin/env python3
"""Runner Demo: Demonstrates how to import and use importance_sampling to fit models."""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from importance_sampling import Sampler, compare_models, load_model
from models import hyper_priors, model_data
from sample_data import get_sample_data

def main():
    print("Loading sample participant data...")
    data = get_sample_data(n_subjects=10, n_trials=60)

    m1_spec = model_data["Learning_Standard"]
    m1_params = {p: hyper_priors[p] for p in m1_spec["params"]}

    # 1. Instantiate Sampler
    model = Sampler(
        data=data,
        model=m1_spec["model"],
        hyper_params=m1_params,
        n_choices=2,
        model_name="Learning_Standard",
    )

    # 2. Fit iteratively
    model.iterative_model_fit(n_iterations=12, n_samples=1000, verbose=True)

    # 3. View summary & diagnostics
    model.summary()
    model.plot_hyper_param_evolution()
    model.plot_posterior_distributions()

    # 4. Save and reload
    model.save_model("saved_models/my_model")
    reloaded = load_model("saved_models/my_model")
    print("Reloaded model successfully!")

if __name__ == "__main__":
    main()
`
  }
];
