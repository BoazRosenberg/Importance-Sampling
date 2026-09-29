"""importance_sampling

A minimal, robust Python package for fitting cognitive and computational models
using Iterative Importance Sampling (IIS).
"""

from importance_sampling.sampler import Sampler, load_model
from importance_sampling.report import create_report, compare_models, compare_parameters
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
    "load_model",
    "save_model",
    "create_report",
    "compare_models",
    "compare_parameters",
    "logsumexp",
    "sigmoid",
    "logit",
    "log_sigmoid",
    "softplus",
    "time_to_text",
]
