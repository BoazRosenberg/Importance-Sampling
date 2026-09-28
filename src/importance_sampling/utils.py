"""Mathematical and utility functions for importance sampling."""

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
    """Convert elapsed seconds into a readable string (HHh MMm SSs)."""
    if np.isnan(time_in_seconds) or time_in_seconds < 0:
        return "00h 00m 00s"
    hours = int(time_in_seconds // 3600)
    minutes = int((time_in_seconds % 3600) // 60)
    seconds = int(np.round(time_in_seconds % 60))
    return f"{hours:02d}h {minutes:02d}m {seconds:02d}s"


def logsumexp(a: np.ndarray | list[float], axis: int | None = None) -> float | np.ndarray:
    """Compute the log of the sum of exponentials stably, ignoring NaNs."""
    arr = np.asarray(a, dtype=np.float64)
    # Mask NaNs with -inf so they don't break the sum
    masked = np.where(np.isnan(arr), -np.inf, arr)
    return _scipy_logsumexp(masked, axis=axis)


def sigmoid(x: np.ndarray | float, b: float = 1.0, a: float = 0.0) -> np.ndarray | float:
    """Numerically stable logistic sigmoid: 1 / (1 + exp(-(a + b * x)))."""
    arr = np.asarray(x, dtype=np.float64)
    return expit(a + b * arr)


def log_sigmoid(x: np.ndarray | float, b: float = 1.0) -> np.ndarray | float:
    """Numerically stable log of sigmoid: -log(1 + exp(-b * x))."""
    arr = np.asarray(x, dtype=np.float64)
    bx = b * arr
    return np.where(bx >= 0, -np.log1p(np.exp(-bx)), bx - np.log1p(np.exp(bx)))


def logit(p: np.ndarray | float, eps: float = 1e-12) -> np.ndarray | float:
    """Log-odds function: log(p / (1 - p))."""
    arr = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(arr / (1.0 - arr))


def softplus(x: np.ndarray | float) -> np.ndarray | float:
    """Smooth approximation to positive constraint: log(1 + exp(x))."""
    arr = np.asarray(x, dtype=np.float64)
    return np.where(arr > 30.0, arr, np.log1p(np.exp(arr)))


def save_model(model: object, filename: str = "", directory: str = "saved_models") -> str:
    """Save a fitted Sampler object to disk."""
    os.makedirs(directory, exist_ok=True)
    if not filename:
        filename = getattr(model, "model_name", "unnamed_model")
    if not filename.endswith(".pkl"):
        filename += ".pkl"
    filepath = os.path.join(directory, filename)
    with open(filepath, "wb") as f:
        _SERIALIZER.dump(model, f)
    return filepath


def load_model(filename: str, directory: str = "saved_models") -> object:
    """Load a previously saved Sampler object from disk."""
    if not filename.endswith(".pkl"):
        filename += ".pkl"
    filepath = os.path.join(directory, filename)
    if not os.path.exists(filepath):
        if os.path.exists(filename):
            filepath = filename
        else:
            raise FileNotFoundError(f"Model file not found at: {filepath}")
    with open(filepath, "rb") as f:
        return _SERIALIZER.load(f)
