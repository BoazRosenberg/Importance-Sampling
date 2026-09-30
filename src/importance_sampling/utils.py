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


def save_model(
    model: object,
    filename: str = "",
    directory: str = "saved_models",
    save_metadata_file: bool = True,
    timestamp: str | None = None,
) -> str:
    """Save a fitted Sampler object to disk along with its metadata JSON file.

    Parameters
    ----------
    model : object
        A Sampler instance to save.
    filename : str, default=""
        Base filename (e.g. 'q_learning'). Defaults to model.model_name.
    directory : str, default="saved_models"
        Target directory to save into.
    save_metadata_file : bool, default=True
        Whether to write companion '{base_name}_metadata.json' alongside the '.pkl' file.
    timestamp : Optional[str], default=None
        Custom timestamp string. Defaults to current local time.

    Returns
    -------
    str
        Full path to the saved pickle file.
    """
    import json
    import time

    os.makedirs(directory, exist_ok=True)
    if not filename:
        filename = getattr(model, "model_name", "unnamed_model")

    if filename.endswith(".pkl"):
        base_name = filename[:-4]
    else:
        base_name = filename
        filename = f"{base_name}.pkl"

    saved_time = timestamp or time.strftime("%Y-%m-%d %H:%M:%S")

    # Generate metadata dictionary
    if hasattr(model, "get_metadata") and callable(getattr(model, "get_metadata")):
        metadata = model.get_metadata(saved_timestamp=saved_time)
    elif hasattr(model, "metadata") and isinstance(getattr(model, "metadata"), dict):
        metadata = dict(getattr(model, "metadata"))
        metadata["saved_at"] = saved_time
    else:
        metadata = {
            "model_name": getattr(model, "model_name", base_name),
            "timestamp": saved_time,
            "saved_at": saved_time,
            "created_at": getattr(model, "creation_time", saved_time),
            "iterations": getattr(model, "iterations", 0),
            "n_subjects": getattr(model, "n_subjects", 0),
            "params": getattr(model, "params", []),
            "total_fit_time_seconds": getattr(model, "total_fit_time", 0.0),
            "total_fit_time_formatted": time_to_text(getattr(model, "total_fit_time", 0.0)),
            "model_code": getattr(model, "model_code", ""),
        }

    # Store directly on the model so it is preserved inside the pickle file
    setattr(model, "metadata", metadata)
    setattr(model, "saved_timestamp", saved_time)

    # 1. Save pickle file
    filepath = os.path.join(directory, filename)
    with open(filepath, "wb") as f:
        _SERIALIZER.dump(model, f)

    # 2. Save metadata JSON file alongside pickle
    if save_metadata_file:
        meta_filepath = os.path.join(directory, f"{base_name}_metadata.json")
        try:
            with open(meta_filepath, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, default=str)
        except Exception:
            # Fallback if complex custom objects exist in metadata
            clean_meta = {
                k: str(v) if not isinstance(v, (int, float, bool, list, dict, type(None))) else v
                for k, v in metadata.items()
            }
            with open(meta_filepath, "w", encoding="utf-8") as f:
                json.dump(clean_meta, f, indent=2)

    return filepath


def load_model(filename: str, directory: str = "saved_models") -> object:
    """Load a previously saved Sampler object from disk, restoring metadata if available."""
    import json

    if not filename.endswith(".pkl"):
        clean_name = filename
        filepath = os.path.join(directory, f"{clean_name}.pkl")
    else:
        clean_name = filename[:-4]
        filepath = os.path.join(directory, filename)

    if not os.path.exists(filepath):
        if os.path.exists(filename):
            filepath = filename
            clean_name = os.path.splitext(filename)[0]
        else:
            raise FileNotFoundError(f"Model file not found at: {filepath}")

    with open(filepath, "rb") as f:
        loaded = _SERIALIZER.load(f)

    # If metadata attribute is missing or empty, attempt to load companion metadata json
    if not hasattr(loaded, "metadata") or not loaded.metadata:
        dir_path = os.path.dirname(filepath)
        base = os.path.splitext(os.path.basename(filepath))[0]
        meta_json_path = os.path.join(dir_path, f"{base}_metadata.json")
        if os.path.exists(meta_json_path):
            try:
                with open(meta_json_path, "r", encoding="utf-8") as f:
                    setattr(loaded, "metadata", json.load(f))
            except Exception:
                pass

    return loaded
