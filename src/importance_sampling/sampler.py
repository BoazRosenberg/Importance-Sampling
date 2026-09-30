"""Iterative Importance Sampling (IIS) algorithm for hierarchical model estimation.

A lightweight, general-purpose module for fitting multi-subject computational, cognitive,
or statistical models using iterative population importance sampling.
"""

from __future__ import annotations

import copy
import inspect
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.special import logsumexp
from tqdm.auto import tqdm

from importance_sampling.utils import (
    save_model as _save_model_func,
    load_model as _load_model_func,
    time_to_text,
)


def _to_single_df(item: Any) -> Any:
    """Helper to convert a single sub-table or record collection to a pandas DataFrame."""
    import pandas as pd
    if isinstance(item, pd.DataFrame):
        return item.copy()
    if isinstance(item, dict):
        try:
            return pd.DataFrame(item)
        except Exception:
            return pd.DataFrame([item])
    if isinstance(item, (list, tuple)):
        try:
            return pd.DataFrame(item)
        except Exception:
            return pd.DataFrame({"value": list(item)})
    if hasattr(item, "to_dict"):
        return pd.DataFrame(item.to_dict())
    return pd.DataFrame({"value": [item]})


def _extract_subject_sub_dfs(subj: Any) -> Dict[Union[int, str], Any]:
    """Parse a single subject's simulated output into a dictionary of sub-DataFrames.

    Supports:
    - Single DataFrame
    - List, tuple, or numpy array of sub-DataFrames / sub-tables: [df_0, df_1, ...]
    - Dict mapping sub-indices to sub-DataFrames: {0: df_0, 1: df_1} or {'train': df_train, 'test': df_test}
    - Dict of trial column arrays: {'choice': [...], 'reward': [...]}
    - List of row record dicts: [{'trial': 1, 'choice': 0}, ...]
    """
    import pandas as pd

    if subj is None:
        return {}

    # Case 1: Already a single DataFrame
    if isinstance(subj, pd.DataFrame):
        return {0: subj.copy()}

    # Case 2: Dict
    if isinstance(subj, dict):
        # Check if dict values are themselves sub-tables/DataFrames or scalar column arrays
        has_subtables = any(
            isinstance(v, (pd.DataFrame, dict))
            or (isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], dict))
            for v in subj.values()
        )
        if has_subtables:
            return {k: _to_single_df(v) for k, v in subj.items()}
        else:
            return {0: _to_single_df(subj)}

    # Case 3: List, tuple, or numpy array
    if isinstance(subj, (list, tuple, np.ndarray)):
        if len(subj) == 0:
            return {}
        first = subj[0]
        if isinstance(first, pd.DataFrame):
            # List of sub-DataFrames!
            return {i: df.copy() for i, df in enumerate(subj)}
        elif isinstance(first, dict):
            # Check if elements are sub-tables (dict with arrays) or row records (dict with scalar values)
            first_has_arrays = any(
                isinstance(v, (list, tuple, np.ndarray, pd.Series)) and len(v) > 1
                for v in first.values()
            )
            if first_has_arrays:
                return {i: _to_single_df(sub_item) for i, sub_item in enumerate(subj)}
            else:
                return {0: pd.DataFrame(subj)}
        elif isinstance(first, (list, tuple, np.ndarray)):
            # List of sub-arrays/sub-tables
            return {i: _to_single_df(sub_item) for i, sub_item in enumerate(subj)}
        else:
            return {0: pd.DataFrame({"simulated_value": list(subj)})}

    return {0: pd.DataFrame({"simulated_value": [subj]})}


def _reconstruct_dataframes(
    sim_list: Sequence[Any],
    sampler: Optional["Sampler"] = None,
    sub_index: Optional[Union[int, str, Sequence[Union[int, str]]]] = None,
    combine_all: bool = False,
    add_subject_col: bool = True,
    add_group_col: bool = True,
    **kwargs: Any,
) -> Any:
    """Reconstruct a list of per-subject simulated data into combined pandas DataFrame(s).

    Supports 3 primary workflows:
    1. Construct ALL sub-dfs together into 1 combined DataFrame (combine_all=True or sub_index="all").
    2. Construct X sub-dfs in each subject's data into X DataFrames across subjects (sub_index=None).
    3. Construct 1 DataFrame from a specific sub_index across all subjects (e.g. sub_index=0 or sub_index="trials").
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required to reconstruct simulated data into DataFrames. "
            "Install pandas via: pip install pandas"
        ) from exc

    # Backwards compatibility alias for target_kind / kinds / sub_indices
    if sub_index is None:
        sub_index = kwargs.get("target_kind", kwargs.get("kinds", kwargs.get("sub_indices", None)))

    if not sim_list:
        return pd.DataFrame()

    def _get_subj_info(idx: int) -> Tuple[Any, Optional[str]]:
        s_id: Any = idx
        group_val: Optional[str] = None
        if sampler is not None:
            if hasattr(sampler, "subjects") and idx < len(sampler.subjects):
                s_id = sampler.subjects[idx]
            if hasattr(sampler, "subject_groups") and sampler.subject_groups:
                first_param = next(iter(sampler.subject_groups))
                s_groups = sampler.subject_groups[first_param]
                if idx < len(s_groups):
                    group_val = s_groups[idx]
        return s_id, group_val

    # Parse each subject into a dictionary of {sub_idx: df}
    per_subject_dfs: List[Dict[Union[int, str], Any]] = [
        _extract_subject_sub_dfs(item) for item in sim_list
    ]

    # Collect all unique sub-indices across subjects in insertion order
    all_sub_indices: List[Union[int, str]] = []
    for s_dfs in per_subject_dfs:
        for k in s_dfs.keys():
            if k not in all_sub_indices:
                all_sub_indices.append(k)

    if not all_sub_indices:
        return pd.DataFrame()

    # Determine if user wants all sub-dfs concatenated together to 1 single master DataFrame
    is_combine_all = combine_all or (
        isinstance(sub_index, str)
        and sub_index.strip().lower() == "all"
        and "all" not in all_sub_indices
    )

    if is_combine_all:
        combined_rows = []
        has_multiple_sub = len(all_sub_indices) > 1 or all_sub_indices != [0]
        for s_idx, s_dfs in enumerate(per_subject_dfs):
            s_id, g_val = _get_subj_info(s_idx)
            for sub_k, df in s_dfs.items():
                cur_df = df.copy()
                if add_subject_col and "subject" not in cur_df.columns and "subject_id" not in cur_df.columns:
                    cur_df.insert(0, "subject", s_id)
                if has_multiple_sub and "sub_index" not in cur_df.columns:
                    insert_pos = 1 if "subject" in cur_df.columns else 0
                    cur_df.insert(insert_pos, "sub_index", sub_k)
                if add_group_col and g_val is not None and "group" not in cur_df.columns:
                    insert_pos = len(cur_df.columns)
                    for col_name in ["sub_index", "subject"]:
                        if col_name in cur_df.columns:
                            insert_pos = cur_df.columns.get_loc(col_name) + 1
                            break
                    cur_df.insert(insert_pos, "group", g_val)
                combined_rows.append(cur_df)
        return pd.concat(combined_rows, ignore_index=True) if combined_rows else pd.DataFrame()

    # If a specific single sub-index was requested (e.g. sub_index=0 or sub_index="phase1")
    if sub_index is not None and not isinstance(sub_index, (list, tuple, set)):
        target_k = sub_index
        matched_k = target_k if target_k in all_sub_indices else None
        if matched_k is None:
            for k in all_sub_indices:
                if str(k) == str(target_k):
                    matched_k = k
                    break
        if matched_k is None:
            raise KeyError(
                f"Sub-index '{target_k}' was not found in subject simulated data. "
                f"Available sub-indices: {all_sub_indices}."
            )

        sub_list = []
        for s_idx, s_dfs in enumerate(per_subject_dfs):
            if matched_k in s_dfs:
                cur_df = s_dfs[matched_k].copy()
                s_id, g_val = _get_subj_info(s_idx)
                if add_subject_col and "subject" not in cur_df.columns and "subject_id" not in cur_df.columns:
                    cur_df.insert(0, "subject", s_id)
                if add_group_col and g_val is not None and "group" not in cur_df.columns:
                    insert_pos = 1 if "subject" in cur_df.columns else 0
                    cur_df.insert(insert_pos, "group", g_val)
                sub_list.append(cur_df)
        return pd.concat(sub_list, ignore_index=True) if sub_list else pd.DataFrame()

    # Multiple sub-indices requested, or sub_index is None
    target_keys = list(sub_index) if isinstance(sub_index, (list, tuple, set)) else all_sub_indices

    result_dict: Dict[Union[int, str], Any] = {}
    for k in target_keys:
        matched_k = k if k in all_sub_indices else next((ak for ak in all_sub_indices if str(ak) == str(k)), None)
        if matched_k is None:
            continue
        sub_list = []
        for s_idx, s_dfs in enumerate(per_subject_dfs):
            if matched_k in s_dfs:
                cur_df = s_dfs[matched_k].copy()
                s_id, g_val = _get_subj_info(s_idx)
                if add_subject_col and "subject" not in cur_df.columns and "subject_id" not in cur_df.columns:
                    cur_df.insert(0, "subject", s_id)
                if add_group_col and g_val is not None and "group" not in cur_df.columns:
                    insert_pos = 1 if "subject" in cur_df.columns else 0
                    cur_df.insert(insert_pos, "group", g_val)
                sub_list.append(cur_df)
        result_dict[k] = pd.concat(sub_list, ignore_index=True) if sub_list else pd.DataFrame()

    # If each subject only has 1 sub-df and no sub_index was explicitly specified, return the DataFrame directly
    if len(result_dict) == 1 and sub_index is None and all_sub_indices == [0]:
        return next(iter(result_dict.values()))

    return result_dict


def _save_reconstructed_dataframes(
    sim_data: Sequence[Any],
    folder: str,
    sub_index: Optional[Union[int, str, Sequence[Union[int, str]]]] = None,
    combine_all: bool = False,
    sampler: Optional["Sampler"] = None,
    file_format: str = "csv",
    prefix: str = "simulated_data",
    **kwargs: Any,
) -> Dict[Union[int, str], str]:
    """Save reconstructed DataFrame(s) to a folder."""
    os.makedirs(folder, exist_ok=True)

    if sub_index is None:
        sub_index = kwargs.get("kinds", kwargs.get("save_kinds", None))

    reconstructed = _reconstruct_dataframes(
        sim_data,
        sampler=sampler,
        sub_index=sub_index,
        combine_all=combine_all,
        **kwargs,
    )

    dfs_to_save: Dict[Union[int, str], Any] = {}
    if hasattr(reconstructed, "to_csv"):
        key = sub_index if (sub_index is not None and not isinstance(sub_index, (list, tuple, set))) else prefix
        dfs_to_save[key] = reconstructed
    elif isinstance(reconstructed, dict):
        dfs_to_save = reconstructed
    else:
        dfs_to_save = {prefix: reconstructed}

    saved_paths: Dict[Union[int, str], str] = {}
    for sub_k, df in dfs_to_save.items():
        if len(dfs_to_save) == 1 and sub_k == prefix:
            fname = f"{prefix}.{file_format}"
        elif isinstance(sub_k, int):
            fname = f"{prefix}_sub_{sub_k}.{file_format}"
        else:
            fname = f"{sub_k}.{file_format}"

        out_path = os.path.join(folder, fname)
        if file_format == "csv":
            df.to_csv(out_path, index=False)
        elif file_format == "parquet":
            df.to_parquet(out_path, index=False)
        elif file_format == "feather":
            df.to_feather(out_path)
        else:
            df.to_csv(out_path, index=False)

        abs_p = os.path.abspath(out_path)
        saved_paths[sub_k] = abs_p
        print(f"Saved simulated data for sub_index '{sub_k}' ({len(df)} rows) to: {abs_p}")

    return saved_paths


class SimulatedDataList(list):
    """A list of subject simulated data with helper methods for DataFrame reconstruction and export.

    Because SimulatedDataList inherits directly from `list`, it functions as a standard Python
    list (e.g. `len(sim_dat)`, `sim_dat[0]`, `for s in sim_dat:`).
    """

    def __init__(self, data_list: Sequence[Any], sampler: Optional["Sampler"] = None):
        super().__init__(data_list)
        self.sampler = sampler

    def to_dataframe(
        self,
        sub_index: Optional[Union[int, str]] = None,
        combine_all: bool = False,
        add_subject_col: bool = True,
        add_group_col: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Reconstruct subject simulated data into DataFrame(s).

        Parameters
        ----------
        sub_index : Optional[Union[int, str]], default=None
            - If an integer or string (e.g. 0, 1, or 'trials'): returns 1 combined DataFrame for that sub_index.
            - If 'all': combines all sub-dfs across subjects into 1 master DataFrame (with 'sub_index' column).
            - If None: returns 1 combined DataFrame (if 1 df per subject) or a dict of {sub_index: df} (if multiple sub-dfs).
        combine_all : bool, default=False
            If True, constructs all sub-dfs across all subjects together into 1 combined DataFrame.
        add_subject_col : bool, default=True
            Whether to add a 'subject' column identifying the subject ID.
        add_group_col : bool, default=True
            Whether to add a 'group' column if the sampler was configured with group differences.
        """
        return _reconstruct_dataframes(
            self,
            sampler=self.sampler,
            sub_index=sub_index,
            combine_all=combine_all,
            add_subject_col=add_subject_col,
            add_group_col=add_group_col,
            **kwargs,
        )

    def to_dataframes(
        self,
        sub_indices: Optional[Union[int, str, Sequence[Union[int, str]]]] = None,
        add_subject_col: bool = True,
        add_group_col: bool = True,
        **kwargs: Any,
    ) -> Dict[Union[int, str], Any]:
        """Reconstruct the X sub-dfs in each subject's data into X combined DataFrames across all subjects.

        Returns
        -------
        Dict[Union[int, str], pd.DataFrame]
            Dictionary mapping each sub_index (e.g. 0, 1, ... or 'phase1', 'phase2') to its combined DataFrame.
        """
        res = _reconstruct_dataframes(
            self,
            sampler=self.sampler,
            sub_index=sub_indices,
            combine_all=False,
            add_subject_col=add_subject_col,
            add_group_col=add_group_col,
            **kwargs,
        )
        if hasattr(res, "to_csv"):
            return {0: res}
        return res

    def save(
        self,
        folder: str,
        sub_index: Optional[Union[int, str, Sequence[Union[int, str]]]] = None,
        combine_all: bool = False,
        file_format: str = "csv",
        prefix: str = "simulated_data",
        **kwargs: Any,
    ) -> Dict[Union[int, str], str]:
        """Save reconstructed DataFrame(s) to a folder.

        Parameters
        ----------
        folder : str
            Destination directory (created if it does not exist).
        sub_index : Optional[Union[int, str, Sequence[Union[int, str]]]], default=None
            Specific sub-index (e.g. 0 or 'phase1') to save. If None, saves all sub-DataFrames.
        combine_all : bool, default=False
            If True, constructs all sub-dfs together into 1 combined file.
        file_format : str, default='csv'
            File format ('csv', 'parquet', or 'feather').
        prefix : str, default='simulated_data'
            Prefix for output filenames.

        Returns
        -------
        Dict[Union[int, str], str]
            Dictionary mapping saved sub-indices to their destination file paths.
        """
        return _save_reconstructed_dataframes(
            self,
            folder=folder,
            sub_index=sub_index,
            combine_all=combine_all,
            sampler=self.sampler,
            file_format=file_format,
            prefix=prefix,
            **kwargs,
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
        group_diff: Optional[Union[Dict[str, str], Sequence[str]]] = None,
        group_column: str = "group",
        params_with_group_diff: Optional[Union[Dict[str, str], Sequence[str]]] = None,
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
        self.hyper_params: Dict[str, Dict[str, Any]] = {}
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

        # Configure parameters with group differences
        effective_group_diff = group_diff if group_diff is not None else params_with_group_diff
        self.group_diff: Dict[str, Any] = {}
        if isinstance(effective_group_diff, dict):
            invalid_params = [p for p in effective_group_diff if p not in self.params]
            if invalid_params:
                raise KeyError(
                    f"Parameters {invalid_params} in group_diff were not found in hyper_params. "
                    f"Available parameters are: {self.params}."
                )
            self.group_diff = {p: col for p, col in effective_group_diff.items()}
        elif isinstance(effective_group_diff, (list, tuple, set)):
            invalid_params = [p for p in effective_group_diff if p not in self.params]
            if invalid_params:
                raise KeyError(
                    f"Parameters {invalid_params} in group_diff were not found in hyper_params. "
                    f"Available parameters are: {self.params}."
                )
            self.group_diff = {p: group_column for p in effective_group_diff}
        elif effective_group_diff is not None:
            raise TypeError(
                f"group_diff must be a dictionary mapping parameter names to column names or paths "
                f"(e.g. {{'alpha': 'condition'}} or {{'alpha': ('subject_data', 'group')}}), "
                f"or a list of parameter names (e.g. ['alpha']), "
                f"got {type(effective_group_diff).__name__}."
            )

        # Backwards compatibility alias
        self.params_with_group_diff = self.group_diff

        def _extract_subject_group(subj_idx: int, subj_data: Any, col_spec: Any, param_name: str) -> str:
            # Helper to unpack scalar value from Series, array, or list
            def _unpack(v: Any) -> Any:
                if hasattr(v, "iloc"):
                    return v.iloc[0] if len(v) > 0 else None
                if isinstance(v, (list, tuple, np.ndarray)):
                    return v[0] if len(v) > 0 else None
                return v

            # 1. Parse path tokens (supports tuple/list, dot-separated strings, or direct keys)
            if isinstance(col_spec, (list, tuple)):
                path_tokens = list(col_spec)
            elif isinstance(col_spec, str):
                has_direct = False
                if hasattr(subj_data, "columns") and col_spec in subj_data.columns:
                    has_direct = True
                elif isinstance(subj_data, dict) and col_spec in subj_data:
                    has_direct = True
                elif hasattr(subj_data, col_spec):
                    has_direct = True

                if not has_direct and ("." in col_spec or "/" in col_spec):
                    delim = "." if "." in col_spec else "/"
                    path_tokens = [tok.strip() for tok in col_spec.split(delim) if tok.strip()]
                else:
                    path_tokens = [col_spec]
            else:
                path_tokens = [col_spec]

            # 2. Try explicit path traversal
            def _traverse(node: Any, tokens: List[Any]) -> Tuple[bool, Any]:
                curr = node
                for tok in tokens:
                    if curr is None:
                        return False, None

                    is_int = isinstance(tok, int) or (isinstance(tok, str) and tok.isdigit())

                    # Sequence / ndarray indexing
                    if is_int and isinstance(curr, (list, tuple, np.ndarray)):
                        idx = int(tok)
                        if 0 <= idx < len(curr):
                            curr = curr[idx]
                            continue
                        return False, None

                    # DataFrame column or positional iloc
                    if hasattr(curr, "columns"):
                        if tok in curr.columns:
                            curr = curr[tok]
                            continue
                        if is_int and int(tok) < len(curr.columns):
                            curr = curr.iloc[:, int(tok)]
                            continue

                    # Dict key
                    if isinstance(curr, dict):
                        if tok in curr:
                            curr = curr[tok]
                            continue
                        if is_int and int(tok) in curr:
                            curr = curr[int(tok)]
                            continue

                    # General __getitem__
                    if hasattr(curr, "__getitem__"):
                        try:
                            curr = curr[tok]
                            continue
                        except Exception:
                            pass

                    # Object attribute
                    if isinstance(tok, str) and hasattr(curr, tok):
                        curr = getattr(curr, tok)
                        continue

                    return False, None

                return True, curr

            found, raw_val = _traverse(subj_data, path_tokens)

            # 3. If explicit path didn't find it, and we have a single column name, search sub-dataframes/dicts
            if not found and len(path_tokens) == 1:
                target_col = path_tokens[0]
                queue = [(subj_data, 0)]
                visited = set()
                while queue:
                    curr_obj, depth = queue.pop(0)
                    if depth > 4:
                        continue
                    obj_id = id(curr_obj)
                    if obj_id in visited:
                        continue
                    visited.add(obj_id)

                    if hasattr(curr_obj, "columns") and target_col in curr_obj.columns:
                        raw_val = curr_obj[target_col]
                        found = True
                        break
                    if isinstance(curr_obj, dict) and target_col in curr_obj:
                        raw_val = curr_obj[target_col]
                        found = True
                        break
                    if hasattr(curr_obj, str(target_col)) and not callable(getattr(curr_obj, str(target_col))):
                        raw_val = getattr(curr_obj, str(target_col))
                        found = True
                        break

                    # Enqueue sub-structures
                    if isinstance(curr_obj, dict):
                        for child in curr_obj.values():
                            if isinstance(child, (dict, list, tuple, np.ndarray)) or hasattr(child, "columns"):
                                queue.append((child, depth + 1))
                    elif isinstance(curr_obj, (list, tuple)):
                        for child in curr_obj:
                            if isinstance(child, (dict, list, tuple, np.ndarray)) or hasattr(child, "columns"):
                                queue.append((child, depth + 1))

            if not found:
                available_preview: List[str] = []
                if hasattr(subj_data, "columns"):
                    available_preview = list(subj_data.columns)
                elif isinstance(subj_data, dict):
                    available_preview = list(subj_data.keys())
                elif isinstance(subj_data, (list, tuple)):
                    available_preview = [f"index {i} ({type(item).__name__})" for i, item in enumerate(subj_data[:5])]

                avail_str = f" Available keys/sub-structures: {available_preview}" if available_preview else ""
                spec_display = ".".join(str(x) for x in col_spec) if isinstance(col_spec, (list, tuple)) else str(col_spec)
                raise KeyError(
                    f"Group column '{spec_display}' specified for parameter '{param_name}' was not found in subject {subj_idx}'s data.{avail_str}\n"
                    f"Tips for multi-dataframe subject data:\n"
                    f"  - If inside a named sub-DataFrame/dict: use group_column=('subject_data', 'group') or 'subject_data.group'\n"
                    f"  - If inside a list of sub-DataFrames: use group_column=(0, 'group') or '0.group'\n"
                    f"  - Or configure per-parameter: group_diff={{'{param_name}': ('subject_data', 'group')}}"
                )

            val = _unpack(raw_val)

            if val is None or (isinstance(val, float) and np.isnan(val)):
                raise ValueError(
                    f"Subject {subj_idx} has a missing or NaN group label in '{col_spec}' for parameter '{param_name}'."
                )

            val_str = str(val).strip()
            if not val_str or val_str.lower() in ("nan", "none", "null"):
                raise ValueError(
                    f"Subject {subj_idx} has an empty or invalid group label ('{val}') in '{col_spec}' for parameter '{param_name}'."
                )

            return val_str

        self.subject_groups: Dict[str, List[str]] = {}
        self.group_names: Dict[str, List[str]] = {}
        self.ref_group: Dict[str, str] = {}
        self.group_means: Dict[str, Dict[str, float]] = {}
        self.group_diffs: Dict[str, Dict[str, float]] = {}
        self.group_means_list: Dict[str, List[Dict[str, float]]] = {}
        self.group_diffs_list: Dict[str, List[Dict[str, float]]] = {}

        for p, col in self.group_diff.items():
            s_groups = [_extract_subject_group(s, d, col, p) for s, d in enumerate(self.data)]
            self.subject_groups[p] = s_groups
            unique_groups = sorted(list(set(s_groups)))

            if len(unique_groups) < 2:
                raise ValueError(
                    f"Parameter '{p}' was configured for group differences on column '{col}', but only {len(unique_groups)} "
                    f"distinct group was found across all {self.n_subjects} subjects: {unique_groups}.\n"
                    f"Group difference estimation requires at least 2 distinct groups (e.g. ['control', 'patient']). "
                    f"If all subjects belong to a single cohort, omit '{p}' from group_diff."
                )

            self.group_names[p] = unique_groups
            self.ref_group[p] = unique_groups[0]
            init_mean = self.hyper_params[p]["mean"]
            self.group_means[p] = {g: float(init_mean) for g in unique_groups}
            self.group_diffs[p] = {f"{g} - {unique_groups[0]}": 0.0 for g in unique_groups[1:]}
            self.group_means_list[p] = [copy.deepcopy(self.group_means[p])]
            self.group_diffs_list[p] = [copy.deepcopy(self.group_diffs[p])]
            self.hyper_params[p]["group_means"] = copy.deepcopy(self.group_means[p])
            self.hyper_params[p]["group_diffs"] = copy.deepcopy(self.group_diffs[p])

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
        raw_samples = {}
        for i, param in enumerate(self.params):
            if param in self.group_diff:
                s_group = self.subject_groups[param][subj]
                center_mean = self.group_means[param][s_group]
            else:
                center_mean = self.hyper_params[param]["mean"]
            raw_samples[param] = (
                std_normals[i] * self.hyper_params[param]["sd"]
                + center_mean
            )

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

    def adjust_hyper_priors(
        self,
        n_samples: int = 1000,
        disable_progress: bool = True,
        leave: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        position: int = 0,
    ) -> None:
        """Run one iteration of importance sampling across all subjects and update priors."""
        subject_samples: Dict[str, List[np.ndarray]] = {param: [] for param in self.params}
        mean_params: Dict[str, List[float]] = {param: [] for param in self.params}
        log_likelihoods: List[float] = []

        subj_indices = range(self.n_subjects)
        pbar = None
        if not disable_progress:
            pbar = tqdm(
                subj_indices,
                desc="  Subjects",
                unit="subj",
                leave=leave,
                file=sys.stdout,
                dynamic_ncols=True,
                position=position,
                mininterval=0.25,
            )
            subj_iter = pbar
        else:
            subj_iter = subj_indices

        for s in subj_iter:
            resample, log_mean_ll, s_mean = self.sample_resample(s, n_samples=n_samples)
            for param in self.params:
                subject_samples[param].append(resample[param])
                mean_params[param].append(s_mean[param])
            log_likelihoods.append(log_mean_ll)
            if progress_callback is not None:
                progress_callback(s + 1, self.n_subjects)

        if pbar is not None and not leave:
            pbar.close()

        # Pool particles across all subjects to update population distribution
        pooled_samples = {
            param: np.concatenate(subject_samples[param])
            for param in self.params
        }

        # Recalculate population mean, standard deviation, and group differences
        updated_priors = {}
        for param in self.params:
            if param in self.group_diff:
                unique_groups = self.group_names[param]
                ref_g = self.ref_group[param]
                g_means: Dict[str, float] = {}
                residuals = []

                for g in unique_groups:
                    subj_indices = [
                        s for s in range(self.n_subjects)
                        if self.subject_groups[param][s] == g
                    ]
                    if subj_indices:
                        g_particles = np.concatenate([subject_samples[param][s] for s in subj_indices])
                        m_g = float(np.mean(g_particles))
                        g_means[g] = m_g
                        residuals.extend([
                            subject_samples[param][s] - m_g for s in subj_indices
                        ])
                    else:
                        g_means[g] = float(self.group_means[param].get(g, 0.0))

                self.group_means[param] = g_means
                ref_mean = g_means[ref_g]
                self.group_diffs[param] = {
                    f"{g} - {ref_g}": float(g_means[g] - ref_mean)
                    for g in unique_groups[1:]
                }

                # Pooled within-group standard deviation
                pooled_res = np.concatenate(residuals) if residuals else pooled_samples[param]
                pooled_sd = float(np.std(pooled_res, ddof=1)) if len(pooled_res) > 1 else float(self.hyper_params[param]["sd"])

                # Grand mean across subjects
                grand_mean = float(np.mean(pooled_samples[param]))

                updated_priors[param] = {
                    "mean": grand_mean,
                    "sd": pooled_sd,
                    "group_means": copy.deepcopy(g_means),
                    "group_diffs": copy.deepcopy(self.group_diffs[param]),
                }
                self.group_means_list[param].append(copy.deepcopy(g_means))
                self.group_diffs_list[param].append(copy.deepcopy(self.group_diffs[param]))
            else:
                updated_priors[param] = {
                    "mean": float(np.mean(pooled_samples[param])),
                    "sd": float(np.std(pooled_samples[param], ddof=1)),
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

        # Base hyperparameters: 2 per parameter (mean and sd)
        # Plus (N_groups - 1) additional hyperparameters for each parameter with group difference
        # Total parameters: 2 + (N_groups - 1) = N_groups + 1 per group-difference parameter
        extra_group_hyper = sum(
            max(0, len(self.group_names[p]) - 1) for p in self.group_diff
        )
        if self.multinormal == "full":
            n_hyper = 2 * self.n_params + self.n_params * (self.n_params - 1) / 2 + extra_group_hyper
        else:
            n_hyper = 2 * self.n_params + extra_group_hyper
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
        verbose: bool = False,
        progress_bar: Union[bool, str] = True,
    ) -> "Sampler":
        """Run the iterative importance sampling estimation loop until convergence.

        Parameters
        ----------
        n_iterations : int, default=10
            Maximum number of iterations.
        n_samples : int, default=1000
            Number of particles/draws sampled per subject on each iteration.
        epsilon : float, default=0.01
            Convergence threshold: stops when evidence change over n_mean iterations < epsilon.
        n_mean : int, default=10
            Window size for moving average convergence check.
        stop_at_convergence : bool, default=True
            Whether to stop early when convergence criterion is met.
        verbose : bool, default=False
            Whether to print detailed diagnostics at each iteration.
        progress_bar : Union[bool, str], default=True
            Progress bar display mode:
            - True (or 'single'): Displays ONE clean in-place progress bar tracking iterations
              with live subject sub-progress in the postfix (no multi-line spam).
            - 'nested': Displays both iteration and subject progress bars.
            - 'subjects': Displays only subject progress bar for each iteration.
            - False: Disables progress bars.
        """
        start = time.time()
        iter_range = range(n_iterations)

        show_iter_bar = progress_bar in (True, "single", "nested")
        show_subj_bar = progress_bar in ("nested", "subjects")

        iter_pbar = (
            tqdm(
                iter_range,
                desc=f"Fitting {self.model_name}",
                unit="iter",
                leave=True,
                file=sys.stdout,
                dynamic_ncols=True,
            )
            if show_iter_bar
            else iter_range
        )

        for i in iter_pbar:
            # Live subject sub-progress callback on the single progress bar
            def _on_subj_progress(curr: int, total: int) -> None:
                if show_iter_bar and not show_subj_bar and hasattr(iter_pbar, "set_postfix"):
                    if curr == total or curr % max(1, total // 8) == 0:
                        p_data: Dict[str, Any] = {"subj": f"{curr}/{total}"}
                        if self.evidence:
                            p_data["Evidence"] = f"{self.evidence[-1]:.2f}"
                            p_data["BIC"] = f"{self.BIC[-1]:.2f}"
                        iter_pbar.set_postfix(p_data, refresh=True)

            # Within-iteration progress across subjects
            self.adjust_hyper_priors(
                n_samples=n_samples,
                disable_progress=not show_subj_bar,
                leave=False,
                progress_callback=_on_subj_progress if (show_iter_bar and not show_subj_bar) else None,
                position=1 if show_subj_bar and show_iter_bar else 0,
            )
            self.iterations += 1

            # Update tqdm postfix with completed iteration diagnostics
            if show_iter_bar and hasattr(iter_pbar, "set_postfix"):
                postfix_data = {
                    "Evidence": f"{self.evidence[-1]:.2f}",
                    "BIC": f"{self.BIC[-1]:.2f}",
                }
                if len(self.evidence_change) > 0:
                    postfix_data["ΔEv"] = f"{self.evidence_change[-1]:+.2f}"
                iter_pbar.set_postfix(postfix_data, refresh=True)

            # Convergence check over window of n_mean iterations
            change = 0.0
            converged = False
            if i >= n_mean:
                change = (self.evidence[-1] - self.evidence[-n_mean]) / n_mean
                if change < epsilon and stop_at_convergence:
                    converged = True

            if converged:
                msg = (
                    f"✨ Converged after {i + 1} iterations. "
                    f"Evidence: {self.evidence[-1]:.4f}, BIC: {self.BIC[-1]:.2f}"
                )
                if show_iter_bar:
                    tqdm.write(msg, file=sys.stdout)
                else:
                    print(msg)
                self.total_fit_time += time.time() - start
                break

            elapsed = time.time() - start
            predicted_duration = (elapsed / (i + 1)) * (n_iterations - i - 1)

            if verbose:
                msg = (
                    f"Iter {i + 1}/{n_iterations} | "
                    f"Evidence: {self.evidence[-1]:.4f} (ΔEv: {self.evidence_change[-1]:+.4f}) | "
                    f"BIC: {self.BIC[-1]:.2f} | "
                    f"Elapsed: {time_to_text(elapsed)} | Remaining: {time_to_text(predicted_duration)}"
                )
                if show_iter_bar:
                    tqdm.write(msg, file=sys.stdout)
                else:
                    print(msg)
        else:
            if verbose:
                msg = (
                    f"Completed all {n_iterations} iterations. "
                    f"Evidence: {self.evidence[-1]:.4f}, BIC: {self.BIC[-1]:.2f}"
                )
                if show_iter_bar:
                    tqdm.write(msg, file=sys.stdout)
                else:
                    print(msg)
            self.total_fit_time += time.time() - start

        if show_iter_bar and hasattr(iter_pbar, "close"):
            iter_pbar.close()

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
        to_df: bool = False,
        sub_index: Optional[Union[int, str, Sequence[Union[int, str]]]] = None,
        combine_all: bool = False,
        save_to_folder: Optional[str] = None,
        save_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[SimulatedDataList, Any]:
        """Simulate data for subjects using fitted parameters.

        By default, returns a `SimulatedDataList` (which is a standard Python list of each subject's
        simulated data as returned by your model). You can inspect, modify, or put it together however you like.

        Flexible Reconstruction & Export Options:
        - `to_df=True`: Reconstructs the per-subject datasets into combined DataFrame(s).
        - `sub_index=0` (or `sub_index="trials"`): Reconstructs/saves ONLY a specific sub-index to 1 DataFrame.
        - `combine_all=True` (or `sub_index="all"`): Constructs all sub-dfs across all subjects together into 1 combined DataFrame.
        - `save_to_folder="path"`: Automatically saves reconstructed DataFrame(s) to CSV in that directory.

        Parameters
        ----------
        mode : str, default='resample'
            Parameter sampling source: 'resample' (from posterior samples), 'hyper_params' (from population priors),
            'override_params' (custom parameters), or 'mean_params' (subject posterior means).
        subjects : Union[str, Sequence[int]], default='all'
            Subjects to simulate.
        override_params : Optional[Dict[str, Any]], default=None
            Parameters to use if mode='override_params'.
        n_samples : int, default=1000
            Number of draws when mode='hyper_params'.
        to_df : bool, default=False
            If True, returns the reconstructed DataFrame(s) instead of the list of subject datas.
        sub_index : Optional[Union[int, str, Sequence[Union[int, str]]]], default=None
            Specific sub-index (e.g. 0, 1, or 'phase1') to reconstruct or save.
            Use 'all' to construct all sub-dfs together into 1 combined DataFrame.
        combine_all : bool, default=False
            If True, constructs all sub-dfs across all subjects together into 1 master DataFrame.
        save_to_folder : Optional[str], default=None
            If provided, saves the reconstructed DataFrame(s) directly to this directory.
        save_dir : Optional[str], default=None
            Alias for save_to_folder.

        Returns
        -------
        Union[SimulatedDataList, pd.DataFrame, Dict[Union[int, str], pd.DataFrame]]
            A SimulatedDataList of per-subject data (which supports .to_dataframe() and .save()),
            or the reconstructed DataFrame(s) if `to_df=True`.
        """
        # Backwards compatibility for save_kinds / kinds
        if sub_index is None:
            sub_index = kwargs.get("save_kinds", kwargs.get("kinds", kwargs.get("sub_indices", None)))

        sim_data = []
        target_subjects = self.subjects if subjects == "all" else list(subjects)
        target_folder = save_to_folder or save_dir

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
                if self.mean_params is not None and s < len(next(iter(self.mean_params.values()), [])):
                    raw_p = {k: np.array([self.mean_params[k][s]]) for k in self.params}
                else:
                    raw_p = {k: np.array([self.hyper_params[k]["mean"]]) for k in self.params}
                transformed_p = {k: self.transformations[k](raw_p[k]) for k in self.params}

            # Invoke model in simulate mode
            sim_res = self._invoke_model(subj_data, transformed_p, mode="simulate", raw_samples=raw_p)
            if sim_res is None:
                sim_res = subj_data

            sim_data.append(sim_res)

        result = SimulatedDataList(sim_data, sampler=self)

        # Save to folder if requested
        if target_folder:
            result.save(folder=target_folder, sub_index=sub_index, combine_all=combine_all)

        # Return reconstructed DataFrame(s) if requested
        if to_df:
            return result.to_dataframe(sub_index=sub_index, combine_all=combine_all)

        return result

    def to_dataframes(
        self,
        sim_data: Sequence[Any],
        sub_index: Optional[Union[int, str, Sequence[Union[int, str]]]] = None,
        combine_all: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Reconstruct subject simulated data into 1 combined DataFrame or X DataFrames for each sub-index."""
        return _reconstruct_dataframes(
            sim_data,
            sampler=self,
            sub_index=sub_index,
            combine_all=combine_all,
            **kwargs,
        )

    def save_simulated_data(
        self,
        sim_data: Sequence[Any],
        folder: str,
        sub_index: Optional[Union[int, str, Sequence[Union[int, str]]]] = None,
        combine_all: bool = False,
        file_format: str = "csv",
        **kwargs: Any,
    ) -> Dict[Union[int, str], str]:
        """Save reconstructed simulated DataFrames directly to a folder."""
        return _save_reconstructed_dataframes(
            sim_data,
            folder=folder,
            sub_index=sub_index,
            combine_all=combine_all,
            sampler=self,
            file_format=file_format,
            **kwargs,
        )

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
