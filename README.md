# importance_sampling

A lightweight, general-purpose Python package for fitting computational and cognitive models to multi-subject data using **Iterative Importance Sampling (IIS)**.

---

## Installation

Install the package directly from GitHub using `pip`:

```bash
pip install git+https://github.com/BoazRosenberg/Importance-Sampling.git
```

For interactive Plotly report generation (`sampler.create_report`), ensure `plotly` is installed:

```bash
pip install plotly
```

---

## How to Use the Package

To fit a computational model, you provide three core inputs:

1. **Input Data (`data`)**: A list containing the dataset for each subject (`[subj_1_data, subj_2_data, subj_3_data, ...]`). Each subject's data can be a dictionary, DataFrame, or array containing trial-by-trial choices and observed outcomes.
2. **A Model (`model`)**: A Python function that takes a single subject's data and a sample of candidate parameter draws, and evaluates trial outcomes in a single vectorized loop.
3. **Parameters and Priors (`hyper_params`)**: A dictionary defining each parameter to estimate, its initial population mean and standard deviation in latent space, and its transformation function into valid parameter bounds.

---

## Example: Full-Information Two-Armed Bandit (Q-Learning)

The following example demonstrates fitting a full-information two-armed bandit model across multiple subjects. On each trial, the participant chooses between Arm 1 and Arm 2, and the outcomes for both arms are revealed. Action values initialize at $Q_1 = 0.5$ and $Q_2 = 0.5$.

```python
import numpy as np
from scipy.special import expit
from importance_sampling import Sampler, sigmoid, softplus, load_model

# -----------------------------------------------------------------------------
# 1. Data: List of datasets for each subject
# Each subject's data contains the columns: trial, choice, reward1, and reward2
# -----------------------------------------------------------------------------
data = [subj_1_data, subj_2_data, subj_3_data]


# -----------------------------------------------------------------------------
# 2. Model Function
# -----------------------------------------------------------------------------
def q_learning_model(subj_data, parameters, mode="log_likelihood"):
    alpha = parameters["lr"]        # 1D array of shape (n_samples,), bounded in [0, 1]
    beta = parameters["inv_temp"]   # 1D array of shape (n_samples,), strictly positive > 0
    n_samples = len(alpha)

    choices = subj_data["choice"]
    reward1 = subj_data["reward1"]
    reward2 = subj_data["reward2"]
    n_trials = len(choices)

    # Initialize Q values to 0.5
    Q1 = np.full(n_samples, 0.5)
    Q2 = np.full(n_samples, 0.5)

    log_likelihood = np.zeros(n_samples)
    p_choices = []

    for i in range(n_trials):
        # Probability of choosing Arm 1 based on sigmoid of difference scaled by beta
        p_choose_1 = expit(beta * (Q1 - Q2))
        p_choose_1 = np.clip(p_choose_1, 1e-12, 1.0 - 1e-12)
        p_choices.append(p_choose_1)

        # Accumulate log-likelihood of observed choice
        if choices[i] == 1:
            log_likelihood += np.log(p_choose_1)
        else:
            log_likelihood += np.log(1.0 - p_choose_1)

        # Update Q1 and Q2
        Q1 += alpha * (reward1[i] - Q1)
        Q2 += alpha * (reward2[i] - Q2)

    if mode == "simulate":
        return np.array(p_choices)
    return log_likelihood


# -----------------------------------------------------------------------------
# 3. Parameters, Priors, and Transformations
# -----------------------------------------------------------------------------
hyper_priors = {
    "lr":       {"mean": 0.0, "sd": 1.0, "transform": sigmoid},   # Sigmoid -> [0, 1]
    "inv_temp": {"mean": 1.0, "sd": 1.0, "transform": softplus},  # Softplus -> (0, inf)
}


# -----------------------------------------------------------------------------
# 4. Initialize & Fit Sampler
# -----------------------------------------------------------------------------
sampler = Sampler(
    data=data,
    model=q_learning_model,
    hyper_params=hyper_priors,
    n_choices=2,               # Choice alternatives per decision (important for BIC scaling)
    model_name="FullInfoQLearning",
)

sampler.iterative_model_fit(n_iterations=12, n_samples=1000)

# Print terminal summary table
sampler.summary()

# Generate an interactive Plotly report widget
sampler.create_report(filename="qlearning_report.html")
```

---

## The Model Function: Arguments, Shapes & Modes

Your model function receives three arguments:

```python
def my_model(subj_data, parameters, mode="log_likelihood"):
```

### Arguments

| Argument | Type | Description |
| :--- | :--- | :--- |
| `subj_data` | `Any` (e.g. `dict`, `DataFrame`) | The dataset for a **single subject** containing observed trial choices and outcomes. |
| `parameters` | `Dict[str, np.ndarray]` | Dictionary mapping parameter names to candidate values. Each parameter is a **1D NumPy array of shape `(n_samples,)`** (e.g., 1,000 draws). Parameters are **automatically transformed** into their valid domain bounds by the `Sampler` *before* entering your model. |
| `mode` | `str` | Either `"log_likelihood"` or `"simulate"`. |

### Behavior by `mode`

* **`mode == "log_likelihood"`**:
  Evaluates the accumulated log-likelihood across all trials for each candidate parameter set.
  * **Return Value**: A **1D NumPy array of shape `(n_samples,)`**, where index $j$ is the total log-likelihood for particle $j$.

* **`mode == "simulate"`**:
  Evaluates choice probabilities across trials for generating simulated behavior.
  * **Return Value**: An array of choice probabilities `p_choices` for each trial. When calling `sampler.simulate()`, these are averaged across parameter samples to return the dataset with trial-by-trial mean choice probabilities.

---

## API Reference & Detailed Specifications

### `Sampler` Class

```python
Sampler(
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
)
```

#### Initialization Parameters

* **`data`** (`Sequence[Any]`):
  List or sequence where `data[s]` contains the empirical dataset for subject $s$.
* **`model`** (`Callable`):
  Model function with signature `model(subj_data, parameters, mode="log_likelihood")`.
* **`hyper_params`** (`Dict[str, Dict[str, Any]]`):
  Initial population hyper-priors in latent standard normal space. Format:
  ```python
  {
      "param_name": {
          "mean": float,       # Latent population mean
          "sd": float,         # Latent population standard deviation
          "transform": func,   # Optional link function (e.g. sigmoid, softplus)
      }
  }
  ```
* **`transformations`** (`Optional[Dict[str, Callable]]`, default=`None`):
  Optional dictionary mapping parameter names to functions that transform latent normal values into valid bounded ranges (e.g., `sigmoid` for $[0, 1]$, `softplus` for $(0, \infty)$). If specified under the `"transform"` key of `hyper_params`, this argument can be omitted.
* **`multinormal`** (`Union[bool, str, List[Tuple[str, str]]]`, default=`False`):
  Structure of the parameter covariance matrix in latent space:
  * `False`: Independent parameters (identity correlation matrix).
  * `"full"`: Estimates the full $P \times P$ correlation matrix between all parameters.
  * `List[Tuple[str, str]]`: Estimates correlation only between specified pairs (e.g., `[("lr", "inv_temp")]`).
* **`n_choices`** (`int`, default=`2`):
  Number of discrete choice options available per decision (e.g., 2 for a two-armed bandit). Used to scale degrees of freedom for BIC calculations:
  $$\text{sample\_size} = \max(1, N_{\text{subjects}} \times \text{n\_choices})$$
* **`model_name`** (`str`, default=`"unnamed_model"`):
  Descriptive name for the model used in summaries, reports, and file saving.
* **`description`** (`str`, default=`""`):
  Optional notes or descriptive documentation for the model.
* **`type`** (`str`, default=`"B"`):
  Model category code (e.g. `"B"` for Behavioral).
* **`random_state`** (`Optional[Union[int, np.random.Generator]]`, default=`None`):
  Seed integer or NumPy Generator for reproducible sampling.
* **`group_diff`** (`Optional[Union[Dict[str, Any], Sequence[str]]]`, default=`None`):
  Enables estimation of between-group differences for specified parameters.
  * Either a dictionary mapping parameter names to the group column or path in subject datasets (e.g. `{"alpha": "condition"}` or `{"alpha": ("subject_data", "group")}`).
  * Or a list/sequence of parameter names (e.g. `["alpha"]`), using `group_column`.
  * **Nested Sub-DataFrame Support**: If each subject's data consists of multiple sub-dataframes (e.g. `{"subject_data": df_info, "task": df_trials}` or `[df_info, df_trials]`):
    * **Tuple path**: `group_column=("subject_data", "group")` or `group_diff={"alpha": ("subject_data", "group")}`.
    * **Dot notation**: `group_column="subject_data.group"` or `group_column="0.group"`.
    * **Automatic deep search**: If you just pass `group_column="group"`, the sampler automatically searches inside all sub-dataframes and sub-dicts to find the column.
  * **Statistical estimation**: For each parameter with $N$ groups, fits $2 + (N - 1)$ hyperparameters:
    1. Grand population mean ($\mu$)
    2. Pooled within-group standard deviation ($\sigma$)
    3. Between-group differences ($\Delta_{k} = \mu_{g_k} - \mu_{g_0}$) for each non-reference group.
  * **BIC Penalization**: Accurately adds $(N - 1)$ degrees of freedom per group-difference parameter to the BIC penalty term.
* **`group_column`** (`Union[str, Tuple, List]`, default=`"group"`):
  Default column name or nested path in subject datasets indicating group membership when `group_diff` is passed as a list (e.g., `"group"`, `"subject_data.group"`, or `("subject_data", "group")`).

---

### Fitting Method: `sampler.iterative_model_fit`

```python
sampler.iterative_model_fit(
    n_iterations: int = 10,
    n_samples: int = 1000,
    epsilon: float = 0.01,
    n_mean: int = 10,
    stop_at_convergence: bool = True,
    verbose: bool = False,
    progress_bar: Union[bool, str] = True,
) -> "Sampler"
```

Runs the iterative importance sampling estimation loop until maximum iterations or evidence stabilization.

#### Parameters

* **`n_iterations`** (`int`, default=`10`):
  Maximum number of importance sampling iterations.
* **`n_samples`** (`int`, default=`1000`):
  Number of candidate parameter draws (particles) sampled per subject on each iteration.
* **`epsilon`** (`float`, default=`0.01`):
  Convergence threshold. Iteration stops when the average change in total evidence over the last `n_mean` iterations falls below `epsilon`:
  $$\frac{\text{evidence}_t - \text{evidence}_{t - n_{\text{mean}}}}{n_{\text{mean}}} < \epsilon$$
* **`n_mean`** (`int`, default=`10`):
  Window size in iterations used for the moving convergence check.
* **`stop_at_convergence`** (`bool`, default=`True`):
  Whether to terminate early when the convergence threshold is reached.
* **`verbose`** (`bool`, default=`False`):
  If `True`, prints clean iteration timestamps, evidence change, and elapsed/remaining durations.
* **`progress_bar`** (`Union[bool, str]`, default=`True`):
  Controls progress bar display:
  * `True` (or `"single"`): Displays **one clean, in-place progress bar** tracking iterations with live subject progress in the postfix (`subj: 32/96`), eliminating multi-line terminal spam.
  * `"nested"`: Displays both outer iteration and inner subject bars.
  * `"subjects"`: Displays only the subject progress bar.
  * `False`: Disables all progress bars.

---

### Simulation Method: `sampler.simulate`

```python
sim_dat = sampler.simulate(
    mode: str = "resample",
    subjects: Union[str, Sequence[int]] = "all",
    override_params: Optional[Dict[str, Any]] = None,
    n_samples: int = 1000,
    to_df: bool = False,
    sub_index: Optional[Union[int, str, Sequence[Union[int, str]]]] = None,
    combine_all: bool = False,
    save_to_folder: Optional[str] = None,
) -> Union[SimulatedDataList, pd.DataFrame, Dict[Union[int, str], pd.DataFrame]]
```

Runs the model in simulate mode using the fitted parameters.

#### Basic Usage: Simple List of Subject Datas

By default, `sampler.simulate()` returns a `SimulatedDataList` (which behaves as a standard Python list containing each subject's simulated output data directly from your model):

```python
# 1. Simple list of subject datasets
sim_dat = sampler.simulate()

# Functions as a standard Python list:
print(len(sim_dat))       # e.g., 96 subjects
print(sim_dat[0])          # Subject 0's simulated data (DataFrame, dict, or list/array of sub-dfs)
```

#### Reconstructing into Combined DataFrames

The object in each subject's data can be a single DataFrame/dict, or a list/array/dict containing $X$ sub-DataFrames (e.g. blocks, phases, or conditions). You can reconstruct them in 3 ways:

1. **Construct the $X$ sub-dfs in each subject's data into $X$ combined DataFrames**:
```python
# Returns a dictionary: {0: df_0, 1: df_1, ...} (or by key names)
dfs = sim_dat.to_dataframes()
block0_df = dfs[0]
block1_df = dfs[1]
```

2. **Construct just one specific sub-index to 1 DataFrame**:
```python
# Combine only sub-index 0 across all subjects:
df_sub0 = sim_dat.to_dataframe(sub_index=0)

# Or directly during simulate:
df_sub0 = sampler.simulate(to_df=True, sub_index=0)
```

3. **Construct ALL sub-dfs across all subjects together into 1 combined DataFrame**:
```python
# Concatenates all sub-dfs across all subjects, with 'subject' and 'sub_index' columns:
master_df = sim_dat.to_dataframe(combine_all=True)  # or sub_index="all"

# Or directly during simulate:
master_df = sampler.simulate(to_df=True, combine_all=True)
```

#### Saving DataFrames Directly to Folder

You can save all reconstructed DataFrames directly to CSV in a folder, or save only one particular sub-index:

```python
# Save all X DataFrames to folder (e.g. simulated_data_sub_0.csv, simulated_data_sub_1.csv):
sim_dat.save("simulated_output")

# Or save only a specific sub-index (e.g. only sub_index 0):
sim_dat.save("simulated_output", sub_index=0)

# Or save all sub-dfs combined into 1 single file:
sim_dat.save("simulated_output", combine_all=True)

# Or save directly during simulate():
sampler.simulate(save_to_folder="simulated_output", sub_index=0)
```

---

### Interactive Plotly Report: `sampler.create_report`

```python
sampler.create_report(
    filename: Optional[str] = None,
    show: bool = True,
    transformed: bool = True,
    renderer: Optional[str] = None,
) -> Any
```

Generates an interactive multi-page diagnostic report widget powered by Plotly. Also available as a standalone function `create_report(sampler, ...)`. When exported to HTML (`filename="model_report.html"`), it renders as an interactive 3-page tabbed web application.

#### Parameters

* **`filename`** (`Optional[str]`, default=`None`):
  If provided (e.g., `"model_report.html"`), exports a standalone, self-contained interactive HTML file that can be opened in any web browser without needing a Python runtime.
* **`show`** (`bool`, default=`True`):
  Whether to render the interactive widget in the current environment (Jupyter notebook, Google Colab, or browser).
* **`transformed`** (`bool`, default=`True`):
  * `True`: Displays hyperparameters in their valid domain space (e.g. learning rate in $[0, 1]$, inverse temperature $> 0$).
  * `False`: Displays hyperparameters in latent normal space.
* **`renderer`** (`Optional[str]`, default=`None`):
  Plotly renderer option (e.g., `'browser'`, `'notebook'`, `'colab'`).

#### Multi-Page Single Model Report Layout

1. **Page 1: General Fit & Convergence**:
   * **Total Evidence (Log Likelihood) & BIC**: Dual-axis trajectory showing total model evidence climbing and BIC minimizing across iterations.
   * **Subject-Level Evidence Spaghetti Plot**: Trajectories of log marginal likelihood for each individual subject, alongside the population mean trajectory.
   * **Convergence Summary Metrics**: Total evidence, BIC, iteration count, duration, and convergence criteria status.
2. **Page 2: Model Parameters & Subject Distributions**:
   * **Model Parameters Summary Table**: Raw parameter names, fitted population means, standard deviations, and ±1 SD credible intervals.
   * **Hyperparameter Evolution Grid (3 Plots per Row)**: Mean trajectories and translucent shaded $\pm 1 \text{ SD}$ bands across iterations for each parameter.
   * **Individual Subject Posterior Means (Scatter Plot per Parameter)**: Subject-by-subject posterior draws plotted along the X-axis for each parameter to visualize between-subject heterogeneity.
   * **Latent Correlation Matrix Heatmap**: Full bivariate correlation heatmap ($r \in [-1.0, 1.0]$) between parameters in latent normal space.
3. **Page 3: Model Code & Run Metadata**:
   * **Run Metadata Cards**: Execution timestamps (start time, finish time, formatted duration), number of iterations, subject counts, particle counts, and system environment (Python & OS versions).
   * **Model Source Code Box**: Embedded, syntax-highlighted Python function code used to fit the model, complete with one-click copy.
   * **Companion JSON Metadata Viewer**: Full JSON dump of the model metadata snapshot saved alongside the model pickle.

---

### Multi-Model Comparison: `compare_models`

```python
from importance_sampling import compare_models

compare_models(
    samplers: Union[Sequence[Sampler], Dict[str, Sampler]],
    filename: Optional[str] = None,
    show: bool = True,
    compare_params: Optional[Sequence[str]] = None,  # Defaults to ALL parameters across models
    transformed: bool = True,
    renderer: Optional[str] = None,
) -> Any
```

Generates an interactive comparative report widget to evaluate two or more fitted models against each other. By default (`compare_params=None`), it automatically extracts and compares **all** parameters across the candidate models.

#### Multi-Page Comparison Layout

1. **Page 1: Likelihood, BICs & Participant Selection**:
   * **Total Evidence & BIC Trajectories**: Overlay of evidence and BIC evolution across iterations for all models.
   * **Final Model Ranking Table**: Ranks models by BIC (lowest is best), reporting $k$ parameters, Final Evidence, Final BIC, and $\Delta\text{BIC}$ relative to the winning model.
   * **Percentage of Participants Best Explained**: Reports the percentage and exact subject counts where each model achieved the highest individual log-marginal likelihood $\ln p(D_s \mid M)$.
2. **Page 2: Parameter Inclusion Matrix & Collapsible Code**:
   * **Parameter Overview Matrix Table**: Parameters listed on the Y-axis, compared models on the X-axis. Cells show whether each parameter is included (with fitted population mean and SD) or excluded (`—`), clearly distinguishing shared parameters from model-specific parameters.
   * **Collapsible Model Functions Code**: Tabbed viewer displaying the exact Python implementation function for each model, with a collapse/expand toggle to preserve vertical space and a one-click copy button.
3. **Page 3: Parameter Evolution Comparison Plots**:
   * **Shared Parameters Plotted Together**: When a parameter repeats across multiple models (e.g. `alpha` or `beta`), all models containing it are displayed on the **same subplot** with distinct color-coded curves and $\pm 1 \text{ SD}$ uncertainty ribbons for direct visual comparison.
   * **Unique Parameters**: If a parameter is specific to a single model, it is plotted individually with a badge indicating its unique status.
   * **Responsive Subplot Grid**: Arranged in 3 columns with tooltips and final estimated values.

---

### Model Persistence: `save_model` & `load_model`

Fitted `Sampler` objects can be serialized to disk and restored across sessions:

```python
# Save model to disk (method on Sampler instance)
# Saves both a model pickle file and a companion .json metadata file
saved_path = sampler.save_model(filename="my_fitted_model", directory="saved_models")

# Load a saved model (imported separately as a standalone function)
from importance_sampling import load_model

loaded_sampler = load_model(filename="my_fitted_model", directory="saved_models")
print(loaded_sampler.evidence[-1])
print(loaded_sampler.mean_params)
print(loaded_sampler.metadata)  # Access stored timestamp, duration, iterations, and model code
```

When saved, `sampler.save_model()` stores:
* The serialized `Sampler` instance pickle (`.pkl`).
* A companion JSON metadata file (`_metadata.json`) containing fit duration, start/end timestamps, number of iterations, convergence criteria, parameter specifications, and the model function source code.
* An internal `sampler.metadata` dictionary for programmatic inspection.

---

## Outputs: Accessible Attributes Reference

All estimation results, subject draws, and metrics are stored directly on the `Sampler` instance:

| Attribute | Type | Description |
| :--- | :--- | :--- |
| `sampler.evidence` | `List[float]` | Total model evidence (log marginal likelihood summed across all subjects) at each iteration. |
| `sampler.subj_evidence` | `List[List[float]]` | 2D list of shape `(n_iterations, n_subjects)` containing the individual log marginal likelihood for each subject across iterations. |
| `sampler.evidence_change` | `List[float]` | Difference in total evidence between successive iterations. |
| `sampler.hyper_params` | `Dict[str, Dict[str, float]]` | Final fitted population distribution in latent normal space: `{"param": {"mean": float, "sd": float}}`. |
| `sampler.hyper_params_list` | `List[Dict]` | History of population hyper-parameters at each iteration from $0$ to $N$. |
| `sampler.mean_params` | `Dict[str, List[float]]` | Weighted posterior mean parameter value for each subject: `sampler.mean_params["lr"][subject_idx]`. |
| `sampler.samples` | `Dict[str, List[np.ndarray]]` | Full array of resampled posterior candidate particles for each subject: `sampler.samples["lr"][subject_idx]`. |
| `sampler.cor_matrix` | `np.ndarray` | Final parameter correlation matrix in latent normal space (shape `(n_params, n_params)`). |
| `sampler.cor_matrices` | `List[np.ndarray]` | Parameter correlation matrices across iterations. |
| `sampler.BIC` | `List[float]` | Bayesian Information Criterion at each iteration (lower is better): $\text{BIC} = -2 \cdot \text{evidence} + k \cdot \ln(N_{\text{choices}} \times N_{\text{subjects}})$. |
| `sampler.mean_accuracy` | `List[float]` | Average log-likelihood per trial choice: $\text{evidence} / (N_{\text{subjects}} \times \text{n\_choices})$. |
| `sampler.iterations` | `int` | Number of completed importance sampling iterations. |
| `sampler.total_fit_time` | `float` | Cumulative model estimation execution time in seconds. |
| `sampler.summary()` | `None` | Prints a formatted summary table of fit metrics and population parameters to the terminal. |

*(For backwards compatibility, `sampler.log_likelihood` aliases `sampler.evidence` and `sampler.log_likelihoods` aliases `sampler.subj_evidence`).*

---

## Built-In Link Functions & Utilities

The package provides numerically stable transformation functions in `importance_sampling`:

* **`sigmoid(x)`**: Maps real numbers $(-\infty, \infty)$ to the open interval $(0, 1)$. Ideal for probabilities and learning rates.
* **`softplus(x)`**: $\ln(1 + e^x)$, smooth mapping $(-\infty, \infty)$ to strictly positive values $(0, \infty)$. Ideal for inverse temperatures, response noise, and diffusion rates.
* **`logit(p)`**: Inverse sigmoid mapping probabilities in $(0, 1)$ to real numbers $(-\infty, \infty)$.
* **`log_sigmoid(x)`**: Numerically stable $\ln(\sigma(x))$.
* **`logsumexp(a)`**: Stable computation of $\ln\left(\sum e^a\right)$.
* **`time_to_text(seconds)`**: Converts elapsed seconds into human-readable format (`"1h 12m 30s"`).

---

## Demo File (`demo.py`)

For a complete, runnable demonstration with realistic multi-subject data, explore **`demo.py`** in this repository.

It contains:
* A `simulate_data` helper function that simulates data for 10 subjects (30 trials each, with `trial`, `choice`, `reward1`, and `reward2` columns).
* The single-loop Q-learning model function.
* Fitting with `Sampler`.
* Recovered evidence and subject parameter inspection.
* Choice simulation via `sampler.simulate()`.
* Model persistence (`save_model` and `load_model`).
* Interactive Plotly report generation (`sampler.create_report`).

You can open and inspect `demo.py` in your code editor as an easy-to-adapt template for your own multi-subject experiments.

---

## License

This project is licensed under the [MIT License](LICENSE).
