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
* **`group_diff`** (`Optional[Union[Dict[str, str], Sequence[str]]]`, default=`None`):
  Enables estimation of between-group differences for specified parameters.
  * Either a dictionary mapping parameter names to the group column name in subject datasets (e.g. `{"alpha": "condition", "beta": "diagnosis"}`).
  * Or a list/sequence of parameter names (e.g. `["alpha"]`), using `group_column`.
  * **Statistical estimation**: For each parameter with $N$ groups, fits $2 + (N - 1)$ hyperparameters:
    1. Grand population mean ($\mu$)
    2. Pooled within-group standard deviation ($\sigma$)
    3. Between-group differences ($\Delta_{k} = \mu_{g_k} - \mu_{g_0}$) for each non-reference group.
  * **BIC Penalization**: Accurately adds $(N - 1)$ degrees of freedom per group-difference parameter to the BIC penalty term.
* **`group_column`** (`str`, default=`"group"`):
  Default column or attribute name in subject datasets indicating group membership when `group_diff` is passed as a list.

---

### Fitting Method: `sampler.iterative_model_fit`

```python
sampler.iterative_model_fit(
    n_iterations: int = 10,
    n_samples: int = 1000,
    epsilon: float = 0.01,
    n_mean: int = 10,
    stop_at_convergence: bool = True,
    verbose: bool = True,
    progress_bar: bool = True,
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
* **`verbose`** (`bool`, default=`True`):
  If `True`, prints iteration progress, elapsed time, predicted time remaining, total evidence, and convergence notices.
* **`progress_bar`** (`bool`, default=`True`):
  If `True`, displays `tqdm` progress bars across iterations and within iterations (for subjects), updating live with Evidence, BIC, and &Delta;Evidence.

---

### Simulation Method: `sampler.simulate`

```python
sampler.simulate(
    mode: str = "resample",
    subjects: Union[str, Sequence[int]] = "all",
    override_params: Optional[Dict[str, Any]] = None,
    n_samples: int = 1000,
) -> List[Any]
```

Runs the model in simulate mode using the fitted parameters.

#### Parameters

* **`mode`** (`str`, default=`"resample"`):
  Source of parameter draws:
  * `"resample"`: Uses each subject's resampled posterior candidate draws (`sampler.samples`).
  * `"hyper_params"`: Draws $N$ candidate parameters from the fitted population normal distribution.
  * `"override_params"`: Uses an explicitly provided parameter dictionary.
* **`subjects`** (`Union[str, Sequence[int]]`, default=`"all"`):
  Either `"all"` or a list of integer subject indices to simulate.
* **`override_params`** (`Optional[Dict[str, Any]]`, default=`None`):
  Custom parameter dictionary to test specific counterfactuals.
* **`n_samples`** (`int`, default=`1000`):
  Number of parameter draws used when `mode="hyper_params"`.

#### Return Value

A list containing each subject's data as it was originally structured, with an added **`mean_choice_probability`** array (1D NumPy array across trials, averaged across parameter samples):

```python
sim_data = sampler.simulate()

# Access mean choice probability for Subject 0:
print(sim_data[0]["mean_choice_probability"])

# Original data keys remain preserved:
print(sim_data[0]["choice"])
print(sim_data[0]["trial"])
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

Generates an interactive diagnostic report widget powered by Plotly. Also available as a standalone function `create_report(sampler, ...)`.

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

#### Report Layout & Visualizations Included

1. **Model Fit & Convergence**:
   * **Total Evidence (Log Likelihood) & BIC**: Dual-axis trajectory showing total model evidence climbing and BIC minimizing across iterations.
   * **Subject-Level Evidence Spaghetti Plot**: Trajectories of log marginal likelihood for each individual subject, alongside the population mean.
2. **Model Parameters Table**:
   * Compact table listing all model parameters by their raw dictionary keys (e.g. `alpha`, `beta`, `decay`, etc.), with final fitted means, standard deviations, and ±1 SD credible bounds.
3. **Hyperparameter Evolution Grid (3 Plots per Row)**:
   * Arranged in a responsive 3-column grid, adding a separate subplot for each selected parameter.
   * **Solid line** represents the population mean trajectory across iterations $0, 1, \dots, N$.
   * **Translucent shaded band** represents $\pm 1 \text{ SD}$ around the mean.
   * Hover tooltips display the exact iteration, mean, $+1 \text{ SD}$, and $-1 \text{ SD}$.
   * Scalable to models with 10+ parameters without vertical clutter.
4. **Individual Subject Posterior Means (Scatter Plot per Parameter)**:
   * Arranged in a separate plot for each selected parameter.
   * **Y-axis**: Subject index / number ($0, 1, \dots, N_{\text{subjects}} - 1$).
   * **X-axis**: The subject's posterior mean value for that parameter.
   * Each dot represents an individual subject, enabling immediate inspection of between-subject spread without displaying raw numbers.
5. **Multinormal Correlation / Covariance Matrix Heatmap**:
   * Located at the end of the report: an interactive correlation heatmap showing latent space parameter correlations ($r \in [-1.0, 1.0]$) with a diverging color scale, cell annotations, and hover values. Useful when fitting models with `multinormal="full"` or paired correlations.

---

### Multi-Model Comparison: `compare_models`

```python
from importance_sampling import compare_models

compare_models(
    samplers: Union[Sequence[Sampler], Dict[str, Sampler]],
    filename: Optional[str] = None,
    show: bool = True,
    compare_params: Optional[Sequence[str]] = None,
    renderer: Optional[str] = None,
) -> Any
```

Generates an interactive comparative report widget to evaluate two or more fitted models against each other.

#### Features Included:
1. **Evidence & BIC Evolution**:
   * Dual subplots showing total evidence (log likelihood) and BIC curves across iterations for all models simultaneously.
2. **Final Fit Comparison Bar Plot & Table**:
   * Side-by-side grouped bar plot of final evidence and BIC.
   * Comprehensive model ranking table with $k$ (parameters), Final Evidence, Final BIC, and $\Delta\text{BIC}$ relative to the winning model.
3. **Parameter Comparison Across Models**:
   * Pass `compare_params=["lr", "inv_temp"]` to inspect shared parameter estimates (population mean $\pm 1 \text{ SD}$ error bars and subject distributions) across the models.
   * Can also be called directly via `compare_parameters([m1, m2], params=["lr"])` to keep the primary report focused.

---

### Model Persistence: `save_model` & `load_model`

Fitted `Sampler` objects can be serialized to disk and restored across sessions:

```python
# Save model to disk (method on Sampler instance)
saved_path = sampler.save_model(filename="my_fitted_model", directory="saved_models")

# Load a saved model (imported separately as a standalone function)
from importance_sampling import load_model

loaded_sampler = load_model(filename="my_fitted_model", directory="saved_models")
print(loaded_sampler.evidence[-1])
print(loaded_sampler.mean_params)
```

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
