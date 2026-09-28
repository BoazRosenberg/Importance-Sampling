# importance_sampling

A lightweight Python package for fitting computational and cognitive models to multi-subject data using **Iterative Importance Sampling (IIS)**.

---

## Installation

Install the package directly from GitHub:

```bash
pip install git+https://github.com/BoazRsnbrg/importance_sampling.git
```

---

## How to Use the Package

To fit a model, you provide three core inputs:

1. **Data (`data`)**: A list containing the dataset for each subject (`[subj_1_data, subj_2_data, ...]`).
2. **Model (`model`)**: A Python function that evaluates trial choices and outcomes for a single subject.
3. **Parameters and Priors (`hyper_params`)**: A dictionary defining the parameters to estimate, their initial population mean and standard deviation in latent space, and their transformation functions into valid bounds.

---

## Example: Full-Information Two-Armed Bandit (Q-Learning)

The following example demonstrates fitting a full-information two-armed bandit model across multiple subjects. On each trial, the participant chooses between Arm 1 and Arm 2, and the outcomes for both arms are revealed. Action values initialize at $Q_1 = 0.5$ and $Q_2 = 0.5$.

```python
import numpy as np
from scipy.special import expit
from importance_sampling import Sampler, sigmoid, softplus

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

        # Update Q values using full outcome information
        Q1 += alpha * (reward1[i] - Q1)
        Q2 += alpha * (reward2[i] - Q2)

    if mode == "simulate":
        return np.array(p_choices)
    return log_likelihood


# -----------------------------------------------------------------------------
# 3. Parameters and Priors
# -----------------------------------------------------------------------------
hyper_priors = {
    "lr":       {"mean": 0.0, "sd": 1.0, "transform": sigmoid},   # Sigmoid -> [0, 1]
    "inv_temp": {"mean": 1.0, "sd": 1.0, "transform": softplus},  # Softplus -> (0, inf)
}


# -----------------------------------------------------------------------------
# 4. Fit the Model
# -----------------------------------------------------------------------------
sampler = Sampler(
    data=data,
    model=q_learning_model,
    hyper_params=hyper_priors,
    n_choices=2,  # Number of choice options (important for BIC calculation)
    model_name="FullInfoQLearning",
)

sampler.iterative_model_fit(n_iterations=12, n_samples=1000)
```

---

## Model Function Requirements

The model function takes three arguments:

```python
def my_model(subj_data, parameters, mode="log_likelihood"):
```

1. **`subj_data`**: The dataset for a single subject (such as a dictionary, DataFrame, or array) containing choices and trial outcomes.
2. **`parameters`**: A dictionary where each parameter is a **1D NumPy array of $N$ samples** (shape `(n_samples,)`). The parameters are automatically transformed into their valid domain bounds before being passed to your model.
3. **`mode`**: A string indicating what the function should calculate:
   - **`"log_likelihood"`**: Returns a 1D NumPy array of total log-likelihoods for each parameter sample (length $N$).
   - **`"simulate"`**: Returns the choice probabilities for each trial. When calling `sampler.simulate()`, these are averaged across the parameter samples to yield the mean choice probabilities for each trial.

---

## Outputs: Fitting Results

All fitting results and posterior estimates are stored directly on the `Sampler` instance:

### 1. General & Subject Evidence
- **`sampler.evidence`**: A list of the total model evidence (log marginal likelihood summed across all subjects) at each iteration.
- **`sampler.subj_evidence`**: A 2D list of shape `(n_iterations, n_subjects)` containing the log marginal likelihood for each subject across iterations.

### 2. Population Hyperparameters
- **`sampler.hyper_params`**: A dictionary containing the final fitted population mean and standard deviation for each parameter:
  ```python
  print(sampler.hyper_params)
  # {'lr': {'mean': -0.412, 'sd': 0.285}, 'inv_temp': {'mean': 1.152, 'sd': 0.241}}
  ```
- **`sampler.hyper_params_list`**: The history of population parameters at each iteration from $0$ to $N$.
- **`sampler.cor_matrix`**: The estimated correlation matrix between parameters in latent normal space.

### 3. Subject-Level Estimates
- **`sampler.mean_params`**: A dictionary mapping each parameter to a list of weighted posterior means for each subject:
  ```python
  subj_0_lr = sampler.mean_params["lr"][0]
  ```
- **`sampler.samples`**: A dictionary mapping each parameter to full arrays of resampled posterior candidate draws for each subject.

### 4. Metrics & Summary
- **`sampler.BIC`**: Bayesian Information Criterion at each iteration (uses `n_choices` to scale degrees of freedom; lower is better).
- **`sampler.iterations`** & **`sampler.total_fit_time`**: Total completed iterations and execution runtime in seconds.
- **`sampler.summary()`**: Prints a formatted summary table of the final fit metrics and population hyper-parameters.

---

## Additional Utilities

### 1. Saving and Loading Models

Fitted models can be saved to disk and reloaded in another script or session:

```python
# Save the fitted model
saved_path = sampler.save_model(filename="my_fitted_model", directory="saved_models")

# Load a saved model (imported as a standalone function)
from importance_sampling import load_model

loaded_sampler = load_model(filename="my_fitted_model", directory="saved_models")
print(loaded_sampler.evidence[-1])
print(loaded_sampler.mean_params)
```

### 2. Simulating Choices (`sampler.simulate`)

The `simulate` method evaluates choice probabilities using the fitted posterior parameters. It returns the original data as it was (the list of subject datasets), with an added **`mean_choice_probability`** array for each subject (averaged across the parameter samples):

```python
sim_data = sampler.simulate()

# Mean choice probability for Subject 0 across trials:
print(sim_data[0]["mean_choice_probability"])

# Original data fields remain intact:
print(sim_data[0]["choice"])
```

---

## Demo File (`demo.py`)

For a runnable, self-contained demonstration with simulated data, check out **`demo.py`**.

While the example above illustrates the workflow, `demo.py` includes a helper function that generates realistic multi-subject data (10 subjects with 30 trials each, containing `trial`, `choice`, `reward1`, and `reward2`), fits the model with the `Sampler`, inspects recovered evidence and subject parameters, simulates choice probabilities, and demonstrates saving and loading. You can open and inspect `demo.py` to try it out and adapt it for your own experiments.

---

## License

This project is licensed under the [MIT License](LICENSE).
