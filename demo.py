#!/usr/bin/env python3
"""demo.py: Hands-on demonstration of importance_sampling.

Open and look through this file to try the package out yourself!
This script implements the exact same Q-learning workflow described in the README,
but with sample data for multiple subjects (10 subjects, 30 trials each):

1. Simulates realistic choice and reward data for 10 subjects.
2. Defines the Q-learning model function in a single clean loop.
3. Fits the hierarchical population hyper-parameters with the Sampler.
4. Inspects the fitted outputs (evidence, population priors, subject estimates).
5. Demonstrates additional utilities:
   - sampler.simulate(): returns the dataset with trial mean_choice_probability
   - sampler.save_model(): persists the fitted model to disk
   - load_model(): loads the saved model back into memory
"""

import sys
import os
import numpy as np
from scipy.special import expit

# Ensure importance_sampling is importable when running locally
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from importance_sampling import Sampler, load_model, sigmoid, softplus


# =============================================================================
# 1. Simulate Sample Data (Full-information two-armed bandit, 30 trials)
# =============================================================================
def simulate_data(n_subjects=10, n_trials=30, seed=42):
    rng = np.random.default_rng(seed)
    data = []
    ground_truth = []

    for s in range(n_subjects):
        true_alpha = float(np.clip(rng.normal(0.40, 0.08), 0.1, 0.9))
        true_beta = float(np.clip(rng.normal(3.0, 0.5), 1.0, 6.0))

        choices = []
        reward1 = []
        reward2 = []

        # Q values initialized to 0.5
        Q1 = 0.5
        Q2 = 0.5

        for _ in range(n_trials):
            # Arm 1 pays 70% of the time, Arm 2 pays 30% of the time
            r1 = 1.0 if rng.random() < 0.70 else 0.0
            r2 = 1.0 if rng.random() < 0.30 else 0.0

            # Probability of choosing arm 1 from sigmoid of difference
            p_choose_1 = float(expit(true_beta * (Q1 - Q2)))
            c = 1 if rng.random() < p_choose_1 else 0

            choices.append(c)
            reward1.append(r1)
            reward2.append(r2)

            # Full information update for both arms
            Q1 += true_alpha * (r1 - Q1)
            Q2 += true_alpha * (r2 - Q2)

        data.append({
            "trial": np.arange(n_trials),
            "choice": np.array(choices, dtype=int),
            "reward1": np.array(reward1, dtype=float),
            "reward2": np.array(reward2, dtype=float),
        })
        ground_truth.append({"alpha": true_alpha, "beta": true_beta})

    return data, ground_truth


# =============================================================================
# 2. Define the Model (Full Information Two-Armed Bandit)
# =============================================================================
def q_learning_model(subj_data, parameters, mode="log_likelihood"):
    """Full-information two-armed bandit Q-learning model.

    Parameters are automatically transformed into their bounds by Sampler:
      - parameters["lr"]: 1D array of shape (n_samples,), bounded in [0, 1]
      - parameters["inv_temp"]: 1D array of shape (n_samples,), strictly positive > 0

    Returns:
      - p_choices (choice probabilities) if mode == "simulate"
      - log_likelihood (total log evidence per sample) if mode == "log_likelihood"
    """
    alpha = parameters["lr"]        # Shape: (n_samples,)
    beta = parameters["inv_temp"]   # Shape: (n_samples,)
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
        # Probability of choosing Arm 1 based on sigmoid of difference using beta
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


# =============================================================================
# 3. Parameters, Priors, and Transformations
# =============================================================================
hyper_priors = {
    "lr":       {"mean": 0.0, "sd": 1.0, "transform": sigmoid},   # Sigmoid -> [0, 1]
    "inv_temp": {"mean": 1.0, "sd": 1.0, "transform": softplus},  # Softplus -> (0, inf)
}


# =============================================================================
# 4. Run the Sampler & Utilities
# =============================================================================
def main():
    print("=" * 65)
    print("  importance_sampling Demo: Full-Information Q-Learning Model")
    print("=" * 65)

    # 1. Create simulated data
    print("\n1. Generating sample data for 10 subjects (30 trials each)...")
    data, ground_truth = simulate_data(n_subjects=10, n_trials=30)

    # 2. Instantiate Sampler
    print("2. Initializing Sampler...")
    sampler = Sampler(
        data=data,
        model=q_learning_model,
        hyper_params=hyper_priors,
        n_choices=2,  # Amount of choices the likelihood is based on (important for BIC calculations)
        model_name="FullInfoQLearning",
        random_state=42,
    )

    # 3. Fit iteratively
    print("3. Fitting model (10 iterations, 800 samples/subject)...")
    sampler.iterative_model_fit(n_iterations=10, n_samples=800, verbose=True)

    # 4. Access Outputs directly on the Sampler
    print("\n4. Fitted Results:")
    sampler.summary()

    print("\n--- Accessible Attributes on Sampler ---")
    print(f"Final Evidence (all subjects): {sampler.evidence[-1]:.4f}")
    print(f"Subject 0 Evidence:            {sampler.subj_evidence[-1][0]:.4f}")
    print(f"Fitted Population lr:          mean={sampler.hyper_params['lr']['mean']:.4f}, sd={sampler.hyper_params['lr']['sd']:.4f}")
    print(f"Subject 0 Mean lr:             {sampler.mean_params['lr'][0]:.4f}")

    # 5. Additional Utilities
    print("\n5. Additional Utilities:")
    # A. Simulate: returns data as it was with mean_choice_probability
    sim_data = sampler.simulate()
    print(f"• simulate(): returned data for {len(sim_data)} subjects.")
    print(f"  Subject 0 mean_choice_probability (first 5 trials): {np.round(sim_data[0]['mean_choice_probability'][:5], 3)}")

    # B. Save model
    saved_path = sampler.save_model("demo_model", directory="saved_models")
    print(f"• save_model(): successfully saved to '{saved_path}'")

    # C. Load model (imported separately)
    loaded_sampler = load_model("demo_model", directory="saved_models")
    print(f"• load_model(): successfully loaded '{loaded_sampler.model_name}' (evidence: {loaded_sampler.evidence[-1]:.4f})")

    print("\nDemo finished successfully!")
    print("=" * 65)


if __name__ == "__main__":
    main()
