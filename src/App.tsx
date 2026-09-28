import React, { useState } from 'react';
import { Copy, Check } from 'lucide-react';

export default function App() {
  const [copied, setCopied] = useState(false);

  const installCmd = 'pip install git+https://github.com/BoazRsnbrg/importance_sampling.git';

  const handleCopy = () => {
    navigator.clipboard.writeText(installCmd);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const codeSnippet = `import numpy as np
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

sampler.iterative_model_fit(n_iterations=12, n_samples=1000)`;

  const utilitiesSnippet = `# 1. Save the fitted model
saved_path = sampler.save_model(filename="my_fitted_model", directory="saved_models")

# 2. Load a saved model (imported as a standalone function)
from importance_sampling import load_model

loaded_sampler = load_model(filename="my_fitted_model", directory="saved_models")
print(loaded_sampler.evidence[-1])
print(loaded_sampler.mean_params)

# 3. Simulating Choices (returns original data with mean choice probability)
sim_data = sampler.simulate()

# Mean choice probability for Subject 0 across trials:
print(sim_data[0]["mean_choice_probability"])

# Original data fields remain intact:
print(sim_data[0]["choice"])`;

  return (
    <div className="min-h-screen bg-[#ffffff] text-[#1f2328] font-sans antialiased">
      {/* Top Header */}
      <header className="border-b border-[#d1d9e0] bg-[#f6f8fa] px-6 py-4">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-lg text-[#0969da] hover:underline cursor-pointer">
              BoazRsnbrg
            </span>
            <span className="text-[#59636e]">/</span>
            <span className="font-bold text-lg text-[#1f2328]">importance_sampling</span>
            <span className="ml-2 px-2 py-0.5 text-xs font-medium text-[#59636e] bg-[#ffffff] border border-[#d1d9e0] rounded-full">
              Public
            </span>
          </div>

          <button
            onClick={handleCopy}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium bg-[#ffffff] hover:bg-[#f3f4f6] text-[#1f2328] border border-[#d1d9e0] rounded-md shadow-xs transition-colors cursor-pointer"
          >
            {copied ? (
              <>
                <Check className="w-3.5 h-3.5 text-[#1a7f37]" />
                <span className="text-[#1a7f37]">Copied!</span>
              </>
            ) : (
              <>
                <Copy className="w-3.5 h-3.5 text-[#59636e]" />
                <span>Copy pip command</span>
              </>
            )}
          </button>
        </div>
      </header>

      {/* Main Container */}
      <main className="max-w-4xl mx-auto px-6 py-8">
        {/* Repo file list preview */}
        <div className="border border-[#d1d9e0] rounded-md mb-8 overflow-hidden bg-white">
          <div className="bg-[#f6f8fa] px-4 py-3 border-b border-[#d1d9e0] text-xs font-semibold text-[#59636e] flex items-center justify-between">
            <span>Repository Files</span>
            <span className="font-mono text-[11px] text-[#59636e]">branch: main</span>
          </div>

          <div className="divide-y divide-[#d1d9e0] text-xs font-mono">
            <div className="px-4 py-2 hover:bg-[#f6f8fa] flex items-center justify-between">
              <span className="text-[#0969da] font-medium">pyproject.toml</span>
              <span className="text-[#59636e]">PEP 517/621 packaging metadata</span>
            </div>
            <div className="px-4 py-2 hover:bg-[#f6f8fa] flex items-center justify-between">
              <span className="text-[#0969da] font-medium">src/importance_sampling/sampler.py</span>
              <span className="text-[#59636e]">Core general-purpose Sampler class</span>
            </div>
            <div className="px-4 py-2 hover:bg-[#f6f8fa] flex items-center justify-between">
              <span className="text-[#0969da] font-medium">src/importance_sampling/utils.py</span>
              <span className="text-[#59636e]">Logsumexp, link functions, time formatting</span>
            </div>
            <div className="px-4 py-2 hover:bg-[#f6f8fa] flex items-center justify-between bg-[#f6f8fa]/60">
              <span className="text-[#0969da] font-semibold">demo.py</span>
              <span className="text-[#59636e]">Self-contained demo with simulated data</span>
            </div>
            <div className="px-4 py-2 hover:bg-[#f6f8fa] flex items-center justify-between">
              <span className="text-[#0969da] font-medium">README.md</span>
              <span className="text-[#59636e]">Documentation &amp; usage walkthrough</span>
            </div>
          </div>
        </div>

        {/* README Box */}
        <article className="border border-[#d1d9e0] rounded-md p-8 bg-white shadow-xs space-y-6 text-sm text-[#1f2328] leading-relaxed">
          <div className="border-b border-[#d1d9e0] pb-4">
            <h1 className="text-3xl font-bold tracking-tight text-[#1f2328]">
              importance_sampling
            </h1>
            <p className="text-sm text-[#59636e] mt-1.5">
              A lightweight Python package for fitting computational and cognitive models to multi-subject data using <strong>Iterative Importance Sampling (IIS)</strong>.
            </p>
          </div>

          {/* Quick Install callout */}
          <div className="p-3 bg-[#f6f8fa] border border-[#d1d9e0] rounded-md font-mono text-xs flex items-center justify-between">
            <span className="text-[#1f2328] select-all">{installCmd}</span>
            <button
              onClick={handleCopy}
              className="text-[#0969da] hover:underline cursor-pointer ml-3 font-sans text-xs"
            >
              {copied ? 'Copied' : 'Copy'}
            </button>
          </div>

          {/* How to Use */}
          <div>
            <h2 className="text-xl font-bold border-b border-[#d1d9e0] pb-2 mb-3">
              How to Use the Package
            </h2>
            <p className="mb-2">To fit a model, you provide three core inputs:</p>
            <ol className="list-decimal pl-6 space-y-2 mb-4">
              <li>
                <strong>Data (<code>data</code>)</strong>: A list containing the dataset for each subject (<code>[subj_1_data, subj_2_data, ...]</code>).
              </li>
              <li>
                <strong>Model (<code>model</code>)</strong>: A Python function that evaluates trial choices and outcomes for a single subject.
              </li>
              <li>
                <strong>Parameters and Priors (<code>hyper_params</code>)</strong>: A dictionary defining the parameters to estimate, their initial population mean and standard deviation in latent space, and their transformation functions into valid bounds.
              </li>
            </ol>
          </div>

          {/* Example Code */}
          <div>
            <h2 className="text-xl font-bold border-b border-[#d1d9e0] pb-2 mb-3">
              Example: Full-Information Two-Armed Bandit (Q-Learning)
            </h2>
            <p className="mb-3 text-xs text-[#59636e]">
              A two-armed bandit with full outcome information (reward1 and reward2 observed). Initial action values start at 0.5. On each trial, arm 1 choice probability is computed via a sigmoid of the difference scaled by beta, followed by value updates to both arms.
            </p>
            <pre className="bg-[#f6f8fa] border border-[#d1d9e0] rounded-md p-4 font-mono text-xs overflow-x-auto text-[#1f2328] leading-relaxed whitespace-pre">
              {codeSnippet}
            </pre>
          </div>

          {/* Model Contract & Dimensions */}
          <div>
            <h2 className="text-xl font-bold border-b border-[#d1d9e0] pb-2 mb-3">
              Model Function Requirements
            </h2>
            <p className="mb-2">
              The model function takes three arguments:
            </p>
            <div className="p-2 bg-[#f6f8fa] border border-[#d1d9e0] rounded font-mono text-xs text-[#0969da] mb-3">
              def my_model(subj_data, parameters, mode="log_likelihood"):
            </div>
            <ol className="list-decimal pl-6 space-y-2 mb-4">
              <li>
                <strong><code>subj_data</code></strong>: The dataset for a single subject (such as a dictionary, DataFrame, or array) containing choices and trial outcomes.
              </li>
              <li>
                <strong><code>parameters</code></strong>: A dictionary where each parameter is a <strong>1D NumPy array of $N$ samples</strong> (shape <code>(n_samples,)</code>). The parameters are automatically transformed into their valid domain bounds before being passed to your model.
              </li>
              <li>
                <strong><code>mode</code></strong>: A string indicating what the function should calculate:
                <ul className="list-disc pl-6 mt-1.5 space-y-1">
                  <li>
                    <strong><code>"log_likelihood"</code></strong>: Returns a 1D NumPy array of total log-likelihoods for each parameter sample (length $N$).
                  </li>
                  <li>
                    <strong><code>"simulate"</code></strong>: Returns the choice probabilities for each trial. When calling <code>sampler.simulate()</code>, these are averaged across the parameter samples to yield the mean choice probabilities for each trial.
                  </li>
                </ul>
              </li>
            </ol>
          </div>

          {/* Outputs: Fitting Results */}
          <div>
            <h2 className="text-xl font-bold border-b border-[#d1d9e0] pb-2 mb-3">
              Outputs: Fitting Results
            </h2>
            <p className="mb-3">
              All fitting results and posterior estimates are stored directly on the <code>Sampler</code> instance:
            </p>

            <div className="space-y-4">
              <div>
                <h3 className="font-semibold text-sm text-[#1f2328] mb-1">
                  1. General &amp; Subject Evidence
                </h3>
                <ul className="list-disc pl-6 space-y-1 text-xs">
                  <li>
                    <code>sampler.evidence</code>: Total model evidence (log marginal likelihood summed across all subjects) at each iteration.
                  </li>
                  <li>
                    <code>sampler.subj_evidence</code>: 2D list of shape <code>(n_iterations, n_subjects)</code> containing the log marginal likelihood for each subject across iterations.
                  </li>
                </ul>
              </div>

              <div>
                <h3 className="font-semibold text-sm text-[#1f2328] mb-1">
                  2. Population Hyperparameters
                </h3>
                <ul className="list-disc pl-6 space-y-1 text-xs">
                  <li>
                    <code>sampler.hyper_params</code>: Dictionary of final fitted population mean and standard deviation for each parameter (<code>{"{param: {'mean': float, 'sd': float}}"}</code>).
                  </li>
                  <li>
                    <code>sampler.hyper_params_list</code>: History of population parameters at each iteration from 0 to N.
                  </li>
                  <li>
                    <code>sampler.cor_matrix</code>: Final correlation matrix between parameters in latent normal space.
                  </li>
                </ul>
              </div>

              <div>
                <h3 className="font-semibold text-sm text-[#1f2328] mb-1">
                  3. Subject-Level Estimates
                </h3>
                <ul className="list-disc pl-6 space-y-1 text-xs">
                  <li>
                    <code>sampler.mean_params</code>: Dictionary mapping each parameter to a list of weighted posterior means for each subject (e.g. <code>sampler.mean_params["lr"][0]</code>).
                  </li>
                  <li>
                    <code>sampler.samples</code>: Dictionary mapping each parameter to full arrays of resampled posterior candidate draws for each subject.
                  </li>
                </ul>
              </div>

              <div>
                <h3 className="font-semibold text-sm text-[#1f2328] mb-1">
                  4. Metrics &amp; Summary
                </h3>
                <ul className="list-disc pl-6 space-y-1 text-xs">
                  <li>
                    <code>sampler.BIC</code>: Bayesian Information Criterion across iterations (uses <code>n_choices</code> to scale degrees of freedom; lower is better).
                  </li>
                  <li>
                    <code>sampler.iterations</code> &amp; <code>sampler.total_fit_time</code>: Total completed iterations and execution runtime in seconds.
                  </li>
                  <li>
                    <code>sampler.summary()</code>: Prints a formatted summary table of the fit.
                  </li>
                </ul>
              </div>
            </div>
          </div>

          {/* Additional Utilities */}
          <div>
            <h2 className="text-xl font-bold border-b border-[#d1d9e0] pb-2 mb-3">
              Additional Utilities
            </h2>
            <ul className="list-disc pl-6 space-y-2 mb-3">
              <li>
                <strong>Saving and Loading Models</strong>: Persist fitted models to disk with <code>sampler.save_model()</code> and reload with <code>load_model()</code> (imported as a standalone function).
              </li>
              <li>
                <strong>Simulating Choices</strong>: <code>sampler.simulate()</code> returns the original dataset as it was (the list of subject datasets), with an added <code>mean_choice_probability</code> array for each subject (averaged across the parameter samples).
              </li>
            </ul>

            <pre className="bg-[#f6f8fa] border border-[#d1d9e0] rounded-md p-4 font-mono text-xs overflow-x-auto text-[#1f2328] leading-relaxed whitespace-pre">
              {utilitiesSnippet}
            </pre>
          </div>

          {/* Demo File: demo.py */}
          <div>
            <h2 className="text-xl font-bold border-b border-[#d1d9e0] pb-2 mb-3">
              Demo File (<code>demo.py</code>)
            </h2>
            <p className="mb-2">
              For a runnable, self-contained demonstration with simulated data, check out <strong><code>demo.py</code></strong>.
            </p>
            <p className="text-xs text-[#59636e]">
              While the example above illustrates the workflow, <code>demo.py</code> includes a helper function that generates realistic multi-subject data (10 subjects with 30 trials each, containing <code>trial</code>, <code>choice</code>, <code>reward1</code>, and <code>reward2</code>), fits the model with the <code>Sampler</code>, inspects recovered evidence and subject parameters, simulates choice probabilities, and demonstrates saving and loading. You can open and inspect <code>demo.py</code> to try it out and adapt it for your own experiments.
            </p>
          </div>
        </article>
      </main>
    </div>
  );
}
