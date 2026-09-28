import React, { useState } from 'react';
import { Plus, Trash2, Copy, Check, Code, Sparkles, SlidersHorizontal } from 'lucide-react';

interface ModelParam {
  name: string;
  priorMean: number;
  priorSd: number;
  transform: 'sigmoid' | 'softplus' | 'exp' | 'identity' | 'logit';
  description: string;
}

export const ModelBuilder: React.FC = () => {
  const [modelName, setModelName] = useState('my_custom_model');
  const [modelType, setModelType] = useState<'bandit' | 'choice' | 'custom'>('bandit');
  const [copied, setCopied] = useState(false);

  const [params, setParams] = useState<ModelParam[]>([
    {
      name: 'lr',
      priorMean: 0.0,
      priorSd: 1.0,
      transform: 'sigmoid',
      description: 'Learning rate in [0, 1]',
    },
    {
      name: 'inv_temp',
      priorMean: 1.0,
      priorSd: 1.0,
      transform: 'softplus',
      description: 'Choice sensitivity beta > 0',
    },
    {
      name: 'perseveration',
      priorMean: 0.0,
      priorSd: 1.0,
      transform: 'identity',
      description: 'Choice stickiness / perseveration weight',
    },
  ]);

  const addParam = () => {
    setParams([
      ...params,
      {
        name: `param_${params.length + 1}`,
        priorMean: 0.0,
        priorSd: 1.0,
        transform: 'identity',
        description: 'Custom parameter',
      },
    ]);
  };

  const removeParam = (index: number) => {
    if (params.length > 1) {
      setParams(params.filter((_, i) => i !== index));
    }
  };

  const updateParam = (index: number, field: keyof ModelParam, value: any) => {
    const updated = [...params];
    updated[index] = { ...updated[index], [field]: value };
    setParams(updated);
  };

  // Generate Python Code
  const generatePythonCode = (): string => {
    const transformMap: Record<string, string> = {
      sigmoid: 'expit',
      softplus: 'lambda x: np.log1p(np.exp(x))',
      exp: 'np.exp',
      identity: 'lambda x: x',
      logit: 'logit',
    };

    const hyperPriorsDict = params
      .map(
        p => `    "${p.name}": {"mean": ${p.priorMean.toFixed(1)}, "sd": ${p.priorSd.toFixed(1)}},`
      )
      .join('\n');

    const transformationsDict = params
      .map(p => `    "${p.name}": ${transformMap[p.transform]},`)
      .join('\n');

    const paramUnpack = params
      .map(p => `    ${p.name} = np.asarray(transformations["${p.name}"](parameters["${p.name}"]), dtype=np.float64)`)
      .join('\n');

    return `import numpy as np
import pandas as pd
from scipy.special import expit
from importance_sampling import Sampler
from importance_sampling.utils import logit

# -----------------------------------------------------------------------------
# 1. Model Likelihood & Simulation Function
# -----------------------------------------------------------------------------
def ${modelName}(subj_data, parameters, transformations=None, mode="loglikelihood"):
    """${modelName}: Custom computational cognitive model.
    
    Parameters:
${params.map(p => `    - ${p.name}: ${p.description}`).join('\n')}
    """
    if transformations is None:
        transformations = {
${transformationsDict}
        }
    
    # 1. Transform candidate parameter samples (shape: N_SAMPLES,)
${paramUnpack}
    n_samples = len(${params[0].name})
    
    # 2. Extract trial records from subject data
    choices = np.asarray(subj_data["choice"], dtype=int)
    rewards = np.asarray(subj_data["reward"], dtype=float)
    n_trials = len(choices)
    
    if mode == "loglikelihood":
        # Vectorized evaluation over all parameter particles
        total_ll = np.zeros(n_samples, dtype=np.float64)
        q = np.full((n_samples, 2), 0.5, dtype=np.float64)
        
        for t in range(n_trials):
            c = choices[t]
            r = rewards[t]
            
            # Softmax choice probability
            diff = q[:, 1] - q[:, 0]
            p1 = expit(inv_temp * diff)
            p1 = np.clip(p1, 1e-12, 1.0 - 1e-12)
            
            # Accumulate trial log-likelihood
            total_ll += np.log(p1) if c == 1 else np.log(1.0 - p1)
            
            # Value update on chosen option
            pe = r - q[:, c]
            q[:, c] += lr * pe
            
        return total_ll

    elif mode == "simulate":
        # Simulate behavioral trials using the posterior mean
        sim_q = np.array([0.5, 0.5])
        sim_choices = []
        for t in range(n_trials):
            p1 = float(expit(float(inv_temp[0]) * (sim_q[1] - sim_q[0])))
            c = 1 if np.random.rand() < p1 else 0
            sim_choices.append(c)
            sim_q[c] += float(lr[0]) * (rewards[t] - sim_q[c])
            
        return pd.DataFrame({"trial": range(n_trials), "sim_choice": sim_choices})
    else:
        raise ValueError(f"Unknown mode '{mode}'")


# -----------------------------------------------------------------------------
# 2. Hyper-Priors & Sampler Configuration
# -----------------------------------------------------------------------------
hyper_priors = {
${hyperPriorsDict}
}

transformations = {
${transformationsDict}
}

# -----------------------------------------------------------------------------
# 3. Fitting with Sampler
# -----------------------------------------------------------------------------
# sampler = Sampler(
#     data=my_data,
#     model=${modelName},
#     hyper_params=hyper_priors,
#     transformations=transformations,
#     n_choices=2,
#     model_name="${modelName}",
# )
# sampler.iterative_model_fit(n_iterations=12, n_samples=1000)
`;
  };

  const generatedCode = generatePythonCode();

  const handleCopy = () => {
    navigator.clipboard.writeText(generatedCode);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
      {/* Left Form: Parameter Configuration */}
      <div className="lg:col-span-5 bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-xl flex flex-col gap-4">
        <div>
          <h2 className="text-base font-bold text-white flex items-center gap-2">
            <SlidersHorizontal className="w-5 h-5 text-indigo-400" />
            Model Specification
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Configure free parameters, link transforms, and initial hyper-priors.
          </p>
        </div>

        {/* Model Name */}
        <div className="flex flex-col gap-1.5">
          <label className="text-xs font-semibold text-slate-300">Model Name</label>
          <input
            type="text"
            value={modelName}
            onChange={e => setModelName(e.target.value.replace(/[^a-zA-Z0-9_]/g, ''))}
            className="px-3 py-2 bg-slate-950 border border-slate-800 rounded-lg text-xs font-mono text-white focus:outline-none focus:border-indigo-500"
          />
        </div>

        {/* Parameters List */}
        <div className="flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <label className="text-xs font-semibold text-slate-300">
              Free Parameters ({params.length})
            </label>
            <button
              onClick={addParam}
              className="flex items-center gap-1 px-2.5 py-1 bg-indigo-600 hover:bg-indigo-500 text-white rounded text-xs font-medium transition-colors"
            >
              <Plus className="w-3.5 h-3.5" />
              Add Parameter
            </button>
          </div>

          <div className="flex flex-col gap-2.5 max-h-[460px] overflow-y-auto pr-1">
            {params.map((p, idx) => (
              <div
                key={idx}
                className="p-3 bg-slate-950 border border-slate-800/80 rounded-lg flex flex-col gap-2 relative group hover:border-slate-700 transition-colors"
              >
                <div className="flex items-center gap-2">
                  <input
                    type="text"
                    value={p.name}
                    onChange={e => updateParam(idx, 'name', e.target.value.replace(/[^a-zA-Z0-9_]/g, ''))}
                    className="flex-1 px-2.5 py-1 bg-slate-900 border border-slate-800 rounded text-xs font-mono text-white font-semibold"
                    placeholder="param_name"
                  />

                  <select
                    value={p.transform}
                    onChange={e => updateParam(idx, 'transform', e.target.value)}
                    className="px-2 py-1 bg-slate-900 border border-slate-800 rounded text-xs text-indigo-300 font-medium"
                  >
                    <option value="sigmoid">Sigmoid [0, 1]</option>
                    <option value="softplus">Softplus (0, ∞)</option>
                    <option value="exp">Exponential (0, ∞)</option>
                    <option value="identity">Identity (-∞, ∞)</option>
                    <option value="logit">Logit</option>
                  </select>

                  {params.length > 1 && (
                    <button
                      onClick={() => removeParam(idx)}
                      className="p-1 text-slate-500 hover:text-red-400 rounded transition-colors"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  )}
                </div>

                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="flex items-center gap-1.5">
                    <span className="text-slate-400 text-[11px]">Prior μ:</span>
                    <input
                      type="number"
                      step="0.1"
                      value={p.priorMean}
                      onChange={e => updateParam(idx, 'priorMean', parseFloat(e.target.value) || 0)}
                      className="w-full px-2 py-0.5 bg-slate-900 border border-slate-800 rounded text-slate-200 font-mono text-[11px]"
                    />
                  </div>
                  <div className="flex items-center gap-1.5">
                    <span className="text-slate-400 text-[11px]">Prior σ:</span>
                    <input
                      type="number"
                      step="0.1"
                      value={p.priorSd}
                      onChange={e => updateParam(idx, 'priorSd', parseFloat(e.target.value) || 1)}
                      className="w-full px-2 py-0.5 bg-slate-900 border border-slate-800 rounded text-slate-200 font-mono text-[11px]"
                    />
                  </div>
                </div>

                <input
                  type="text"
                  value={p.description}
                  onChange={e => updateParam(idx, 'description', e.target.value)}
                  className="px-2 py-0.5 bg-slate-900 border border-slate-800 rounded text-[11px] text-slate-400"
                  placeholder="Parameter description..."
                />
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Right Column: Code Generator Output */}
      <div className="lg:col-span-7 bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-xl flex flex-col h-[640px]">
        <div className="px-5 py-3 border-b border-slate-800 bg-slate-950 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Code className="w-4 h-4 text-emerald-400" />
            <h3 className="text-xs font-bold text-white font-mono">{modelName}.py</h3>
          </div>

          <button
            onClick={handleCopy}
            className="flex items-center gap-1.5 px-3 py-1 bg-indigo-600 hover:bg-indigo-500 active:bg-indigo-700 text-white text-xs font-semibold rounded-md shadow transition-colors"
          >
            {copied ? (
              <>
                <Check className="w-3.5 h-3.5" />
                <span>Copied Python Code!</span>
              </>
            ) : (
              <>
                <Copy className="w-3.5 h-3.5" />
                <span>Copy Python Code</span>
              </>
            )}
          </button>
        </div>

        <div className="flex-1 overflow-auto p-4 bg-slate-950 font-mono text-xs text-slate-200">
          <pre className="leading-relaxed">
            <code>{generatedCode}</code>
          </pre>
        </div>
      </div>
    </div>
  );
};
