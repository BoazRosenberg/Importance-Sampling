import React from 'react';
import {
  BookOpen,
  CheckCircle2,
  GitBranch,
  Terminal,
  Zap,
  ArrowRight,
  ShieldCheck,
  Cpu,
  Layers,
} from 'lucide-react';

export const Documentation: React.FC = () => {
  return (
    <div className="flex flex-col gap-8 max-w-5xl mx-auto">
      {/* Hero Overview */}
      <div className="bg-gradient-to-br from-indigo-950/60 via-slate-900 to-slate-950 border border-indigo-900/40 rounded-2xl p-8 shadow-2xl relative overflow-hidden">
        <div className="relative z-10">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/30 text-indigo-300 text-xs font-semibold mb-4">
            <BookOpen className="w-3.5 h-3.5" />
            Iterative Importance Sampling (IIS) Guide
          </div>
          <h1 className="text-2xl lg:text-3xl font-extrabold text-white tracking-tight">
            Hierarchical Bayesian Estimation Without MCMC Bottlenecks
          </h1>
          <p className="text-sm text-slate-300 mt-2 max-w-2xl leading-relaxed">
            In computational cognitive modeling, researchers frequently encounter complex,
            simulation-based processes (reinforcement learning, drift-diffusion, heuristics) where
            gradients are unavailable or Stan / MCMC chains suffer from divergent transitions.
            Iterative Importance Sampling offers an elegant, vectorized, and fast alternative.
          </p>
        </div>
      </div>

      {/* The 4-Step IIS Loop */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl">
        <h2 className="text-lg font-bold text-white mb-6 flex items-center gap-2">
          <Layers className="w-5 h-5 text-indigo-400" />
          The Iterative Importance Sampling Algorithm
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="p-4 bg-slate-950 rounded-xl border border-slate-800 flex flex-col gap-2">
            <div className="w-7 h-7 rounded-lg bg-indigo-600/20 border border-indigo-500/30 flex items-center justify-center text-xs font-bold text-indigo-300">
              1
            </div>
            <h3 className="text-xs font-bold text-white uppercase tracking-wider">Candidate Proposal</h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              Sample $M$ particles from the current population normal proposal:
              <br />
              <code className="text-indigo-400 font-mono text-[11px]">
                θ ~ N(μ_pop, Σ_pop)
              </code>
            </p>
          </div>

          <div className="p-4 bg-slate-950 rounded-xl border border-slate-800 flex flex-col gap-2">
            <div className="w-7 h-7 rounded-lg bg-sky-600/20 border border-sky-500/30 flex items-center justify-center text-xs font-bold text-sky-300">
              2
            </div>
            <h3 className="text-xs font-bold text-white uppercase tracking-wider">Likelihood Evaluation</h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              Transform particles via link functions and evaluate subject log-likelihood:
              <br />
              <code className="text-sky-400 font-mono text-[11px]">
                ln p(D_s | θ_i)
              </code>
            </p>
          </div>

          <div className="p-4 bg-slate-950 rounded-xl border border-slate-800 flex flex-col gap-2">
            <div className="w-7 h-7 rounded-lg bg-emerald-600/20 border border-emerald-500/30 flex items-center justify-center text-xs font-bold text-emerald-300">
              3
            </div>
            <h3 className="text-xs font-bold text-white uppercase tracking-wider">Importance Weighting</h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              Compute normalized weights using numerically stable logsumexp:
              <br />
              <code className="text-emerald-400 font-mono text-[11px]">
                w_i = exp(LL_i - LSE)
              </code>
            </p>
          </div>

          <div className="p-4 bg-slate-950 rounded-xl border border-slate-800 flex flex-col gap-2">
            <div className="w-7 h-7 rounded-lg bg-purple-600/20 border border-purple-500/30 flex items-center justify-center text-xs font-bold text-purple-300">
              4
            </div>
            <h3 className="text-xs font-bold text-white uppercase tracking-wider">Population Update</h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              Resample particles, pool across all participants, and compute new:
              <br />
              <code className="text-purple-400 font-mono text-[11px]">
                μ^(t+1) and σ^(t+1)
              </code>
            </p>
          </div>
        </div>
      </div>

      {/* Codebase Touch-ups & Modernization */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl">
        <h2 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
          <ShieldCheck className="w-5 h-5 text-emerald-400" />
          Code Modernization & Fixes Implemented
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
          <div className="p-3.5 bg-slate-950 rounded-lg border border-slate-800">
            <div className="font-semibold text-emerald-400 flex items-center gap-2 mb-1">
              <CheckCircle2 className="w-4 h-4" />
              Replaced Deprecated Pandas .append()
            </div>
            <p className="text-slate-400">
              In Pandas 2.0+, <code className="text-slate-300">DataFrame.append</code> was permanently removed.
              All data accumulation has been replaced with modern, efficient <code className="text-slate-300">pd.concat()</code>.
            </p>
          </div>

          <div className="p-3.5 bg-slate-950 rounded-lg border border-slate-800">
            <div className="font-semibold text-emerald-400 flex items-center gap-2 mb-1">
              <CheckCircle2 className="w-4 h-4" />
              Numerically Stable Logsumexp & NaN Masking
            </div>
            <p className="text-slate-400">
              Replaced fragile manual reductions with NaN-protected <code className="text-slate-300">scipy.special.logsumexp</code>,
              preventing underflow/overflow when candidate particles produce extreme values.
            </p>
          </div>

          <div className="p-3.5 bg-slate-950 rounded-lg border border-slate-800">
            <div className="font-semibold text-emerald-400 flex items-center gap-2 mb-1">
              <CheckCircle2 className="w-4 h-4" />
              Fixed Indexing Inconsistencies
            </div>
            <p className="text-slate-400">
              Fixed off-by-one errors like <code className="text-slate-300">s - 1</code> in subject simulation routines,
              ensuring 0-indexed compliance throughout all loops.
            </p>
          </div>

          <div className="p-3.5 bg-slate-950 rounded-lg border border-slate-800">
            <div className="font-semibold text-emerald-400 flex items-center gap-2 mb-1">
              <CheckCircle2 className="w-4 h-4" />
              PEP 517/621 pyproject.toml Packaging
            </div>
            <p className="text-slate-400">
              Configured a modern <code className="text-slate-300">pyproject.toml</code> with full metadata, optional dev dependencies,
              type hints, and test runners, ready for <code className="text-slate-300">pip install -e .</code>.
            </p>
          </div>
        </div>
      </div>

      {/* GitHub & PyPI Release Checklist */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl">
        <h2 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
          <GitBranch className="w-5 h-5 text-indigo-400" />
          How to Publish to GitHub & PyPI
        </h2>

        <div className="flex flex-col gap-3 font-mono text-xs">
          <div className="p-3 bg-slate-950 rounded-lg border border-slate-800 flex items-start gap-3">
            <span className="w-6 h-6 rounded bg-indigo-900/60 text-indigo-300 flex items-center justify-center font-bold shrink-0">
              1
            </span>
            <div className="flex-1">
              <div className="text-white font-sans font-semibold mb-1">Push to GitHub:</div>
              <div className="bg-slate-900 p-2 rounded text-slate-300 select-all overflow-x-auto">
                git init<br />
                git add .<br />
                git commit -m "feat: modern importance_sampling v0.2.0 package"<br />
                git remote add origin https://github.com/BoazRsnbrg/importance_sampling.git<br />
                git push -u origin main
              </div>
            </div>
          </div>

          <div className="p-3 bg-slate-950 rounded-lg border border-slate-800 flex items-start gap-3">
            <span className="w-6 h-6 rounded bg-indigo-900/60 text-indigo-300 flex items-center justify-center font-bold shrink-0">
              2
            </span>
            <div className="flex-1">
              <div className="text-white font-sans font-semibold mb-1">Build Distribution Wheel & Source Archive:</div>
              <div className="bg-slate-900 p-2 rounded text-slate-300 select-all overflow-x-auto">
                pip install build twine<br />
                python -m build
              </div>
            </div>
          </div>

          <div className="p-3 bg-slate-950 rounded-lg border border-slate-800 flex items-start gap-3">
            <span className="w-6 h-6 rounded bg-indigo-900/60 text-indigo-300 flex items-center justify-center font-bold shrink-0">
              3
            </span>
            <div className="flex-1">
              <div className="text-white font-sans font-semibold mb-1">Upload to PyPI (Python Package Index):</div>
              <div className="bg-slate-900 p-2 rounded text-slate-300 select-all overflow-x-auto">
                twine upload dist/*
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
