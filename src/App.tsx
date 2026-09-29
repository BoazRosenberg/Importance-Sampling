import React, { useState } from 'react';
import {
  Copy,
  Check,
  BarChart3,
  BookOpen,
  Activity,
  TrendingUp,
  Layers,
  Grid,
} from 'lucide-react';

// Simulated multi-parameter dataset (e.g. 6-8 parameters common in computational models)
const PARAM_KEYS = ['alpha', 'beta', 'decay', 'bias', 'noise', 'pers'] as const;
type ParamKey = typeof PARAM_KEYS[number];

interface ParamTrajectory {
  iters: number[];
  means: number[];
  sdUpper: number[];
  sdLower: number[];
}

const mockTrajectories: Record<ParamKey, ParamTrajectory> = {
  alpha: {
    iters: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    means: [0.500, 0.462, 0.435, 0.418, 0.410, 0.405, 0.402, 0.400, 0.398, 0.397, 0.397, 0.396],
    sdUpper: [0.731, 0.684, 0.632, 0.589, 0.554, 0.528, 0.509, 0.495, 0.485, 0.478, 0.474, 0.471],
    sdLower: [0.269, 0.261, 0.258, 0.264, 0.276, 0.291, 0.304, 0.312, 0.318, 0.322, 0.325, 0.327],
  },
  beta: {
    iters: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    means: [1.31, 1.84, 2.21, 2.48, 2.67, 2.81, 2.91, 2.97, 3.01, 3.03, 3.04, 3.05],
    sdUpper: [2.12, 2.68, 3.06, 3.31, 3.48, 3.61, 3.70, 3.76, 3.80, 3.82, 3.84, 3.85],
    sdLower: [0.69, 1.15, 1.49, 1.76, 1.96, 2.11, 2.21, 2.28, 2.32, 2.35, 2.36, 2.37],
  },
  decay: {
    iters: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    means: [0.10, 0.14, 0.18, 0.21, 0.23, 0.24, 0.25, 0.25, 0.26, 0.26, 0.26, 0.26],
    sdUpper: [0.25, 0.28, 0.30, 0.32, 0.33, 0.33, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34],
    sdLower: [0.02, 0.04, 0.07, 0.10, 0.13, 0.15, 0.16, 0.16, 0.18, 0.18, 0.18, 0.18],
  },
  bias: {
    iters: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    means: [0.00, 0.05, 0.09, 0.12, 0.14, 0.15, 0.15, 0.16, 0.16, 0.16, 0.16, 0.16],
    sdUpper: [0.35, 0.36, 0.37, 0.36, 0.35, 0.34, 0.33, 0.32, 0.31, 0.30, 0.30, 0.30],
    sdLower: [-0.35, -0.26, -0.19, -0.12, -0.07, -0.04, -0.03, -0.00, 0.01, 0.02, 0.02, 0.02],
  },
  noise: {
    iters: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    means: [0.05, 0.08, 0.11, 0.13, 0.14, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15],
    sdUpper: [0.15, 0.18, 0.20, 0.21, 0.21, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22],
    sdLower: [0.01, 0.02, 0.03, 0.05, 0.07, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08],
  },
  pers: {
    iters: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    means: [0.00, 0.10, 0.18, 0.24, 0.28, 0.31, 0.32, 0.33, 0.34, 0.34, 0.34, 0.34],
    sdUpper: [0.50, 0.52, 0.54, 0.55, 0.55, 0.54, 0.53, 0.52, 0.51, 0.51, 0.50, 0.50],
    sdLower: [-0.50, -0.32, -0.18, -0.07, 0.01, 0.08, 0.11, 0.14, 0.17, 0.17, 0.18, 0.18],
  },
};

// 10 subjects posterior means across all parameters
const mockSubjMeans: Record<ParamKey, number[]> = {
  alpha: [0.38, 0.42, 0.35, 0.45, 0.39, 0.33, 0.44, 0.37, 0.41, 0.40],
  beta:  [3.21, 2.85, 3.42, 2.65, 3.10, 3.65, 2.78, 3.30, 2.92, 3.05],
  decay: [0.24, 0.28, 0.22, 0.29, 0.25, 0.21, 0.27, 0.23, 0.26, 0.25],
  bias:  [0.18, 0.12, 0.22, 0.09, 0.15, 0.25, 0.11, 0.19, 0.14, 0.16],
  noise: [0.14, 0.17, 0.12, 0.19, 0.15, 0.11, 0.18, 0.13, 0.16, 0.15],
  pers:  [0.36, 0.31, 0.42, 0.25, 0.33, 0.45, 0.29, 0.38, 0.32, 0.34],
};

const mockEvidence = [-212.4, -188.2, -172.5, -161.8, -154.2, -149.1, -145.7, -143.5, -142.4, -141.8, -141.5, -141.3];
const mockBIC = [434.1, 385.7, 354.3, 332.9, 317.7, 307.5, 300.7, 296.3, 294.1, 292.9, 292.3, 291.9];

const mockSubjSpaghetti = [
  [-21.2, -18.7, -17.1, -16.0, -15.3, -14.8, -14.4, -14.2, -14.1, -14.0, -13.9, -13.9],
  [-21.5, -19.0, -17.4, -16.3, -15.6, -15.1, -14.8, -14.6, -14.5, -14.4, -14.4, -14.4],
  [-20.8, -18.1, -16.6, -15.5, -14.8, -14.2, -13.8, -13.5, -13.4, -13.3, -13.2, -13.2],
  [-22.1, -19.6, -18.0, -16.9, -16.2, -15.8, -15.5, -15.3, -15.2, -15.1, -15.1, -15.1],
  [-21.0, -18.6, -17.0, -15.9, -15.2, -14.7, -14.3, -14.1, -14.0, -13.9, -13.8, -13.8],
  [-20.5, -17.8, -16.2, -15.1, -14.3, -13.8, -13.4, -13.2, -13.0, -12.9, -12.9, -12.9],
  [-21.8, -19.2, -17.6, -16.6, -15.9, -15.4, -15.1, -14.9, -14.8, -14.7, -14.7, -14.7],
  [-20.9, -18.4, -16.8, -15.8, -15.0, -14.5, -14.1, -13.9, -13.7, -13.6, -13.6, -13.6],
  [-21.4, -18.9, -17.3, -16.2, -15.5, -15.0, -14.6, -14.4, -14.3, -14.2, -14.2, -14.2],
  [-21.2, -18.5, -16.9, -15.8, -15.1, -14.5, -14.1, -13.8, -13.7, -13.6, -13.5, -13.5],
];

// Correlation matrix (multinormal)
const mockCorMatrix: Record<ParamKey, Record<ParamKey, number>> = {
  alpha: { alpha: 1.0,  beta: -0.42, decay: 0.28, bias: -0.15, noise: 0.31, pers: -0.22 },
  beta:  { alpha: -0.42, beta: 1.0,  decay: -0.19, bias: 0.34, noise: -0.25, pers: 0.41 },
  decay: { alpha: 0.28,  beta: -0.19, decay: 1.0,  bias: -0.08, noise: 0.12, pers: -0.14 },
  bias:  { alpha: -0.15, beta: 0.34, decay: -0.08, bias: 1.0,  noise: -0.18, pers: 0.29 },
  noise: { alpha: 0.31,  beta: -0.25, decay: 0.12, bias: -0.18, noise: 1.0,  pers: -0.09 },
  pers:  { alpha: -0.22, beta: 0.41, decay: -0.14, bias: 0.29, noise: -0.09, pers: 1.0 },
};

export default function App() {
  const [activeTab, setActiveTab] = useState<'report' | 'docs'>('report');
  const [copied, setCopied] = useState(false);

  // Selected parameters from the table (supports 1 or 2 selected parameters)
  const [selectedParams, setSelectedParams] = useState<ParamKey[]>(['alpha', 'beta']);
  const [hoveredIter, setHoveredIter] = useState<number | null>(null);
  const [hoveredSubj, setHoveredSubj] = useState<number | null>(null);

  const installCmd = 'pip install git+https://github.com/BoazRsnbrg/importance_sampling.git';

  const handleCopy = () => {
    navigator.clipboard.writeText(installCmd);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Toggle selection of a parameter in the table
  const handleSelectParam = (param: ParamKey) => {
    if (selectedParams.includes(param)) {
      if (selectedParams.length > 1) {
        setSelectedParams(selectedParams.filter((p) => p !== param));
      }
    } else {
      if (selectedParams.length < 2) {
        setSelectedParams([...selectedParams, param]);
      } else {
        // Replace second parameter
        setSelectedParams([selectedParams[0], param]);
      }
    }
  };

  // Helper chart coordinates
  const chartW = 580;
  const chartH = 220;
  const pad = { top: 20, right: 25, bottom: 30, left: 50 };

  const primaryParam = selectedParams[0];
  const secondaryParam = selectedParams[1] || null;

  const currentTraj = mockTrajectories[primaryParam];
  const yMin = Math.min(...currentTraj.sdLower);
  const yMax = Math.max(...currentTraj.sdUpper);
  const yRange = yMax - yMin || 1;

  const getX = (it: number) => pad.left + (it / 11) * (chartW - pad.left - pad.right);
  const getY = (val: number) =>
    chartH - pad.bottom - ((val - yMin) / yRange) * (chartH - pad.top - pad.bottom);

  // Path for ribbon
  const upperPts = currentTraj.iters.map((it, idx) => `${getX(it)},${getY(currentTraj.sdUpper[idx])}`);
  const lowerPts = currentTraj.iters
    .slice()
    .reverse()
    .map((it, idx) => {
      const orig = 11 - idx;
      return `${getX(it)},${getY(currentTraj.sdLower[orig])}`;
    });
  const ribbonD = `M ${upperPts.join(' L ')} L ${lowerPts.join(' L ')} Z`;
  const meanPathD = currentTraj.iters
    .map((it, idx) => `${idx === 0 ? 'M' : 'L'} ${getX(it)},${getY(currentTraj.means[idx])}`)
    .join(' ');

  // Evidence coordinates
  const evMin = Math.min(...mockEvidence);
  const evMax = Math.max(...mockEvidence);
  const evRange = evMax - evMin || 1;
  const getEvY = (val: number) =>
    chartH - pad.bottom - ((val - evMin) / evRange) * (chartH - pad.top - pad.bottom);
  const evidenceD = mockEvidence
    .map((val, idx) => `${idx === 0 ? 'M' : 'L'} ${getX(idx)},${getEvY(val)}`)
    .join(' ');

  // Histogram calculation for 1 parameter
  const rawVals = mockSubjMeans[primaryParam];
  const histMin = Math.min(...rawVals);
  const histMax = Math.max(...rawVals);
  const numBins = 6;
  const binWidth = (histMax - histMin) / numBins || 0.1;
  const bins = Array.from({ length: numBins }, (_, b) => {
    const start = histMin + b * binWidth;
    const end = start + binWidth;
    const count = rawVals.filter((v) => (b === numBins - 1 ? v >= start && v <= end : v >= start && v < end)).length;
    return { start, end, count };
  });
  const maxBinCount = Math.max(...bins.map((b) => b.count), 1);

  // Scatter plot calculation for 2 parameters
  const xParamVals = mockSubjMeans[primaryParam];
  const yParamVals = secondaryParam ? mockSubjMeans[secondaryParam] : [];
  const scatXMin = Math.min(...xParamVals);
  const scatXMax = Math.max(...xParamVals);
  const scatXRange = scatXMax - scatXMin || 1;
  const scatYMin = secondaryParam ? Math.min(...yParamVals) : 0;
  const scatYMax = secondaryParam ? Math.max(...yParamVals) : 1;
  const scatYRange = scatYMax - scatYMin || 1;

  const getScatX = (val: number) => pad.left + ((val - scatXMin) / scatXRange) * (chartW - pad.left - pad.right);
  const getScatY = (val: number) => chartH - pad.bottom - ((val - scatYMin) / scatYRange) * (chartH - pad.top - pad.bottom);

  return (
    <div className="min-h-screen bg-[#f6f8fa] text-[#1f2328] font-sans antialiased">
      {/* Top Header */}
      <header className="border-b border-[#d1d9e0] bg-white sticky top-0 z-30 shadow-2xs">
        <div className="max-w-6xl mx-auto px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="font-semibold text-lg text-[#0969da] hover:underline cursor-pointer">
              BoazRsnbrg
            </span>
            <span className="text-[#59636e]">/</span>
            <span className="font-bold text-lg text-[#1f2328]">importance_sampling</span>
            <span className="px-2 py-0.5 text-xs font-medium text-[#59636e] bg-[#f6f8fa] border border-[#d1d9e0] rounded-full">
              v0.2.0
            </span>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex bg-[#f6f8fa] p-1 border border-[#d1d9e0] rounded-lg text-xs font-medium">
              <button
                onClick={() => setActiveTab('report')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md transition-all cursor-pointer ${
                  activeTab === 'report'
                    ? 'bg-white text-[#0969da] shadow-2xs font-semibold'
                    : 'text-[#59636e] hover:text-[#1f2328]'
                }`}
              >
                <BarChart3 className="w-3.5 h-3.5" />
                <span>Interactive Report Explorer</span>
              </button>
              <button
                onClick={() => setActiveTab('docs')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md transition-all cursor-pointer ${
                  activeTab === 'docs'
                    ? 'bg-white text-[#0969da] shadow-2xs font-semibold'
                    : 'text-[#59636e] hover:text-[#1f2328]'
                }`}
              >
                <BookOpen className="w-3.5 h-3.5" />
                <span>Documentation &amp; API</span>
              </button>
            </div>

            <button
              onClick={handleCopy}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium bg-[#ffffff] hover:bg-[#f6f8fa] text-[#1f2328] border border-[#d1d9e0] rounded-md shadow-2xs cursor-pointer"
            >
              {copied ? (
                <>
                  <Check className="w-3.5 h-3.5 text-[#1a7f37]" />
                  <span className="text-[#1a7f37]">Copied!</span>
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5 text-[#59636e]" />
                  <span>Copy pip</span>
                </>
              )}
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-8">
        {activeTab === 'report' && (
          <div className="space-y-6">
            {/* Top Dashboard Summary Card */}
            <div className="bg-white border border-[#d1d9e0] rounded-xl p-6 shadow-2xs flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <span className="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-[#0969da]/10 text-[#0969da] border border-[#0969da]/20">
                    Live Report Widget
                  </span>
                  <span className="text-xs text-[#59636e]">
                    Generated from <code>sampler.create_report()</code>
                  </span>
                </div>
                <h1 className="text-2xl font-bold text-[#1f2328]">
                  Model Diagnostics &amp; Estimation Results
                </h1>
                <p className="text-sm text-[#59636e] mt-1">
                  10 Subjects • 6 Fitted Parameters • Converged in 12 Iterations (1,000 particles/subject)
                </p>
              </div>

              <div className="flex items-center gap-3 font-mono">
                <div className="bg-[#f6f8fa] border border-[#d1d9e0] px-4 py-2 rounded-lg text-center">
                  <div className="text-[11px] uppercase tracking-wider font-semibold text-[#59636e]">Evidence</div>
                  <div className="text-lg font-bold text-[#0969da]">-141.30</div>
                </div>
                <div className="bg-[#f6f8fa] border border-[#d1d9e0] px-4 py-2 rounded-lg text-center">
                  <div className="text-[11px] uppercase tracking-wider font-semibold text-[#59636e]">BIC</div>
                  <div className="text-lg font-bold text-[#1a7f37]">291.90</div>
                </div>
              </div>
            </div>

            {/* PARAMETER TABLE SELECTOR (Works seamlessly for lots of parameters) */}
            <div className="bg-white border border-[#d1d9e0] rounded-xl p-5 shadow-2xs space-y-3">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-sm font-bold text-[#1f2328] uppercase tracking-wider flex items-center gap-2">
                    <Grid className="w-4 h-4 text-[#0969da]" />
                    Model Parameters Table
                  </h2>
                  <p className="text-xs text-[#59636e]">
                    Click to select <strong>1 parameter</strong> (to inspect evolution and view distribution histogram) or <strong>2 parameters</strong> (to inspect 2D subject scatter plot).
                  </p>
                </div>
                <div className="text-xs font-mono text-[#0969da] bg-[#0969da]/5 px-2.5 py-1 rounded border border-[#0969da]/20">
                  Active: <strong>{selectedParams.join(', ')}</strong>
                </div>
              </div>

              <div className="overflow-x-auto border border-[#eaeef2] rounded-lg">
                <table className="w-full text-left text-xs">
                  <thead className="bg-[#f6f8fa] text-[#59636e] border-b border-[#eaeef2] font-mono">
                    <tr>
                      <th className="py-2.5 px-4 font-semibold">Select</th>
                      <th className="py-2.5 px-4 font-semibold">Parameter Key</th>
                      <th className="py-2.5 px-4 font-semibold">Fitted Mean</th>
                      <th className="py-2.5 px-4 font-semibold">Fitted SD</th>
                      <th className="py-2.5 px-4 font-semibold">±1 SD Range</th>
                      <th className="py-2.5 px-4 font-semibold">Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#eaeef2]">
                    {PARAM_KEYS.map((key) => {
                      const isSelected = selectedParams.includes(key);
                      const isPrimary = selectedParams[0] === key;
                      const isSecondary = selectedParams[1] === key;
                      const traj = mockTrajectories[key];
                      const finalMean = traj.means[11];
                      const finalSD = (traj.sdUpper[11] - traj.sdLower[11]) / 2;

                      return (
                        <tr
                          key={key}
                          onClick={() => handleSelectParam(key)}
                          className={`cursor-pointer transition-colors ${
                            isSelected ? 'bg-[#0969da]/8 hover:bg-[#0969da]/12' : 'hover:bg-[#f6f8fa]'
                          }`}
                        >
                          <td className="py-2.5 px-4">
                            <input
                              type="checkbox"
                              checked={isSelected}
                              onChange={() => {}}
                              className="rounded text-[#0969da] cursor-pointer"
                            />
                          </td>
                          <td className="py-2.5 px-4 font-mono font-bold text-[#1f2328]">
                            {key}
                          </td>
                          <td className="py-2.5 px-4 font-mono text-[#0969da]">
                            {finalMean.toFixed(3)}
                          </td>
                          <td className="py-2.5 px-4 font-mono text-[#59636e]">
                            {finalSD.toFixed(3)}
                          </td>
                          <td className="py-2.5 px-4 font-mono text-[#59636e]">
                            [{traj.sdLower[11].toFixed(2)}, {traj.sdUpper[11].toFixed(2)}]
                          </td>
                          <td className="py-2.5 px-4">
                            {isPrimary && (
                              <span className="px-2 py-0.5 rounded text-[11px] font-semibold bg-[#0969da] text-white">
                                Primary (X)
                              </span>
                            )}
                            {isSecondary && (
                              <span className="px-2 py-0.5 rounded text-[11px] font-semibold bg-[#8250df] text-white">
                                Secondary (Y)
                              </span>
                            )}
                            {!isSelected && (
                              <span className="text-[11px] text-[#8c959f]">Click to inspect</span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            {/* SECTION 1: HYPERPARAMETER EVOLUTION */}
            <div className="bg-white border border-[#d1d9e0] rounded-xl p-6 shadow-2xs space-y-4">
              <div className="flex items-center justify-between pb-3 border-b border-[#d1d9e0]">
                <div>
                  <h2 className="text-base font-bold text-[#1f2328] flex items-center gap-2">
                    <TrendingUp className="w-5 h-5 text-[#0969da]" />
                    Hyperparameter Evolution: <code>{primaryParam}</code>
                  </h2>
                  <p className="text-xs text-[#59636e]">
                    Solid line indicates population mean; shaded area represents ±1 SD ribbon across iterations.
                  </p>
                </div>

                <div className="text-xs font-mono text-[#59636e]">
                  Parameter: <strong>{primaryParam}</strong> (final mean: {currentTraj.means[11].toFixed(3)})
                </div>
              </div>

              {/* Chart */}
              <div className="border border-[#eaeef2] rounded-lg p-3 bg-white">
                <svg
                  viewBox={`0 0 ${chartW} ${chartH}`}
                  className="w-full h-56 select-none"
                  onMouseLeave={() => setHoveredIter(null)}
                >
                  {/* Grid Lines */}
                  {[0, 0.25, 0.5, 0.75, 1.0].map((frac, idx) => {
                    const yPos = pad.top + frac * (chartH - pad.top - pad.bottom);
                    const val = yMax - frac * yRange;
                    return (
                      <g key={idx}>
                        <line x1={pad.left} y1={yPos} x2={chartW - pad.right} y2={yPos} stroke="#eaeef2" />
                        <text x={pad.left - 8} y={yPos + 4} textAnchor="end" fontSize="10" fill="#8c959f" fontFamily="monospace">
                          {val.toFixed(2)}
                        </text>
                      </g>
                    );
                  })}

                  {/* X Axis */}
                  {currentTraj.iters.map((it) => (
                    <g key={it}>
                      <line x1={getX(it)} y1={chartH - pad.bottom} x2={getX(it)} y2={chartH - pad.bottom + 4} stroke="#8c959f" />
                      <text x={getX(it)} y={chartH - pad.bottom + 15} textAnchor="middle" fontSize="10" fill="#8c959f" fontFamily="monospace">
                        {it}
                      </text>
                    </g>
                  ))}

                  {/* Shaded +/- 1 SD Ribbon */}
                  <path d={ribbonD} fill="rgba(9, 105, 218, 0.16)" />

                  {/* Mean Line */}
                  <path d={meanPathD} fill="none" stroke="#0969da" strokeWidth="2.5" />

                  {/* Markers */}
                  {currentTraj.iters.map((it, idx) => (
                    <circle
                      key={it}
                      cx={getX(it)}
                      cy={getY(currentTraj.means[idx])}
                      r={hoveredIter === it ? 5 : 3.5}
                      fill="#0969da"
                      stroke="#fff"
                      strokeWidth="1.5"
                      className="cursor-pointer"
                      onMouseEnter={() => setHoveredIter(it)}
                    />
                  ))}
                </svg>

                {hoveredIter !== null && (
                  <div className="mt-2 p-2 bg-[#f6f8fa] border border-[#d1d9e0] rounded text-xs font-mono flex items-center justify-between">
                    <span>Iteration: <strong>{hoveredIter}</strong></span>
                    <span>Mean: <strong className="text-[#0969da]">{currentTraj.means[hoveredIter].toFixed(4)}</strong></span>
                    <span>+1 SD: <strong>{currentTraj.sdUpper[hoveredIter].toFixed(4)}</strong></span>
                    <span>-1 SD: <strong>{currentTraj.sdLower[hoveredIter].toFixed(4)}</strong></span>
                  </div>
                )}
              </div>
            </div>

            {/* SECTION 2: MODEL FIT & SUBJECT SPAGHETTI PLOT */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Total Evidence & BIC */}
              <div className="bg-white border border-[#d1d9e0] rounded-xl p-5 shadow-2xs space-y-3">
                <div className="pb-2 border-b border-[#d1d9e0]">
                  <h3 className="text-sm font-bold text-[#1f2328] flex items-center gap-2">
                    <Activity className="w-4 h-4 text-[#0969da]" />
                    Total Model Evidence &amp; BIC
                  </h3>
                  <p className="text-xs text-[#59636e]">Evidence climbing to maximum; BIC minimizing.</p>
                </div>

                <svg viewBox={`0 0 ${chartW} ${chartH}`} className="w-full h-48 select-none">
                  {[0, 0.5, 1.0].map((frac, idx) => {
                    const yPos = pad.top + frac * (chartH - pad.top - pad.bottom);
                    const val = evMax - frac * evRange;
                    return (
                      <g key={idx}>
                        <line x1={pad.left} y1={yPos} x2={chartW - pad.right} y2={yPos} stroke="#eaeef2" />
                        <text x={pad.left - 8} y={yPos + 4} textAnchor="end" fontSize="10" fill="#8c959f" fontFamily="monospace">
                          {val.toFixed(0)}
                        </text>
                      </g>
                    );
                  })}
                  <path d={evidenceD} fill="none" stroke="#0969da" strokeWidth="2.5" />
                  {mockEvidence.map((ev, idx) => (
                    <circle key={idx} cx={getX(idx)} cy={getEvY(ev)} r="3" fill="#0969da" stroke="#fff" strokeWidth="1" />
                  ))}
                </svg>
                <div className="text-[11px] font-mono text-[#59636e] flex justify-between border-t border-[#eaeef2] pt-1.5">
                  <span>Start: {mockEvidence[0].toFixed(1)}</span>
                  <span className="text-[#0969da] font-bold">Final: {mockEvidence[11].toFixed(1)}</span>
                </div>
              </div>

              {/* Subject Evidence Spaghetti Curves */}
              <div className="bg-white border border-[#d1d9e0] rounded-xl p-5 shadow-2xs space-y-3">
                <div className="pb-2 border-b border-[#d1d9e0]">
                  <h3 className="text-sm font-bold text-[#1f2328] flex items-center gap-2">
                    <Layers className="w-4 h-4 text-[#8250df]" />
                    Subject-Level Evidence Spaghetti Plot
                  </h3>
                  <p className="text-xs text-[#59636e]">Per-subject log-likelihood trajectories (Subjects 0–9).</p>
                </div>

                <svg viewBox={`0 0 ${chartW} ${chartH}`} className="w-full h-48 select-none">
                  {[0, 0.5, 1.0].map((frac, idx) => {
                    const yPos = pad.top + frac * (chartH - pad.top - pad.bottom);
                    const val = -13.0 - frac * 9;
                    return (
                      <g key={idx}>
                        <line x1={pad.left} y1={yPos} x2={chartW - pad.right} y2={yPos} stroke="#eaeef2" />
                        <text x={pad.left - 8} y={yPos + 4} textAnchor="end" fontSize="10" fill="#8c959f" fontFamily="monospace">
                          {val.toFixed(1)}
                        </text>
                      </g>
                    );
                  })}
                  {mockSubjSpaghetti.map((traj, s) => {
                    const d = traj
                      .map((val, idx) => {
                        const y = chartH - pad.bottom - ((val - -22.5) / 10) * (chartH - pad.top - pad.bottom);
                        return `${idx === 0 ? 'M' : 'L'} ${getX(idx)},${y}`;
                      })
                      .join(' ');
                    const isHovered = hoveredSubj === s;
                    return (
                      <path
                        key={s}
                        d={d}
                        fill="none"
                        stroke={isHovered ? '#8250df' : 'rgba(89, 99, 110, 0.3)'}
                        strokeWidth={isHovered ? 2.5 : 1}
                        className="cursor-pointer transition-all"
                        onMouseEnter={() => setHoveredSubj(s)}
                        onMouseLeave={() => setHoveredSubj(null)}
                      />
                    );
                  })}
                </svg>
                <div className="text-[11px] font-mono text-[#59636e] flex justify-between border-t border-[#eaeef2] pt-1.5">
                  <span>{hoveredSubj !== null ? `Hovering Subject ${hoveredSubj}` : 'Hover over a line to highlight'}</span>
                  <span>10 Subjects</span>
                </div>
              </div>
            </div>

            {/* SECTION 3: INDIVIDUAL SUBJECT POSTERIOR MEANS (HISTOGRAM OR 2D SCATTER PLOT) */}
            <div className="bg-white border border-[#d1d9e0] rounded-xl p-6 shadow-2xs space-y-4">
              <div className="pb-3 border-b border-[#d1d9e0] flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                <div>
                  <h2 className="text-base font-bold text-[#1f2328] flex items-center gap-2">
                    <BarChart3 className="w-5 h-5 text-[#1a7f37]" />
                    Individual Subject Posterior Means
                  </h2>
                  <p className="text-xs text-[#59636e]">
                    {selectedParams.length === 1
                      ? `Showing Histogram distribution of subject means for parameter '${primaryParam}'.`
                      : `Showing 2D Scatter plot of subject means: '${primaryParam}' (X) vs '${secondaryParam}' (Y).`}
                  </p>
                </div>
                <div className="text-xs font-mono bg-[#f6f8fa] border border-[#d1d9e0] px-3 py-1 rounded">
                  {selectedParams.length === 1 ? 'Mode: Single Parameter Histogram' : 'Mode: 2-Parameter Subject Scatter'}
                </div>
              </div>

              {/* RENDER HISTOGRAM IF 1 PARAMETER SELECTED */}
              {selectedParams.length === 1 && (
                <div className="border border-[#eaeef2] rounded-lg p-4 bg-white">
                  <div className="text-xs text-[#59636e] mb-2 font-mono flex justify-between">
                    <span>Parameter: <strong>{primaryParam}</strong></span>
                    <span>Range: [{histMin.toFixed(2)}, {histMax.toFixed(2)}]</span>
                  </div>

                  <svg viewBox={`0 0 ${chartW} ${chartH}`} className="w-full h-56 select-none">
                    {/* Bins */}
                    {bins.map((bin, idx) => {
                      const barW = (chartW - pad.left - pad.right) / numBins - 6;
                      const barX = pad.left + idx * ((chartW - pad.left - pad.right) / numBins) + 3;
                      const barH = (bin.count / maxBinCount) * (chartH - pad.top - pad.bottom);
                      const barY = chartH - pad.bottom - barH;
                      return (
                        <g key={idx}>
                          <rect
                            x={barX}
                            y={barY}
                            width={barW}
                            height={barH}
                            fill="#0969da"
                            opacity={0.85}
                            rx={3}
                          />
                          <text
                            x={barX + barW / 2}
                            y={chartH - pad.bottom + 14}
                            textAnchor="middle"
                            fontSize="9"
                            fill="#59636e"
                            fontFamily="monospace"
                          >
                            {bin.start.toFixed(2)}
                          </text>
                          {bin.count > 0 && (
                            <text
                              x={barX + barW / 2}
                              y={barY - 5}
                              textAnchor="middle"
                              fontSize="10"
                              fontWeight="bold"
                              fill="#0969da"
                              fontFamily="monospace"
                            >
                              {bin.count}
                            </text>
                          )}
                        </g>
                      );
                    })}
                  </svg>
                  <p className="text-center text-xs text-[#59636e] mt-1 font-mono">
                    Subject Posterior Mean Bins for '{primaryParam}'
                  </p>
                </div>
              )}

              {/* RENDER 2D SCATTER PLOT IF 2 PARAMETERS SELECTED */}
              {selectedParams.length === 2 && secondaryParam && (
                <div className="border border-[#eaeef2] rounded-lg p-4 bg-white">
                  <div className="text-xs text-[#59636e] mb-2 font-mono flex justify-between">
                    <span>X-Axis: <strong>{primaryParam}</strong></span>
                    <span>Y-Axis: <strong>{secondaryParam}</strong></span>
                  </div>

                  <svg viewBox={`0 0 ${chartW} ${chartH}`} className="w-full h-56 select-none">
                    {/* Grid */}
                    {[0, 0.5, 1.0].map((frac, idx) => {
                      const yPos = pad.top + frac * (chartH - pad.top - pad.bottom);
                      const val = scatYMax - frac * scatYRange;
                      return (
                        <g key={idx}>
                          <line x1={pad.left} y1={yPos} x2={chartW - pad.right} y2={yPos} stroke="#eaeef2" />
                          <text x={pad.left - 8} y={yPos + 4} textAnchor="end" fontSize="10" fill="#8c959f" fontFamily="monospace">
                            {val.toFixed(2)}
                          </text>
                        </g>
                      );
                    })}

                    {/* Subject Dots */}
                    {xParamVals.map((xVal, s) => {
                      const yVal = yParamVals[s];
                      const cx = getScatX(xVal);
                      const cy = getScatY(yVal);
                      const isHovered = hoveredSubj === s;

                      return (
                        <g key={s} onMouseEnter={() => setHoveredSubj(s)} onMouseLeave={() => setHoveredSubj(null)}>
                          <circle
                            cx={cx}
                            cy={cy}
                            r={isHovered ? 7 : 5}
                            fill={isHovered ? '#8250df' : '#0969da'}
                            stroke="#fff"
                            strokeWidth={1.5}
                            className="cursor-pointer transition-all"
                          />
                          <text
                            x={cx}
                            y={cy - 9}
                            textAnchor="middle"
                            fontSize="9"
                            fontFamily="monospace"
                            fill="#1f2328"
                            fontWeight="bold"
                          >
                            S{s}
                          </text>
                        </g>
                      );
                    })}
                  </svg>
                  <p className="text-center text-xs text-[#59636e] mt-1 font-mono">
                    Subject Coordinates: {primaryParam} vs {secondaryParam} (10 Subjects)
                  </p>
                </div>
              )}
            </div>

            {/* SECTION 4: MULTINORMAL CORRELATION / COVARIANCE MATRIX HEATMAP */}
            <div className="bg-white border border-[#d1d9e0] rounded-xl p-6 shadow-2xs space-y-4">
              <div className="pb-3 border-b border-[#d1d9e0]">
                <h2 className="text-base font-bold text-[#1f2328] flex items-center gap-2">
                  <Grid className="w-5 h-5 text-[#cf222e]" />
                  Multinormal Parameter Correlation Matrix Heatmap
                </h2>
                <p className="text-xs text-[#59636e]">
                  Correlation / covariance matrix between all parameters in latent space (for multinormal model estimation).
                </p>
              </div>

              {/* Heatmap Grid */}
              <div className="overflow-x-auto border border-[#eaeef2] rounded-lg p-4 bg-white flex justify-center">
                <table className="border-collapse font-mono text-xs">
                  <thead>
                    <tr>
                      <th className="p-2 text-[#59636e]"></th>
                      {PARAM_KEYS.map((col) => (
                        <th key={col} className="p-2 text-center text-[#1f2328] font-bold">
                          {col}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {PARAM_KEYS.map((rowKey) => (
                      <tr key={rowKey}>
                        <td className="p-2 font-bold text-[#1f2328] text-right">{rowKey}</td>
                        {PARAM_KEYS.map((colKey) => {
                          const r = mockCorMatrix[rowKey][colKey];
                          // Color mapping (-1 to +1)
                          let bg = 'rgba(255, 255, 255, 1)';
                          let textColor = '#1f2328';
                          if (r > 0) {
                            bg = `rgba(9, 105, 218, ${Math.min(1, r * 0.75 + 0.1)})`;
                            if (r > 0.4) textColor = '#ffffff';
                          } else if (r < 0) {
                            bg = `rgba(207, 34, 46, ${Math.min(1, Math.abs(r) * 0.75 + 0.1)})`;
                            if (r < -0.3) textColor = '#ffffff';
                          } else {
                            bg = '#f6f8fa';
                          }

                          return (
                            <td
                              key={colKey}
                              style={{ backgroundColor: bg, color: textColor }}
                              className="p-3 text-center border border-[#eaeef2] font-semibold text-xs transition-colors cursor-pointer"
                              title={`Correlation(${rowKey}, ${colKey}) = ${r.toFixed(2)}`}
                            >
                              {r.toFixed(2)}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* DOCUMENTATION & API TAB */}
        {activeTab === 'docs' && (
          <div className="space-y-6">
            <article className="border border-[#d1d9e0] rounded-md p-8 bg-white shadow-2xs space-y-6 text-sm text-[#1f2328] leading-relaxed">
              <div className="border-b border-[#d1d9e0] pb-4">
                <h1 className="text-3xl font-bold tracking-tight text-[#1f2328]">
                  importance_sampling
                </h1>
                <p className="text-sm text-[#59636e] mt-1.5">
                  A lightweight Python package for fitting computational and cognitive models using <strong>Iterative Importance Sampling (IIS)</strong>.
                </p>
              </div>

              {/* Install callout */}
              <div className="p-3 bg-[#f6f8fa] border border-[#d1d9e0] rounded-md font-mono text-xs flex items-center justify-between">
                <span>{installCmd}</span>
                <button onClick={handleCopy} className="text-[#0969da] hover:underline cursor-pointer">
                  {copied ? 'Copied' : 'Copy'}
                </button>
              </div>

              <div>
                <h2 className="text-xl font-bold border-b border-[#d1d9e0] pb-2 mb-3">
                  How to Use the Package
                </h2>
                <p className="mb-2">To fit a model, you provide three core inputs:</p>
                <ol className="list-decimal pl-6 space-y-2 mb-4">
                  <li><strong>Data (<code>data</code>)</strong>: A list containing datasets for each subject.</li>
                  <li><strong>Model (<code>model</code>)</strong>: A Python function evaluating trial outcomes for a single subject.</li>
                  <li><strong>Parameters &amp; Priors (<code>hyper_params</code>)</strong>: A dictionary of initial latent mean, SD, and link functions.</li>
                </ol>
              </div>

              <div>
                <h2 className="text-xl font-bold border-b border-[#d1d9e0] pb-2 mb-3">
                  Interactive Reporting (<code>sampler.create_report</code>)
                </h2>
                <p className="mb-2">
                  Generate an interactive Plotly report with a single line:
                </p>
                <pre className="bg-[#f6f8fa] border border-[#d1d9e0] p-3 rounded font-mono text-xs">
                  sampler.create_report(filename="model_report.html")
                </pre>
                <ul className="list-disc pl-6 space-y-1.5 text-xs text-[#59636e] mt-2">
                  <li><strong>Hyperparameter Evolution</strong>: Solid line for mean, shaded area for ±1 SD for any number of parameters.</li>
                  <li><strong>Model Fit</strong>: Total evidence, BIC, and subject-level spaghetti trajectories.</li>
                  <li><strong>Individual Subject Means</strong>: Histogram for single parameter or 2D scatter plot for parameter pairs (no raw numbers).</li>
                  <li><strong>Multinormal Correlation Matrix</strong>: Heatmap visualization of parameter covariances in latent space.</li>
                </ul>
              </div>
            </article>
          </div>
        )}
      </main>
    </div>
  );
}
